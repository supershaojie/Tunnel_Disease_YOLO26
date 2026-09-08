"""Executable structure, initialization, branch gradient and serialization audits."""

import argparse
import copy
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

import torch

from tools.experiments import b19_common as common
from tools.experiments.msi_experiment import V1
from ultralytics import YOLO
from ultralytics.nn.modules import SPPF
from ultralytics.nn.modules.msi_c2psa import MSI, PSABlock_MSI
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import YAML


@contextmanager
def bypass(model):
    """Zero only MSI output with a removable hook for post-training inference diagnosis."""
    handles = [
        m.register_forward_hook(lambda m, inputs, output: torch.zeros_like(output))
        for m in model.modules()
        if isinstance(m, MSI)
    ]
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def audit(baseline, candidate, weights, experiment=V1):
    """Require precisely four added tensors and unchanged common tensors, graph and scaling."""
    report = common.audit_weights(baseline, candidate, weights, new_prefix="model.10.m.0.msi.")
    block = candidate.model[10]
    assert type(block) is experiment.block_type and type(candidate.model[9]) is SPPF
    assert [i for i, (a, b) in enumerate(zip(baseline.model, candidate.model)) if type(a) is not type(b)] == [10]
    assert len(baseline.model) == len(candidate.model) == 24
    assert candidate.model[21].f == [-1, 10] and candidate.model[-1].f == [16, 19, 22]
    assert candidate.stride.tolist() == baseline.stride.tolist() == [8, 16, 32]
    assert len(block.m) == 1 and block.c == 128
    attn = block.m[0].attn
    assert (attn.num_heads, attn.key_dim, attn.head_dim) == (2, 32, 64)
    expected = {"model.10.m.0.msi." + k for k in ("dw3.weight", "dw5.weight", "project.weight", "project.bias")}
    assert set(candidate.state_dict()) - set(baseline.state_dict()) == expected == set(report["new_parameters"])
    assert report["added_parameters"] == 20864
    assert report["baseline_parameters"] == 2504190 and report["candidate_parameters"] == 2525054
    report["expected_missing_keys_vs_native_single_class"] = sorted(expected)
    report["dimensions"] = dict(channels=256, half_channels=128, repeats=1, heads=2, ffn_expansion=256)
    report["shared_tensors"] = {
        k: dict(shape=list(v.shape), equal=True, pretrained=k in report["loaded_keys"])
        for k, v in baseline.state_dict().items()
    }
    return report


def branch_checks():
    """Check both shortcut formulas, each identity channel, one BN update and delayed data gradients."""
    rows = []
    for shortcut in (True, False):
        block = PSABlock_MSI(64, num_heads=1, shortcut=shortcut).train()
        msi = block.msi
        x = torch.randn(2, 64, 7, 11)
        for conv in (msi.dw3, msi.dw5):
            torch.testing.assert_close(conv(x), x, atol=0, rtol=0)
        assert not msi.project.weight.count_nonzero() and not msi.project.bias.count_nonzero()
        copy_block = copy.deepcopy(block)
        with torch.no_grad():
            msi.project.weight.normal_(std=0.01)
        copy_block.load_state_dict(block.state_dict())
        count = []
        handle = block.ffn[0].register_forward_hook(lambda *args: count.append(1))
        actual = block(x)
        handle.remove()
        z = x + copy_block.attn(x) if shortcut else copy_block.attn(x)
        u = copy_block.ffn[0](z)
        v = copy_block.ffn[1](u)
        ua, ub = u.chunk(2, 1)
        expected = (z + v if shortcut else v) + copy_block.msi.project(
            torch.nn.functional.gelu(copy_block.msi.dw3(ua), approximate="none") * copy_block.msi.dw5(ub)
        )
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        assert count == [1]
        assert all(m.num_batches_tracked.item() == 1 for m in block.modules() if isinstance(m, torch.nn.BatchNorm2d))
        rows.append(dict(shortcut=shortcut, formula_exact=True, bn_updates=1, expansion_calls=1))
    block = PSABlock_MSI(64, num_heads=1).train()
    optimizer = torch.optim.SGD(block.parameters(), lr=0.01, weight_decay=0)
    gradients = []
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        (block(x) * torch.randn_like(x)).mean().backward()
        row = {k: float(p.grad.norm()) for k, p in block.msi.named_parameters()}
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in block.msi.parameters())
        assert row["project.weight"] > 0
        assert (
            (row["dw3.weight"] == row["dw5.weight"] == 0)
            if step == 0
            else (row["dw3.weight"] > 0 and row["dw5.weight"] > 0)
        )
        gradients.append(row)
        optimizer.step()
    return dict(formula=rows, data_gradients_without_weight_decay=gradients)


def structural_checks(directory, experiment=V1):
    """Check exact shared initialization/RNG and complete square and stride-aligned rectangular outputs."""
    expected = YAML.load(common.ROOT / "ultralytics/cfg/models/26/yolo26.yaml")
    expected["nc"] = 1
    expected["backbone"][10][2] = "C2PSA_MSI"
    assert YAML.load(experiment.model) == expected
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        baseline = DetectionModel(common.baseline_architecture(), verbose=False).eval()
        rng = torch.get_rng_state()
        torch.manual_seed(42)
        candidate = DetectionModel(str(experiment.model), nc=1, verbose=False).eval()
        assert torch.equal(rng, torch.get_rng_state())
        report = audit(baseline, candidate, None)
        report.update(constructor_rng_equal=True, branch=branch_checks(), shapes=[])
        for h, w in ((640, 640), (384, 672)):
            x = torch.randn(1, 3, h, w)
            shape = []
            handle = candidate.model[10].register_forward_hook(
                lambda m, inputs, output: shape.append(list(output.shape))
            )
            with torch.no_grad():
                common.assert_close_tree(baseline(x), candidate(x), 0, 0)
            handle.remove()
            assert shape == [[1, 256, h // 32, w // 32]]
            report["shapes"].append(dict(input=list(x.shape), layer10=shape[0], initial_output_exact=True))
    common.write_json(Path(directory) / "structural.json", report)
    print({k: report[k] for k in ("baseline_parameters", "candidate_parameters", "added_parameters", "dimensions")})
    return report


@torch.no_grad()
def gpu_initialization_check(weight, directory):
    """Compare untrained pretrained models at batch=32, loading and releasing them sequentially on GPU."""
    import gc

    from ultralytics.models.yolo.detect import DetectionTrainer
    from ultralytics.nn.tasks import load_checkpoint
    from ultralytics.utils.torch_utils import init_seeds
    from tools.experiments.run_b19_msi_c2psa import AuditedTrainer

    probe = object.__new__(AuditedTrainer)
    probe.args = common.get_cfg(overrides=common.REFERENCE["args"])
    probe.data = dict(nc=1, channels=3, names={0: "crack"})
    source, _ = load_checkpoint(weight)
    x = torch.rand(32, 3, 640, 640, generator=torch.Generator().manual_seed(42)).cuda()
    outputs, memory = [], {}
    for name, cfg in (("native", common.baseline_architecture()), ("msi", str(V1.model))):
        init_seeds(42, deterministic=True)
        model = DetectionTrainer.get_model(probe, cfg, source, False).cuda().float().eval()
        torch.cuda.reset_peak_memory_stats()
        result = model(x)
        outputs.append({k: result[1]["one2one"][k].cpu() for k in ("boxes", "scores")})
        memory[name] = dict(
            peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved()
        )
        del result, model
        gc.collect()
        torch.cuda.empty_cache()
    comparisons = []
    try:
        common.assert_close_tree(*outputs, 0, 0, report=comparisons)
    finally:
        common.write_json(
            Path(directory) / "initialization_gpu.json",
            dict(
                batch=32,
                imgsz=640,
                dtype="FP32",
                trained=False,
                comparisons=comparisons,
                sequential_gpu_models=True,
                memory=memory,
                conditions=common.computation_conditions(),
            ),
        )


def save_reload_check(model, directory, experiment=V1):
    """Serialize native FP16 EMA and compare its quantized reference in a fresh YOLO process."""
    directory = Path(directory)
    snapshot = copy.deepcopy(model).cpu().half().eval()
    snapshot.criterion = None
    assert snapshot.model[10].m[0].msi.project.weight.count_nonzero()
    args = snapshot.args if isinstance(snapshot.args, dict) else vars(snapshot.args)
    path = directory / "preflight.pt"
    torch.save(dict(model=None, ema=snapshot, train_args=args), path)
    snapshot.float()
    with torch.no_grad():
        x = torch.randn(1, 3, 64, 96)
        torch.save(dict(x=x, state=snapshot.state_dict(), raw=snapshot(x)), directory / "reload_reference.pt")
    result = subprocess.run(
        [sys.executable, "-m", "tools.experiments.verify_b19_msi_c2psa", str(path)],
        cwd=common.ROOT,
        env={**os.environ, "PYTHONPATH": str(common.ROOT)},
        capture_output=True,
        text=True,
    )
    (directory / "reload.log").write_text(result.stdout + result.stderr, encoding="utf-8")
    result.check_returncode()
    return dict(
        checkpoint=str(path),
        sha256=common.sha256(path),
        fresh_process=True,
        checkpoint_precision="FP16 EMA, reloaded FP32",
        state_exact=True,
        fuse=True,
    )


def reload_in_process(path):
    """Verify nonzero trained MSI survives YOLO loading, fuse, AutoBackend and actual one-to-one inference."""
    from ultralytics.nn.autobackend import AutoBackend

    path = Path(path)
    reference = torch.load(path.with_name("reload_reference.pt"), map_location="cpu", weights_only=False)
    model = YOLO(path).model
    assert type(model.model[10]) is V1.block_type
    common.assert_close_tree(reference["state"], model.state_dict(), 0, 0)
    report = []
    with torch.no_grad():
        before = model(reference["x"])
        common.assert_close_tree(reference["raw"], before, 0, 0, report=report)
        fused = copy.deepcopy(model).fuse(verbose=False)
        after = fused(reference["x"])
        assert after[1]["one2many"] == {}
        common.assert_close_tree(before[1]["one2one"], after[1]["one2one"], 1e-4, 1e-4, report=report)
        head = fused.model[-1]
        common.assert_close_tree(
            head._inference(before[1]["one2one"]), head._inference(after[1]["one2one"]), 1e-4, 1e-4, report=report
        )
        backend = AutoBackend(model=copy.deepcopy(model), device=torch.device("cpu"), fp16=False)
        common.assert_close_tree(list(after), backend(reference["x"]), 0, 0, report=report)
        with bypass(fused):
            disabled = fused(reference["x"])
        differences = {
            k: float((after[1]["one2one"][k] - disabled[1]["one2one"][k]).abs().max()) for k in ("boxes", "scores")
        }
        assert max(differences.values()) > 0, "MSI must affect actual one-to-one inference"
    common.write_json(
        path.with_name("reload_check.json"),
        dict(
            passed=True,
            comparisons=report,
            one2one_bypass_max_abs=differences,
            conditions=common.computation_conditions(),
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    reload_in_process(parser.parse_args().checkpoint)
