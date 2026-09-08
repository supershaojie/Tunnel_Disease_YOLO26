"""Executable MPDF graph, initialization, numerical and serialization audits."""

import argparse
import copy
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

import torch
from torch.nn import functional as F

from tools.experiments import b19_common as common
from tools.experiments.mpdf_experiment import V1
from ultralytics import YOLO
from ultralytics.nn.modules.mpdf_p3 import MPDFP3
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import YAML


@contextmanager
def bypass(model):
    """Temporarily return native [U,L] for trained inference diagnosis, preserving all weights."""
    handle = model.model[15].register_forward_hook(lambda m, inputs, output: torch.cat(inputs[0][:2], dim=1))
    try:
        yield
    finally:
        handle.remove()


def audit(baseline, candidate, weights, experiment=V1):
    """Require the only graph change and every additional tensor to belong to node 15."""
    report = common.audit_weights(baseline, candidate, weights, new_prefix="model.15.", layer=15)
    assert len(baseline.model) == len(candidate.model) == 24
    assert [i for i, (a, b) in enumerate(zip(baseline.model, candidate.model)) if type(a) is not type(b)] == [15]
    assert type(candidate.model[15]) is MPDFP3
    for i, (a, b) in enumerate(zip(baseline.model, candidate.model)):
        if i != 15:
            assert a.f == b.f
    assert candidate.model[15].f == [14, 4, 13] and {14, 4, 13} <= set(candidate.save)
    assert candidate.model[14].mode == "nearest" and candidate.model[14].scale_factor == 2
    assert candidate.model[-1].f == [16, 19, 22] and candidate.model[-1].end2end
    assert candidate.stride.tolist() == baseline.stride.tolist() == [8, 16, 32]
    expected = {"model.15." + k for k in ("reduce.weight", "dw.weight", "project.weight", "project.bias")}
    assert set(candidate.state_dict()) - set(baseline.state_dict()) == expected == set(report["new_parameters"])
    assert report["added_parameters"] == 33440
    assert (report["baseline_parameters"], report["candidate_parameters"]) == (2504190, 2537630)
    report.update(
        expected_missing_keys_vs_native_single_class=sorted(expected),
        save=candidate.save,
        dimensions=dict(c_low=128, c_high=128, hidden=32, q=640, coefficients=384, output=256),
    )
    return report


def error_stats(value, scale, path, arithmetic_dtype=None):
    """Record arithmetic error and a scale-aware floating-point roundoff bound, without projection."""
    error = value.detach().double().abs()
    arithmetic_dtype = arithmetic_dtype or value.dtype
    limit = 8 * torch.finfo(arithmetic_dtype).eps * max(float(scale), torch.finfo(arithmetic_dtype).tiny)
    row = dict(
        path=path,
        shape=list(value.shape),
        dtype=str(value.dtype),
        finite=bool(torch.isfinite(value).all()),
        max_abs=float(error.max()),
        mean_abs=float(error.mean()),
        scale=float(scale),
        arithmetic_dtype=str(arithmetic_dtype),
        roundoff_bound=limit,
        outside_tolerance=int((error > limit).sum()),
    )
    assert row["finite"] and row["outside_tolerance"] == 0, row
    return row


def formula_checks(device="cpu", amp=False, dtype=None):
    """Compare independent matrix Haar and indexed inverse references, including two distinct channels."""
    rows = []
    dtype = dtype or (torch.float64 if device == "cpu" else torch.float32)
    for magnitude in (1.0, 1000.0):
        m = MPDFP3(2, 2).to(device=device, dtype=dtype)
        high = torch.randn(2, 2, 3, 5, device=device, dtype=dtype)
        low = torch.randn(2, 2, 6, 10, device=device, dtype=dtype)
        up = F.interpolate(high, scale_factor=2, mode="nearest")
        original = [t.clone() for t in (up, low, high)]
        torch.testing.assert_close(m([up, low, high]), torch.cat((up, low), 1), atol=0, rtol=0)
        # Matrix multiplication independently defines all four orthonormal Haar rows.
        matrix = low.new_tensor([[1, 1, 1, 1], [1, -1, 1, -1], [1, 1, -1, -1], [1, -1, -1, 1]]) / 2
        phases = low.reshape(2, 2, 3, 2, 5, 2).permute(0, 1, 2, 4, 3, 5).reshape(2, 2, 3, 5, 4)
        bands = (phases @ matrix.T).permute(4, 0, 1, 2, 3)
        expected_q = torch.cat([high, *bands], 1)
        saved = {}
        handle = m.reduce.register_forward_pre_hook(lambda module, args: saved.update(q=args[0].clone()))
        coefficients = torch.randn(2, 6, 3, 5, device=device, dtype=dtype) * magnitude
        inject = m.project.register_forward_hook(lambda module, args, output: coefficients.to(output.dtype))
        with torch.autocast(device_type="cuda", enabled=amp):
            result = m([up, low, high])
        handle.remove()
        inject.remove()
        torch.testing.assert_close(saved["q"], expected_q)
        cast = coefficients.half() if amp else coefficients
        # Explicit channel-specific placement, never PixelShuffle in the reference.
        reference = torch.zeros_like(up, dtype=cast.dtype)
        inverse = matrix[1:].to(cast.dtype)
        for channel in range(2):
            terms = cast[:, [channel, 2 + channel, 4 + channel]].permute(0, 2, 3, 1)
            values = terms @ inverse
            for phase in range(4):
                reference[:, channel, phase // 2 :: 2, phase % 2 :: 2] = values[..., phase]
        torch.testing.assert_close(
            result[:, :2],
            up + reference,
            rtol=8 * torch.finfo(cast.dtype).eps,
            atol=8 * torch.finfo(cast.dtype).eps * magnitude,
        )
        torch.testing.assert_close(result[:, 2:], low, atol=0, rtol=0)
        for old, new in zip(original, (up, low, high)):
            torch.testing.assert_close(old, new, atol=0, rtol=0)
        correction = result[:, :2] - up
        rows.append(
            error_stats(
                F.avg_pool2d(correction, 2),
                max(float(reference.abs().max()), float(up.abs().max())),
                "pool(R)",
                cast.dtype,
            )
        )
        rows.append(
            error_stats(
                F.avg_pool2d(result[:, :2], 2) - high,
                max(float(result.abs().max()), float(high.abs().max())),
                "pool(U+R)-H",
                cast.dtype,
            )
        )
    # Isolated Bx, By, Bd use distinct values per channel to expose phase/channel transpositions.
    for band, signs in enumerate(((1, -1, 1, -1), (1, 1, -1, -1), (1, -1, -1, 1))):
        m = MPDFP3(2, 2).double()
        with torch.no_grad():
            m.project.bias[2 * band : 2 * band + 2] = torch.tensor([2.0, 6.0])
        x = torch.zeros(1, 2, 4, 6, dtype=torch.float64)
        y = m([x, x, x[:, :, ::2, ::2]])[:, :2]
        for phase, sign in enumerate(signs):
            expected = y.new_tensor([1.0, 3.0]).view(1, 2, 1, 1).expand(1, 2, 2, 3) * sign
            torch.testing.assert_close(y[:, :, phase // 2 :: 2, phase % 2 :: 2], expected, atol=0, rtol=0)
    return rows


def structural_checks(directory, experiment=V1):
    """Check graph identity, construction RNG, square/rectangular outputs and full backward paths."""
    expected = YAML.load(common.ROOT / "ultralytics/cfg/models/26/yolo26.yaml")
    expected["nc"] = 1
    expected["head"][4] = [[14, 4, 13], 1, "MPDFP3", [32]]
    assert YAML.load(common.ROOT / "ultralytics/cfg/models/26/yolo26n-mpdf-p3-v1.yaml") == expected
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        baseline = DetectionModel(common.baseline_architecture(), verbose=False).eval()
        rng = torch.get_rng_state()
        torch.manual_seed(42)
        candidate = DetectionModel(str(experiment.model), nc=1, verbose=False).eval()
        assert torch.equal(rng, torch.get_rng_state())
        report = audit(baseline, candidate, None)
        report.update(
            constructor_rng_equal=True, formula=formula_checks() + formula_checks(dtype=torch.float32), shapes=[]
        )
        for h, w in ((640, 640), (384, 672)):
            x = torch.randn(1, 3, h, w)
            shapes = {}
            handles = [
                candidate.model[i].register_forward_hook(lambda m, args, out, i=i: shapes.update({i: list(out.shape)}))
                for i in (4, 13, 14, 15, 16)
            ]
            with torch.no_grad():
                common.assert_close_tree(baseline(x), candidate(x), 0, 0)
            for handle in handles:
                handle.remove()
            assert shapes == {
                4: [1, 128, h // 8, w // 8],
                13: [1, 128, h // 16, w // 16],
                14: [1, 128, h // 8, w // 8],
                15: [1, 256, h // 8, w // 8],
                16: [1, 64, h // 8, w // 8],
            }
            candidate.zero_grad(set_to_none=True)
            raw = candidate(x)[1]
            # Native one-to-one features stay detached; the normal one-to-many loss path trains the neck.
            assert all(not t.requires_grad for t in raw["one2one"]["feats"])
            (raw["one2many"]["scores"].square().mean() + raw["one2many"]["boxes"].square().mean()).backward()
            branch = candidate.model[15]
            assert branch.project.weight.grad.isfinite().all() and branch.project.weight.grad.norm() > 0
            assert branch.reduce.weight.grad.count_nonzero() == branch.dw.weight.grad.count_nonzero() == 0
            report["shapes"].append(dict(input=list(x.shape), layers=shapes, initial_output_exact=True, backward=True))
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
    from tools.experiments.run_b19_mpdf_p3 import AuditedTrainer

    probe = object.__new__(AuditedTrainer)
    probe.args = common.get_cfg(overrides=common.REFERENCE["args"])
    probe.data = dict(nc=1, channels=3, names={0: "crack"})
    source, _ = load_checkpoint(weight)
    x = torch.rand(32, 3, 640, 640, generator=torch.Generator().manual_seed(42)).cuda()
    outputs, memory = [], {}
    for name, cfg in (("native", common.baseline_architecture()), ("mpdf", str(V1.model))):
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
    assert snapshot.model[15].project.weight.count_nonzero()
    args = snapshot.args if isinstance(snapshot.args, dict) else vars(snapshot.args)
    path = directory / "preflight.pt"
    torch.save(dict(model=None, ema=snapshot, train_args=args), path)
    snapshot.float()
    with torch.no_grad():
        x = torch.randn(1, 3, 64, 96)
        torch.save(dict(x=x, state=snapshot.state_dict(), raw=snapshot(x)), directory / "reload_reference.pt")
    result = subprocess.run(
        [sys.executable, "-m", "tools.experiments.verify_b19_mpdf_p3", str(path)],
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
    """Verify nonzero trained MPDF survives YOLO loading, fuse, AutoBackend and actual one-to-one inference."""
    from ultralytics.nn.autobackend import AutoBackend

    path = Path(path)
    reference = torch.load(path.with_name("reload_reference.pt"), map_location="cpu", weights_only=False)
    model = YOLO(path).model
    assert type(model.model[15]) is V1.block_type
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
        assert max(differences.values()) > 0, "MPDF must affect actual one-to-one inference"
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
