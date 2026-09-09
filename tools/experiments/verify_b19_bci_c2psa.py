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
from tools.experiments.bci_experiment import V1
from ultralytics import YOLO
from ultralytics.nn.modules import SPPF
from ultralytics.nn.modules.bci_c2psa import BCI
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import YAML


@contextmanager
def bypass(model):
    """Zero only BCI output with a removable hook for post-training inference diagnosis."""
    handles = [
        m.register_forward_hook(lambda m, inputs, output: torch.zeros_like(output))
        for m in model.modules()
        if isinstance(m, BCI)
    ]
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def audit(baseline, candidate, weights, experiment=V1):
    """Require precisely six added tensors and unchanged common tensors, graph and scaling."""
    report = common.audit_weights(baseline, candidate, weights, new_prefix="model.10.bci.")
    block = candidate.model[10]
    assert type(block) is experiment.block_type and type(candidate.model[9]) is SPPF
    assert [i for i, (a, b) in enumerate(zip(baseline.model, candidate.model)) if type(a) is not type(b)] == [10]
    assert len(baseline.model) == len(candidate.model) == 24
    assert candidate.model[21].f == [-1, 10] and candidate.model[-1].f == [16, 19, 22]
    assert candidate.stride.tolist() == baseline.stride.tolist() == [8, 16, 32]
    assert candidate.model[-1].reg_max == 1 and candidate.model[-1].end2end
    assert len(block.m) == 1 and block.c == 128
    attn = block.m[0].attn
    assert (attn.num_heads, attn.key_dim, attn.head_dim) == (2, 32, 64)
    expected = {
        "model.10.bci." + k for k in ("pq.weight", "dwq.weight", "pk.weight", "dwk.weight", "pv.weight", "po.weight")
    }
    assert set(candidate.state_dict()) - set(baseline.state_dict()) == expected == set(report["new_parameters"])
    assert report["added_parameters"] == 16960
    assert report["baseline_parameters"] == 2504190 and report["candidate_parameters"] == 2521150
    report["expected_missing_keys_vs_native_single_class"] = sorted(expected)
    report["dimensions"] = {"channels": 256, "half_channels": 128, "repeats": 1, "heads": 2, "ffn_expansion": 256}
    report["shared_tensors"] = {
        k: {"shape": list(v.shape), "equal": True, "pretrained": k in report["loaded_keys"]}
        for k, v in baseline.state_dict().items()
    }
    report["pretrained_matched_keys"] = len(report.pop("loaded_keys"))
    return report


def branch_checks():
    """Compare the production math against an independent float64 scalar reference."""
    import math

    q = torch.tensor([[[1.0, 2.0, -3.0, 4.0, 0.0], [3.0, -2.0, 5.0, 1.0, 8.0], [-1.0, 4.0, 2.0, 9.0, 0.0]]])
    k = torch.tensor([[[4.0, -2.0, 7.0, 1.0, 3.0], [8.0, 0.0, 2.0, -3.0, 4.0], [5.0, 2.0, -1.0, 3.0, 7.0]]])
    v = torch.arange(15).reshape(1, 3, 5).float().square() / 7

    def reference(q, k, v):
        qs, ks = [], []
        for source, dest in ((q[0].double().tolist(), qs), (k[0].double().tolist(), ks)):
            for row in source:
                centered = [x - sum(row) / len(row) for x in row]
                denom = max(math.sqrt(sum(x * x for x in centered)), 1e-6)
                dest.append([x / denom for x in centered])
        att = []
        for row in qs:
            logits = [4 * sum(x * y for x, y in zip(row, col)) for col in ks]
            exps = [math.exp(x - max(logits)) for x in logits]
            att.append([x / sum(exps) for x in exps])
        delta = [
            [sum(att[i][j] * float(v[0, j, n]) for j in range(3)) - float(v[0, i, n]) for n in range(5)]
            for i in range(3)
        ]
        return torch.tensor([delta]), torch.tensor([att])

    expected, expected_a = reference(q, k, v)
    delta, attention, _ = BCI.interaction(q, k, v)
    torch.testing.assert_close(delta, expected, atol=3e-6, rtol=1e-6)
    torch.testing.assert_close(attention, expected_a, atol=2e-7, rtol=1e-6)
    assert attention.shape == (1, 3, 3)
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        for magnitude in (0.0, 1e-7):
            low_q = (torch.ones_like(q) + magnitude * q).to(dtype)
            low_k = (torch.ones_like(k) + magnitude * k).to(dtype)
            d, a, _ = BCI.interaction(low_q, low_k, v.to(dtype))
            assert torch.isfinite(d).all() and torch.isfinite(a).all()
            torch.testing.assert_close(a.sum(-1), torch.ones(1, 3), atol=1e-7, rtol=1e-6)
    branch = BCI()
    a, b = torch.randn(2, 128, 7, 11), torch.randn(2, 128, 7, 11)
    old_a, old_b = a.clone(), b.clone()
    assert branch(a, b).shape == b.shape and not branch(a, b).count_nonzero()
    with torch.no_grad():
        branch.po.weight.normal_(std=0.01)
    result = branch(a, b)
    changed = branch(a + torch.randn_like(a), b)
    assert not torch.equal(result, changed)
    assert torch.equal(a, old_a) and torch.equal(b, old_b)
    for training in (True, False):
        from ultralytics.nn.modules import C2PSA, C2PSA_BCI

        native = C2PSA(256, 256).train(training)
        candidate = C2PSA_BCI(256, 256).train(training)
        candidate.load_state_dict(native.state_dict(), strict=False)
        x = torch.randn(2, 256, 7, 11)
        torch.testing.assert_close(native(x), candidate(x), atol=0, rtol=0)
    return {
        "independent_float64_reference": True,
        "constant_low_precision_finite": True,
        "rectangular": True,
        "bypass_conditioned": True,
        "native_train_eval_exact": True,
    }


def structural_checks(directory, experiment=V1):
    """Check exact shared initialization/RNG and complete square and stride-aligned rectangular outputs."""
    expected = YAML.load(common.ROOT / "ultralytics/cfg/models/26/yolo26.yaml")
    expected["nc"] = 1
    expected["backbone"][10][2] = "C2PSA_BCI"
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
        for h, w in ((640, 640), (640, 960)):
            x = torch.randn(1, 3, h, w)
            shapes = {}
            expected_nodes = {
                4: (128, 8),
                9: (256, 32),
                10: (256, 32),
                13: (128, 16),
                14: (128, 8),
                15: (256, 8),
                16: (64, 8),
                17: (64, 16),
                19: (128, 16),
                20: (128, 32),
                22: (256, 32),
            }
            handles = [
                candidate.model[index].register_forward_hook(
                    lambda m, inputs, output, index=index, shapes=shapes: shapes.__setitem__(index, list(output.shape))
                )
                for index in expected_nodes
            ]
            try:
                with torch.no_grad():
                    common.assert_close_tree(baseline(x), candidate(x), 0, 0)
            finally:
                for handle in handles:
                    handle.remove()
            assert shapes == {index: [1, c, h // stride, w // stride] for index, (c, stride) in expected_nodes.items()}
            report["shapes"].append({"input": list(x.shape), "nodes": shapes, "initial_output_exact": True})
        with torch.no_grad():
            x = torch.randn(2, 3, 64, 96)
            common.assert_close_tree(copy.deepcopy(baseline).train()(x), copy.deepcopy(candidate).train()(x), 0, 0)
        report["full_network_train_initial_exact"] = True
    common.write_json(Path(directory) / "structural.json", report)
    print({k: report[k] for k in ("baseline_parameters", "candidate_parameters", "added_parameters", "dimensions")})
    return report


@torch.no_grad()
def gpu_initialization_check(weight, directory):
    """Compare untrained pretrained models at batch=32, loading and releasing them sequentially on GPU."""
    import gc

    from tools.experiments.run_b19_bci_c2psa import AuditedTrainer
    from ultralytics.models.yolo.detect import DetectionTrainer
    from ultralytics.nn.tasks import load_checkpoint
    from ultralytics.utils.torch_utils import init_seeds

    probe = object.__new__(AuditedTrainer)
    probe.args = common.get_cfg(overrides=common.REFERENCE["args"])
    probe.data = {"nc": 1, "channels": 3, "names": {0: "crack"}}
    source, _ = load_checkpoint(weight)
    x = torch.rand(32, 3, 640, 640, generator=torch.Generator().manual_seed(42)).cuda()
    outputs, memory = [], {}
    for name, cfg in (("native", common.baseline_architecture()), ("bci", str(V1.model))):
        init_seeds(42, deterministic=True)
        model = DetectionTrainer.get_model(probe, cfg, source, False).cuda().float().eval()
        torch.cuda.reset_peak_memory_stats()
        result = model(x)

        def cpu(value):
            if isinstance(value, torch.Tensor):
                return value.cpu()
            if isinstance(value, dict):
                return {k: cpu(v) for k, v in value.items()}
            if isinstance(value, (tuple, list)):
                return type(value)(cpu(v) for v in value)
            return value

        outputs.append(cpu(result))
        memory[name] = {
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        }
        del result, model
        gc.collect()
        torch.cuda.empty_cache()
    comparisons = []
    try:
        common.assert_close_tree(*outputs, 0, 0, report=comparisons)
    finally:
        common.write_json(
            Path(directory) / "initialization_gpu.json",
            {
                "batch": 32,
                "imgsz": 640,
                "dtype": "FP32",
                "trained": False,
                "comparisons": comparisons,
                "sequential_gpu_models": True,
                "memory": memory,
                "conditions": common.computation_conditions(),
            },
        )


def save_reload_check(model, directory, experiment=V1):
    """Serialize native FP16 EMA and compare its quantized reference in a fresh YOLO process."""
    directory = Path(directory)
    snapshot = copy.deepcopy(model).cpu().half().eval()
    snapshot.criterion = None
    assert snapshot.model[10].bci.po.weight.count_nonzero()
    args = snapshot.args if isinstance(snapshot.args, dict) else vars(snapshot.args)
    path = directory / "preflight.pt"
    torch.save({"model": None, "ema": snapshot, "train_args": args}, path)
    snapshot.float()
    with torch.no_grad():
        x = torch.randn(1, 3, 64, 96)
        torch.save(
            {"x": x, "state": snapshot.state_dict(), "raw": snapshot(x), "threads": torch.get_num_threads()},
            directory / "reload_reference.pt",
        )
    result = subprocess.run(
        [sys.executable, "-m", "tools.experiments.verify_b19_bci_c2psa", str(path)],
        cwd=common.ROOT,
        env={**os.environ, "PYTHONPATH": str(common.ROOT)},
        check=False,
        capture_output=True,
        text=True,
    )
    (directory / "reload.log").write_text(result.stdout + result.stderr, encoding="utf-8")
    result.check_returncode()
    return {
        "checkpoint": str(path),
        "sha256": common.sha256(path),
        "fresh_process": True,
        "checkpoint_precision": "FP16 EMA, reloaded FP32",
        "state_exact": True,
        "fuse": True,
    }


def reload_in_process(path):
    """Verify nonzero trained BCI survives YOLO loading, fuse, AutoBackend and actual one-to-one inference."""
    from ultralytics.nn.autobackend import AutoBackend

    path = Path(path)
    reference = torch.load(path.with_name("reload_reference.pt"), map_location="cpu", weights_only=False)
    torch.set_num_threads(reference["threads"])
    model = YOLO(path).model
    assert type(model.model[10]) is V1.block_type
    common.assert_close_tree(reference["state"], model.state_dict(), 0, 0)
    report = []
    with torch.no_grad():
        before = model(reference["x"])
        common.assert_close_tree(reference["raw"], before, 0, 0, report=report)
        native = DetectionModel(common.baseline_architecture(), verbose=False).eval()
        native.load_state_dict({k: v for k, v in model.state_dict().items() if ".bci." not in k}, strict=True)
        native_before = native(reference["x"])
        native_fused = copy.deepcopy(native).fuse(verbose=False)
        native_after = native_fused(reference["x"])
        common.assert_close_tree(native_before[1]["one2one"], native_after[1]["one2one"], 1e-4, 1e-4, report=report)
        common.assert_close_tree(
            native_fused.model[-1]._inference(native_before[1]["one2one"]),
            native_fused.model[-1]._inference(native_after[1]["one2one"]),
            1e-4,
            1e-4,
            report=report,
        )
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
        assert max(differences.values()) > 0, "BCI must affect actual one-to-one inference"
    common.write_json(
        path.with_name("reload_check.json"),
        {
            "passed": True,
            "comparisons": report,
            "one2one_bypass_max_abs": differences,
            "conditions": common.computation_conditions(),
        },
    )


def local_validation(data, weight, directory):
    """Run development checks using real labeled images, never issue a server preflight receipt."""
    from tools.experiments.finish_b19_bci_c2psa import evaluate, fixed_subset
    from tools.experiments.run_b19_bci_c2psa import AuditedTrainer
    from ultralytics.data import build_dataloader
    from ultralytics.data.dataset import YOLODataset
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.nn.tasks import load_checkpoint
    from ultralytics.utils.torch_utils import ModelEMA, init_seeds

    torch.set_num_threads(4)
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    report = {"server_preflight": False, "formal_training": False, "checks": {}}
    try:
        structural_checks(directory)
        source, _ = load_checkpoint(weight)
        assert common.sha256(weight) == common.PRETRAINED_SHA256
        probe = object.__new__(AuditedTrainer)
        probe.experiment = V1
        probe.args = common.get_cfg(overrides=common.REFERENCE["args"])
        probe.data = {"nc": 1, "channels": 3, "names": {0: "crack"}}
        init_seeds(42, deterministic=True)
        model = probe.get_model(str(V1.model), source, False)
        common.write_json(directory / "weights.json", probe.weight_audit)
        model.args = probe.args
        subset, images = fixed_subset(data, directory / "real_val", 2)
        import shutil

        copied = []
        for image in images:
            destination = directory / "isolated_data/images" / image.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(image, destination)
            label = Path(str(image).replace(f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}")).with_suffix(".txt")
            target = directory / "isolated_data/labels" / label.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(label, target)
            copied.append(destination)
        subset.with_suffix(".txt").write_text("\n".join(map(str, copied)) + "\n")
        dataset = YOLODataset(
            img_path=str(subset.with_suffix(".txt")),
            imgsz=96,
            batch_size=2,
            augment=False,
            hyp=probe.args,
            rect=False,
            data=check_det_dataset(str(subset)),
            task="detect",
        )
        loader = build_dataloader(dataset, batch=2, workers=0, shuffle=False)
        batch = next(iter(loader))
        loader.close()
        batch["img"] = batch["img"].float() / 255
        optimizer = probe.build_optimizer(model, "MuSGD", lr=0.01, momentum=0.937, decay=0.0005, iterations=52600)
        params = dict(model.model[10].bci.named_parameters())
        groups = []
        for name, p in params.items():
            memberships = [g for g in optimizer.param_groups for q in g["params"] if q is p]
            assert len(memberships) == 1 and memberships[0]["use_muon"] and memberships[0]["lr"] == 0.01
            groups.append({"name": name, "group": memberships[0]["param_group"], "lr": 0.01})
        ema = ModelEMA(model)
        rows = []
        effective = set()
        model.train()
        for step in range(8):
            optimizer.zero_grad()
            loss, _ = model(batch)
            loss.sum().backward()
            assert torch.isfinite(loss).all()
            grads = {k: float(p.grad.norm()) for k, p in params.items()}
            assert all(torch.isfinite(p.grad).all() for p in params.values())
            if step == 0:
                assert grads["po.weight"] > 0 and all(v == 0 for k, v in grads.items() if k != "po.weight")
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            old = {k: p.detach().clone() for k, p in params.items()}
            optimizer.step()
            ema.update(model)
            changes = {k: float((p - old[k]).abs().max()) for k, p in params.items()}
            effective.update(k for k in params if grads[k] > 0 and changes[k] > 0)
            rows.append({"step": step + 1, "loss": loss.detach().tolist(), "task_gradients": grads, "updates": changes})
            if len(effective) == 6:
                break
        assert len(effective) == 6
        report["checks"]["real_detection_musgd"] = {
            "batch": 2,
            "imgsz": 96,
            "steps": rows,
            "groups": groups,
            "recipe": "development fixed learning rate; not native warmup server preflight",
            "real_images": list(map(str, images)),
        }
        ema.update_attr(model, include=["yaml", "nc", "args", "names", "stride"])
        report["checks"]["reload_fuse"] = save_reload_check(ema.ema, directory)
        report["checks"]["validator"] = evaluate(
            directory / "preflight.pt", subset, directory / "validator", "val", device="cpu", batch=2, workers=0
        )
        del optimizer, ema
        if torch.cuda.is_available():
            cuda_rows = []
            original = probe.get_model(str(V1.model), source, False).cuda()
            original.args = probe.args
            for height, width in ((640, 640), (640, 960)):
                for amp in (False, True):
                    test = copy.deepcopy(original).train()
                    real = dict(batch)
                    real["img"] = torch.nn.functional.interpolate(batch["img"][:1], size=(height, width)).cuda()
                    mask = batch["batch_idx"] == 0
                    for key in ("cls", "bboxes", "batch_idx"):
                        real[key] = batch[key][mask].cuda()
                    torch.cuda.reset_peak_memory_stats()
                    with torch.autocast("cuda", enabled=amp):
                        loss, _ = test(real)
                    loss.sum().backward()
                    assert torch.isfinite(loss).all()
                    assert all(
                        p.grad is not None and torch.isfinite(p.grad).all() for p in test.model[10].bci.parameters()
                    )
                    cuda_rows.append(
                        {
                            "shape": [1, 3, height, width],
                            "amp": amp,
                            "loss": loss.detach().tolist(),
                            "po_gradient": float(test.model[10].bci.po.weight.grad.norm()),
                            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                        }
                    )
                    del test, loss
            from torch.utils._python_dispatch import TorchDispatchMode

            class ObserveMath(TorchDispatchMode):
                def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                    if str(func) in ("aten.bmm.default", "aten._softmax.default"):
                        assert args[0].dtype == torch.float32
                        self.seen.append(str(func))
                    return func(*args, **(kwargs or {}))

            math = ObserveMath()
            math.seen = []
            with math, torch.autocast("cuda"):
                q = torch.randn(1, 32, 35, device="cuda", dtype=torch.float16)
                d, att, corr = BCI.interaction(q, q.roll(1, -1), q)
                assert d.dtype == att.dtype == corr.dtype == torch.float32
            assert math.seen.count("aten.bmm.default") == 2 and "aten._softmax.default" in math.seen
            report["checks"]["cuda"] = cuda_rows
            report["checks"]["autocast_fp32_math"] = math.seen
        report["backend"] = common.computation_conditions()
        report["flops_profiler_gflops"] = {}
        for name, cfg in (("native", common.baseline_architecture()), ("bci", str(V1.model))):
            measured = DetectionModel(cfg, nc=1, verbose=False).eval()
            with torch.no_grad(), torch.profiler.profile(with_flops=True) as profiler:
                measured(torch.randn(1, 3, 640, 640))
            report["flops_profiler_gflops"][name] = sum(row.flops for row in profiler.key_averages()) / 1e9
        report["flops_scope"] = (
            "Actual 1x3x640x640, unfused eval, operators counted by PyTorch profiler; not all scalar operations"
        )
        report["passed"] = True
    except Exception:
        import traceback

        report["passed"] = False
        report["failure"] = traceback.format_exc()
        raise
    finally:
        common.write_json(directory / "local_validation.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, nargs="?")
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--data", type=Path)
    parser.add_argument("--weight", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.local:
        local_validation(args.data, args.weight, args.output)
    elif args.checkpoint:
        reload_in_process(args.checkpoint)
    else:
        parser.error("Provide checkpoint or --local --data --weight --output")
