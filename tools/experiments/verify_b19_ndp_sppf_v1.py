"""NDP formula, graph, RNG, checkpoint and native lifecycle validation."""

import argparse
import copy
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

import torch

from tools.experiments import b19_common as common
from tools.experiments.ndp_experiment import V1
from ultralytics import YOLO
from ultralytics.nn.modules import SPPF, SPPF_NDP
from ultralytics.nn.tasks import DetectionModel, load_checkpoint
from ultralytics.utils import YAML


@contextmanager
def bypass(model):
    """Disable only the added residual using a temporary final-projection output hook."""
    handles = [
        m.ndp_out.register_forward_hook(lambda m, inputs, output: torch.zeros_like(output))
        for m in model.modules()
        if isinstance(m, SPPF_NDP)
    ]
    try:
        yield
    finally:
        for h in handles:
            h.remove()


def audit(baseline, candidate, weights, experiment=V1):
    """Permit only the layer-9 class and its two added projections to differ."""
    report = common.audit_weights(baseline, candidate, weights, new_prefix="model.9.ndp_")
    assert len(candidate.model) == len(baseline.model) == 24
    assert [i for i, (a, b) in enumerate(zip(baseline.model, candidate.model)) if type(a) is not type(b)] == [9]
    m = candidate.model[9]
    assert type(m) is experiment.block_type and m.n == 3 and m.add
    assert isinstance(m.cv1.act, torch.nn.Identity)
    assert (m.ndp_in.in_channels, m.ndp_in.out_channels, m.ndp_out.out_channels) == (128, 16, 256)
    assert candidate.model[-1].f == baseline.model[-1].f == [16, 19, 22]
    assert candidate.model[-1].end2end and candidate.model[-1].nc == 1
    assert candidate.stride.tolist() == baseline.stride.tolist() == [8, 16, 32]
    assert all(a.f == b.f for a, b in zip(baseline.model, candidate.model))
    expected = {"model.9.ndp_in.weight", "model.9.ndp_out.weight"}
    assert set(report["new_parameters"]) == set(candidate.state_dict()) - set(baseline.state_dict()) == expected
    assert report["added_parameters"] == 14336
    assert (report["baseline_parameters"], report["candidate_parameters"]) == (2504190, 2518526)
    report["shared_tensors"] = {
        k: {"shape": list(v.shape), "equal": True, "pretrained": k in report["loaded_keys"]}
        for k, v in baseline.state_dict().items()
    }
    report["pretrained_matched_keys"] = len(report.pop("loaded_keys"))
    return report


def branch_checks():
    """Compare independent FP64 valid-window loops, mathematical properties and dispatched dtypes."""
    import math
    from ultralytics.nn.modules.ndp_sppf import ndp_window
    from torch.utils._python_dispatch import TorchDispatchMode

    def reference(u, k):
        rows = []
        for y in range(u.shape[-2]):
            cols = []
            for x in range(u.shape[-1]):
                v = (
                    u[..., max(0, y - k // 2) : y + k // 2 + 1, max(0, x - k // 2) : x + k // 2 + 1]
                    .double()
                    .flatten(-2)
                )
                d = v - v.mean(-1, keepdim=True)
                score = 2 * torch.tanh(d / (d.square().mean(-1, keepdim=True) + 1e-4).sqrt())
                cols.append((score.softmax(-1) * d).sum(-1))
            rows.append(torch.stack(cols, -1))
        return torch.stack(rows, -2)

    errors, grads = [], []
    for shape in ((2, 3, 3, 7), (1, 2, 9, 4), (1, 1, 1, 1)):
        for k in (5, 9, 13):
            u = (torch.arange(math.prod(shape)).reshape(shape).float().sin()).requires_grad_()
            actual, stats = ndp_window(u, k, diagnostics=True)
            expected = reference(u, k)
            torch.testing.assert_close(actual.double(), expected, atol=2e-6, rtol=2e-5)
            ga = torch.autograd.grad(actual.sum(), u, retain_graph=True)[0]
            gr = torch.autograd.grad(expected.sum(), u)[0]
            torch.testing.assert_close(ga, gr, atol=3e-6, rtol=3e-5)
            assert torch.isfinite(ga).all()
            if shape[-2:] != (1, 1):
                assert ga.norm() > 0
            torch.testing.assert_close(ndp_window(u.detach() + 11, k), actual.detach(), atol=3e-6, rtol=2e-5)
            torch.testing.assert_close(ndp_window(torch.full_like(u, 7), k), torch.zeros_like(u), atol=1e-6, rtol=0)
            torch.testing.assert_close(stats["weight_sum"], torch.ones_like(u))
            assert stats["weight_ratio"].max() <= math.exp(4) * (1 + 1e-6)
            errors.append({"shape": shape, "k": k, "max_abs": float((actual - expected).abs().max())})
            grads.append({"shape": shape, "k": k, "max_abs": float((ga - gr).abs().max())})
    examples = []
    for count, target in ((1, 0.275157), (8, 0.583010)):
        u = torch.zeros(1, 1, 5, 5)
        u.flatten()[:count] = 1
        e = ndp_window(u, 5)[0, 0, 2, 2]
        assert abs(float(e) - target) < 1e-6
        permuted = u.flatten()[torch.randperm(25)].reshape_as(u)
        torch.testing.assert_close(ndp_window(permuted, 5)[0, 0, 2, 2], e)
        examples.append(float(e))
    dtype_rows = []

    class Observe(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if any(t in str(func) for t in ("im2col", "sum.dim", "sqrt", "tanh", "_softmax")):
                self.rows.append((str(func), str(args[0].dtype)))
            return func(*args, **(kwargs or {}))

    for device, dtype in [("cpu", torch.bfloat16)] + ([("cuda", torch.float16)] if torch.cuda.is_available() else []):
        observer = Observe()
        observer.rows = []
        u = torch.randn(1, 2, 4, 7, device=device, dtype=dtype, requires_grad=True)
        with observer, torch.autocast(device_type=device, dtype=dtype):
            out = ndp_window(u, 13)
        out.float().sum().backward()
        assert out.dtype == dtype and torch.isfinite(out).all() and torch.isfinite(u.grad).all()
        assert all(
            d == "torch.float32" for op, d in observer.rows if not op.startswith("aten.sum.dim") or d != "torch.bool"
        )
        dtype_rows.append({"device": device, "output_dtype": str(out.dtype), "operations": observer.rows})
    stress = []
    for scale in (0.0, 1e6, 1e15, 1e30):
        u = (torch.tensor([[-1.0, 1.0], [0.5, -0.5]]) * scale)[None, None].requires_grad_()
        out = ndp_window(u, 13)
        out.sum().backward()
        row = {
            "scale": scale,
            "output_finite": bool(torch.isfinite(out).all()),
            "gradient_finite": bool(torch.isfinite(u.grad).all()),
            "centered_variance_finite": bool(torch.isfinite((u.detach() - u.detach().mean()).square().mean())),
            "reference_max_abs_error": float((out.double() - reference(u.detach(), 13)).abs().max()),
        }
        if scale <= 1e15:
            assert row["output_finite"] and row["gradient_finite"]
        # FP32 centered-square overflow at extreme finite magnitudes is reported, never repaired with nan_to_num.
        stress.append(row)
    m = SPPF_NDP(16, 16, 5, 3, True).train()
    native = SPPF(16, 16, 5, 3, True).train()
    native.load_state_dict({k: v for k, v in m.state_dict().items() if not k.startswith("ndp_")})
    calls = []
    hook = m.cv1.register_forward_hook(lambda *args: calls.append(1))
    x = torch.randn(2, 16, 3, 7)
    torch.testing.assert_close(m(x), native(x), atol=1e-6, rtol=1e-5)
    hook.remove()
    assert len(calls) == 1
    for k, v in native.state_dict().items():
        torch.testing.assert_close(m.state_dict()[k], v)
    return {
        "reference": errors,
        "gradient_reference": grads,
        "examples": examples,
        "fp32_dispatch": dtype_rows,
        "stress": stress,
        "cv1_calls": len(calls),
        "optimization": "unblocked sequential scales; no recomputation or mask cache",
    }


def structural_checks(directory, experiment=V1):
    """Compare native train/eval outputs and the complete 640 and rectangular feature graph."""
    expected = YAML.load(common.ROOT / "ultralytics/cfg/models/26/yolo26.yaml")
    expected["nc"] = 1
    expected["backbone"][9][2] = "SPPF_NDP"
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
            comparisons = []
            nodes = {9: (256, 32), 10: (256, 32), 16: (64, 8), 19: (128, 16), 22: (256, 32)}
            handles = [
                candidate.model[i].register_forward_hook(
                    lambda m, ins, out, i=i, shapes=shapes: shapes.__setitem__(i, list(out.shape))
                )
                for i in nodes
            ]
            with torch.no_grad():
                common.assert_close_tree(baseline(x), candidate(x), 1e-6, 1e-5, report=comparisons)
            for handle in handles:
                handle.remove()
            assert shapes == {i: [1, c, h // stride, w // stride] for i, (c, stride) in nodes.items()}
            report["shapes"].append({"input": list(x.shape), "nodes": shapes, "comparisons": comparisons})
        with torch.no_grad():
            x = torch.randn(2, 3, 64, 96)
            comparisons = []
            common.assert_close_tree(
                copy.deepcopy(baseline).train()(x), copy.deepcopy(candidate).train()(x), 1e-6, 1e-5, report=comparisons
            )
        report["train_comparisons"] = comparisons
        report["unsupported_native_grid"] = {}
        for name, network in (("native", baseline), ("ndp", candidate)):
            try:
                with torch.no_grad():
                    network(torch.zeros(1, 3, 65, 97))
            except (RuntimeError, ValueError) as error:
                report["unsupported_native_grid"][name] = {"input": [1, 3, 65, 97], "error": str(error)}
        assert set(report["unsupported_native_grid"]) == {"native", "ndp"}

        report["layers"] = [
            {"i": m.i, "from": m.f, "type": m.type, "parameters": sum(p.numel() for p in m.parameters())}
            for m in candidate.model
        ]
        candidate.info(verbose=True)
    common.write_json(Path(directory) / "structural.json", report)
    return report


@torch.no_grad()
def gpu_initialization_check(weight, directory):
    """Compare untrained pretrained models at batch=32, loading and releasing them sequentially on GPU."""
    import gc

    from tools.experiments.run_b19_ndp_sppf_v1 import AuditedTrainer
    from ultralytics.models.yolo.detect import DetectionTrainer
    from ultralytics.utils.torch_utils import init_seeds

    probe = object.__new__(AuditedTrainer)
    probe.args = common.get_cfg(overrides=common.REFERENCE["args"])
    probe.data = {"nc": 1, "channels": 3, "names": {0: "crack"}}
    source, _ = load_checkpoint(weight)
    x = torch.rand(32, 3, 640, 640, generator=torch.Generator().manual_seed(42)).cuda()
    outputs, memory = [], {}
    for name, cfg in (("native", common.baseline_architecture()), ("ndp", str(V1.model))):
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
        common.assert_close_tree(*outputs, 1e-6, 1e-5, report=comparisons)
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
    assert snapshot.model[9].ndp_out.weight.count_nonzero()
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
        [sys.executable, "-m", "tools.experiments.verify_b19_ndp_sppf_v1", str(path)],
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
    """Verify nonzero trained NDP survives YOLO loading, fuse, AutoBackend and actual one-to-one inference."""
    from ultralytics.nn.autobackend import AutoBackend

    path = Path(path)
    reference = torch.load(path.with_name("reload_reference.pt"), map_location="cpu", weights_only=False)
    torch.set_num_threads(reference["threads"])
    model = YOLO(path).model
    assert type(model.model[9]) is V1.block_type
    common.assert_close_tree(reference["state"], model.state_dict(), 0, 0)
    report = []
    with torch.no_grad():
        before = model(reference["x"])
        common.assert_close_tree(reference["raw"], before, 1e-6, 1e-5, report=report)
        native = DetectionModel(common.baseline_architecture(), verbose=False).eval()
        native.load_state_dict({k: v for k, v in model.state_dict().items() if "model.9.ndp_" not in k}, strict=True)
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
        common.assert_close_tree(list(after), backend(reference["x"]), 1e-6, 1e-5, report=report)
        with bypass(fused):
            disabled = fused(reference["x"])
        differences = {
            k: float((after[1]["one2one"][k] - disabled[1]["one2one"][k]).abs().max()) for k in ("boxes", "scores")
        }
        assert max(differences.values()) > 0, "NDP must affect actual one-to-one inference"
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
    from tools.experiments.finish_b19_ndp_sppf_v1 import evaluate, fixed_subset
    from tools.experiments.run_b19_ndp_sppf_v1 import AuditedTrainer
    from ultralytics.data import build_dataloader
    from ultralytics.data.dataset import YOLODataset
    from ultralytics.data.utils import check_det_dataset
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
        params = {k: p for k, p in model.model[9].named_parameters() if k.startswith("ndp_")}
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
                assert grads["ndp_out.weight"] > 0 and all(v == 0 for k, v in grads.items() if k != "ndp_out.weight")
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            old = {k: p.detach().clone() for k, p in params.items()}
            optimizer.step()
            ema.update(model)
            changes = {k: float((p - old[k]).abs().max()) for k, p in params.items()}
            effective.update(k for k in params if grads[k] > 0 and changes[k] > 0)
            rows.append({"step": step + 1, "loss": loss.detach().tolist(), "task_gradients": grads, "updates": changes})
            if len(effective) == 2:
                break
        assert len(effective) == 2
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
                    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in test.model[9].parameters())
                    cuda_rows.append(
                        {
                            "shape": [1, 3, height, width],
                            "amp": amp,
                            "loss": loss.detach().tolist(),
                            "po_gradient": float(test.model[9].ndp_out.weight.grad.norm()),
                            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                        }
                    )
                    del test, loss
            report["checks"]["cuda"] = cuda_rows
        torch.set_num_threads(4)
        report["backend"] = common.computation_conditions()
        report["flops_profiler_gflops"] = {}
        for name, cfg in (("native", common.baseline_architecture()), ("ndp", str(V1.model))):
            measured = DetectionModel(cfg, nc=1, verbose=False).eval()
            with torch.no_grad(), torch.profiler.profile(with_flops=True) as profiler:
                measured(torch.randn(1, 3, 640, 640))
            report["flops_profiler_gflops"][name] = sum(row.flops for row in profiler.key_averages()) / 1e9
        report["flops_scope"] = (
            "Actual 1x3x640x640, unfused eval, operators counted by PyTorch profiler; not all scalar operations"
        )
        import time

        report["latency_cpu_ms"] = {}
        for name, cfg in (("native", common.baseline_architecture()), ("ndp", str(V1.model))):
            measured = DetectionModel(cfg, nc=1, verbose=False).eval()
            x = torch.randn(1, 3, 640, 640)
            with torch.no_grad():
                for _ in range(3):
                    measured(x)
                start = time.perf_counter()
                for _ in range(10):
                    measured(x)
            report["latency_cpu_ms"][name] = (time.perf_counter() - start) * 100
        report["latency_scope"] = (
            "CPU 4 threads; 3 warmup + 10 timed forwards, batch1 640, unfused FP32; not server performance"
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
