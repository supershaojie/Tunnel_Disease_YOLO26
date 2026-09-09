"""BDI formula, graph, RNG, checkpoint and native lifecycle validation."""

import argparse
import copy
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

import torch

from tools.experiments import b19_common as common
from tools.experiments.bdi_experiment import V1
from ultralytics import YOLO
from ultralytics.nn.modules import Concat_BDI_P3
from ultralytics.nn.tasks import DetectionModel, load_checkpoint
from ultralytics.utils import YAML


@contextmanager
def bypass(model):
    """Disable only the added residual using a temporary final-projection output hook."""
    handles = [
        m.po.register_forward_hook(lambda m, inputs, output: torch.zeros_like(output))
        for m in model.modules()
        if isinstance(m, Concat_BDI_P3)
    ]
    try:
        yield
    finally:
        for h in handles:
            h.remove()


def audit(baseline, candidate, weights, experiment=V1):
    """Check every shared state and allow only the layer-15 replacement and new P2 edge."""
    report = common.audit_weights(baseline, candidate, weights, new_prefix="model.15.")
    assert len(candidate.model) == len(baseline.model) == 24
    assert [i for i, (a, b) in enumerate(zip(baseline.model, candidate.model)) if type(a) is not type(b)] == [15]
    m = candidate.model[15]
    assert type(m) is experiment.block_type and m.f == [14, 4, 2] and 2 in candidate.save
    assert candidate.model[16].cv1.conv.in_channels == 256
    assert candidate.model[-1].f == [16, 19, 22] and candidate.model[-1].end2end
    assert candidate.stride.tolist() == baseline.stride.tolist() == [8, 16, 32]
    assert (m.pin.in_channels, m.pin.out_channels, m.po.out_channels) == (64, 16, 128)
    expected = {"model.15." + k for k in ("pin.weight", "dw.weight", "po.weight")}
    assert set(report["new_parameters"]) == expected
    assert set(candidate.state_dict()) - set(baseline.state_dict()) == expected | {"model.15.k3", "model.15.k5"}
    assert report["added_parameters"] == 3216
    assert (report["baseline_parameters"], report["candidate_parameters"]) == (2504190, 2507406)
    report["shared_tensors"] = {
        k: {"shape": list(v.shape), "equal": True, "pretrained": k in report["loaded_keys"]}
        for k, v in baseline.state_dict().items()
    }
    report["pretrained_matched_keys"] = len(report.pop("loaded_keys"))
    return report


def branch_checks():
    """Independent scalar-clamped reference covers borders, odd downsampling and band endpoints."""
    m = Concat_BDI_P3(128, 128, 64)
    assert sum(p.numel() for p in m.parameters()) == 3216
    assert set(dict(m.named_buffers())) == {"k3", "k5"}
    x = torch.arange(16 * 7 * 9).reshape(1, 16, 7, 9).float() / 100
    errors = []
    for n, coeff, stride in ((3, [1, 2, 1], 1), (5, [1, 4, 6, 4, 1], 1), (3, [1, 2, 1], 2)):
        kernel = getattr(m, f"k{n}")
        torch.testing.assert_close(kernel.sum((1, 2, 3)), torch.ones(16))
        actual = m.blur(x, kernel, stride)
        ref = torch.zeros_like(actual, dtype=torch.float64)
        for y in range(ref.shape[-2]):
            for z in range(ref.shape[-1]):
                for i in range(n):
                    for j in range(n):
                        ref[:, :, y, z] += (
                            x[
                                :, :, min(max(y * stride + i - n // 2, 0), 6), min(max(z * stride + j - n // 2, 0), 8)
                            ].double()
                            * coeff[i]
                            * coeff[j]
                            / sum(coeff) ** 2
                        )
        torch.testing.assert_close(actual.double(), ref, atol=2e-6, rtol=2e-6)
        errors.append(float((actual - ref).abs().max()))
    torch.testing.assert_close(m.bandpass(torch.ones(1, 16, 11, 13)), torch.zeros(1, 16, 11, 13), atol=1e-7, rtol=0)
    chess = (
        ((torch.arange(11)[:, None] + torch.arange(13)[None, :]) % 2 * 2 - 1).float()[None, None].repeat(1, 16, 1, 1)
    )
    d = m.bandpass(chess)
    torch.testing.assert_close(d[:, :, 2:-2, 2:-2], torch.zeros_like(d[:, :, 2:-2, 2:-2]), atol=1e-7, rtol=0)
    assert d[:, :, :2, :2].abs().max() > 0
    torch.testing.assert_close(m.bandpass(x), 4 * (m.blur(x, m.k3) - m.blur(x, m.k5)))
    assert m.blur(torch.randn(1, 16, 31, 47), m.k3, 2).shape == (1, 16, 16, 24)
    s, l, f2 = torch.randn(1, 128, 16, 24), torch.randn(1, 128, 16, 24), torch.randn(1, 64, 31, 47, requires_grad=True)
    torch.testing.assert_close(m([s, l, f2]), torch.cat([s, l], 1), atol=1e-7, rtol=1e-6)
    with torch.no_grad():
        m.po.weight.normal_(std=0.01)
    result = m([s, l, f2])
    torch.testing.assert_close(result[:, :128], s, atol=1e-7, rtol=1e-6)
    result.square().mean().backward()
    grads = {k: float(p.grad.norm()) for k, p in m.named_parameters()}
    assert all(v > 0 for v in grads.values()) and f2.grad.norm() > 0
    dtype_rows = []
    from torch.utils._python_dispatch import TorchDispatchMode

    class Observe(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if str(func) == "aten.convolution.default":
                self.rows.append([str(args[0].dtype), str(args[1].dtype)])
            return func(*args, **(kwargs or {}))

    for device, dtype in [("cpu", torch.bfloat16)] + ([("cuda", torch.float16)] if torch.cuda.is_available() else []):
        block = copy.deepcopy(m).to(device)
        u = torch.randn(1, 16, 11, 13, device=device, dtype=dtype)
        observer = Observe()
        observer.rows = []
        with observer, torch.autocast(device_type=device, dtype=dtype):
            d = block.bandpass(u)
            down = block.blur(d, block.k3, 2)
        assert d.dtype == down.dtype == dtype and observer.rows == [["torch.float32", "torch.float32"]] * 3
        dtype_rows.append({"device": device, "feature_dtype": str(dtype), "actual_fixed_convolutions": observer.rows})
    return {
        "reference_max_abs": errors,
        "checkerboard_boundary_max": float(m.bandpass(chess).abs().max()),
        "synthetic_gradients": grads,
        "p2_gradient": float(f2.grad.norm()),
        "autocast": dtype_rows,
    }


def structural_checks(directory, experiment=V1):
    """Compare native train/eval outputs and the complete 640 and rectangular feature graph."""
    expected = YAML.load(common.ROOT / "ultralytics/cfg/models/26/yolo26.yaml")
    expected["nc"] = 1
    expected["head"][4] = [[14, 4, 2], 1, "Concat_BDI_P3", []]
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
            nodes = {2: (64, 4), 4: (128, 8), 14: (128, 8), 15: (256, 8), 16: (64, 8), 19: (128, 16), 22: (256, 32)}
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
        for name, network in (("native", baseline), ("bdi", candidate)):
            try:
                with torch.no_grad():
                    network(torch.zeros(1, 3, 65, 97))
            except (RuntimeError, ValueError) as error:
                report["unsupported_native_grid"][name] = {"input": [1, 3, 65, 97], "error": str(error)}
        assert set(report["unsupported_native_grid"]) == {"native", "bdi"}

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

    from tools.experiments.run_b19_bdi_p3_v1 import AuditedTrainer
    from ultralytics.models.yolo.detect import DetectionTrainer
    from ultralytics.utils.torch_utils import init_seeds

    probe = object.__new__(AuditedTrainer)
    probe.args = common.get_cfg(overrides=common.REFERENCE["args"])
    probe.data = {"nc": 1, "channels": 3, "names": {0: "crack"}}
    source, _ = load_checkpoint(weight)
    x = torch.rand(32, 3, 640, 640, generator=torch.Generator().manual_seed(42)).cuda()
    outputs, memory = [], {}
    for name, cfg in (("native", common.baseline_architecture()), ("bdi", str(V1.model))):
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
    assert snapshot.model[15].po.weight.count_nonzero()
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
        [sys.executable, "-m", "tools.experiments.verify_b19_bdi_p3_v1", str(path)],
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
    """Verify nonzero trained BDI survives YOLO loading, fuse, AutoBackend and actual one-to-one inference."""
    from ultralytics.nn.autobackend import AutoBackend

    path = Path(path)
    reference = torch.load(path.with_name("reload_reference.pt"), map_location="cpu", weights_only=False)
    torch.set_num_threads(reference["threads"])
    model = YOLO(path).model
    assert type(model.model[15]) is V1.block_type
    common.assert_close_tree(reference["state"], model.state_dict(), 0, 0)
    report = []
    with torch.no_grad():
        before = model(reference["x"])
        common.assert_close_tree(reference["raw"], before, 1e-6, 1e-5, report=report)
        native = DetectionModel(common.baseline_architecture(), verbose=False).eval()
        native.load_state_dict({k: v for k, v in model.state_dict().items() if "model.15." not in k}, strict=True)
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
        assert max(differences.values()) > 0, "BDI must affect actual one-to-one inference"
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
    from tools.experiments.finish_b19_bdi_p3_v1 import evaluate, fixed_subset
    from tools.experiments.run_b19_bdi_p3_v1 import AuditedTrainer
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
        params = dict(model.model[15].named_parameters())
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
            if len(effective) == 3:
                break
        assert len(effective) == 3
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
                    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in test.model[15].parameters())
                    cuda_rows.append(
                        {
                            "shape": [1, 3, height, width],
                            "amp": amp,
                            "loss": loss.detach().tolist(),
                            "po_gradient": float(test.model[15].po.weight.grad.norm()),
                            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                        }
                    )
                    del test, loss
            report["checks"]["cuda"] = cuda_rows
        report["backend"] = common.computation_conditions()
        report["flops_profiler_gflops"] = {}
        for name, cfg in (("native", common.baseline_architecture()), ("bdi", str(V1.model))):
            measured = DetectionModel(cfg, nc=1, verbose=False).eval()
            with torch.no_grad(), torch.profiler.profile(with_flops=True) as profiler:
                measured(torch.randn(1, 3, 640, 640))
            report["flops_profiler_gflops"][name] = sum(row.flops for row in profiler.key_averages()) / 1e9
        report["flops_scope"] = (
            "Actual 1x3x640x640, unfused eval, operators counted by PyTorch profiler; not all scalar operations"
        )
        import time

        report["latency_cpu_ms"] = {}
        for name, cfg in (("native", common.baseline_architecture()), ("bdi", str(V1.model))):
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
