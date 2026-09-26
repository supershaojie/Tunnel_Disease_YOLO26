"""Finite QCA engineering checks; synthetic smoke results are not detection validation.

Run from the experiment checkout: python -m tunnel_project.qca.check_qca --weights /path/yolo26n.pt.
Use --groups to rerun only an affected group after fixing a failure. Temporary checkpoints are discarded.
"""

import argparse
import hashlib
import json
import os
import tempfile
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

os.environ["YOLO_OFFLINE"] = "true"
os.environ["YOLO_AUTOINSTALL"] = "false"

import numpy as np
import torch
from PIL import Image

import ultralytics
from ultralytics import YOLO
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.nn.modules.block import Attention, C2PSA
from ultralytics.nn.modules.qca import C2PSA_QCA, QCAAttention
from ultralytics.nn.tasks import DetectionModel, load_checkpoint
from ultralytics.utils import YAML
from ultralytics.utils.torch_utils import ModelEMA

ROOT = Path(__file__).resolve().parents[2]
QCA_YAML = ROOT / "ultralytics/cfg/models/26/yolo26n-qca.yaml"
NATIVE_YAML = ROOT / "ultralytics/cfg/models/26/yolo26n.yaml"
EXPECTED_WEIGHTS = "9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef"


def error(actual, expected, atol=1e-6, rtol=1e-5, check=True):
    """Report absolute and tolerance-normalized errors, including near-zero values."""
    actual, expected = actual.detach().float(), expected.detach().float()
    assert actual.shape == expected.shape
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
    difference = (actual - expected).abs()
    result = {
        "max_abs": difference.max().item(),
        "max_tolerance_ratio": (difference / (atol + rtol * expected.abs())).max().item(),
        "atol": atol,
        "rtol": rtol,
    }
    if check:
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    return result


def synchronize(native, qca):
    """Copy every shared parameter AND buffer before equivalence comparisons."""
    state = native.state_dict()
    target = qca.state_dict()
    assert set(state) <= set(target)
    assert all(target[key].shape == value.shape for key, value in state.items())
    result = qca.load_state_dict(state, strict=False)
    assert not result.unexpected_keys
    assert all(key.endswith(".theta") or key == "theta" for key in result.missing_keys)
    assert all(torch.equal(qca.state_dict()[key], value) for key, value in state.items())


def theta_parameters(model):
    return {name: value for name, value in model.named_parameters() if name.endswith(".theta") or name == "theta"}


def reference(module, x):
    """Independent FP32 direct signed-attention formula with explicit valid-neighbor means."""
    batch, channels, height, width = x.shape
    q, k, v = (
        module.qkv(x)
        .reshape(batch, module.num_heads, -1, height * width)
        .split([module.key_dim, module.key_dim, module.head_dim], dim=2)
    )
    spatial = q.reshape(batch, module.num_heads, module.key_dim, height, width)
    context = torch.stack(
        [
            spatial[..., max(0, y - 1) : y + 2, max(0, z - 1) : z + 2].mean(dim=(-2, -1))
            for y in range(height)
            for z in range(width)
        ],
        dim=-1,
    )
    attention = ((q.transpose(-2, -1) @ k) * module.scale).softmax(-1)
    contextual = ((context.transpose(-2, -1) @ k) * module.scale).softmax(-1)
    effective = attention + (0.25 * module.theta.tanh()).view(1, -1, 1, 1) * (attention - contextual)
    output = (v @ effective.transpose(-2, -1)).reshape(batch, channels, height, width)
    return module.proj(output + module.pe(v.reshape(batch, channels, height, width))), effective, context


def formula_checks():
    results = []
    for channels, heads, height, width in [(32, 1, 5, 7), (64, 2, 5, 7), (64, 4, 1, 7), (32, 2, 7, 1), (32, 2, 1, 1)]:
        module = QCAAttention(channels, num_heads=heads).eval()
        with torch.no_grad():
            module.theta.copy_(torch.linspace(-0.4, 0.4, heads) if heads > 1 else torch.tensor([0.4]))
        x = torch.randn(2, channels, height, width)
        with torch.no_grad():
            expected, effective, context = reference(module, x)
            observed = module(x)
            q = module.qkv(x).reshape(2, heads, -1, height * width)[:, :, : module.key_dim]
            pooled = torch.nn.functional.avg_pool2d(
                q.reshape(2 * heads, module.key_dim, height, width), 3, 1, 1, count_include_pad=False
            ).reshape_as(context)
            pool_error = error(pooled, context)
            row_error = error(effective.sum(-1), torch.ones_like(effective.sum(-1)))
        results.append(
            {
                "shape": list(x.shape),
                "heads": heads,
                "forward": error(observed, expected),
                "valid_pool": pool_error,
                "row_sum": row_error,
            }
        )
    return results


def zero_module_check(native, qca, shape, device="cpu", amp=False):
    native, qca = native.to(device).eval(), qca.to(device).eval()
    synchronize(native, qca)
    x = torch.randn(*shape, device=device, requires_grad=True)
    y = x.detach().clone().requires_grad_(True)
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
        expected, observed = native(x), qca(y)
    tol = dict(atol=2e-3, rtol=2e-3) if amp else dict(atol=1e-6, rtol=1e-5)
    forward = error(observed, expected, **tol)
    probe = torch.randn_like(expected)
    (expected * probe).sum().backward()
    (observed * probe).sum().backward()
    gradients = {"input": error(y.grad, x.grad, **tol)}
    native_parameters = dict(native.named_parameters())
    for name, value in qca.named_parameters():
        if name in native_parameters:
            gradients[name] = error(value.grad, native_parameters[name].grad, **tol)
    theta = {name: value.grad.abs().max().item() for name, value in theta_parameters(qca).items()}
    assert all(np.isfinite(value) and value > 0 for value in theta.values()), theta
    return {
        "forward": forward,
        "gradient_max_abs": max(v["max_abs"] for v in gradients.values()),
        "gradient_max_tolerance_ratio": max(v["max_tolerance_ratio"] for v in gradients.values()),
        "theta_gradient_abs_max": theta,
        "native_gradient_tensors_checked": len(gradients),
    }


def construct_pair(weights):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        native = DetectionModel(str(NATIVE_YAML), nc=1, verbose=False)
        native_rng = torch.get_rng_state().clone()
        torch.manual_seed(42)
        qca = DetectionModel(str(QCA_YAML), nc=1, verbose=False)
        qca_rng = torch.get_rng_state().clone()
    assert torch.equal(native_rng, qca_rng), "QCA construction changes native RNG consumption"
    assert qca.yaml["scale"] == "n"
    assert type(native.model[10]) is C2PSA and type(qca.model[10]) is C2PSA_QCA
    before_native, before_qca = native.state_dict(), qca.state_dict()
    assert all(torch.equal(value, before_qca[key]) for key, value in before_native.items())
    source, _ = load_checkpoint(str(weights), device="cpu", fuse=False)
    native.names = qca.names = {0: "crack"}
    native.load(source, verbose=False)
    qca.load(source, verbose=False)
    ns, qs, ss = native.state_dict(), qca.state_dict(), source.state_dict()
    new_keys = sorted(set(qs) - set(ns))
    assert new_keys and all(key.startswith("model.10.m.") and key.endswith(".attn.theta") for key in new_keys)
    assert not set(ns) - set(qs)
    assert all(value.shape == qs[key].shape and torch.equal(value, qs[key]) for key, value in ns.items())
    transferred = [key for key, value in ns.items() if key in ss and value.shape == ss[key].shape]
    assert all(torch.equal(qs[key], ss[key]) and torch.equal(ns[key], ss[key]) for key in transferred)
    gaps = {
        key: {"source": list(ss[key].shape) if key in ss else None, "target": list(value.shape)}
        for key, value in ns.items()
        if key not in transferred
    }
    assert all(".cv3." in key or ".one2one_cv3." in key for key in gaps), gaps
    modules = {
        name: {"heads": module.num_heads, "key_dim": module.key_dim, "head_dim": module.head_dim}
        for name, module in qca.named_modules()
        if isinstance(module, QCAAttention)
    }
    added = sum(p.numel() for p in theta_parameters(qca).values())
    assert all(torch.count_nonzero(p) == 0 for p in theta_parameters(qca).values())
    assert added == sum(value["heads"] for value in modules.values())
    np_native, np_qca = sum(p.numel() for p in native.parameters()), sum(p.numel() for p in qca.parameters())
    assert np_qca - np_native == added
    synchronize(native, qca)
    report = {
        "nc": 1,
        "scale": qca.yaml["scale"],
        "stride": qca.stride.tolist(),
        "cls_remap": True,
        "target_names": qca.names,
        "rng_identical": True,
        "common_tensors_equal": len(ns),
        "source_transferred_keys": sorted(transferred),
        "native_existing_shape_gaps": gaps,
        "experiment_new_keys": new_keys,
        "accidentally_lost_native_keys": [],
        "modules": modules,
        "parameters_native": np_native,
        "parameters_qca": np_qca,
        "added_parameters": added,
    }
    return native.eval(), qca.eval(), report


def raw(model, x, branch="one2one"):
    prediction = model(x)
    outputs = prediction[1] if isinstance(prediction, tuple) else prediction
    return outputs[branch]


def compare_raw(actual, expected, **tolerance):
    return {key: error(actual[key], expected[key], **tolerance) for key in ("boxes", "scores")}


def initial_checks(native, qca):
    results = {
        "attention_fp32": zero_module_check(Attention(64, 2), QCAAttention(64, 2), (2, 64, 5, 7)),
        "c2psa_fp32_two_repeats": zero_module_check(C2PSA(256, 256, n=2), C2PSA_QCA(256, 256, n=2), (2, 256, 5, 7)),
    }
    x = torch.rand(1, 3, 160, 224)
    with torch.no_grad():
        native_raw, qca_raw = native(x)[1], qca(x)[1]
        results["full_model_fp32_raw"] = {
            branch: compare_raw(qca_raw[branch], native_raw[branch]) for branch in ("one2many", "one2one")
        }
    if torch.cuda.is_available():
        results["attention_cuda_fp16"] = zero_module_check(
            Attention(64, 2), QCAAttention(64, 2), (2, 64, 5, 7), "cuda", True
        )
        n_cuda, q_cuda = deepcopy(native).cuda(), deepcopy(qca).cuda()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            nr, qr = n_cuda(x.cuda())[1], q_cuda(x.cuda())[1]
            results["full_model_cuda_fp16_raw"] = {
                branch: compare_raw(qr[branch], nr[branch], atol=2e-3, rtol=2e-3) for branch in ("one2many", "one2one")
            }
            output_640 = q_cuda(torch.rand(1, 3, 640, 640, device="cuda"))[0]
        assert torch.isfinite(output_640).all()
        results["single_640_no_grad"] = {
            "shape": list(output_640.shape),
            "dtype": str(output_640.dtype),
            "device": "cuda",
        }
        del n_cuda, q_cuda
        torch.cuda.empty_cache()
    else:
        with torch.no_grad():
            output_640 = qca(torch.rand(1, 3, 640, 640))[0]
        assert torch.isfinite(output_640).all()
        results["single_640_no_grad"] = {"shape": list(output_640.shape), "device": "cpu"}
        results["cuda_fp16"] = "NOT_VERIFIED: CUDA unavailable"
    return results


def serialization_checks(qca, temporary):
    model = deepcopy(qca).eval()
    with torch.no_grad():
        for value in theta_parameters(model).values():
            value.fill_(0.4)
        x = torch.rand(1, 3, 160, 224)
        baseline, nonzero = raw(qca, x), raw(model, x)
        effect = max((nonzero[key] - baseline[key]).abs().max().item() for key in ("boxes", "scores"))
        assert effect > 0, "Nonzero QCA must affect raw predictions"
        copied, ema = deepcopy(model), ModelEMA(model)
        for instance in (copied, ema.ema):
            assert all(
                torch.equal(value, theta_parameters(instance)[key]) for key, value in theta_parameters(model).items()
            )
        ema.update(model)
        ema_result = compare_raw(raw(ema.ema, x), nonzero)
        fused = deepcopy(model).fuse(verbose=False)
        assert all(torch.equal(value, theta_parameters(fused)[key]) for key, value in theta_parameters(model).items())
        fusion = compare_raw(raw(fused, x), nonzero, atol=1e-4, rtol=1e-4)
        assert not any(isinstance(module, torch.nn.BatchNorm2d) for module in fused.modules())
        wrapper = YOLO(str(QCA_YAML), task="detect")
        wrapper.model = model
        checkpoint = temporary / "qca-nonzero.pt"
        wrapper.save(checkpoint)
        loaded = YOLO(str(checkpoint), task="detect")
        rounded = deepcopy(model).half().float().eval()
        expected_theta = float(torch.tensor(0.4).half().float())
        assert all(
            torch.equal(value, torch.full_like(value, expected_theta))
            for value in theta_parameters(loaded.model).values()
        )
        reloaded = raw(loaded.model.eval(), x)
        reload_error = compare_raw(reloaded, raw(rounded, x))
        quantization = {key: error(reloaded[key], nonzero[key], check=False) for key in ("boxes", "scores")}
        prediction = loaded.predict(np.zeros((160, 224, 3), dtype=np.uint8), imgsz=224, device="cpu", verbose=False)
        assert len(prediction) == 1 and torch.isfinite(prediction[0].boxes.data).all()
        assert all(
            torch.equal(value, torch.full_like(value, expected_theta))
            for value in theta_parameters(loaded.model).values()
        )
    return {
        "theta_before_save": 0.4,
        "theta_after_native_half_save": expected_theta,
        "nonzero_effect_max_abs": effect,
        "deepcopy_theta_preserved": True,
        "ema": ema_result,
        "fuse_raw_one2one": fusion,
        "fuse_note": "Native end-to-end fuse removes the one2many detection head.",
        "reload_vs_native_half_rounded_state": reload_error,
        "save_half_quantization_error": quantization,
        "default_predict_after_reload": True,
        "all_original_bn_fused": True,
    }


def trainer_checks(weights, temporary):
    if not torch.cuda.is_available():
        return {"status": "NOT_VERIFIED", "reason": "CUDA unavailable; no local training substituted"}
    dataset = temporary / "synthetic"
    rng = np.random.default_rng(42)
    for split, count in (("train", 6), ("val", 2)):
        (dataset / "images" / split).mkdir(parents=True)
        (dataset / "labels" / split).mkdir(parents=True)
        for index in range(count):
            pixels = rng.integers(0, 256, (160, 160, 3), dtype=np.uint8)
            Image.fromarray(pixels).save(dataset / "images" / split / f"{index}.png")
            (dataset / "labels" / split / f"{index}.txt").write_text("0 0.5 0.5 0.3 0.5\n", encoding="utf-8")
    data_file = dataset / "data.yaml"
    YAML.save(data_file, {"path": str(dataset), "train": "images/train", "val": "images/val", "names": {0: "crack"}})
    record = {
        "status": "PASSED",
        "dataset": "6 synthetic nonempty training images; 2 synthetic validation images",
        "actual_optimizer_steps": 0,
        "batches": [],
        "theta_history": [],
        "theta_groups": {},
    }

    def setup(trainer):
        assert trainer.amp and trainer.model.yaml["scale"] == "n" and trainer.model.nc == 1
        assert type(trainer.model.model[10]) is C2PSA_QCA
        assert type(trainer.optimizer).__name__ == "MuSGD"
        assert trainer.accumulate == 1
        assert not trainer.args.freeze
        assert all(param.requires_grad for name, param in trainer.model.named_parameters() if ".dfl" not in name)
        theta = theta_parameters(trainer.model)
        all_ids = [id(param) for group in trainer.optimizer.param_groups for param in group["params"]]
        assert len(all_ids) == len(set(all_ids))
        assert all(id(param) in all_ids for param in trainer.model.parameters() if param.requires_grad)
        for name, value in theta.items():
            matches = [
                group for group in trainer.optimizer.param_groups if any(param is value for param in group["params"])
            ]
            assert len(matches) == 1
            group = matches[0]
            assert group["param_group"] == "weight" and not group.get("use_muon", False)
            assert group["lr"] == trainer.args.lr0
            record["theta_groups"][name] = {
                key: group[key] for key in ("param_group", "lr", "weight_decay", "use_muon")
            }
        record["theta_history"].append({name: value.detach().cpu().tolist() for name, value in theta.items()})
        # This engineering-only scale avoids spending extra batches on scaler warmup.
        trainer.scaler = torch.amp.GradScaler("cuda", init_scale=128.0)

        def before_step(optimizer, args, kwargs):
            grads = [param.grad for param in trainer.model.parameters() if param.grad is not None]
            assert grads and all(torch.isfinite(value).all() for value in grads)
            assert all(
                value.grad is not None and torch.isfinite(value.grad).all() and value.grad.abs().max() > 0
                for value in theta.values()
            )

        def after_step(optimizer, args, kwargs):
            record["actual_optimizer_steps"] += 1
            assert all(torch.isfinite(param).all() for param in trainer.model.parameters())
            record["theta_history"].append({name: value.detach().cpu().tolist() for name, value in theta.items()})

        trainer.optimizer.register_step_pre_hook(before_step)
        trainer.optimizer.register_step_post_hook(after_step)

    def batch_end(trainer):
        assert torch.isfinite(trainer.loss).all()
        record["batches"].append({"loss": trainer.loss.detach().item(), "scale": trainer.scaler.get_scale()})

    overrides = dict(
        model=str(QCA_YAML),
        pretrained=str(weights),
        data=str(data_file),
        epochs=1,
        batch=2,
        nbs=2,
        imgsz=160,
        workers=0,
        device="0",
        optimizer="MuSGD",
        lr0=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        amp=True,
        warmup_epochs=0.0,
        close_mosaic=0,
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
        plots=False,
        save=False,
        val=True,
        cache=False,
        project=str(temporary / "runs"),
        name="qca-smoke",
        exist_ok=True,
        seed=42,
        deterministic=True,
        resume=False,
    )
    # The independent FP16 checks above replace the download-based AMP probe only in this process.
    # No plotting is requested, so font downloads and external reporting integrations are unnecessary.
    with patch("ultralytics.engine.trainer.check_amp", return_value=True), patch(
        "ultralytics.data.utils.check_font", return_value=None
    ), patch("ultralytics.utils.callbacks.add_integration_callbacks", return_value=None):
        trainer = DetectionTrainer(overrides=overrides)
        trainer.add_callback("on_pretrain_routine_end", setup)
        trainer.add_callback("on_train_batch_end", batch_end)
        trainer.train()
    assert record["actual_optimizer_steps"] == 3, record
    assert len(record["batches"]) == 3
    assert all(any(abs(v) > 0 for v in values) for values in record["theta_history"][-1].values())
    assert record["theta_history"][0] != record["theta_history"][-1]
    assert all(name in trainer.ema.ema.state_dict() for name in record["theta_groups"])
    record["smoke_only_overrides"] = overrides
    record["amp_probe"] = "Independent QCA CUDA FP16 check; download-based stock probe skipped in this test process"
    record["grad_scaler_initial_scale_smoke_only"] = 128.0
    record["all_recorded_losses_gradients_parameters_finite"] = True
    record["formal_training"] = "NOT_STARTED"
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--groups", default="formula,initial,serialization,trainer")
    parser.add_argument("--output", type=Path, default=Path("qca-checks.json"))
    args = parser.parse_args()
    groups = set(args.groups.split(","))
    assert groups <= {"formula", "initial", "serialization", "trainer"}, groups
    assert Path(ultralytics.__file__).resolve().parent == ROOT / "ultralytics", ultralytics.__file__
    weights = args.weights.resolve()
    digest = hashlib.sha256(weights.read_bytes()).hexdigest()
    assert digest == EXPECTED_WEIGHTS, digest
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    torch.manual_seed(42)
    report = {
        "import_path": ultralytics.__file__,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "weights_sha256": digest,
        "groups": sorted(groups),
        "formal_training": "NOT_STARTED",
    }

    def progress(group):
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps({"completed_group": group, "report": str(args.output.resolve())}), flush=True)

    try:
        if "formula" in groups:
            report["formula"] = formula_checks()
            progress("formula")
        native, qca, report["construction"] = construct_pair(weights)
        progress("construction")
        if "initial" in groups:
            report["initial"] = initial_checks(native, qca)
            progress("initial")
        with tempfile.TemporaryDirectory(prefix="qca-check-") as temp:
            if "serialization" in groups:
                report["serialization"] = serialization_checks(qca, Path(temp))
                progress("serialization")
            if "trainer" in groups:
                if "initial" not in groups and torch.cuda.is_available():
                    report["trainer_amp_precheck"] = zero_module_check(
                        Attention(64, 2), QCAAttention(64, 2), (2, 64, 5, 7), "cuda", True
                    )
                report["trainer"] = trainer_checks(weights, Path(temp))
                progress("trainer")
        report["status"] = "PASSED"
    except Exception as exc:
        report["status"] = "FAILED"
        report["failure"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps({"status": report["status"], "report": str(args.output.resolve()), "groups": sorted(groups)}))


if __name__ == "__main__":
    main()
