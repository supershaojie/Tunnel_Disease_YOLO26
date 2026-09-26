"""Finite APA A-D engineering checks; synthetic data is not an accuracy evaluation."""

import argparse
import hashlib
import json
import sys
import tempfile
import warnings
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import ultralytics
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.nn.modules.apa import AxisPairedAlignment, DetectAPA
from ultralytics.nn.tasks import DetectionModel, load_checkpoint
from ultralytics.utils import YAML
from ultralytics.utils.torch_utils import ModelEMA, init_seeds

ROOT = Path(__file__).resolve().parents[2]
MODEL_YAML = ROOT / "ultralytics/cfg/models/26/yolo26n-apa.yaml"
WEIGHT_HASH = "9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef"


def error(actual, expected, tolerance=0.0):
    """Check finite tensors and report measured absolute error."""
    assert torch.isfinite(actual).all() and actual.shape == expected.shape
    maximum = float((actual.float() - expected.float()).abs().max())
    assert maximum <= tolerance, (maximum, tolerance)
    return maximum


def modules(model):
    """Get registered APA groups in module order."""
    return [m for m in model.modules() if isinstance(m, AxisPairedAlignment)]


def affine(height, width):
    """Express the prompt's absolute affine boundary fields as native ltrb."""
    i, j = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    x1, y1 = 1 + 0.1 * j + 0.2 * i, 2 + 0.3 * j + 0.1 * i
    return torch.stack((j + 0.5 - x1, i + 0.5 - y1, x1 + 3 - j - 0.5, y1 + 2 - i - 0.5))[None]


def coordinates():
    """A: affine values, signs, border clamp and H=1."""
    module = AxisPairedAlignment(4, 4)
    with torch.no_grad():
        module.offsets[-1].bias.copy_(torch.atanh(torch.tensor([0.5, -0.5, -0.5, 0.5])))
    result = {}
    for height, width in ((3, 5), (1, 5)):
        raw = affine(height, width)
        i, j = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
        dx = 0.1 * ((j + 0.25).clamp(0, width - 1) - j) + 0.2 * ((i - 0.25).clamp(0, height - 1) - i)
        dy = 0.3 * ((j - 0.25).clamp(0, width - 1) - j) + 0.1 * ((i + 0.25).clamp(0, height - 1) - i)
        actual = module(raw, torch.zeros_like(raw), torch.zeros_like(raw))
        result[f"{height}x{width}_all_pixels_max_error"] = error(
            actual, raw + torch.stack((-dx, -dy, dx, dy))[None], 2e-6
        )
        error(actual[:, 0] + actual[:, 2], torch.full_like(actual[:, 0], 3), 2e-6)
        error(actual[:, 1] + actual[:, 3], torch.full_like(actual[:, 1], 2), 2e-6)
    return result


def loading(weights):
    """Classify nc adaptation gaps, verify each loaded tensor and synchronize ALL common state."""
    source, _ = load_checkpoint(weights, device="cpu")
    native = DetectionModel("yolo26n.yaml", nc=1, verbose=False)
    model = DetectionModel(str(MODEL_YAML), nc=1, verbose=False)
    for target in (native, model):
        target.names = {0: "crack"}
        target.load(source, verbose=False)
    original, base, changed = source.state_dict(), native.state_dict(), model.state_dict()
    loaded = {k for k, v in base.items() if k in original and v.shape == original[k].shape}
    added = sorted(set(changed) - set(base))
    unexpected = sorted(k for k in base if k not in changed or base[k].shape != changed[k].shape)
    assert not unexpected and all(".apa." in k or ".one2one_apa." in k for k in added)
    for key in loaded:
        error(base[key], original[key])
        error(changed[key], original[key])
    model.load_state_dict(base, strict=False)
    for key in base:
        error(model.state_dict()[key], base[key])
    head = model.model[-1]
    assert isinstance(head, DetectAPA) and head.reg_max == 1 and head.end2end and head.nl == 3
    assert model.yaml["scale"] == "n" and head.stride.tolist() == [8, 16, 32]
    groups = modules(model)
    ids = [id(p) for m in groups for p in m.parameters()]
    assert len(groups) == 6 and len(ids) == len(set(ids))
    return (
        model,
        native,
        dict(
            source_nc=source.model[-1].nc,
            target_nc=1,
            native_loadable_tensor_count=len(loaded),
            native_existing_gaps=sorted(set(base) - loaded),
            experiment_added_keys=added,
            unexpected_native_gaps=unexpected,
            common_tensors_explicitly_synchronized=len(base),
            apa_group_count=6,
            added_parameters=sum(p.numel() for m in groups for p in m.parameters()),
            total_parameters=sum(p.numel() for p in model.parameters()),
        ),
    )


def equivalence(model, native, device):
    """B: exact identity, invalid source exclusion/center preservation, explicit non-finite failures."""
    result = {}
    for dtype in (torch.float32, torch.float16):
        raw = affine(3, 5).to(dtype)
        raw[:, 2, 0, 0] = -raw[:, 0, 0, 0]
        raw[:, 2, 0, 1] = -raw[:, 0, 0, 1] - 1
        raw[:, 3, 0, 2] = -raw[:, 1, 0, 2] - 1
        result[f"zero_valid_invalid_{dtype}"] = error(AxisPairedAlignment.align(raw, torch.zeros_like(raw)), raw)
    raw = affine(3, 5)
    raw[:, 2, 0, 3] = -raw[:, 0, 0, 3] - 1
    offsets = torch.tensor([0.25, -0.25, -0.25, 0.25])[None, :, None, None].expand_as(raw)
    actual = AxisPairedAlignment.align(raw, offsets)
    # Remaining horizontal weights at (1,2): .1875,.5625,.1875; mass=.9375, delta=-.02.
    result["invalid_neighbor_max_error"] = error(
        actual[0, :, 1, 2], raw[0, :, 1, 2] + torch.tensor([0.02, 0.05, -0.02, -0.05]), 2e-6
    )
    error(actual[0, :, 0, 3], raw[0, :, 0, 3])
    failures = []
    for value in (float("nan"), float("inf")):
        broken = raw.clone()
        broken[0, 0, 1, 1] = value
        try:
            AxisPairedAlignment.align(broken, offsets)
        except ValueError as exc:
            assert "non-finite" in str(exc) and "1, 1" in str(exc)
            failures.append(str(exc))
        else:
            raise AssertionError("Non-finite raw accepted")
    result["nonfinite_errors"] = failures
    model, native = model.to(device).eval(), native.to(device).eval()
    with torch.no_grad():
        image = torch.rand(1, 3, 96, 128, device=device)
        actual, expected = model(image)[1], native(image)[1]
    for branch in ("one2many", "one2one"):
        assert set(actual[branch]) == set(expected[branch]) == {"boxes", "scores", "feats"}
        for name in ("boxes", "scores"):
            result[f"{branch}_{name}_pre_topk_max_error"] = error(actual[branch][name], expected[branch][name])
        assert len(actual[branch]["feats"]) == 3
    return result


def routes(model, device):
    """Use spatially varying features to expose each APA gradient and native O2O detach."""
    head = deepcopy(model.model[-1]).to(device).train()
    for module in modules(head):
        torch.nn.init.normal_(module.offsets[-1].weight, std=0.01)
    channels = [t[0].conv.in_channels for t in head.cv2]
    result = {}
    for branch, groups, towers in (("one2many", head.apa, head.cv3), ("one2one", head.one2one_apa, head.one2one_cv3)):
        head.zero_grad(set_to_none=True)
        features = [
            torch.randn(2, c, h, w, device=device, requires_grad=True)
            for c, h, w in zip(channels, (12, 6, 3), (16, 8, 4))
        ]
        head(features)[branch]["boxes"].square().mean().backward()
        reaches = any(x.grad is not None and x.grad.abs().sum() > 0 for x in features)
        assert reaches == (branch == "one2many")
        final = [float(m.offsets[-1].weight.grad.abs().sum()) for m in groups]
        cls = [float(t[0][1].conv.weight.grad.abs().sum()) for t in towers]
        assert all(v > 0 for v in final + cls)
        result[branch] = dict(
            input_gradient=bool(reaches),
            apa_final_weight_gradient_l1=final,
            classification_tower_localization_gradient_l1=cls,
        )
    return result


def training(weights, directory, device):
    """C: genuine Trainer setup/dataloader/loss/optimizer_step, exactly three observed updates."""
    directory = Path(directory)
    rng = np.random.default_rng(42)
    for split, count in (("train", 6), ("val", 2)):
        for kind in ("images", "labels"):
            (directory / kind / split).mkdir(parents=True)
        for index in range(count):
            Image.fromarray(rng.integers(0, 256, (96, 128, 3), dtype=np.uint8)).save(
                directory / f"images/{split}/{index}.png"
            )
            (directory / f"labels/{split}/{index}.txt").write_text("0 0.5 0.5 0.45 0.55\n", encoding="utf-8")
    data = directory / "data.yaml"
    YAML.save(data, dict(path=str(directory), train="images/train", val="images/val", names={0: "crack"}))
    overrides = dict(
        model=str(MODEL_YAML),
        pretrained=str(weights),
        data=str(data),
        project=str(directory),
        name="smoke",
        exist_ok=True,
        device="0" if device.type == "cuda" else "cpu",
        epochs=1,
        batch=2,
        nbs=2,
        imgsz=128,
        workers=0,
        optimizer="MuSGD",
        lr0=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=0,
        seed=42,
        deterministic=True,
        amp=False,
        plots=False,
        save=False,
        val=False,
        mosaic=0.0,
        close_mosaic=0,
        cache=False,
    )
    trainer = DetectionTrainer(overrides=overrides, _callbacks=defaultdict(list))
    trainer._setup_train()
    model = trainer.model.train()
    assert isinstance(model.model[-1], DetectAPA) and model.model[-1].nc == 1
    groups = modules(model)
    counts = Counter(id(p) for group in trainer.optimizer.param_groups for p in group["params"])
    assert type(trainer.optimizer).__name__ == "MuSGD" and len(groups) == 6
    assert all(counts[id(p)] == 1 and p.requires_grad for p in model.parameters())
    assert trainer.accumulate == 1
    before = [m.offsets[-1].weight.detach().clone() for m in groups]
    steps, losses, final, upstream = [], [], [], []
    handle = trainer.optimizer.register_step_post_hook(lambda *_: steps.append(len(steps) + 1))
    for index, batch in enumerate(trainer.train_loader):
        batch = trainer.preprocess_batch(batch)
        loss, _ = model(batch)
        loss = loss.sum()
        assert torch.isfinite(loss)
        loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        final.append([float(m.offsets[-1].weight.grad.abs().sum()) for m in groups])
        upstream.append(
            [float(m.reg_proj[0].weight.grad.abs().sum() + m.cls_proj[0].weight.grad.abs().sum()) for m in groups]
        )
        losses.append(float(loss.detach()))
        trainer.optimizer_step()
        if index == 2:
            break
    handle.remove()
    assert len(steps) == 3 and any(v > 0 for v in final[0])
    assert all(v == 0 for v in upstream[0]) and any(v > 0 for row in upstream[1:] for v in row)
    changed = [bool(torch.count_nonzero(m.offsets[-1].weight.detach() - old)) for m, old in zip(groups, before)]
    assert any(changed)
    amp = dict(status="NOT_RUN: CUDA unavailable")
    if device.type == "cuda":
        model.zero_grad(set_to_none=True)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with torch.autocast("cuda", dtype=torch.float16):
                amp_loss, _ = model(batch)
            amp_loss.sum().backward()
        assert torch.isfinite(amp_loss).all()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        amp = dict(
            status="PASS: native loss AMP forward/backward, no extra update",
            warnings=sorted({str(w.message) for w in caught}),
            loss=amp_loss.detach().tolist(),
        )
    return model, dict(
        trainer_smoke="native setup/dataloader/loss/optimizer_step with synthetic labels; no epoch validation",
        actual_optimizer_steps=len(steps),
        optimizer=type(trainer.optimizer).__name__,
        all_parameters_once=True,
        losses=losses,
        first_final_gradient_l1=final[0],
        first_projection_gradient_l1=upstream[0],
        post_update_projection_gradient_l1=upstream[1:],
        final_weight_changed_by_group=changed,
        amp=amp,
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
        smoke_overrides=overrides,
        gradient_routes=routes(model, device),
    )


def delivery(model, directory, device):
    """D: nonzero pre-topk effect through deepcopy, EMA, save/reload, native fuse/default predict."""
    model = model.to(device).eval()
    with torch.no_grad():
        for module in modules(model):
            module.offsets[-1].bias.copy_(torch.atanh(torch.tensor([0.5, -0.5, -0.5, 0.5], device=device)))
    image = torch.rand(1, 3, 96, 128, device=device)
    zero = deepcopy(model)
    for module in modules(zero):
        torch.nn.init.zeros_(module.offsets[-1].weight)
        torch.nn.init.zeros_(module.offsets[-1].bias)
    with torch.no_grad():
        output, raw = model(image)
        zero_raw = zero(image)[1]
        effect = float((raw["one2one"]["boxes"] - zero_raw["one2one"]["boxes"]).abs().max())
        assert effect > 1e-5
        error(raw["one2one"]["scores"], zero_raw["one2one"]["scores"])
        copied_error = error(deepcopy(model)(image)[0], output)
        ema = ModelEMA(model)
        ema.update(model)
        ema_output, ema_raw = ema.ema(image)
        ema_raw_error = error(ema_raw["one2one"]["boxes"], raw["one2one"]["boxes"], 2e-5)
        ema_error = error(ema_output, output, 2e-3)
    checkpoint = Path(directory) / "nonzero_apa.pt"
    train_args = model.args if isinstance(model.args, dict) else vars(model.args)
    torch.save(dict(model=deepcopy(model).cpu(), train_args=train_args), checkpoint)
    reloaded, _ = load_checkpoint(checkpoint, device=device)
    for key, value in model.state_dict().items():
        error(reloaded.state_dict()[key], value)
    with torch.no_grad():
        reload_error = error(reloaded(image)[0], output)
        fused = reloaded.fuse(verbose=False)
        fused_output, fused_raw = fused(image)
        fuse_raw_error = error(fused_raw["one2one"]["boxes"], raw["one2one"]["boxes"], 2e-4)
        fuse_error = error(fused_output, output, 2e-3)
        assert len(modules(fused)) == 3
        full = fused(torch.rand(1, 3, 640, 640, device=device))[0]
        assert full.shape == (1, 300, 6) and torch.isfinite(full).all()
    wrapper = YOLO(str(checkpoint))
    predicted = wrapper.predict(source=image, verbose=False, device="0" if device.type == "cuda" else "cpu", save=False)
    assert len(predicted) == 1 and isinstance(wrapper.model.model[-1], DetectAPA) and len(modules(wrapper.model)) == 3
    return dict(
        nonzero_pre_topk_max_effect=effect,
        classification_max_error=0.0,
        deepcopy_max_error=copied_error,
        ema_max_error=ema_error,
        ema_raw_max_error=ema_raw_error,
        reload_all_state_tensors_exact=True,
        reload_max_error=reload_error,
        fused_raw_max_error=fuse_raw_error,
        fused_prediction_max_error=fuse_error,
        fused_remaining_apa_groups=3,
        single_640_output_shape=list(full.shape),
        default_predict="PASS: default native fuse with nonzero checkpoint",
        checkpoint="one temporary FP32 checkpoint removed after checks",
    )


def main():
    """Run finite groups and persist one factual JSON report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/apa_head/validation.json")
    parser.add_argument("--groups", default="ABCD", help="Subset of A/B/C/D for targeted rechecks")
    args = parser.parse_args()
    assert Path(ultralytics.__file__).resolve() == ROOT / "ultralytics/__init__.py"
    assert hashlib.sha256(args.weights.read_bytes()).hexdigest() == WEIGHT_HASH
    torch.set_num_threads(4)
    init_seeds(42, deterministic=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    report = dict(
        status="RUNNING",
        formal_training="NOT_STARTED",
        import_path=ultralytics.__file__,
        torch=torch.__version__,
        cuda=torch.version.cuda,
        device=str(device),
        gpu=torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        weights_sha256=WEIGHT_HASH,
        limitations=[
            "Synthetic engineering smoke only; no real-data val/test or accuracy claims.",
            "No full training epoch, formal 200e, B32 memory test, export or latency benchmark.",
            "Functional grid_sample/mask/normalization/FP32 coordinates cost is not captured by Conv FLOP counters.",
            "Native deterministic=True uses warn_only=True; CUDA grid_sample backward is not bitwise deterministic.",
        ],
    )
    if args.groups != "ABCD" and args.output.exists():
        report = {**json.loads(args.output.read_text(encoding="utf-8")), **report}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with tempfile.TemporaryDirectory(prefix="apa-validation-", dir=args.weights.parent) as directory:
            if "A" in args.groups:
                report["A"] = coordinates()
            model, native, report["loading"] = loading(args.weights)
            model.args = get_cfg()
            if "B" in args.groups:
                report["B"] = equivalence(model, native, device)
            del native
            if "C" in args.groups:
                model, report["C"] = training(args.weights, directory, device)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
            if "D" in args.groups:
                report["D"] = delivery(model, directory, device)
        report["warnings"] = sorted({str(w.message) for w in caught})
    report["status"] = "PASS"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(dict(status=report["status"], groups=args.groups, report=str(args.output)), indent=2))


if __name__ == "__main__":
    main()
