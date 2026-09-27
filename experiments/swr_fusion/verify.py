"""Finite, synthetic SWR checks; never trains on the dataset or evaluates its val/test images."""

from __future__ import annotations

# ruff: noqa: E402 - bootstrap this worktree and disable auto-install before Ultralytics imports.

import argparse
import json
import os
import random
import subprocess
import sys
import tempfile
from collections import Counter
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ["YOLO_AUTOINSTALL"] = "false"

import numpy as np
import torch

from experiments.swr_fusion.train import (
    RECIPE,
    SWRTrainer,
    disable_oom_retry,
    environment_report,
    resolve_config,
    rng_digest,
)
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.modules.swr import SWRFusion, haar_inverse, haar_split
from ultralytics.nn.tasks import load_checkpoint
from ultralytics.utils import LOGGER
from ultralytics.utils.torch_utils import ModelEMA


def compare(actual: torch.Tensor, expected: torch.Tensor, atol: float = 1e-6, rtol: float = 1e-5) -> dict:
    """Assert numerical agreement and report both maximum absolute and stabilized relative error."""
    a, b = actual.detach().float(), expected.detach().float()
    torch.testing.assert_close(a, b, atol=atol, rtol=rtol)
    return {
        "max_abs": float((a - b).abs().max()),
        "max_rel": float(((a - b).abs() / b.abs().clamp_min(1e-6)).max()),
        "atol": atol,
        "rtol": rtol,
    }


def math_checks() -> dict:
    """Independently check signs, phases, channel grouping, initialization and invalid geometry."""
    report = {}
    x = torch.randn(2, 3, 6, 10, requires_grad=True)
    bands = haar_split(x)
    report["rectangular_roundtrip"] = compare(haar_inverse(bands), x)
    report["energy"] = compare(sum(t.square().sum() for t in bands), x.square().sum(), atol=1e-4)
    haar_inverse(bands).sum().backward()
    compare(x.grad, torch.ones_like(x))
    constant = torch.randn(2, 3, 3, 5).repeat_interleave(2, -2).repeat_interleave(2, -1)
    assert all(torch.count_nonzero(h) == 0 for h in haar_split(constant)[1:])
    signs = torch.tensor([[1, -1, -1, 1], [1, -1, 1, -1], [1, 1, -1, -1], [1, 1, 1, 1]], dtype=torch.float32) / 2
    for phase in range(4):
        impulse = torch.zeros(1, 3, 2, 2)
        impulse[0, :, phase // 2, phase % 2] = torch.tensor([1.0, 2.0, 4.0])
        transformed = haar_split(impulse)
        compare(torch.stack(transformed, -1).reshape(3, 4), torch.tensor([1.0, 2.0, 4.0])[:, None] * signs[phase])
        compare(haar_inverse(transformed), impulse)

    module = SWRFusion(128, 128, 64).eval()
    assert sum(p.numel() for p in module.parameters()) == 80008
    assert torch.count_nonzero(module.low_delta.weight) == 0
    assert all(torch.count_nonzero(g[-1].weight) == torch.count_nonzero(g[-1].bias) == 0 for g in module.band_gates)
    inputs = [torch.randn(2, 128, 6, 10), torch.randn(2, 128, 3, 5)]
    originals = [t.clone() for t in inputs]
    with torch.no_grad():
        lp, _, reconstructed, gains, _ = module._paths(inputs)
        report["initial_reconstruction"] = compare(reconstructed, lp)
        assert all(torch.equal(g, torch.ones_like(g)) for g in gains)
        low, *high = haar_split(lp)
        for band in range(3):
            modified = deepcopy(module)
            logits = torch.linspace(-2, 2, 8)
            modified.band_gates[band][-1].bias.copy_(logits)
            actual = modified._paths(inputs)
            # Explicit channel -> group mapping, independent of repeat_interleave.
            expected_high = [h.clone() for h in high]
            for channel in range(64):
                expected_high[band][:, channel] *= 2 * logits[channel // 8].sigmoid()
            report[f"nonzero_band_{band}"] = compare(actual[2], haar_inverse((low, *expected_high)))
        modified = deepcopy(module)
        modified.low_delta.weight.copy_(torch.eye(64).reshape(64, 64, 1, 1) * 0.125)
        sp = modified.p_s(inputs[1])
        context = modified.context_dw(modified.context_mix(torch.cat((low, sp), 1)))
        report["nonzero_low"] = compare(modified._paths(inputs)[2], haar_inverse((low + context * 0.125, *high)))
        assert all(torch.equal(a, b) for a, b in zip(inputs, originals))
        assert module(inputs).shape == (2, 64, 6, 10)
    for h, w, sh, sw in ((7, 10, 3, 5), (6, 9, 3, 5), (6, 10, 2, 5)):
        try:
            module([torch.zeros(1, 128, h, w), torch.zeros(1, 128, sh, sw)])
        except ValueError as error:
            assert "requires" in str(error)
        else:
            raise AssertionError("Invalid 2x geometry was accepted")
    report["phase_constant_grouping_input_integrity_and_invalid_shapes"] = "PASS"
    return report


def trace(model: torch.nn.Module, image: torch.Tensor) -> tuple[dict, dict]:
    """Capture raw one-to-one predictions and count all required inference branches."""
    counts, shapes, diagnostics = Counter(), {}, {}
    module = model.model[16]
    handles = []
    for name in ("low_delta", "band_gates.0", "band_gates.1", "band_gates.2", "blocks.0", "blocks.1"):
        layer = module.get_submodule(name)

        def count(_layer, _inputs, _outputs, name=name):
            counts[name] += 1

        handles.append(layer.register_forward_hook(count))

    def fusion(_layer, args):
        paths = args[0].split(64, 1)
        assert len(paths) == 3 and all(torch.isfinite(t).all() for t in paths)
        counts["reconstruction_at_fuse"] += 1

    handles.append(module.fuse.register_forward_pre_hook(fusion))

    def diagnostic(_layer, args):
        diagnostics.update(module.diagnostics(args[0]))

    # Separate explicit diagnostics after counted forward: no extra counted reconstruction.
    for i in (4, 13, 16, 17):

        def shape(_layer, _args, output, i=i):
            shapes[str(i)] = list(output.shape)

        handles.append(model.model[i].register_forward_hook(shape))

    def head_input(_layer, args):
        shapes["detect_inputs"] = [list(t.shape) for t in args[0]]

    handles.append(model.model[-1].register_forward_pre_hook(head_input))
    captured = []
    handles.append(module.register_forward_pre_hook(lambda _m, args: captured.append(args[0])))
    try:
        with torch.no_grad():
            output = model(image)
    finally:
        for handle in handles:
            handle.remove()
    assert all(
        counts[k] == 1
        for k in (
            "low_delta",
            "band_gates.0",
            "band_gates.1",
            "band_gates.2",
            "blocks.0",
            "blocks.1",
            "reconstruction_at_fuse",
        )
    )
    diagnostic(module, (captured[0],))
    raw = output if isinstance(output, dict) else output[1]
    return {k: raw["one2one"][k].detach().cpu() for k in ("boxes", "scores")}, {
        "calls": dict(counts),
        "shapes": shapes,
        "diagnostics": diagnostics,
        "model_dtype": str(next(model.parameters()).dtype),
        "input_dtype": str(image.dtype),
    }


def smoke(model: torch.nn.Module, device: str, amp: bool) -> tuple[torch.nn.Module, torch.nn.Module, dict]:
    """Perform three real native-loss/MuSGD updates, capped at 12 microbatches per precision."""
    model = deepcopy(model).to(device).train()
    trainer = SWRTrainer.__new__(SWRTrainer)
    trainer.model, trainer.args, trainer.data = model, model.args, {"nc": 1}
    trainer.optimizer = trainer.build_optimizer(model, name="MuSGD", lr=0.01, momentum=0.937, decay=0.0005)
    trainer.scaler = torch.amp.GradScaler("cuda", enabled=amp)
    trainer.ema = ModelEMA(model)
    counts = Counter(id(p) for group in trainer.optimizer.param_groups for p in group["params"])
    named = {k: p for k, p in model.named_parameters() if k.startswith("model.16.")}
    assert all(p.requires_grad and counts[id(p)] == 1 for p in named.values())
    observed, records, effective = set(), [], []
    step_count = []
    handle = trainer.optimizer.register_step_post_hook(lambda *_: step_count.append(1))
    for microbatch in range(12):
        batch = {
            "img": torch.rand(2, 3, 96, 128, device=device),
            "batch_idx": torch.tensor([0.0, 1.0], device=device),
            "cls": torch.zeros(2, 1, device=device),
            "bboxes": torch.tensor([[0.45, 0.5, 0.3, 0.4], [0.6, 0.4, 0.25, 0.2]], device=device),
        }
        before = {k: p.detach().clone() for k, p in named.items()}
        with torch.autocast(device_type="cuda" if amp else "cpu", dtype=torch.float16, enabled=amp):
            predictions = model(batch["img"])
            assert set(predictions) == {"one2many", "one2one"}
            assert all(not x.requires_grad for x in predictions["one2one"]["feats"])
            assert all(x.requires_grad for x in predictions["one2many"]["feats"])
            loss, _ = model.loss(batch, predictions)
            loss = loss.sum()
        if not torch.isfinite(loss):
            raise AssertionError(f"Nonfinite {device} loss at microbatch {microbatch}")
        trainer.scaler.scale(loss).backward()
        scale = trainer.scaler.get_scale()
        gradient = {
            k: float(p.grad.detach().float().norm() / scale) if p.grad is not None else 0.0 for k, p in named.items()
        }
        # A cloned optimizer with zero CURRENT task gradients isolates decay and existing momentum.
        control = deepcopy(trainer.optimizer)
        control.muon, control.sgd = trainer.optimizer.muon, trainer.optimizer.sgd
        task_control = {}
        for group, control_group in zip(trainer.optimizer.param_groups, control.param_groups):
            for p, q in zip(group["params"], control_group["params"]):
                q.grad = torch.zeros_like(q) if p.grad is not None else None
                task_control[id(p)] = q
        control.step()
        old_steps = len(step_count)
        trainer.optimizer_step()  # Native unscale, norm clipping, step, zero_grad and EMA.
        updated = len(step_count) > old_steps
        changed = {k: float((p.detach() - before[k]).abs().max()) for k, p in named.items()}
        task_effect = {k: float((p.detach() - task_control[id(p)]).abs().max()) for k, p in named.items()}
        if updated:
            effective.append(microbatch)
            observed.update(k for k in named if gradient[k] > 0 and changed[k] > 0 and task_effect[k] > 0)
        records.append(
            {
                "microbatch": microbatch,
                "loss": float(loss.detach()),
                "actual_optimizer_step": updated,
                "scale_before": scale,
                "scale_after": trainer.scaler.get_scale(),
                "task_grad_norm": gradient,
                "update_max_abs": changed,
                "task_effect_max_abs": task_effect,
            }
        )
        del control, task_control
        if len(effective) == 3:
            break
    handle.remove()
    assert len(effective) == 3, "Fewer than three actual optimizer updates within finite smoke budget"
    assert observed == named.keys(), f"No effective task-gradient update: {sorted(named.keys() - observed)}"
    assert trainer.ema.updates >= 3
    return (
        model.eval(),
        trainer.ema.ema,
        {
            "device": device,
            "amp": amp,
            "batch": 2,
            "imgsz": [96, 128],
            "synthetic_nonempty_labels": True,
            "formal_recipe_modified": False,
            "effective_updates": len(effective),
            "microbatches": len(records),
            "all_new_parameters_once_in_native_optimizer": True,
            "all_new_parameters_have_task_update": True,
            "steps": records,
        },
    )


def lifecycle(learned: torch.nn.Module, ema: torch.nn.Module, temp: Path) -> dict:
    """Check learned state through deepcopy, EMA, FP32/native saves, subprocess reload and native fusion."""
    model = deepcopy(learned).float().cpu().eval()
    image = torch.randn(1, 3, 160, 192)
    baseline, details = trace(model, image)
    report = {"unfused": details}
    for name, variant in (("deepcopy", deepcopy(model)), ("ema", deepcopy(ema).float().cpu().eval())):
        raw, extra = trace(variant, image)
        if name == "deepcopy":
            extra["errors"] = {k: compare(raw[k], baseline[k]) for k in raw}
        assert torch.count_nonzero(variant.model[16].low_delta.weight) > 0
        assert all(torch.count_nonzero(g[-1].weight) > 0 for g in variant.model[16].band_gates)
        report[name] = extra
    checkpoint, fixture = temp / "learned.pt", temp / "fixture.pt"
    model.criterion = None
    torch.save({"model": model, "train_args": vars(model.args)}, checkpoint)
    torch.save({"image": image}, fixture)
    reloaded, _ = load_checkpoint(str(checkpoint))
    raw, report["reload"] = trace(reloaded.eval(), image)
    report["reload"]["errors"] = {k: compare(raw[k], baseline[k]) for k in raw}
    assert all(torch.equal(v, reloaded.state_dict()[k]) for k, v in model.state_dict().items())

    wrapper = YOLO(str(checkpoint))
    native_checkpoint = temp / "native_save.pt"
    wrapper.save(str(native_checkpoint))
    native, _ = load_checkpoint(str(native_checkpoint))
    quantized, _ = trace(deepcopy(model).half().float(), image)
    raw, report["native_half_save"] = trace(native.eval(), image)
    report["native_half_save"]["roundtrip_errors"] = {k: compare(raw[k], quantized[k]) for k in raw}
    report["native_half_save"]["fp16_storage_rounding_max_abs"] = {
        k: float((raw[k] - baseline[k]).abs().max()) for k in raw
    }

    output = temp / "fresh_process.pt"
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--reload",
            str(checkpoint),
            "--fixture",
            str(fixture),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=True,
    )
    fresh = torch.load(output, weights_only=False)
    report["fresh_process"] = fresh["details"]
    report["fresh_process"]["errors"] = {k: compare(fresh["raw"][k], baseline[k]) for k in baseline}
    fused = deepcopy(model).fuse(verbose=False).eval()
    raw, report["fused"] = trace(fused, image)
    report["fused"]["errors"] = {k: compare(raw[k], baseline[k], atol=2e-4, rtol=2e-4) for k in raw}
    for k, v in model.model[16].state_dict().items():
        if k.startswith(("low_delta.", "band_gates.")):
            assert torch.equal(v, fused.model[16].state_dict()[k])
    assert fused.model[-1].cv2 is None and fused.model[-1].cv3 is None
    trainer = SWRTrainer.__new__(SWRTrainer)
    trainer.args, trainer.data = model.args, {"nc": 1, "names": {0: "crack"}, "channels": 3}
    restored = trainer.get_model(weights=reloaded, verbose=False)
    assert all(torch.equal(v, restored.state_dict()[k]) for k, v in reloaded.state_dict().items())
    report["learned_trainer_load_exact"] = True
    report["public_predictor"] = predict_contract(model, temp)
    return report


def entry_contracts(data_path: str, weights_path: str, temp: Path) -> dict:
    """Check entry rejection, metric interpretation and the actual native OOM exception branch."""
    from experiments.swr_fusion.train import dataset_inventory
    from experiments.swr_fusion.validate import metric_summary
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.utils import YAML

    options = SimpleNamespace(
        data=data_path,
        weights=weights_path,
        baseline_args=str(RECIPE),
        project=str(temp.resolve()),
        name="must-not-exist",
        device="0",
    )
    config, _ = resolve_config(options)
    target = Path(config["save_dir"])
    assert not target.exists()
    target.mkdir()
    try:
        resolve_config(options)
    except FileExistsError:
        pass
    else:
        raise AssertionError("Existing output was accepted")
    options.name = "drift-probe"
    wrong = YAML.load(RECIPE)
    wrong["batch"] = 16
    wrong_path = temp / "wrong-recipe.yaml"
    YAML.save(wrong_path, wrong)
    options.baseline_args = str(wrong_path)
    try:
        resolve_config(options)
    except ValueError:
        pass
    else:
        raise AssertionError("Recipe drift was accepted")
    metric = SimpleNamespace(
        mp=0.7,
        mr=0.6,
        f1=[0.646],
        map50=0.9,
        map=0.5,
        all_ap=np.array([[0.3, 0.8, 0.5]]),
        px=np.array([0.0, 0.5, 1.0]),
        p_curve=np.array([[0.2, 0.6, 1.0]]),
        r_curve=np.array([[1.0, 0.6, 0.2]]),
    )
    summary = metric_summary(metric, torch.tensor([0.5, 0.75, 0.95]))
    assert summary["ap75"] == 0.8 and abs(summary["fixed_confidence"]["0.25"]["precision"] - 0.4) < 1e-12

    class OneBatch:
        num_workers = 0

        def __len__(self):
            return 1

        def __iter__(self):
            yield {}

    trainer = SWRTrainer.__new__(SWRTrainer)
    trainer.args = SimpleNamespace(**config)
    trainer.train_loader = OneBatch()
    trainer.world_size, trainer.start_epoch, trainer.epochs, trainer.batch_size = 0, 0, 200, 32
    trainer.save_dir, trainer.plot_idx, trainer.amp = temp, [], False
    trainer.loss_names = ("box_loss", "cls_loss", "dfl_loss")
    trainer._setup_train = lambda: None
    trainer._model_train = lambda: None
    trainer.scheduler = SimpleNamespace(step=lambda: None)
    trainer.optimizer = SimpleNamespace(zero_grad=lambda: None, param_groups=[])
    trainer.callbacks = {event: [] for event in ("on_train_start", "on_train_epoch_start", "on_train_batch_start")}
    trainer.callbacks["on_train_epoch_start"].append(disable_oom_retry)

    def fail_batch(_batch):
        raise torch.cuda.OutOfMemoryError("Synthetic OOM policy probe; no allocation or training")

    trainer.preprocess_batch = fail_batch
    try:
        trainer._do_train()
    except torch.cuda.OutOfMemoryError:
        assert trainer.batch_size == trainer.args.batch == 32
    else:
        raise AssertionError("Native trainer swallowed the synthetic OOM")
    return {
        "configuration_rejection_and_metric_checks": True,
        "native_oom_propagates_with_batch32": True,
        "dataset": dataset_inventory(check_det_dataset(data_path, autodownload=False)),
    }


def predict_contract(model: torch.nn.Module, temp: Path) -> dict:
    """Run public YOLO prediction with nonzero SWR state and native automatic fusion."""
    checkpoint = temp / "predict-contract.pt"
    model = deepcopy(model).cpu().float().eval()
    model.criterion = None
    torch.save({"model": model, "train_args": vars(model.args)}, checkpoint)
    predictor = YOLO(str(checkpoint))
    counts, handles = Counter(), []
    names = ("low_delta", "band_gates.0", "band_gates.1", "band_gates.2", "blocks.0", "blocks.1", "fuse")
    for name in names:

        def count(_module, _inputs, _outputs, name=name):
            counts[name] += 1

        handles.append(predictor.model.model[16].get_submodule(name).register_forward_hook(count))
    try:
        result = predictor.predict(
            torch.rand(1, 3, 160, 192),
            imgsz=[160, 192],
            device="cpu",
            quantize=None,
            save=False,
            verbose=False,
            project=str(temp),
            name="prediction",
        )
    finally:
        for handle in handles:
            handle.remove()
    assert len(result) == 1 and all(counts[k] > 0 for k in names)
    assert predictor.model.is_fused()
    return {"calls": dict(counts), "native_fused": True, "result_count": len(result)}


def main():
    """Write a detailed finite validation report, or perform the isolated checkpoint-reload worker."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default="yolo26n.pt")
    parser.add_argument("--data")
    parser.add_argument("--baseline-args", default=str(RECIPE))
    parser.add_argument("--device", default="0")
    parser.add_argument("--output", required=True)
    parser.add_argument("--reload")
    parser.add_argument("--fixture")
    args = parser.parse_args()
    os.chdir(ROOT)
    torch.set_num_threads(4)
    LOGGER.setLevel("WARNING")
    if args.reload:
        model, _ = load_checkpoint(args.reload)
        image = torch.load(args.fixture, weights_only=True)["image"]
        raw, details = trace(model.eval(), image)
        torch.save({"raw": raw, "details": details}, args.output)
        return
    if not args.data:
        parser.error("--data is required for the real production setup_model audit")
    report = {
        "formal_training": "NOT_STARTED",
        "environment": environment_report(),
        "dataset_accuracy_evaluation": "NOT_RUN",
        "server_batch32_img640": "UNVERIFIED",
    }
    with tempfile.TemporaryDirectory(prefix="swr-verify-") as directory:
        temp = Path(os.path.relpath(directory, ROOT))
        config, report["configuration"] = resolve_config(
            SimpleNamespace(
                data=args.data,
                weights=args.weights,
                baseline_args=args.baseline_args,
                project=str(temp),
                name="setup",
                device=args.device,
            )
        )
        trainer = SWRTrainer(overrides=config)
        before_setup = (
            torch.get_rng_state(),
            random.getstate(),
            np.random.get_state(),
            torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
        )
        trainer.setup_model()
        trainer.set_model_attributes()
        fresh_model = trainer.model
        report["initialization"] = trainer.initialization_report
        torch.set_rng_state(before_setup[0])
        random.setstate(before_setup[1])
        np.random.set_state(before_setup[2])
        if before_setup[3]:
            torch.cuda.set_rng_state_all(before_setup[3])
        pretrained, _ = load_checkpoint(args.weights)
        reference = DetectionTrainer.get_model(
            trainer, cfg=deepcopy(pretrained.yaml), weights=pretrained, verbose=False
        )
        assert rng_digest() == report["initialization"]["rng_after_swr"], (
            rng_digest(),
            report["initialization"]["rng_after_swr"],
        )
        assert all(
            torch.equal(v, fresh_model.state_dict()[k])
            for k, v in reference.state_dict().items()
            if not k.startswith("model.16.")
        )
        report["independent_native_rng_and_state_audit"] = True
        report["haar_and_gates"] = math_checks()
        report["parameters"] = {
            "native_unfused": sum(p.numel() for p in reference.parameters()),
            "swr_unfused": sum(p.numel() for p in fresh_model.parameters()),
            "removed_layer16": sum(p.numel() for p in reference.model[16].parameters()),
            "new_layer16": sum(p.numel() for p in fresh_model.model[16].parameters()),
            "native_fused": sum(p.numel() for p in deepcopy(reference).fuse(verbose=False).parameters()),
            "swr_fused": sum(p.numel() for p in deepcopy(fresh_model).fuse(verbose=False).parameters()),
        }
        report["parameters"]["new_layer_breakdown"] = {
            k: sum(p.numel() for p in module.parameters()) for k, module in fresh_model.model[16].named_children()
        }
        report["wiring"] = {}
        inference = deepcopy(fresh_model).eval()
        for h, w in ((640, 640), (320, 512)):
            _, report["wiring"][f"{h}x{w}"] = trace(inference, torch.randn(1, 3, h, w))
        assert report["wiring"]["640x640"]["shapes"] == {
            "4": [1, 128, 80, 80],
            "13": [1, 128, 40, 40],
            "16": [1, 64, 80, 80],
            "17": [1, 64, 40, 40],
            "detect_inputs": [[1, 64, 80, 80], [1, 128, 40, 40], [1, 256, 20, 20]],
        }
        learned, ema, report["cpu_smoke"] = smoke(fresh_model, "cpu", False)
        report["cpu_lifecycle"] = lifecycle(learned, ema, temp)
        if torch.cuda.is_available() and args.device != "cpu":
            learned, ema, report["cuda_amp_smoke"] = smoke(fresh_model, f"cuda:{args.device}", True)
            report["cuda_learned_lifecycle_cpu_fp32"] = lifecycle(learned, ema, temp)
        else:
            report["cuda_amp_smoke"] = "UNVERIFIED"
        disable_oom_retry(trainer)
        assert trainer._oom_retries == 3
        report["oom_retry_budget"] = trainer._oom_retries
        report["entry_contracts"] = entry_contracts(args.data, args.weights, temp)
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(output), "status": "PASS", "parameters": report["parameters"]}, indent=2))


if __name__ == "__main__":
    main()
