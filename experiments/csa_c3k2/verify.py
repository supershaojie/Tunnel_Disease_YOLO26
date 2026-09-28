"""Bounded CSA verification: native construction, coordinate oracle, task updates and checkpoint lifecycle."""

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import tempfile
import warnings
from collections import Counter
from copy import deepcopy
from pathlib import Path

os.environ["YOLO_AUTOINSTALL"] = "false"
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.ops import deform_conv2d

from experiments.csa_c3k2.common import (
    MODEL_YAML,
    ROOT,
    audit_data,
    differences,
    environment,
    isolated_rng,
    parameter_count,
    recipe,
    rng_digest,
    rng_state,
    sha256,
    write_report,
)
from experiments.csa_c3k2.train import make_trainer
from ultralytics import YOLO
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.nn.modules import CSAUnit, CurveSampler
from ultralytics.utils import YAML
from ultralytics.utils.checks import check_amp
from ultralytics.utils.torch_utils import ModelEMA, autocast, init_seeds


def close(actual, expected, atol=2e-5, rtol=2e-5):
    """Report numerical error and enforce the same-device, same-dtype comparison tolerance."""
    delta = (actual - expected).abs()
    result = {
        "max_absolute_error": float(delta.max()),
        "max_relative_error": float((delta / expected.abs().clamp_min(1e-6)).max()),
        "atol": atol,
        "rtol": rtol,
    }
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    return result


def state_digest(model):
    """Hash all parameters and buffers to prove smoke and diagnostics do not change the pristine model."""
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def coordinate_checks():
    """Check offset math and actual installed operator directions/groups against a scalar bilinear oracle."""
    h, w = 5, 9
    deltas = torch.tensor(
        [
            [0.6, 0.7, 0.8, -0.2, 0.3, -0.4],
            [-0.3, 0.1, 0.4, 0.2, -0.5, 0.1],
            [0.1, -0.4, 0.2, 0.4, 0.2, -0.1],
            [-0.5, -0.2, 0.1, -0.1, 0.4, 0.3],
        ]
    )
    sequence = torch.stack(
        (
            deltas[:, 0] + deltas[:, 1] + deltas[:, 2],
            deltas[:, 0] + deltas[:, 1],
            deltas[:, 0],
            torch.zeros(4),
            deltas[:, 3],
            deltas[:, 3] + deltas[:, 4],
            deltas[:, 3] + deltas[:, 4] + deltas[:, 5],
        ),
        dim=1,
    )
    results = {}
    for direction in ("horizontal", "vertical"):
        sampler = CurveSampler(direction=direction).eval()
        x = torch.randn(2, 32, h, w)
        offsets = torch.zeros(2, 56, h, w)
        raw = deform_conv2d(x, offsets, sampler.kernel.weight, padding=sampler.kernel.padding)
        ordinary = F.conv2d(x, sampler.kernel.weight, padding=sampler.kernel.padding)
        results[direction + "_zero_raw"] = close(raw, ordinary)
        results[direction + "_zero_forward"] = close(sampler(x), sampler.bn(ordinary))
        with torch.no_grad():
            sampler.offset[-1].bias.copy_(deltas.atanh().flatten())
            sampler.kernel.weight.zero_()
            for channel in range(32):
                point = channel % 7
                ky, kx = (0, point) if direction == "horizontal" else (point, 0)
                sampler.kernel.weight[channel, channel, ky, kx] = 1
        steps, actual_sequence, offsets = sampler.curve_offsets(sampler.offset(x[:1]))
        close(steps, deltas[None, :, :, None, None].expand_as(steps), atol=1e-6)
        close(actual_sequence, sequence[None, :, :, None, None].expand_as(actual_sequence), atol=1e-6)
        assert actual_sequence[:, :, 3].count_nonzero() == 0
        assert actual_sequence.abs().max() > 1 and torch.diff(actual_sequence, dim=2).abs().max() <= 1
        pairs = offsets.reshape(1, 4, 7, 2, h, w)
        assert pairs[:, :, :, 1 if direction == "horizontal" else 0].count_nonzero() == 0
        for pattern in ("ramp", "impulse"):
            source = torch.zeros(1, 32, h, w)
            for channel in range(32):
                if pattern == "ramp":
                    source[0, channel] = channel * 100 + 10 * torch.arange(h)[:, None] + torch.arange(w)[None, :]
                else:
                    source[0, channel, channel % h, (channel * 2) % w] = channel + 1
            expected = torch.zeros_like(source)
            for channel in range(32):
                point, group = channel % 7, channel // 8
                for row in range(h):
                    for col in range(w):
                        sy = row + (float(sequence[group, point]) if direction == "horizontal" else point - 3)
                        sx = col + (point - 3 if direction == "horizontal" else float(sequence[group, point]))
                        y0, x0 = math.floor(sy), math.floor(sx)
                        for yy in (y0, y0 + 1):
                            for xx in (x0, x0 + 1):
                                if 0 <= yy < h and 0 <= xx < w:
                                    expected[0, channel, row, col] += (
                                        source[0, channel, yy, xx] * (1 - abs(sy - yy)) * (1 - abs(sx - xx))
                                    )
            actual = deform_conv2d(source, offsets, sampler.kernel.weight, padding=sampler.kernel.padding)
            results[direction + "_" + pattern] = close(actual, expected, atol=1e-3, rtol=2e-5)
    return {
        "status": "PASS",
        "shape": [h, w],
        "checks": results,
        "oracle": "independent scalar bilinear sampling with zero padding; 4 distinct groups, fractional offsets, every channel",
    }


def selector_checks(pristine):
    """Verify actual initialized selectors and independent nonuniform consecutive-channel group routing."""
    results = []
    for unit in pristine.model[4].blocks:
        features = tuple(
            torch.full((2, 32, 5, 9), value) + torch.arange(32)[None, :, None, None] / 100 for value in (1.0, 3.0, 9.0)
        )
        logits = unit.selector(torch.cat(features, 1))
        close(logits.reshape(2, 4, 3, 5, 9).softmax(2), torch.full((2, 4, 3, 5, 9), 1 / 3))
        uniform = CSAUnit.aggregate(features, logits)
        copy = deepcopy(unit)
        weights = torch.tensor([[0.7, 0.2, 0.1], [0.1, 0.7, 0.2], [0.2, 0.1, 0.7], [0.2, 0.5, 0.3]])
        with torch.no_grad():
            copy.selector[-1].bias.copy_(weights.log().flatten())
        actual = copy.aggregate(features, copy.selector(torch.cat(features, 1)))
        expected = torch.empty_like(actual)
        for channel in range(32):
            expected[:, channel] = sum(
                weights[channel // 8, branch] * features[branch][:, channel] for branch in range(3)
            )
        assert not torch.allclose(actual, uniform)
        results.append(close(actual, expected))
    return {"status": "PASS", "initial_weights": "1/3", "group_channels": 8, "nonuniform_checks": results}


def wiring_checks(pristine):
    """Observe the real graph at square and stride-aligned rectangular resolutions."""
    model = deepcopy(pristine).eval()
    reports = []
    for h, w in ((640, 640), (384, 640)):
        observed, counts, handles = {}, Counter(), []

        def record(name):
            def hook(module, inputs, output):
                observed[name] = list(output.shape)
                counts[name] += 1

            return hook

        for i in (3, 4, 5):
            handles.append(model.model[i].register_forward_hook(record(str(i))))
        for name, module in model.model[4].named_modules():
            if isinstance(module, (CSAUnit, CurveSampler)) or name == "stem" or name.endswith("reduce"):
                handles.append(module.register_forward_hook(record(name)))

        def detect_inputs(module, inputs):
            observed["Detect"] = [list(x.shape) for x in inputs[0]]

        handles.append(model.model[-1].register_forward_pre_hook(detect_inputs))
        try:
            with torch.no_grad():
                model(torch.rand(1, 3, h, w))
        finally:
            for handle in handles:
                handle.remove()
        assert observed["3"] == [1, 64, h // 8, w // 8]
        assert observed["4"] == observed["stem"] == [1, 128, h // 8, w // 8]
        assert observed["5"] == [1, 128, h // 16, w // 16]
        assert observed["Detect"] == [[1, c, h // stride, w // stride] for c, stride in ((64, 8), (128, 16), (256, 32))]
        for block in range(2):
            assert observed[f"blocks.{block}"] == [1, 64, h // 8, w // 8]
            assert observed[f"blocks.{block}.reduce"] == [1, 32, h // 8, w // 8]
            assert counts[f"blocks.{block}.horizontal"] == counts[f"blocks.{block}.vertical"] == 1
        reports.append({"input": [1, 3, h, w], "shapes": observed, "calls": dict(counts)})
    original = YAML.load(ROOT / "ultralytics/cfg/models/26/yolo26.yaml")
    modified = YAML.load(MODEL_YAML)
    assert original["head"] == modified["head"] and model.model[-1].f == [16, 19, 22]
    assert all(original["backbone"][i] == modified["backbone"][i] for i in range(11) if i != 4)
    return {"status": "PASS", "graphs": reports, "Detect_from": model.model[-1].f}


def parameter_audit(pristine, reference):
    """Compare every new leaf parameter count to independent static formulas, then native fused counts."""
    expected = {"stem": 64 * 128 + 2 * 128, "merge": 256 * 128 + 2 * 128}
    unit = {
        "reduce": 64 * 32 + 64,
        "local": 32 * 32 * 9 + 64,
        "project": 32 * 64 + 128,
        "selector.0": 96 * 16,
        "selector.2": 16 * 9,
        "selector.4": 16 * 12 + 12,
    }
    for direction in ("horizontal", "vertical"):
        unit.update(
            {
                f"{direction}.kernel": 32 * 32 * 7,
                f"{direction}.bn": 64,
                f"{direction}.offset.0": 32 * 9,
                f"{direction}.offset.2": 32 * 24 + 24,
            }
        )
    for i in range(2):
        expected.update({f"blocks.{i}.{key}": value for key, value in unit.items()})
    modules = dict(pristine.model[4].named_modules())
    measured = {key: parameter_count(modules[key]) for key in expected}
    assert expected == measured and sum(expected.values()) == parameter_count(pristine.model[4]) == 105624
    result = {"status": "PASS", "static_by_module": expected, "measured_by_module": measured}
    for label, model in (("b19", reference), ("csa", pristine)):
        fused = deepcopy(model).eval().fuse(verbose=False)
        result[label] = {
            "unfused": parameter_count(model),
            "fused": parameter_count(fused),
            "layer4_unfused": parameter_count(model.model[4]),
            "layer4_fused": parameter_count(fused.model[4]),
            "O2M_removed_by_native_fuse": fused.model[-1].cv2 is None and fused.model[-1].cv3 is None,
        }
    assert result["b19"]["unfused"] == 2504190 and result["b19"]["fused"] == 2375031
    result["net_delta"] = {
        k: result["csa"][k] - result["b19"][k] for k in ("unfused", "fused", "layer4_unfused", "layer4_fused")
    }
    result["FLOPs"] = (
        "UNVERIFIED: THOP does not count functional deform_conv2d, coordinates, softmax and weighted aggregation; its model summary is incomplete, not zero cost"
    )
    return result


def task_smoke(pristine, trainer, device, amp):
    """Require three actual task-gradient updates within 24 microbatches using native loss and MuSGD grouping."""
    model = deepcopy(pristine).to(device).train()
    model.args = deepcopy(trainer.args)
    model.requires_grad_(True)
    optimizer = trainer.build_optimizer(
        model, name="MuSGD", lr=model.args.lr0, momentum=model.args.momentum, decay=model.args.weight_decay
    )
    params = dict(model.model[4].named_parameters())
    membership = Counter(id(p) for group in optimizer.param_groups for p in group["params"])
    assert all(membership[id(p)] == 1 for p in model.parameters())
    group_report = {}
    for name, p in params.items():
        group = next(g for g in optimizer.param_groups if any(p is q for q in g["params"]))
        assert group["lr"] == model.args.lr0
        group_report[name] = {k: group.get(k) for k in ("param_group", "lr", "weight_decay", "use_muon")}
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    rows, updates, ever_gradient, changed_with_task_gradient = [], 0, set(), set()
    batch = {
        "img": torch.rand(2, 3, 64, 96, device=device),
        "batch_idx": torch.tensor([0, 1], device=device),
        "cls": torch.zeros(2, 1, device=device),
        "bboxes": torch.tensor([[0.5, 0.5, 0.5, 0.4], [0.4, 0.45, 0.3, 0.5]], device=device),
    }
    for microbatch in range(24):
        optimizer.zero_grad(set_to_none=True)
        before = {k: p.detach().clone() for k, p in params.items()}
        scale_before = scaler.get_scale()
        with autocast(enabled=amp, device=device.type):
            predictions = model(batch["img"])
            assert set(predictions) == {"one2many", "one2one"}
            loss, items = model.loss(batch, predictions)
            loss = loss.sum()
            if microbatch == 0:
                stem = model.model[4].stem.conv.weight
                detached = torch.autograd.grad(
                    model.criterion.one2one.loss(predictions["one2one"], batch)[0].sum(),
                    stem,
                    allow_unused=True,
                    retain_graph=True,
                )[0]
                assert detached is None, "Native O2O feature detach semantics changed"
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        all_finite = bool(torch.isfinite(loss)) and all(
            p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()
        )
        gradients = {
            name: float(p.grad.detach().float().norm()) if p.grad is not None else 0.0 for name, p in params.items()
        }
        if all_finite:
            ever_gradient.update(name for name, norm in gradients.items() if norm > 0)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        scaler.step(optimizer)
        scaler.update()
        deltas = {k: float((p.detach() - before[k]).abs().max()) for k, p in params.items()}
        applied = (
            all_finite
            and scaler.get_scale() >= scale_before
            and any(deltas[k] > 0 and gradients[k] > 0 for k in params)
        )
        if applied:
            updates += 1
            changed_with_task_gradient.update(k for k in params if deltas[k] > 0 and gradients[k] > 0)
        rows.append(
            {
                "microbatch": microbatch + 1,
                "loss": float(loss.detach()) if torch.isfinite(loss) else None,
                "loss_items": items.detach().float().cpu().tolist(),
                "unscaled_gradients_finite": all_finite,
                "gradient_norms": {k: v if math.isfinite(v) else None for k, v in gradients.items()},
                "parameter_max_changes": {k: v if math.isfinite(v) else None for k, v in deltas.items()},
                "scale_before": scale_before,
                "scale_after": scaler.get_scale(),
                "effective_update": applied,
            }
        )
        if updates >= 3:
            break
    missing = sorted(set(params) - ever_gradient)
    no_task_change = sorted(set(params) - changed_with_task_gradient)
    result = {
        "status": "PASS" if updates == 3 and not missing and not no_task_change else "FAIL",
        "device": str(device),
        "amp": amp,
        "smoke_shape": [2, 3, 64, 96],
        "formal_recipe_changed": False,
        "data": "synthetic nonempty crack boxes; not dataset accuracy",
        "effective_updates": updates,
        "microbatches": rows,
        "native_loss": type(model.criterion).__name__,
        "O2O_feature_detach": "PASS",
        "native_optimizer_groups": group_report,
        "missing_task_gradients": missing,
        "no_task_driven_change": no_task_change,
        "note": "Zero terminal predictor weights imply zero upstream gradients on the first step; later steps must activate them",
    }
    model.criterion = None
    return result, model.cpu().eval()


def lifecycle_checks(trained, temp, device):
    """Save learned/nonzero EMA state and reload through the real evaluation/prediction entry in a fresh process."""
    model = deepcopy(trained).eval()
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, CurveSampler):
                module.offset[-1].bias.add_(torch.linspace(-0.4, 0.3, 24))
                module.bn.bias.add_(0.12)
                module.bn.running_mean.add_(0.15)
            elif isinstance(module, CSAUnit):
                module.selector[-1].bias.add_(torch.tensor([0.3, -0.2, 0.1] * 4))
    ema = ModelEMA(model)
    with torch.no_grad():
        model.model[4].blocks[0].horizontal.offset[-1].bias.add_(0.01)
    ema.update(model)
    ema.ema.args = vars(trained.args).copy()
    saved = deepcopy(ema.ema).half()
    checkpoint = Path(temp) / "learned-ema.pt"
    torch.save({"model": None, "ema": saved, "updates": ema.updates, "train_args": vars(trained.args)}, checkpoint)
    # Also exercise the native public save() method; no custom class lives in this script's __main__ namespace.
    wrapper = YOLO(str(MODEL_YAML), verbose=False)
    wrapper.model, wrapper.ckpt = deepcopy(ema.ema), {}
    public_checkpoint = Path(temp) / "public-save.pt"
    wrapper.save(public_checkpoint)
    comparison = deepcopy(saved).float().eval()
    x = torch.rand(1, 3, 64, 96)
    with torch.no_grad():
        raw = comparison(x)[1]["one2one"]
    bundle = Path(temp) / "expected.pt"
    torch.save({"input": x, "boxes": raw["boxes"], "scores": raw["scores"], "state": comparison.state_dict()}, bundle)
    output = Path(temp) / "reload.json"
    script = """
import json,sys,torch
from pathlib import Path
from unittest.mock import patch
from torchvision.ops import deform_conv2d
from experiments.csa_c3k2.validate import load_for_evaluation
from experiments.csa_c3k2.common import Diagnostics,write_report
from experiments.csa_c3k2.verify import close
from ultralytics import YOLO
from ultralytics.nn.modules import CurveSampler
torch.set_num_threads(4)
checkpoint,bundle,output,public,device=sys.argv[1:]
expected=torch.load(bundle,weights_only=True)
wrapper=load_for_evaluation(checkpoint)
model=wrapper.model.float().eval()
assert all(torch.equal(model.state_dict()[k],v) for k,v in expected['state'].items())
public_model=load_for_evaluation(public).model
assert all(torch.equal(public_model.state_dict()[k],v) for k,v in expected['state'].items())
assert all(m.offset[-1].bias.count_nonzero()>0 for m in model.modules() if isinstance(m,CurveSampler))
x=expected['input']
with torch.no_grad():
    before=model(x)[1]['one2one']
    errors={k:close(before[k],expected[k]) for k in ('boxes','scores')}
    diagnostics=Diagnostics(model,1)
    model(x)
    diagnostics.close()
    assert len(diagnostics.records)==8
    with patch('torchvision.ops.deform_conv2d',wraps=deform_conv2d) as calls:
        model.fuse(verbose=False)
        after=model(x)[1]['one2one']
        assert calls.call_count==4
    errors['native_fuse']={k:close(after[k],before[k],atol=1e-4,rtol=1e-4) for k in ('boxes','scores')}
    model.fuse(verbose=False)
    again=model(x)[1]['one2one']
    errors['repeated_fuse']={k:close(again[k],after[k],atol=0,rtol=0) for k in ('boxes','scores')}
    torch.save({'model':model,'train_args':{}},str(Path(output).with_suffix('.pt')))
    fused_reload=YOLO(str(Path(output).with_suffix('.pt'))).model.eval()
    errors['fused_reload']={k:close(fused_reload(x)[1]['one2one'][k],after[k]) for k in ('boxes','scores')}
    half_status='UNVERIFIED'
    if device!='cpu':
        half=load_for_evaluation(checkpoint).model.to(device).half().eval()
        y=half(x.to(device).half())[1]['one2one']
        assert all(torch.isfinite(y[k]).all() and y[k].dtype==torch.float16 for k in ('boxes','scores'))
        half.fuse(verbose=False)
        y=half(x.to(device).half())[1]['one2one']
        assert all(torch.isfinite(y[k]).all() for k in ('boxes','scores'))
        half_status='PASS'
    results=wrapper.predict(source=x,device='cpu',imgsz=96,half=False,quantize=None,verbose=False,save=False)
    assert len(results)==1
write_report(output,{'status':'PASS','errors':errors,'EMA_and_public_save_state_equal':True,'restricted_load':True,'fused_deform_calls':4,'half_model_forward_and_fuse':half_status,'predict':'PASS','diagnostics':diagnostics.records})
"""
    child_env = dict(os.environ, ULTRALYTICS_SAFE_LOAD="true", YOLO_AUTOINSTALL="false", PYTHONPATH=str(ROOT))
    completed = subprocess.run(
        [sys.executable, "-c", script, str(checkpoint), str(bundle), str(output), str(public_checkpoint), str(device)],
        cwd=ROOT,
        env=child_env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    if completed.returncode:
        raise RuntimeError(
            f"New-process lifecycle failed ({completed.returncode}): {completed.stdout}\n{completed.stderr}"
        )
    report = json.loads(output.read_text(encoding="utf-8"))
    report["subprocess_output"] = completed.stdout + completed.stderr
    return report


def oom_checks(trainer):
    """Check experimental fail-fast behavior and retained native first-epoch retry policy without allocating memory."""
    before = trainer.batch_size
    for error in (
        torch.cuda.OutOfMemoryError("synthetic memory-policy test"),
        RuntimeError("CUDNN_STATUS_INTERNAL_ERROR"),
    ):
        try:
            trainer._handle_train_batch_failure(error, 0)
        except RuntimeError as raised:
            assert "fixed batch=32" in str(raised) and trainer.batch_size == before and trainer.args.batch == before
        else:
            raise AssertionError("CSA memory failure was swallowed")
    native = object.__new__(BaseTrainer)
    native.start_epoch, native._oom_retries, native.batch_size = 0, 0, 32
    native.args = deepcopy(trainer.args)
    native._handle_train_batch_failure(torch.cuda.OutOfMemoryError("synthetic memory-policy test"), 0)
    assert native.batch_size == native.args.batch == 16 and native._oom_retries == 1
    return {"status": "PASS", "experimental_batch_after_failure": before, "native_retry_batch": native.batch_size}


def restricted_method_checks():
    """Allow only the native fused bound methods, retaining rejection of unrelated instance attributes."""
    import pickle

    from ultralytics.nn.modules import Conv
    from ultralytics.nn.tasks import _SafeLoad

    handler = next(item[0] for item in _SafeLoad._build() if isinstance(item, tuple) and item[1] == "builtins.getattr")
    module = Conv(3, 8)
    assert handler(module, "forward_fuse").__self__ is module
    for obj, name in ((module, "train"), (module, "__dict__"), (torch.nn.Linear(2, 2), "forward_fuse")):
        try:
            handler(obj, name)
        except pickle.UnpicklingError:
            continue
        raise AssertionError(f"Unexpected restricted-load instance permission: {type(obj).__name__}.{name}")
    return {"status": "PASS", "native_forward_fuse": "allowed", "other_instance_attributes": "rejected"}


def main():
    """Execute only the prescribed finite checks and write PASS/FAIL/UNVERIFIED evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "weights", "baseline-args", "report"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    os.chdir(ROOT)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    report = {
        "formal_training": "NOT_STARTED",
        "full_val_test": "NOT_STARTED",
        "server_B32_640": "UNVERIFIED",
        "environment": environment(),
        "checks": {},
    }

    def check(name, action):
        print(f"CSA verify: {name}", flush=True)
        try:
            before_rng = rng_digest()
            with isolated_rng(), warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = action()
            assert rng_digest() == before_rng, f"{name} polluted external RNG"
            result["warnings"] = sorted({str(w.message) for w in caught})
        except Exception as error:
            result = {"status": "FAIL", "error": f"{type(error).__name__}: {error}"}
        report["checks"][name] = result
        write_report(args.report, report)
        print(f"CSA verify: {name} {result['status']}", flush=True)
        return result

    config, report["configuration"] = recipe(args)
    if Path(args.report).resolve().is_relative_to(Path(config["save_dir"])):
        raise ValueError("Verification report must be outside the formal output directory")
    try:
        check("dataset", lambda: audit_data(config["data"]))
        init_seeds(42, deterministic=True)
        with tempfile.TemporaryDirectory(prefix="csa-verify-") as temp:
            trainer = make_trainer(config, Path(temp) / "production")
            setup_start = rng_state()
            trainer.setup_model()
            trainer.set_model_attributes()
            pristine = trainer.model
            production_rng = rng_digest()
            baseline = dict(
                config,
                model=config["pretrained"],
                pretrained=True,
                save_dir=str(Path(temp) / "native"),
                project=str(temp),
                name="native",
            )
            reference_trainer = DetectionTrainer(cfg=baseline, overrides={})
            # The native Events singleton consumes Python RNG on its first import in Trainer.__init__.
            # Compare the two real setup_model paths from the same recorded entry state, including that draw.
            random.setstate(setup_start["python"])
            np.random.set_state(setup_start["numpy"])
            torch.set_rng_state(setup_start["cpu"])
            if setup_start["cuda"]:
                torch.cuda.set_rng_state_all(setup_start["cuda"])
            reference_trainer.setup_model()
            reference_trainer.set_model_attributes()
            reference = reference_trainer.model
            report["rng_comparison"] = {
                "production": production_rng,
                "native": rng_digest(),
                "production_internal": trainer.transfer_report["rng_after_reference"],
                "scope": "Identical recorded RNG at entry to both real setup_model calls; native one-time Events import is outside this construction comparison",
            }
            assert production_rng == rng_digest(), "Independent native setup_model RNG differs"
            assert all(
                torch.equal(v, pristine.state_dict()[k])
                for k, v in reference.state_dict().items()
                if not k.startswith("model.4.")
            )
            report["checks"]["production_state_rng"] = dict(
                trainer.transfer_report, independent_native_setup_rng_equal=True
            )
            report["actual_trainer_args"] = vars(trainer.args).copy()
            report["actual_differences"] = differences(config, vars(trainer.args))
            pristine_hash, weight_hash = state_digest(pristine), sha256(config["pretrained"])
            check("coordinates", coordinate_checks)
            check("selector", lambda: selector_checks(pristine))
            check("graph", lambda: wiring_checks(pristine))
            check("parameters", lambda: parameter_audit(pristine, reference))
            check("oom_policy", lambda: oom_checks(trainer))
            check("restricted_bound_methods", restricted_method_checks)
            trained = {}

            def smoke(device, amp):
                result, model = task_smoke(pristine, trainer, device, amp)
                trained[str(device)] = model
                return result

            check("cpu_fp32", lambda: smoke(torch.device("cpu"), False))
            cuda_device = trainer.device if trainer.device.type == "cuda" else None
            if cuda_device is not None:

                def native_amp():
                    passed = check_amp(deepcopy(pristine).to(cuda_device).eval())
                    return {"status": "PASS" if passed else "FAIL", "native_result": passed}

                check("native_amp", native_amp)
                check("cuda_amp", lambda: smoke(cuda_device, True))
            else:
                report["checks"]["cuda_amp"] = {
                    "status": "UNVERIFIED",
                    "reason": "No CUDA device selected/available; run --device 0 on the target environment",
                }
            if trained:
                check(
                    "lifecycle",
                    lambda: lifecycle_checks(
                        trained.get(str(cuda_device), trained["cpu"]), temp, cuda_device or torch.device("cpu")
                    ),
                )
            assert state_digest(pristine) == pristine_hash and sha256(config["pretrained"]) == weight_hash
            report["checks"]["isolation"] = {
                "status": "PASS",
                "pristine_parameters_and_BN_unchanged": True,
                "original_checkpoint_unchanged": True,
                "each_check_restores_RNG": True,
            }
    except Exception as error:
        report["checks"]["setup"] = {"status": "FAIL", "error": f"{type(error).__name__}: {error}"}
    report["status"] = "FAIL" if any(x["status"] == "FAIL" for x in report["checks"].values()) else "PASS"
    report["reproducibility"] = (
        "Native deterministic=True, warn_only=True retained. Captured warnings are reported; no bitwise reproducibility claim."
    )
    write_report(args.report, report)
    print(f"CSA verification {report['status']}: {Path(args.report).resolve()}")
    return 1 if report["status"] == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
