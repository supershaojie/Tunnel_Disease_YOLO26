"""Bounded structural, native-loss and checkpoint-lifecycle verification; never run full val/test or formal training."""

import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import traceback
from collections import Counter
from copy import copy, deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.sce_fusion.train import (
    ROOT,
    construct,
    parser,
    remap_key,
    rng_fingerprint,
    write_report,
)
from experiments.sce_fusion.validate import (
    Diagnostics,
    FP32Validator,
    load_for_evaluation,
    parameter_audit,
    summarize_metrics,
)

import numpy as np
import torch

from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.modules import Detect, Index, SCEContextBlock, SCEFusion, SCERouter
from ultralytics.nn.tasks import DetectionModel, load_checkpoint
from ultralytics.utils import YAML
from ultralytics.utils.checks import check_amp
from ultralytics.utils.torch_utils import ModelEMA


def require(value, message):
    """Raise a check failure that remains active even when Python assertions are disabled."""
    if not value:
        raise AssertionError(message)


def comparison(actual, expected, atol=1e-6, rtol=1e-5):
    """Report maximum absolute/relative error with explicit tolerances."""
    a, b = actual.detach().float(), expected.detach().float()
    delta = (a - b).abs()
    result = {
        "max_absolute": float(delta.max()),
        "max_relative": float((delta / b.abs().clamp_min(1e-8)).max()),
        "atol": atol,
        "rtol": rtol,
    }
    require(torch.allclose(a, b, atol=atol, rtol=rtol), f"Numerical mismatch: {result}")
    return result


def rectangle():
    """Return the specified legal non-square feature pyramid."""
    return [torch.randn(2, c, h, w) for c, h, w in ((64, 12, 20), (128, 6, 10), (256, 3, 5))]


def module_checks():
    """Check input contracts, source isolation, non-mutation and controlled residual passthrough."""
    module = SCEFusion([64, 128, 256]).eval()
    x = rectangle()
    originals = [v.clone() for v in x]
    identities = [id(v) for v in x]
    parameter_ids = {id(p) for p in module.parameters()}
    y = module(x)
    require(isinstance(y, tuple) and len(y) == 3, "Default forward must return a three-tensor tuple")
    require([id(v) for v in x] == identities, "Input list was modified")
    for original, before, out in zip(x, originals, y):
        require(torch.equal(original, before), "Input feature was modified in-place")
        require(
            (out.shape, out.dtype, out.device) == (original.shape, original.dtype, original.device), "Output contract"
        )
    require(parameter_ids == {id(p) for p in module.parameters()}, "Forward created parameters")
    blocks = [b for c in module.contexts for b in c]
    require(len(blocks) == 6 and all(isinstance(b, SCEContextBlock) for b in blocks), "Expected six context blocks")
    pointers = [p.data_ptr() for b in blocks for p in b.parameters()]
    require(len(pointers) == len(set(pointers)), "Contexts share parameter storage")
    projected, encoded, router_inputs = {}, {}, {}
    handles = []

    def context_input(i):
        def capture(m, args):
            projected[i] = args[0].clone()

        return capture

    def context_output(i):
        def capture(m, args, out):
            encoded[i] = out.clone()

        return capture

    def route_input(i):
        def capture(m, args):
            router_inputs[i] = tuple(v.clone() for v in args)

        return capture

    for i in range(3):
        handles += [
            module.contexts[i].register_forward_pre_hook(context_input(i)),
            module.contexts[i].register_forward_hook(context_output(i)),
            module.routers[i].register_forward_pre_hook(route_input(i)),
        ]
    with torch.no_grad():
        module(x)
    for handle in handles:
        handle.remove()
    for i, (j, k) in enumerate(SCEFusion.sources):
        for actual, expected in zip(router_inputs[i], (projected[i], encoded[j], encoded[k])):
            require(torch.equal(actual, expected), "Router target/source order mismatch")
        require(torch.equal(module.contexts[i](projected[i]), encoded[i]), "Context reads another source")
    rejected = []
    for name, bad in (
        ("channel", [x[0][:, :63], x[1], x[2]]),
        ("height", [x[0][:, :, :-1], x[1], x[2]]),
        ("width", [x[0][:, :, :, :-1], x[1], x[2]]),
        ("batch", [x[0][:1], x[1], x[2]]),
        ("device", [x[0].to("meta"), x[1], x[2]]),
        ("dtype", [x[0].half(), x[1], x[2]]),
    ):
        try:
            module(bad)
        except ValueError as error:
            rejected.append({"case": name, "message": str(error)})
        else:
            raise AssertionError(f"Invalid {name} accepted")
    passthrough = deepcopy(module)
    with torch.no_grad():
        passthrough.lambdas.zero_()
    require(all(torch.equal(a, b) for a, b in zip(x, passthrough(x))), "lambda=0 is not exact passthrough")
    require(torch.equal(module.lambdas, torch.full((3,), 0.1)), "Formal residual initial value changed")
    half = deepcopy(module).half()
    with torch.no_grad():
        half_outputs = half([v.half() for v in x])
    require(all(v.dtype == torch.float16 and torch.isfinite(v).all() for v in half_outputs), "model.half dtype path")
    return {
        "shapes": [list(v.shape) for v in y],
        "rejected_inputs": rejected,
        "contexts_independent": True,
        "router_order": SCEFusion.sources,
        "lambda_zero_exact": True,
        "model_half": "PASS",
        "parameters_created_in_forward": False,
    }


def router_checks():
    """Use distinct source/channel markers and distinct per-group probabilities, not just global means."""
    router = SCERouter().eval()
    x = torch.randn(2, 64, 6, 10)
    first = torch.arange(64).reshape(1, 64, 1, 1).expand_as(x).float() + 2
    second = first * 3 + 7
    comparison(router.probabilities(x, first, second), torch.full((2, 8, 3, 6, 10), 1 / 3))
    uniform = comparison(router(x, first, second), (first + second) / 3, atol=1e-5)
    values = torch.tensor([[g + 1, 10 - g, 2 * g + 3] for g in range(8)], dtype=torch.float32)
    p = values / values.sum(1, keepdim=True)
    with torch.no_grad():
        router.logits[-1].bias.copy_(p.log().reshape(-1))
    actual_p = router.probabilities(x, first, second)
    comparison(actual_p, p[None, :, :, None, None].expand_as(actual_p))
    expected = p[:, 0].repeat_interleave(8)[None, :, None, None] * first
    expected += p[:, 1].repeat_interleave(8)[None, :, None, None] * second
    groups = comparison(router(x, first, second), expected, atol=2e-5)
    return {
        "uniform": uniform,
        "nonuniform_group_test": groups,
        "group_probabilities": p.tolist(),
        "softmax_axis": 2,
        "contiguous_channels_per_group": 8,
        "nonzero_candidates_renormalized": False,
    }


def axis_checks():
    """Check non-square pooling, broadcasting, independent axes and seven-tap impulse direction."""
    block = SCEContextBlock().eval()
    z = torch.zeros(2, 64, 11, 17)
    z[:, :, 5, :] = 2
    z[:, :, :, 8] += 3
    observed = {}

    def capture(name):
        def hook(m, args):
            observed[name] = args[0].clone()

        return hook

    handles = [
        block.height[0].register_forward_pre_hook(capture("height")),
        block.width[0].register_forward_pre_hook(capture("width")),
    ]
    with torch.no_grad():
        actual = block(z)
    for handle in handles:
        handle.remove()
    u = block.local(z)
    comparison(observed["height"], torch.cat((u.mean(3, keepdim=True), u.max(3, keepdim=True).values), 1))
    comparison(observed["width"], torch.cat((u.mean(2, keepdim=True), u.max(2, keepdim=True).values), 1))
    ah = block.height(observed["height"]).sigmoid()
    aw = block.width(observed["width"]).sigmoid()
    expected = z + block.channel_out(block.channel_mix(u + u * ah + u * aw))
    broadcast = comparison(actual, expected)
    require(not torch.allclose(actual, z), "Axial processing collapsed to identity")
    require(
        not {p.data_ptr() for p in block.height.parameters()} & {p.data_ptr() for p in block.width.parameters()},
        "Axis parameters are shared",
    )
    impulses = []
    for conv, shape, index, start in (
        (block.height[-1], (1, 64, 11, 1), 5, 2),
        (block.width[-1], (1, 64, 1, 17), 8, 5),
    ):
        isolated = deepcopy(conv)
        with torch.no_grad():
            isolated.weight.fill_(1)
            isolated.bias.zero_()
            signal = torch.zeros(shape)
            signal.flatten(2)[:, :, index] = 1
            response = isolated(signal).flatten(2)
            expected = torch.zeros_like(response)
            expected[:, :, start : start + 7] = 1
            comparison(response, expected, atol=0, rtol=0)
        impulses.append({"kernel": list(conv.kernel_size), "nonzero_indices": list(range(start, start + 7))})
    return {
        "descriptor_shapes": {k: list(v.shape) for k, v in observed.items()},
        "broadcast": broadcast,
        "impulses": impulses,
    }


def native_checks(trainer, weights):
    """Compare production construction against a separately seeded native nc=1 reference."""
    source, _ = load_checkpoint(weights)
    python_rng, numpy_rng, cpu_rng, cuda_rng = trainer.reference_rng
    random.setstate(python_rng)
    np.random.set_state(numpy_rng)
    torch.set_rng_state(cpu_rng)
    if cuda_rng:
        torch.cuda.set_rng_state_all(cuda_rng)
    reference = DetectionTrainer.get_model(
        trainer, cfg=deepcopy(source.yaml), weights=source, verbose=trainer.reference_verbose
    )
    independent_rng = rng_fingerprint()
    require(independent_rng == trainer.transfer_report["rng_after_sce"], "Production RNG differs from native reference")
    target = trainer.model.state_dict()
    for key, value in reference.state_dict().items():
        require(torch.equal(value, target[remap_key(key)]), f"Native reference differs at {key}")
    native_optimizer = DetectionTrainer.build_optimizer(
        trainer, reference, name="MuSGD", lr=0.01, momentum=0.937, decay=0.0005
    )
    sce_optimizer = trainer.build_optimizer(trainer.model, name="MuSGD", lr=0.01, momentum=0.937, decay=0.0005)

    def memberships(model, optimizer):
        names = {id(p): n for n, p in model.named_parameters()}
        result = {}
        for group in optimizer.param_groups:
            for param in group["params"]:
                name = names[id(param)]
                require(name not in result, f"Duplicate optimizer parameter: {name}")
                result[name] = {
                    k: group.get(k) for k in ("lr", "weight_decay", "momentum", "nesterov", "use_muon", "param_group")
                }
        return result

    old, new = memberships(reference, native_optimizer), memberships(trainer.model, sce_optimizer)
    for key, group in old.items():
        require(new[remap_key(key)] == group, f"Changed native optimizer rule for {key}")
    for key, param in trainer.model.model[23].named_parameters():
        group = new["model.23." + key]
        require(group["lr"] == 0.01, f"Special SCE learning rate: {key}")
        require(bool(group["use_muon"]) == (param.ndim >= 2), f"Non-native MuSGD grouping: {key}")
    batch = trainer.batch_size
    trainer._oom_retries = 0
    try:
        trainer._oom_retries += 1
    except RuntimeError as error:
        oom = str(error)
    else:
        raise AssertionError("Fixed-batch trainer accepted an OOM retry")
    require(trainer.batch_size == batch == 32, "OOM request mutated the formal batch")
    return {
        "independent_reference_rng": independent_rng,
        "all_native_states_equal": True,
        "native_optimizer_parameters_compared": len(old),
        "total_parameters_grouped_once": len(new),
        "relocated_classification_lr": new["model.27.cv3.0.2.weight"]["lr"],
        "oom_failure": oom,
    }


def graph_checks(pristine):
    """Trace one real 640 forward, checking cached features, routing nodes and native detach semantics."""
    model = deepcopy(pristine).eval()
    require(type(model.model[-1]) is Detect and model.model[-1].i == 27, "Detect graph metadata")
    require(model.stride.tolist() == [8, 16, 32], "Incorrect automatically inferred strides")
    native_yaml = YAML.load(ROOT / "ultralytics/cfg/models/26/yolo26.yaml")
    require(
        model.yaml["backbone"] + model.yaml["head"][:12] == native_yaml["backbone"] + native_yaml["head"][:12],
        "Original layers 0..22 changed",
    )
    require(set((16, 19, 22, 23, 24, 25, 26)).issubset(model.save), "Missing savelist nodes")
    cache, counts, seen = {}, Counter(), {}
    handles = []

    def capture(i):
        def hook(m, args, out):
            counts[i] += 1
            if i in (16, 19, 22):
                cache[i] = (out, out.clone())
            seen[i] = out

        return hook

    for i in (16, 19, 22, 23, 24, 25, 26, 27):
        handles.append(model.model[i].register_forward_hook(capture(i)))

    def detect_input(m, args):
        require(all(a is seen[24 + i] for i, a in enumerate(args[0])), "Detect did not consume all Index outputs")

    handles.append(model.model[27].register_forward_pre_hook(detect_input))
    with torch.no_grad():
        model(torch.rand(1, 3, 640, 640))
    for handle in handles:
        handle.remove()
    require(all(n == 1 for n in counts.values()), "SCE or graph nodes executed more than once")
    require(all(torch.equal(a, b) for a, b in cache.values()), "Cached original feature mutated")
    for i in range(3):
        require(type(model.model[24 + i]) is Index and seen[24 + i] is seen[23][i], "Index recomputed its output")
    head = deepcopy(model.model[-1]).train()
    features = [v.requires_grad_() for v in rectangle()]
    preds = head(features)
    require(all(not v.requires_grad for v in preds["one2one"]["feats"]), "Native O2O detach changed")
    require(all(v.requires_grad for v in preds["one2many"]["feats"]), "Native O2M gradient path detached")
    bad = deepcopy(model.yaml)
    bad["head"][-4][3][0] = 128
    try:
        DetectionModel(bad, verbose=False)
    except ValueError as error:
        parser_error = str(error)
    else:
        raise AssertionError("Parser accepted an incorrect Index channel declaration")
    return {
        "calls": dict(counts),
        "sce_output_shapes": [list(v.shape) for v in seen[23]],
        "stride": model.stride.tolist(),
        "index_error": parser_error,
        "native_detach": "PASS",
    }


def smoke(trainer, device, amp):
    """Complete three real native-loss updates, with a hard cap of 24 microbatches per precision."""
    model = deepcopy(trainer.model).to(device).train()
    for p in model.parameters():
        p.requires_grad_(True)
    optimizer = trainer.build_optimizer(
        model, name="MuSGD", lr=trainer.args.lr0, momentum=trainer.args.momentum, decay=trainer.args.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    sce = model.model[23]
    names = dict(sce.named_parameters())
    ids = Counter(id(p) for group in optimizer.param_groups for p in group["params"])
    require(all(ids[id(p)] == 1 for p in names.values()), "SCE parameter missing or duplicated in native optimizer")
    initial = {k: v.detach().clone() for k, v in names.items()}
    gradient_hits = {k: False for k in names}
    lambda_hits = torch.zeros(3, dtype=torch.bool, device=device)
    rows, updates = [], 0
    for microbatch in range(24):
        optimizer.zero_grad(set_to_none=True)
        batch = {
            "img": torch.rand(2, 3, 96, 160, device=device),
            "batch_idx": torch.tensor([0, 1], device=device),
            "cls": torch.zeros(2, 1, device=device),
            "bboxes": torch.tensor([[0.45, 0.5, 0.3, 0.65], [0.6, 0.4, 0.35, 0.5]], device=device),
        }
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            loss, items = model(batch)
            total = loss.sum()
        scaler.scale(total).backward()
        scaler.unscale_(optimizer)
        finite = bool(
            torch.isfinite(total) and all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        )
        norms = {
            k: float(p.grad.float().norm()) if p.grad is not None and torch.isfinite(p.grad).all() else None
            for k, p in names.items()
        }
        before = {k: p.detach().clone() for k, p in names.items()}
        scale_before = scaler.get_scale()
        if finite:
            for k, p in names.items():
                gradient_hits[k] |= p.grad is not None and bool(torch.count_nonzero(p.grad))
            lambda_hits |= sce.lambdas.grad != 0
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        scaler.step(optimizer)
        scaler.update()
        changed_with_gradient = [
            k for k, p in names.items() if norms[k] is not None and norms[k] > 0 and not torch.equal(before[k], p)
        ]
        effective = finite and bool(changed_with_gradient) and scaler.get_scale() >= scale_before
        if effective:
            updates += 1
        rows.append(
            {
                "microbatch": microbatch + 1,
                "loss": float(total) if torch.isfinite(total) else None,
                "loss_items": items.detach().float().cpu().tolist() if torch.isfinite(items).all() else None,
                "finite_unscaled_gradients": finite,
                "gradient_norms": norms,
                "scale_before": scale_before,
                "scale_after": scaler.get_scale(),
                "effective_update": effective,
                "changed_with_task_gradient": changed_with_gradient,
            }
        )
        if updates >= 3 and all(gradient_hits.values()) and bool(lambda_hits.all()):
            break
    missing = [k for k, active in gradient_hits.items() if not active]
    changed = {k: float((p.detach() - initial[k]).abs().max()) for k, p in names.items()}
    require(
        updates >= 3 and not missing and bool(lambda_hits.all()),
        f"Incomplete task learning: updates={updates}, missing={missing}",
    )
    require(all(v > 0 for v in changed.values()), "Some SCE parameters did not change after receiving task gradients")
    require(type(model.criterion).__name__ == "E2ELoss", "Smoke bypassed native O2M/O2O loss")
    result = {
        "device": str(device),
        "amp": amp,
        "batch": 2,
        "image_size": [96, 160],
        "synthetic_nonempty_boxes": True,
        "criterion": type(model.criterion).__name__,
        "optimizer": type(optimizer).__name__,
        "effective_updates": updates,
        "microbatch_cap": 24,
        "microbatches": rows,
        "all_new_parameters_grouped_once": True,
        "task_gradient_reached_every_parameter": True,
        "lambda_task_gradients": lambda_hits.cpu().tolist(),
        "parameter_max_changes": changed,
        "router_upstream_after_zero_output_update": True,
        "formal_recipe_modified": False,
    }
    return result, (model, optimizer, scaler)


def raw_o2o(model, image):
    """Compare raw O2O values rather than top-k order or the O2M branch removed by native fuse."""
    with torch.no_grad():
        output = model(image)[1]["one2one"]
    return torch.cat((output["boxes"], output["scores"]), 1)


def bounded_evaluation(trainer, model, directory):
    """Exercise the native FP32 validation/metric path on two copied nonempty val samples only."""
    subset = Path(directory) / "val_subset"
    (subset / "images").mkdir(parents=True)
    (subset / "labels").mkdir()
    selected = []
    for source in sorted(Path(trainer.data["val"]).iterdir()):
        label = Path(str(source).replace(f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}")).with_suffix(".txt")
        if source.is_file() and label.is_file() and label.read_text().strip():
            shutil.copyfile(source, subset / "images" / source.name)
            shutil.copyfile(label, subset / "labels" / label.name)
            selected.append(source.name)
            if len(selected) == 2:
                break
    require(len(selected) == 2, "Missing two nonempty validation samples")
    data = subset / "data.yaml"
    YAML.save(data, {"path": str(subset), "train": "images", "val": "images", "names": {0: "crack"}})
    validator = FP32Validator(
        args=dict(
            data=str(data),
            project=str(directory),
            name="bounded_eval",
            device="cpu",
            imgsz=160,
            batch=2,
            workers=0,
            conf=0.001,
            iou=0.7,
            max_det=300,
            rect=True,
            augment=False,
            half=False,
            quantize=None,
            end2end=True,
            plots=False,
            task="detect",
            mode="val",
        )
    )
    validator(model=model)
    metrics = summarize_metrics(validator)
    require(len(metrics["curves"]) == 4 and set(metrics["fixed_confidence"]) == {"0.25", "0.5"}, "Metric/curve schema")
    return {
        "samples": selected,
        "count": validator.seen,
        "model_dtype": validator.actual_model_dtype,
        "input_dtype": validator.actual_input_dtype,
        "AP75_iou": 0.75,
        "metrics_and_four_curves": "PASS",
        "scope": "Two samples at 160/B2, only an entry-point check; not a dataset accuracy result",
    }


def lifecycle(trainer, learned, directory):
    """Use native EMA/save plus actual evaluation loading, including a fresh restricted-loading process."""
    model, optimizer, scaler = learned
    model.eval()
    ema = ModelEMA(model)
    with torch.no_grad():
        model.model[23].lambdas.copy_(torch.tensor([0.2, -0.3, 0.4]))
        for i, router in enumerate(model.model[23].routers):
            router.logits[-1].bias.copy_(torch.linspace(-1 + i * 0.2, 1 + i * 0.3, 24))
    ema.update(model)
    ema.update_attr(model, include=["yaml", "nc", "args", "names", "stride"])
    require(not torch.equal(ema.ema.model[23].lambdas, torch.full((3,), 0.1)), "EMA lost learned lambda")
    writer = copy(trainer)
    writer.model, writer.optimizer, writer.scaler, writer.ema = model, optimizer, scaler, ema
    writer.wdir = Path(directory) / "checkpoint"
    writer.last, writer.best = writer.wdir / "last.pt", writer.wdir / "best.pt"
    writer.epoch, writer.fitness, writer.best_fitness, writer.metrics, writer.save_period = 2, 0.1, 0.1, {}, -1
    writer.save_model()
    expected_model = deepcopy(ema.ema).half().float().eval()
    expected_model.criterion = None
    image = torch.rand(1, 3, 96, 160)
    expected = raw_o2o(expected_model, image)
    reloaded = load_for_evaluation(writer.last)
    state_equal = all(
        torch.equal(value, reloaded.model.state_dict()[key]) for key, value in expected_model.state_dict().items()
    )
    require(state_equal, "Native checkpoint did not preserve all FP16-serialized EMA state")
    reload_error = comparison(raw_o2o(reloaded.model, image), expected, atol=0, rtol=0)
    rebuilt = trainer.get_model(weights=reloaded.model, verbose=False)
    require(
        all(torch.equal(v, rebuilt.state_dict()[k]) for k, v in reloaded.model.state_dict().items()),
        "SCE checkpoint reconstruction remapped/overwrote learned parameters",
    )
    fuse_error = comparison(raw_o2o(reloaded.model.fuse(verbose=False), image), expected, atol=3e-4, rtol=2e-4)
    fused = raw_o2o(reloaded.model, image)
    repeat_error = comparison(raw_o2o(reloaded.model.fuse(verbose=False), image), fused, atol=0, rtol=0)
    # Warm up the normal user predictor before installing sample-limited diagnostic observers.
    reloaded.predict(image, imgsz=160, device="cpu", save=False, verbose=False)
    observer = Diagnostics(reloaded.model.model[23], 1)
    try:
        predictions = reloaded.predict(image, imgsz=160, device="cpu", save=False, verbose=False)
    finally:
        observer.close()
    require(len(predictions) == 1 and len(observer.records) == 1, "Default predict or diagnostic sample bound")
    require(
        any(max(t["candidate_mean"]) - min(t["candidate_mean"]) > 0.01 for t in observer.records[0]["targets"]),
        "Lifecycle tested only uniform routing",
    )
    native_eval = bounded_evaluation(trainer, reloaded.model, directory)
    np.save(Path(directory) / "input.npy", image.numpy())
    script = """
import json, sys
from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(2)
from experiments.sce_fusion.validate import load_for_evaluation
from experiments.sce_fusion.verify import raw_o2o
from ultralytics.nn.tasks import _SafeLoad
folder = Path(sys.argv[1])
model = load_for_evaluation(folder / 'checkpoint/last.pt')
image = torch.from_numpy(np.load(folder / 'input.npy'))
np.save(folder / 'fresh.npy', raw_o2o(model.model, image).numpy())
model.model.fuse(verbose=False)
np.save(folder / 'fresh_fused.npy', raw_o2o(model.model, image).numpy())
(folder / 'fresh.json').write_text(json.dumps({'restricted': _SafeLoad.restricted(), 'lambda': model.model.model[23].lambdas.detach().tolist()}))
"""
    process = subprocess.run(
        [sys.executable, "-c", script, str(directory)],
        cwd=ROOT,
        env={**os.environ, "ULTRALYTICS_SAFE_LOAD": "true", "YOLO_AUTOINSTALL": "false"},
        capture_output=True,
        text=True,
        timeout=180,
    )
    require(process.returncode == 0, f"Fresh process failed: {process.stdout}\n{process.stderr}")
    fresh = json.loads((Path(directory) / "fresh.json").read_text())
    require(fresh["restricted"], "Fresh-process checkpoint loading was not restricted")
    fresh_error = comparison(torch.from_numpy(np.load(Path(directory) / "fresh.npy")), expected, atol=1e-5, rtol=1e-5)
    fresh_fused_error = comparison(
        torch.from_numpy(np.load(Path(directory) / "fresh_fused.npy")), expected, atol=3e-4, rtol=2e-4
    )
    return {
        "native_ema_and_save": True,
        "serialized_dtype": "FP16 (native); comparison reference quantized identically",
        "reload": reload_error,
        "fuse": fuse_error,
        "repeat_fuse": repeat_error,
        "fresh_process": fresh,
        "fresh_process_error": fresh_error,
        "fresh_fused_error": fresh_fused_error,
        "default_predict": "PASS",
        "bounded_native_validation": native_eval,
        "checkpoint_remapping_bypassed": True,
        "diagnostics": observer.records,
    }


def main():
    """Write per-check PASS/FAIL/UNVERIFIED results and stop after the finite scope is complete."""
    options = parser(verify=True).parse_args()
    if options.device is None and not torch.cuda.is_available():
        options.device = "cpu"
    torch.set_num_threads(min(4, torch.get_num_threads()))
    report = {"formal_training": "NOT_STARTED", "checks": {}}

    def check(name, function):
        try:
            value = function()
            report["checks"][name] = {"status": "PASS", "details": value}
            print(f"SCE VERIFY {name}: PASS", flush=True)
            return value
        except Exception as error:
            report["checks"][name] = {
                "status": "FAIL",
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            }
            print(f"SCE VERIFY {name}: FAIL: {error}", flush=True)
        finally:
            write_report(options.report, report)

    try:
        with tempfile.TemporaryDirectory(prefix="sce-verify-") as directory:
            trainer, construction = construct(options, directory)
            report.update(construction)
            pristine = {k: v.clone() for k, v in trainer.model.state_dict().items()}
            check("production_reference_rng_optimizer", lambda: native_checks(trainer, options.weights))
            check("module_contracts", module_checks)
            check("routing_math", router_checks)
            check("axial_context", axis_checks)
            check("graph_640_and_detach", lambda: graph_checks(trainer.model))

            def parameters():
                result = parameter_audit(trainer.model)
                require(
                    result["unfused"]["new_module"] == result["unfused"]["net_increase"] == 280891,
                    "SCE parameter count",
                )
                require(result["unfused"]["b19"] == 2504190 and result["fused"]["b19"] == 2375031, "b19 counts")
                return result

            check("parameters", parameters)
            cpu_learned = []

            def cpu_smoke():
                details, learned = smoke(trainer, torch.device("cpu"), False)
                cpu_learned.append(learned)
                return details

            check("cpu_fp32_task_updates", cpu_smoke)
            if torch.cuda.is_available() and options.device != "cpu":
                device = torch.device("cuda:" + (options.device or "0"))

                def cuda_smoke():
                    amp_model = deepcopy(trainer.model).to(device)
                    amp_pass = check_amp(amp_model)
                    require(amp_pass, "Native AMP check failed; no override applied")
                    del amp_model
                    details, _ = smoke(trainer, device, True)
                    details["native_amp_check"] = amp_pass
                    return details

                check("cuda_amp_task_updates", cuda_smoke)
            else:
                report["checks"]["cuda_amp_task_updates"] = {
                    "status": "UNVERIFIED",
                    "reason": "CUDA unavailable or --device cpu",
                }
            if cpu_learned:
                check("learned_lifecycle", lambda: lifecycle(trainer, cpu_learned[0], directory))
            else:
                report["checks"]["learned_lifecycle"] = {"status": "UNVERIFIED", "reason": "CPU task smoke failed"}
            check(
                "pristine_production_state",
                lambda: require(
                    all(torch.equal(v, trainer.model.state_dict()[k]) for k, v in pristine.items()),
                    "Verification polluted production state",
                ),
            )
            report["checks"]["server_b32_i640"] = {
                "status": "UNVERIFIED",
                "reason": "No server/B32 formal run in development",
            }
            report["checks"]["full_val_test"] = {
                "status": "UNVERIFIED",
                "reason": "Explicitly outside this development task",
            }
            report["optional_tools"] = (
                "profile/visualize/embed at tuple node 23 are outside the supported default path; no FLOPs completeness claim"
            )
            report["status"] = "FAIL" if any(x["status"] == "FAIL" for x in report["checks"].values()) else "PASS"
    except Exception as error:
        report.update(status="FAIL", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        raise
    finally:
        write_report(options.report, report)
    if report["status"] == "FAIL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
