"""Bounded integration checks for the frozen DTR+SCE experiment; no formal training or full dataset evaluation."""

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
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from experiments.dtr_sce.train import (
    NEW,
    REMOVED,
    ROOT,
    DTRSCETrainer,
    configuration,
    construct,
    identity,
    parser,
    write_report,
)
from experiments.dtr_sce.validate import (
    FP32Validator,
    diagnostic_samples,
    load_for_evaluation,
    numerical_settings,
    parameter_audit,
    save_evaluation,
)
from experiments.sce_fusion.train import SCETrainer, remap_key, rng_fingerprint, verified_weights
from experiments.sce_fusion.verify import comparison, raw_o2o, require
from ultralytics.data.utils import img2label_paths
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel, load_checkpoint
from ultralytics.utils import YAML
from ultralytics.utils.checks import check_amp
from ultralytics.utils.files import WorkingDirectory
from ultralytics.utils.torch_utils import ModelEMA


def memberships(model, optimizer):
    """Audit optimizer membership by parameter identity, then describe each native group."""
    names = {id(p): n for n, p in model.named_parameters()}
    counts = Counter(id(p) for g in optimizer.param_groups for p in g["params"])
    require(counts == Counter(id(p) for p in model.parameters()), "Optimizer omitted or duplicated parameters")
    return {
        names[id(p)]: {k: g.get(k) for k in ("lr", "weight_decay", "momentum", "nesterov", "use_muon", "param_group")}
        for g in optimizer.param_groups
        for p in g["params"]
    }


def native_checks(trainer):
    """Rebuild the native reference from its exact RNG boundary and compare states and optimizer policy."""
    source, _ = load_checkpoint(trainer.args.pretrained)
    py, np_state, cpu, cuda = trainer.reference_rng
    saved_py, saved_np = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng(devices=list(range(len(cuda)))):
            random.setstate(py)
            np.random.set_state(np_state)
            torch.set_rng_state(cpu)
            if cuda:
                torch.cuda.set_rng_state_all(cuda)
            reference = DetectionTrainer.get_model(
                trainer, cfg=deepcopy(source.yaml), weights=source, verbose=trainer.reference_verbose
            )
            independent = rng_fingerprint()
    finally:
        random.setstate(saved_py)
        np.random.set_state(saved_np)
    require(independent == trainer.transfer_report["rng_after_sce"], "Native reference RNG differs")
    target = trainer.model.state_dict()
    for key, value in reference.state_dict().items():
        if not key.startswith(REMOVED):
            require(torch.equal(value, target[remap_key(key)]), f"Native state mismatch: {key}")
    args = {"name": "MuSGD", "lr": 0.01, "momentum": 0.937, "decay": 0.0005}
    old = memberships(reference, DetectionTrainer.build_optimizer(trainer, reference, **args))
    new = memberships(trainer.model, trainer.build_optimizer(trainer.model, **args))
    for key, group in old.items():
        if not key.startswith(REMOVED):
            require(new[remap_key(key)] == group, f"Native optimizer policy changed: {key}")
    norms = tuple(v for k, v in torch.nn.__dict__.items() if "Norm" in k)
    modules = dict(trainer.model.named_modules())
    for name, p in trainer.model.named_parameters():
        if not name.startswith(NEW):
            continue
        module = modules[name.rsplit(".", 1)[0]]
        group = "muon" if p.ndim >= 2 else "bias" if "bias" in name else "bn" if isinstance(module, norms) else "weight"
        expected_decay = 0.0005 if group in {"muon", "weight"} else 0
        require(
            new[name]["param_group"] == group
            and new[name]["weight_decay"] == expected_decay
            and new[name]["lr"] == 0.01
            and bool(new[name]["use_muon"]) == (p.ndim >= 2),
            f"Special treatment of a new parameter: {name}",
        )
    counts = trainer.transfer_report
    for key, expected in {
        "native_total": 2504190,
        "removed_native": 117632,
        "retained_total": 2386558,
        "retained_cv1_cv2": 132096,
        "new_dtr": 432366,
        "new_sce": 280891,
    }.items():
        require(counts[key]["parameter_elements"] == expected, f"Unexpected migration count: {key}")
    classification = {k: g for k, g in new.items() if k.startswith("model.27.") and "cv3" in k}
    require(classification and all(g["lr"] == 0.03 for g in classification.values()), "Lost classification 3x LR")
    return {
        "independent_reference_rng": independent,
        "native_states_equal": True,
        "preserved_native_parameter_tensors": len(old) - sum(k.startswith(REMOVED) for k in old),
        "total_parameter_tensors_grouped_once": len(new),
        "classification_lr_multiplier": 3,
        "groups": new,
        "parameters": parameter_audit(trainer.model),
        "temporary_optimizer_view_registered": any(k.startswith("view.") for k in trainer.model.state_dict()),
    }


def graph_checks(pristine):
    """Trace both legal input shapes and verify cache identity, DTR/SCE execution and native detach."""
    model = deepcopy(pristine).cpu().eval()
    identity(model)
    require({10, 16, 19, 22, 23, 24, 25, 26}.issubset(model.save), "Missing graph cache entries")
    dtr, sce = model.model[10], model.model[23]
    require(len(dtr.dtr_blocks) == 2 and dtr.c == 128, "DTR depth/width changed")
    groups = [list(block.parameters()) for block in dtr.dtr_blocks]
    require(all(a.data_ptr() != b.data_ptr() for a, b in zip(*groups)), "DTR shares block parameters")
    contexts = [p.data_ptr() for source in sce.contexts for block in source for p in block.parameters()]
    require(len(contexts) == len(set(contexts)) and sum(len(x) for x in sce.contexts) == 6, "SCE context sharing")
    require(torch.equal(sce.lambdas, torch.full((3,), 0.1)), "SCE initialization changed")
    for block in dtr.dtr_blocks:
        require(block.heads == 2 and block.head_dim == 64 and block.splits == (32, 32, 64), "DTR dimensions")
        block.diagnostics_enabled = True
    traces = {}
    for h, w in ((640, 640), (608, 864)):
        seen, calls, grid = {}, Counter(), []
        handles = []

        def capture(index, seen=seen, calls=calls, h=h, w=w):
            def hook(module, inputs, output):
                seen[index] = output
                calls[index] += 1
                if index == 10:
                    require(inputs[0].shape == output.shape == (1, 256, h // 32, w // 32), "DTR shape")

            return hook

        for i in (10, 16, 19, 22, 23, 24, 25, 26, 27):
            handles.append(model.model[i].register_forward_hook(capture(i)))

        def sce_input(module, inputs, seen=seen):
            require(all(a is seen[i] for a, i in zip(inputs[0], (16, 19, 22))), "SCE bypassed original cache")

        def detect_input(module, inputs, seen=seen):
            require(all(a is seen[24 + i] for i, a in enumerate(inputs[0])), "Detect bypassed Index")

        handles.append(sce.register_forward_pre_hook(sce_input))
        handles.append(model.model[27].register_forward_pre_hook(detect_input))
        for block in sce.contexts:
            handles.append(
                block.register_forward_pre_hook(lambda module, inputs, grid=grid: grid.append(list(inputs[0].shape)))
            )
        try:
            with torch.no_grad():
                model(torch.rand(1, 3, h, w))
        finally:
            for handle in handles:
                handle.remove()
        require(calls == Counter({i: 1 for i in (10, 16, 19, 22, 23, 24, 25, 26, 27)}), "Nodes not executed once")
        require(grid == [[1, 64, h // 16, w // 16]] * 3, "SCE exchange grid changed")
        for offset, (i, c, stride) in enumerate(((16, 64, 8), (19, 128, 16), (22, 256, 32))):
            require(seen[i].shape == seen[24 + offset].shape == (1, c, h // stride, w // stride), "Pyramid shapes")
            require(seen[24 + offset] is seen[23][offset], "Index did not select SCE output")
        diagnostics = [deepcopy(b.diagnostics) for b in dtr.dtr_blocks]
        require(
            all(d["attention_shape"] == [1, 2, h * w // 1024, 50] and d["gate_mean"] == 0.5 for d in diagnostics),
            "DTR attention or gate initialization",
        )
        traces[f"{h}x{w}"] = {
            "calls": dict(calls),
            "dtr": diagnostics,
            "exchange_grid": grid,
            "sce_outputs": [list(v.shape) for v in seen[23]],
        }
    head = deepcopy(model.model[-1]).train()
    features = [torch.rand(2, c, 320 // s, 320 // s, requires_grad=True) for c, s in ((64, 8), (128, 16), (256, 32))]
    predictions = head(features)
    one = torch.autograd.grad(predictions["one2one"]["scores"].sum(), features, retain_graph=True, allow_unused=True)
    many = torch.autograd.grad(predictions["one2many"]["scores"].sum(), features)
    require(all(g is None for g in one) and all(g.count_nonzero() for g in many), "Native detach changed")
    invalid = deepcopy(model.yaml)
    invalid["head"][-4][3][0] = 128
    try:
        DetectionModel(invalid, verbose=False)
    except ValueError:
        pass
    else:
        raise AssertionError("Parser accepted invalid Index channel metadata")
    return {"shapes": traces, "o2o_detach": "PASS", "invalid_Index_rejected": True}


def smoke(trainer, device, amp):
    """Count only finite task-caused updates, using a zero-current-gradient optimizer control."""
    model = deepcopy(trainer.model).to(device).train()
    optimizer = trainer.build_optimizer(model, name="MuSGD", lr=0.01, momentum=0.937, decay=0.0005)
    memberships(model, optimizer)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    parameters = {k: p for k, p in model.named_parameters() if k.startswith(NEW)}
    evidence = {k: {"gradient": 0.0, "update": 0.0, "task_contribution": 0.0} for k in parameters}
    rows, updates, steps = [], 0, []
    hook = optimizer.register_step_post_hook(lambda *args: steps.append(True))
    try:
        for microbatch in range(24):
            optimizer.zero_grad(set_to_none=True)
            batch = {
                "img": torch.rand(2, 3, 320, 320, device=device),
                "batch_idx": torch.tensor([0, 0, 1, 1], device=device),
                "cls": torch.zeros(4, 1, device=device),
                "bboxes": torch.tensor(
                    [[0.35, 0.4, 0.2, 0.35], [0.7, 0.7, 0.15, 0.2], [0.4, 0.55, 0.3, 0.2], [0.7, 0.3, 0.15, 0.25]],
                    device=device,
                ),
            }
            with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                loss, components = model(batch)
                total = loss.sum()
            require(torch.isfinite(total), "Nonfinite native detection loss")
            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            finite = all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
            require(amp or finite, "Nonfinite FP32 gradients")
            gradients = (
                {
                    k: p.grad.detach().float().abs().max().item() if p.grad is not None else 0.0
                    for k, p in parameters.items()
                }
                if finite
                else {}
            )
            before = {k: p.detach().clone() for k, p in parameters.items()}
            controls = {}
            if finite:
                control = type(optimizer)(deepcopy(optimizer.param_groups), muon=optimizer.muon, sgd=optimizer.sgd)
                control.load_state_dict(deepcopy(optimizer.state_dict()))
                names = {id(p): k for k, p in parameters.items()}
                for live_group, zero_group in zip(optimizer.param_groups, control.param_groups):
                    for live, zero in zip(live_group["params"], zero_group["params"]):
                        zero.grad = torch.zeros_like(zero) if live.grad is not None else None
                        if id(live) in names:
                            controls[names[id(live)]] = zero
                control.step()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            old_steps, old_scale = len(steps), scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            changed = {"dtr": False, "sce": False}
            if len(steps) > old_steps and finite:
                for k, p in parameters.items():
                    values = {
                        "gradient": gradients[k],
                        "update": (p.detach() - before[k]).abs().max().item(),
                        "task_contribution": (p.detach() - controls[k]).abs().max().item(),
                    }
                    for name, value in values.items():
                        evidence[k][name] = max(evidence[k][name], value)
                    changed["dtr" if k.startswith(NEW[0]) else "sce"] |= all(v > 0 for v in values.values())
            effective = all(changed.values())
            updates += int(effective)
            rows.append(
                {
                    "microbatch": microbatch + 1,
                    "loss": total.item(),
                    "components": components.detach().tolist(),
                    "finite_unscaled_gradients": bool(finite),
                    "scale_before": old_scale,
                    "scale_after": scaler.get_scale(),
                    "optimizer_step": len(steps) > old_steps,
                    "task_updates": changed,
                    "effective_update": effective,
                }
            )
            if finite:
                del control, controls
            if updates == 3:
                break
    finally:
        hook.remove()
    require(updates == 3, f"Only {updates} task updates in 24 microbatches")
    require(type(model.criterion).__name__ == "E2ELoss", "Native E2E detection loss bypassed")
    for key, values in evidence.items():
        if key.startswith(NEW[1]):
            require(all(v > 0 and np.isfinite(v) for v in values.values()), f"SCE parameter without task update: {key}")
    families = (
        "local_in",
        "local_dw",
        "local_out",
        "ln_attn",
        "ln_ffn",
        "tokenizer.score_in",
        "tokenizer.score_dw",
        "tokenizer.score_out",
        "q_proj",
        "k_proj",
        "v_proj",
        "ctx_proj",
        "pos_mlp.0",
        "pos_mlp.2",
        "type_bias",
        "context_gate.0",
        "context_gate.2",
        "ffn_in",
        "ffn_dw",
        "ffn_out",
    )
    for block in range(2):
        for family in families:
            require(
                any(
                    all(v > 0 and np.isfinite(v) for v in row.values())
                    for k, row in evidence.items()
                    if k.startswith(f"{NEW[0]}{block}.{family}")
                ),
                f"DTR block {block} family {family} lacks task update",
            )
    return {
        "device": str(device),
        "amp": amp,
        "batch": 2,
        "imgsz": 320,
        "microbatch_cap": 24,
        "effective_updates": updates,
        "microbatches": rows,
        "parameters": evidence,
        "criterion": type(model.criterion).__name__,
        "optimizer": type(optimizer).__name__,
        "scope": "Independent synthetic nonempty-box copy; no formal recipe changes",
        "zero_gradient_note": "Softmax-common DTR score/position biases may have theoretical zero gradients",
    }, (model, optimizer, scaler)


def bounded_evaluation(trainer, model, scratch):
    """Check the real FP32 evaluation schema on two copied nonempty samples at B2/160 only."""
    subset = Path(scratch) / "subset"
    (subset / "images").mkdir(parents=True)
    (subset / "labels").mkdir()
    selected = []
    for source in sorted(Path(trainer.data["val"]).iterdir()):
        label = Path(img2label_paths([str(source)])[0])
        if source.is_file() and label.is_file() and label.read_text().strip():
            shutil.copyfile(source, subset / "images" / source.name)
            shutil.copyfile(label, subset / "labels" / label.name)
            selected.append(str(subset / "images" / source.name))
            if len(selected) == 2:
                break
    require(len(selected) == 2, "Two nonempty validation samples unavailable")
    data = subset / "data.yaml"
    YAML.save(data, {"path": str(subset), "train": "images", "val": "images", "names": {0: "crack"}})
    target = Path(scratch) / "schema"
    settings = numerical_settings()
    validator = FP32Validator(
        args={
            "data": str(data),
            "save_dir": str(target),
            "device": "cpu",
            "imgsz": 160,
            "batch": 2,
            "workers": 0,
            "conf": 0.001,
            "iou": 0.7,
            "max_det": 300,
            "rect": True,
            "augment": False,
            "quantize": None,
            "end2end": True,
            "plots": True,
            "task": "detect",
            "mode": "val",
            "seed": 42,
            "deterministic": True,
        }
    )
    validator(model=model.model)
    result = save_evaluation(
        validator, target, {"scope": "Two copied samples, not accuracy evidence", "numerical_settings": settings}
    )
    curves = np.load(target / "curves.npz")
    require({"all_ap", "ap_class_index", "p_curve", "r_curve", "f1_curve", "px"}.issubset(curves.files), "Curve schema")
    require(result["metrics"]["images"] == 2 and result["metrics"]["targets"] > 0, "Metric sample/target counts")
    required = [
        "BoxPR_curve.png",
        "BoxP_curve.png",
        "BoxR_curve.png",
        "BoxF1_curve.png",
        "confusion_matrix.png",
        "val_batch0_pred.jpg",
        "metrics_exact.json",
    ]
    require(all((target / name).is_file() for name in required), "Missing native plot/schema artifacts")
    records = diagnostic_samples(model, selected[:1], "cpu")
    require(len(records) == 1 and len(records[0]["dtr"]) == 2 and records[0]["sce"], "Combined diagnostic schema")
    require(not any(m._forward_hooks or m._forward_pre_hooks for m in model.model.modules()), "Observer leaked hooks")
    return {
        "scope": "2 copied nonempty val samples at 160/B2; no full val/test",
        "schema": "PASS",
        "model_dtype": result["model_dtype"],
        "input_dtype": result["input_dtype"],
        "diagnostics": records,
        "native_plots": required,
    }


def lifecycle(trainer, learned, scratch):
    """Use native EMA/save and real checkpoint owners, preserving noninitial state through fusion and a child process."""
    model, optimizer, scaler = learned
    model.eval()
    ema = ModelEMA(model)
    with torch.no_grad():
        model.model[23].lambdas.copy_(torch.tensor([0.2, -0.3, 0.4]))
        for i, router in enumerate(model.model[23].routers):
            router.logits[-1].bias.copy_(torch.linspace(-1 + 0.2 * i, 1 + 0.3 * i, 24))
    ema.update(model)
    ema.update_attr(model, include=["yaml", "nc", "args", "names", "stride"])
    for block in ema.ema.model[10].dtr_blocks:
        require(block.context_gate[-1].weight.count_nonzero() > 0, "EMA lost learned DTR gates")
    require(not torch.equal(ema.ema.model[23].lambdas, torch.full((3,), 0.1)), "EMA lost learned SCE lambdas")
    writer = copy(trainer)
    writer.model, writer.optimizer, writer.scaler, writer.ema = model, optimizer, scaler, ema
    writer.wdir = Path(scratch) / "checkpoint"
    writer.last, writer.best = writer.wdir / "last.pt", writer.wdir / "best.pt"
    writer.epoch, writer.fitness, writer.best_fitness, writer.metrics, writer.save_period = 2, 0.1, 0.1, {}, -1
    writer.save_model()
    expected = deepcopy(ema.ema).half().float().eval()
    expected.criterion = None
    image = torch.rand(1, 3, 192, 288)
    output = raw_o2o(expected, image)
    reload = load_for_evaluation(writer.last)
    require(
        all(torch.equal(v, reload.model.state_dict()[k]) for k, v in expected.state_dict().items()),
        "EMA state mismatch",
    )
    require(
        all(p.data_ptr() != q.data_ptr() for p, q in zip(expected.parameters(), reload.model.parameters())),
        "Shared state",
    )
    reload_error = comparison(raw_o2o(reload.model, image), output, atol=0, rtol=0)
    copied_error = comparison(raw_o2o(deepcopy(reload.model), image), output, atol=0, rtol=0)
    restored = object.__new__(DTRSCETrainer)
    restored.args, restored.data, restored.model = copy(trainer.args), trainer.data, str(writer.last)
    restored.args.pretrained = "must-not-be-read-or-downloaded.pt"
    restored.setup_model()
    require(
        all(torch.equal(v, restored.model.state_dict()[k]) for k, v in expected.state_dict().items()),
        "setup_model reset learned state",
    )
    fuse_error = comparison(raw_o2o(reload.model.fuse(verbose=False), image), output, atol=3e-4, rtol=2e-4)
    fused = raw_o2o(reload.model, image)
    repeat = comparison(raw_o2o(reload.model.fuse(verbose=False), image), fused, atol=0, rtol=0)
    require(not any(isinstance(m, torch.nn.BatchNorm2d) for m in reload.model.modules()), "BN not folded")
    require(
        sum(isinstance(m, torch.nn.LayerNorm) for m in reload.model.modules()) == 4, "DTR LayerNorm changed during fuse"
    )
    validation = bounded_evaluation(trainer, reload, scratch)
    np.save(Path(scratch) / "image.npy", image.numpy())
    script = """
import json, sys
from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(4)
from experiments.dtr_sce.validate import load_for_evaluation
from experiments.sce_fusion.verify import raw_o2o
from ultralytics.nn.tasks import _SafeLoad
folder = Path(sys.argv[1])
model = load_for_evaluation(folder / 'checkpoint/last.pt')
x = torch.from_numpy(np.load(folder / 'image.npy'))
np.save(folder / 'fresh.npy', raw_o2o(model.model, x).numpy())
np.save(folder / 'fresh_fused.npy', raw_o2o(model.model.fuse(verbose=False), x).numpy())
predictions = model.predict(x, imgsz=(192, 288), device='cpu', save=False, verbose=False)
(folder / 'fresh.json').write_text(json.dumps({'restricted': _SafeLoad.restricted(), 'predictions': len(predictions),
    'lambda': model.model.model[23].lambdas.detach().tolist(),
    'dtr_gate_nonzero': [int(b.context_gate[-1].weight.count_nonzero()) for b in model.model.model[10].dtr_blocks]}))
"""
    process = subprocess.run(
        [sys.executable, "-c", script, str(scratch)],
        cwd=ROOT,
        env={**os.environ, "ULTRALYTICS_SAFE_LOAD": "true", "YOLO_AUTOINSTALL": "false"},
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    require(process.returncode == 0, f"Child checkpoint load failed: {process.stdout}\n{process.stderr}")
    fresh = json.loads((Path(scratch) / "fresh.json").read_text())
    require(
        fresh["restricted"] and fresh["predictions"] == 1 and all(fresh["dtr_gate_nonzero"]), "Fresh learned-state load"
    )
    return {
        "native_EMA_save": True,
        "reference_quantization": "Same native FP16 serialization, reloaded as FP32",
        "state_and_storage": "PASS",
        "reload": reload_error,
        "deepcopy": copied_error,
        "native_fuse": fuse_error,
        "repeat_fuse": repeat,
        "explicit_checkpoint_owns_state": True,
        "fresh": fresh,
        "fresh_error": comparison(torch.from_numpy(np.load(Path(scratch) / "fresh.npy")), output, atol=1e-5, rtol=1e-5),
        "fresh_fuse_error": comparison(
            torch.from_numpy(np.load(Path(scratch) / "fresh_fused.npy")), output, atol=3e-4, rtol=2e-4
        ),
        "bounded_evaluation": validation,
    }


def policy_checks(options, scratch):
    """Exercise actual OOM handling and rejection of changed recipes, colliding runs and report paths."""

    class Loader(list):
        num_workers = 0

    class OOMModel(torch.nn.Module):
        def forward(self, batch):
            raise torch.cuda.OutOfMemoryError("Combination verification injected OOM")

    active = object.__new__(DTRSCETrainer)
    active.world_size, active.start_epoch, active.epochs, active.batch_size = 0, 0, 200, 32
    active.args = SimpleNamespace(warmup_epochs=0, imgsz=640, time=None, close_mosaic=0, compile=False, batch=32)
    active.train_loader, active.save_dir = Loader([{}]), str(Path(scratch) / "injected-oom")
    active.optimizer, active.scheduler = SimpleNamespace(zero_grad=lambda: None), SimpleNamespace(step=lambda: None)
    active._setup_train = active._model_train = lambda: None
    active.run_callbacks = lambda event: None
    active.progress_string = lambda: "Injected OOM verification"
    active.preprocess_batch = lambda batch: batch
    active.model, active.amp = OOMModel(), False
    try:
        active._do_train()
    except RuntimeError as error:
        require("automatic batch reduction is disabled" in str(error), f"Wrong OOM failure: {error}")
        require(active.batch_size == active.args.batch == 32, "OOM mutated batch")
    else:
        raise AssertionError("OOM did not abort")
    failures = []
    for key, value in (("patience", 0), ("mixup", 0.0), ("resume", True), ("workers", 0)):
        altered = YAML.load(options.baseline_args)
        altered[key] = value
        archive = Path(scratch) / f"invalid-{key}.yaml"
        YAML.save(archive, altered)
        changed = copy(options)
        changed.baseline_args = str(archive)
        try:
            configuration(changed)
        except ValueError:
            failures.append(key)
        else:
            raise AssertionError(f"Changed recipe accepted: {key}")
    changed = copy(options)
    changed.report = str(Path(options.project) / options.name / "report.json")
    try:
        configuration(changed)
    except ValueError:
        failures.append("report_inside_formal_run")
    else:
        raise AssertionError("Report allowed inside formal directory")
    try:
        construct(options, scratch)  # The production dry-run construction directory already exists.
    except FileExistsError:
        failures.append("existing_output")
    else:
        raise AssertionError("Existing output was overwritten or renamed")
    return {
        "actual_native_OOM_loop": "PASS",
        "batch_after_OOM": 32,
        "rejected": failures,
        "formal_recipe": "unchanged",
        "native_trainer_modified": False,
    }


def fresh_rng(options, scratch):
    """Compare complete cold native and combination setup chains in independent processes."""
    resolved, _ = configuration(options)
    recipe = Path(scratch) / "recipe.json"
    write_report(recipe, resolved)
    script = """
import json, sys
from pathlib import Path
from experiments.dtr_sce.train import DTRSCETrainer, ROOT
from experiments.sce_fusion.train import rng_fingerprint, write_report
from ultralytics.models.yolo.detect import DetectionTrainer
import torch
torch.set_num_threads(4)
recipe, kind, output = sys.argv[1:]
args = json.loads(Path(recipe).read_text(encoding='utf-8'))
args.update(project=str(Path(output).parent), name=kind, save_dir=str(Path(output).parent / kind))
if kind == 'native':
    args['model'] = args['pretrained']
trainer = (DetectionTrainer if kind == 'native' else DTRSCETrainer)(overrides=args)
trainer.setup_model()
trainer.set_model_attributes()
write_report(output, rng_fingerprint())
"""
    states = {}
    for kind in ("native", "combination"):
        output = Path(scratch) / f"{kind}-rng.json"
        child = subprocess.run(
            [sys.executable, "-c", script, str(recipe), kind, str(output)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
        require(child.returncode == 0, f"RNG child failed: {child.stdout}\n{child.stderr}")
        states[kind] = json.loads(output.read_text(encoding="utf-8"))
    require(states["native"] == states["combination"], f"Cold production RNG differs: {states}")
    return states


def variant_checks(trainer, scratch):
    """Check all four native checkpoint identities and the unchanged original SCE migration path."""
    args = vars(trainer.args).copy()
    args.update(
        model=str(ROOT / "experiments/sce_fusion/yolo26n-sce.yaml"),
        save_dir=str(Path(scratch) / "sce-regression"),
        name="sce-regression",
    )
    sce = SCETrainer(overrides=args)
    sce.setup_model()
    sce.set_model_attributes()
    require(
        sce.transfer_report["intentional_removals"] == []
        and sce.transfer_report["retained_total"]["parameter_elements"] == 2504190,
        "Shared migration changed SCE-only retention",
    )
    result = {}
    for variant in ("b19", "dtr", "sce", "dtr_sce"):
        if variant == "sce":
            model = sce.model
        elif variant == "dtr_sce":
            model = deepcopy(trainer.model)
        else:
            cfg = deepcopy(trainer.model.yaml)
            cfg["head"] = cfg["head"][:-5] + [[[16, 19, 22], 1, "Detect", ["nc"]]]
            if variant == "b19":
                cfg["backbone"][10] = [-1, 2, "C2PSA", [1024]]
            model = DetectionModel(cfg, nc=1, verbose=False)
            model.names = {0: "crack"}
        path = Path(scratch) / f"{variant}.pt"
        torch.save({"model": model, "train_args": {"task": "detect"}}, path)
        restored = load_for_evaluation(path, variant)
        result[variant] = parameter_audit(restored.model, variant)
        for wrong in {"b19", "dtr", "sce", "dtr_sce"} - {variant}:
            try:
                identity(restored.model, wrong)
            except ValueError:
                pass
            else:
                raise AssertionError(f"{variant} checkpoint accepted as {wrong}")
    return {"four_checkpoint_variants": result, "twelve_wrong_identities_rejected": True, "SCE_regression": "PASS"}


def main():
    """Write PASS/FAIL/UNVERIFIED per check and never reserve the formal output name."""
    options = parser(verify=True).parse_args()
    configuration(options)  # Reject unsafe report destinations before the report writer owns that path.
    torch.set_num_threads(min(4, torch.get_num_threads()))
    report = {"formal_training": "NOT_STARTED", "checks": {}}

    def check(name, function):
        print(f"DTR+SCE VERIFY {name}: RUNNING", flush=True)
        try:
            value = function()
            report["checks"][name] = {"status": "PASS", "details": value}
            print(f"DTR+SCE VERIFY {name}: PASS", flush=True)
            return value
        except Exception as error:  # noqa: BLE001 -- Bounded verification records every failed check and exits nonzero.
            report["checks"][name] = {
                "status": "FAIL",
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            }
            print(f"DTR+SCE VERIFY {name}: FAIL: {error}", flush=True)
        finally:
            write_report(options.report, report)

    try:
        with tempfile.TemporaryDirectory(prefix="dtr-sce-verify-") as scratch:
            trainer, construction = construct(options, scratch)
            report.update(construction)
            pristine = {k: v.clone() for k, v in trainer.model.state_dict().items()}
            original_recipe = vars(trainer.args).copy()
            check("production_mapping_rng_optimizer_counts", lambda: native_checks(trainer))
            check("cold_process_rng", lambda: fresh_rng(options, scratch))
            check("graph_640_rectangle_detach", lambda: graph_checks(trainer.model))
            check("recipe_output_OOM", lambda: policy_checks(options, scratch))
            learned = []

            def fp32():
                details, state = smoke(trainer, torch.device("cpu"), False)
                learned.append(state)
                return details

            check("fp32_task_updates", fp32)
            if trainer.device.type == "cuda":

                def amp():
                    weights = Path(scratch) / "yolo26n.pt"
                    shutil.copyfile(trainer.args.pretrained, weights)
                    verified_weights(weights)
                    with WorkingDirectory(scratch):
                        require(check_amp(deepcopy(trainer.model).to(trainer.device)), "Native AMP preflight failed")
                    details, _ = smoke(trainer, trainer.device, True)
                    details["native_amp_check"] = "PASS"
                    return details

                check("cuda_amp_task_updates", amp)
            else:
                report["checks"]["cuda_amp_task_updates"] = {"status": "UNVERIFIED", "reason": "No CUDA selected"}
            if learned:
                check("learned_lifecycle_and_bounded_schema", lambda: lifecycle(trainer, learned[0], scratch))
            else:
                report["checks"]["learned_lifecycle_and_bounded_schema"] = {
                    "status": "UNVERIFIED",
                    "reason": "FP32 smoke failed",
                }
            check("four_variants_and_SCE_regression", lambda: variant_checks(trainer, scratch))
            check(
                "pristine_production_state",
                lambda: require(
                    all(torch.equal(v, trainer.model.state_dict()[k]) for k, v in pristine.items())
                    and vars(trainer.args) == original_recipe,
                    "Smoke changed the production state/recipe",
                ),
            )
            for name, reason in {
                "server_batch32_640": "Local development only; no server training",
                "full_val_test": "Explicitly outside this task",
                "linux_tmux_retention": "No local Linux/tmux runtime; shell syntax alone is insufficient",
            }.items():
                report["checks"][name] = {"status": "UNVERIFIED", "reason": reason}
            report["status"] = "FAIL" if any(row["status"] == "FAIL" for row in report["checks"].values()) else "PASS"
    except Exception as error:
        report.update(status="FAIL", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        raise
    finally:
        write_report(options.report, report)
    print(json.dumps({"status": report["status"], "formal_training": "NOT_STARTED", "report": options.report}))
    if report["status"] == "FAIL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
