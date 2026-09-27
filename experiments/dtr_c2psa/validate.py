"""Finite DTR verification, or explicitly requested future FP32 val/test evaluation."""

# ruff: noqa: E402 -- Direct script execution must select this worktree and disable auto-install before imports.

import argparse
import hashlib
import json
import logging
import os
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
import torch.nn as nn
import torch.nn.functional as F

from experiments.dtr_c2psa.train import (
    ADDED,
    REMOVED,
    RUN_NAME,
    DTRTrainer,
    audit_data,
    environment,
    preflight,
    rng_state,
    setup_dry_run,
    sha256,
    write_report,
)
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer, DetectionValidator
from ultralytics.nn.modules.dtr import C2PSA_DTR, DTRBlock, DualRegionTokenizer
from ultralytics.nn.tasks import load_checkpoint
from ultralytics.utils import LOGGER
from ultralytics.utils.checks import check_amp
from ultralytics.utils.torch_utils import ModelEMA, autocast
from ultralytics.utils.torch_utils import get_num_params as count


def compare(expected, actual, atol=1e-5, rtol=1e-4):
    """Assert a measured numerical comparison and return absolute/relative errors and tolerances."""
    a, b = expected.detach().float().cpu(), actual.detach().float().cpu()
    torch.testing.assert_close(a, b, atol=atol, rtol=rtol)
    difference = (a - b).abs()
    return {
        "max_abs": difference.max().item(),
        "max_rel": (difference / a.abs().clamp_min(1e-6)).max().item(),
        "atol": atol,
        "rtol": rtol,
    }


def raw_one2one(model, image):
    """Compare raw O2O boxes/logits, avoiding top-k ties and the fused O2M removal."""
    with torch.no_grad():
        prediction = model(image)[1]["one2one"]
    return torch.cat((prediction["boxes"], prediction["scores"]), 1)


def called_forward(model, image):
    """Count real DTRBlock calls and record aggregate diagnostics for this single forward."""
    counts = [0, 0]
    hooks = []
    for i, block in enumerate(model.model[10].dtr_blocks):
        block.diagnostics_enabled = True

        def count(module, inputs, output, index=i):
            counts[index] += 1

        hooks.append(block.register_forward_hook(count))
    try:
        prediction = raw_one2one(model, image)
    finally:
        for hook in hooks:
            hook.remove()
    assert counts == [1, 1], counts
    diagnostics = [deepcopy(b.diagnostics) for b in model.model[10].dtr_blocks]
    for block in model.model[10].dtr_blocks:
        block.diagnostics_enabled = False
    return prediction, {"block_calls": counts, "blocks": diagnostics}


def check_regions():
    """Use independent explicit spatial references for coverage, centers, pooling order, and gradients."""
    results = {}
    tokenizer = DualRegionTokenizer(8)
    for h, w in ((20, 20), (19, 27), (3, 5), (1, 1)):
        regions = tokenizer.regions(h, w)
        coverage = torch.zeros(h, w, dtype=torch.long)
        values = torch.arange(2 * 8 * h * w, dtype=torch.float32).reshape(2, 8, h, w) / (h * w)
        values.requires_grad_()
        scores = torch.linspace(-3, 3, h * w).reshape(1, 1, h, w).expand(2, 1, h, w).clone().requires_grad_()
        tokens, stats = tokenizer.pool(values, scores, regions, diagnostics=True)
        expected, centers, weight_sums = [], [], []
        for y0, y1, x0, x1 in regions:
            assert y1 > y0 and x1 > x0
            coverage[y0:y1, x0:x1] += 1
            coordinates = [(y, x) for y in range(y0, y1) for x in range(x0, x1)]
            center = [np.mean([x for y, x in coordinates]), np.mean([y for y, x in coordinates])]
            assert center == [(x0 + x1 - 1) / 2, (y0 + y1 - 1) / 2]
            centers.append(center)
            region_values = torch.stack([values[:, :, y, x] for y, x in coordinates], -1)
            region_scores = torch.stack([scores[:, 0, y, x] for y, x in coordinates], -1)
            weights = torch.softmax(region_scores, -1)
            weight_sums.append(weights.sum(-1))
            expected.extend(
                (region_values.sum(-1) / len(coordinates), torch.einsum("bcn,bn->bc", region_values, weights))
            )
        assert torch.equal(coverage, torch.ones_like(coverage))
        assert tokens.shape == (2, 2 * min(5, h) * min(5, w), 8)
        formula_error = compare(torch.stack(expected, 1), tokens)
        compare(torch.ones_like(torch.stack(weight_sums)), torch.stack(weight_sums))
        uniform, _ = tokenizer.pool(values, torch.ones_like(scores), regions)
        compare(uniform[:, 0::2], uniform[:, 1::2])
        compare(uniform[:, 0::2], tokens[:, 0::2])
        if h > 5 and w > 5:
            assert (tokens[:, 1::2] - uniform[:, 1::2]).abs().max() > 0.001
            value_grad, score_grad = torch.autograd.grad(
                tokens[:, 1::2].square().sum(), (values, scores), retain_graph=True
            )
            assert value_grad.abs().max() > 0 and score_grad.abs().max() > 0
            mean_value_grad, mean_score_grad = torch.autograd.grad(
                tokens[:, 0::2].sum(), (values, scores), allow_unused=True
            )
            assert mean_value_grad.abs().max() > 0
            assert mean_score_grad is None or mean_score_grad.abs().max() == 0
        results[f"{h}x{w}"] = {"tokens": list(tokens.shape), "centers_xy": centers, "formula": formula_error, **stats}
    return results


def check_position_and_initialization():
    """Verify dx/dy direction, common rectangular scale, head/type pairing, and initial conditions."""
    block = DTRBlock()
    h, w = 19, 27
    regions = block.tokenizer.regions(h, w)
    assert block.position_bias(h, w, regions, torch.device("cpu")).count_nonzero() == 0
    assert block.tokenizer.score_out.weight.std() > 0
    assert block.local_out.conv.weight.count_nonzero() > 0
    for norm in (block.ln_attn.norm, block.ln_ffn.norm):
        assert norm.eps == 1e-6 and torch.equal(norm.weight, torch.ones(128))
        assert norm.bias.count_nonzero() == 0
    z = torch.randn(2, 128, 4, 7)
    compare(F.layer_norm(z.permute(0, 2, 3, 1), (128,), eps=1e-6).permute(0, 3, 1, 2), block.ln_attn(z))
    compare(torch.full_like(z, 0.5), block.context_gate(torch.cat((z, z), 1)).sigmoid(), atol=0, rtol=0)
    with torch.no_grad():
        block.pos_mlp[0].weight.zero_()
        block.pos_mlp[0].bias.zero_()
        block.pos_mlp[0].weight[:2].copy_(torch.eye(2))
        block.pos_mlp[2].weight.zero_()
        block.pos_mlp[2].weight[0, :2] = torch.tensor([1.0, 2.0])
        block.pos_mlp[2].weight[1, :2] = torch.tensor([-3.0, 4.0])
        block.type_bias.copy_(torch.tensor([[0.1, 0.2], [-0.3, 0.4]]))
    actual = block.position_bias(h, w, regions, torch.device("cpu"))
    expected = torch.empty_like(actual)
    type_bias = block.type_bias.detach()
    for index, (y0, y1, x0, x1) in enumerate(regions):
        for y in range(h):
            for x in range(w):
                dx = torch.tensor(((x0 + x1 - 1) / 2 - x) / 26)
                dy = torch.tensor(((y0 + y1 - 1) / 2 - y) / 26)
                for kind in range(2):
                    expected[0, y * w + x, 2 * index + kind] = F.silu(dx) + 2 * F.silu(dy) + type_bias[0, kind]
                    expected[1, y * w + x, 2 * index + kind] = -3 * F.silu(dx) + 4 * F.silu(dy) + type_bias[1, kind]
    error = compare(expected, actual)
    actual.square().sum().backward()
    assert all(p.grad is not None for p in block.pos_mlp.parameters())
    return {"nonzero_rectangular_bias": error, "zero_initial_position_bias": True, "initial_gate": 0.5}


def check_wiring(model, device):
    """Run actual 640 and 608x864 forwards and verify layer widths, connections, and independent blocks."""
    model = deepcopy(model).eval().to(device)
    layer = model.model[10]
    assert isinstance(layer, C2PSA_DTR) and len(layer.dtr_blocks) == 2 and layer.c == 128
    assert model.model[-1].f == [16, 19, 22] and model.end2end
    assert set(dict(layer.dtr_blocks[0].named_parameters())) == set(dict(layer.dtr_blocks[1].named_parameters()))
    assert all(
        a.data_ptr() != b.data_ptr() for a, b in zip(layer.dtr_blocks[0].parameters(), layer.dtr_blocks[1].parameters())
    )
    for block in layer.dtr_blocks:
        assert block.heads == 2 and block.head_dim == 64 and block.splits == (32, 32, 64)
        assert [d.conv.kernel_size for d in block.local_dw] == [(3, 3), (5, 5), (7, 7)]
        assert [d.conv.groups for d in block.local_dw] == [32, 32, 64]
        assert block.ffn_in.out_channels == 512 and block.ffn_dw.groups == 256
        assert block.q_proj.out_channels == block.k_proj.out_features == block.v_proj.out_features == 128
    results = {}
    for h, w in ((640, 640), (608, 864)):
        shapes = {}
        handles = []
        for i in (10, 16, 19, 22):

            def capture(module, inputs, output, index=i):
                shapes[str(index)] = {"input": list(inputs[0].shape), "output": list(output.shape)}

            handles.append(model.model[i].register_forward_hook(capture))
        _, diagnostics = called_forward(model, torch.rand(1, 3, h, w, device=device))
        for handle in handles:
            handle.remove()
        assert shapes["10"]["input"] == shapes["10"]["output"] == [1, 256, h // 32, w // 32]
        for i, c, stride in ((16, 64, 8), (19, 128, 16), (22, 256, 32)):
            assert shapes[str(i)]["output"] == [1, c, h // stride, w // stride]
        assert diagnostics["blocks"][0]["attention_shape"] == [1, 2, (h // 32) * (w // 32), 50]
        results[f"{h}x{w}"] = {"shapes": shapes, **diagnostics}
    return results


def optimizer_audit(model, optimizer):
    """Verify exact native optimizer membership, including LN, BN, raw projections, and the 2D type bias."""
    membership = Counter(id(p) for g in optimizer.param_groups for p in g["params"])
    assert membership == Counter(id(p) for p in model.parameters())
    groups = {id(p): g for g in optimizer.param_groups for p in g["params"]}
    result = {}
    for name, parameter in model.named_parameters():
        if not name.startswith(ADDED):
            continue
        assert parameter.requires_grad
        group = groups[id(parameter)]
        if ".norm.weight" in name or ".bn.weight" in name:
            assert group["param_group"] == "bn" and group["weight_decay"] == 0
        if ".norm.bias" in name:
            assert group["param_group"] == "bias" and group["weight_decay"] == 0
        if name.endswith("type_bias"):
            assert group["param_group"] == "muon"  # Native rule checks ndim >= 2 before the bias name.
        result[name] = {"group": group["param_group"], "lr": group["lr"], "weight_decay": group["weight_decay"]}
    return result


def smoke_updates(initial, device, amp, budget=24):
    """Stop after three real native-loss/MuSGD updates, at most 24 independent synthetic microbatches."""
    trainer = object.__new__(DTRTrainer)
    trainer.model = deepcopy(initial).to(device).train()
    trainer.args = trainer.model.args
    trainer.optimizer = trainer.build_optimizer(trainer.model, name="MuSGD", lr=0.01, momentum=0.937, decay=0.0005)
    groups = optimizer_audit(trainer.model, trainer.optimizer)
    trainer.scaler = torch.amp.GradScaler("cuda", enabled=amp)
    trainer.ema = ModelEMA(trainer.model)
    updates = []
    step_calls = []
    step_hook = trainer.optimizer.register_step_post_hook(lambda *args: step_calls.append(True))
    parameters = {k: p for k, p in trainer.model.named_parameters() if k.startswith(ADDED)}
    evidence = {k: {"task_gradient_max": 0.0, "update_max": 0.0, "task_contribution_max": 0.0} for k in parameters}
    detach_checked = False
    for microbatch in range(budget):
        batch = {
            "img": torch.rand(2, 3, 320, 320, device=device),
            "batch_idx": torch.tensor([0, 0, 1, 1], device=device),
            "cls": torch.zeros(4, 1, device=device),
            "bboxes": torch.tensor(
                [[0.35, 0.4, 0.2, 0.35], [0.7, 0.7, 0.15, 0.2], [0.4, 0.55, 0.3, 0.2], [0.7, 0.3, 0.15, 0.25]],
                device=device,
            ),
        }
        with autocast(amp, device.type):
            preds = trainer.model(batch["img"])
            if not detach_checked:
                feature = preds["one2many"]["feats"][0]
                one = torch.autograd.grad(
                    preds["one2one"]["scores"].sum(), feature, retain_graph=True, allow_unused=True
                )[0]
                many = torch.autograd.grad(preds["one2many"]["scores"].sum(), feature, retain_graph=True)[0]
                assert one is None and many.abs().max() > 0
                detach_checked = True
            loss, components = trainer.model.loss(batch, preds)
            loss = loss.sum()
        assert torch.isfinite(loss), (microbatch, loss)
        trainer.scaler.scale(loss).backward()
        gradients = {
            k: (p.grad.detach().float().abs().max() / trainer.scaler.get_scale()).item() if p.grad is not None else 0.0
            for k, p in parameters.items()
        }
        old = {k: p.detach().clone() for k, p in parameters.items()}
        # Compare the actual step with identical optimizer history but zero current task gradients.
        # This excludes changes caused solely by weight decay or previously accumulated momentum.
        control = type(trainer.optimizer)(
            deepcopy(trainer.optimizer.param_groups), muon=trainer.optimizer.muon, sgd=trainer.optimizer.sgd
        )
        control.load_state_dict(deepcopy(trainer.optimizer.state_dict()))
        controls = {}
        names_by_id = {id(p): k for k, p in parameters.items()}
        for live_group, control_group in zip(trainer.optimizer.param_groups, control.param_groups):
            for live, zero in zip(live_group["params"], control_group["params"]):
                zero.grad = torch.zeros_like(zero) if live.grad is not None else None
                if id(live) in names_by_id:
                    controls[names_by_id[id(live)]] = zero
        control.step()
        previous_steps = len(step_calls)
        trainer.optimizer_step()
        if len(step_calls) > previous_steps:
            assert all(torch.isfinite(p).all() for p in trainer.model.parameters())
            for key, parameter in parameters.items():
                e = evidence[key]
                e["task_gradient_max"] = max(e["task_gradient_max"], gradients[key])
                e["update_max"] = max(e["update_max"], (parameter.detach() - old[key]).abs().max().item())
                e["task_contribution_max"] = max(
                    e["task_contribution_max"], (parameter.detach() - controls[key]).abs().max().item()
                )
            updates.append(
                {
                    "microbatch": microbatch + 1,
                    "loss": loss.item(),
                    "components": components.detach().cpu().tolist(),
                    "scaler_scale": trainer.scaler.get_scale(),
                }
            )
        del control, controls, old, loss, preds
        if len(updates) == 3:
            break
    step_hook.remove()
    assert len(updates) == 3, f"Only {len(updates)} effective updates within the 24-microbatch limit"
    families = (
        "local_in",
        "local_dw",
        "local_out",
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
        "ln_attn",
        "ln_ffn",
        "ffn_in",
        "ffn_dw",
        "ffn_out",
    )
    for block in range(2):
        for family in families:
            matches = [v for k, v in evidence.items() if k.startswith(f"{ADDED}{block}.{family}")]
            assert matches and any(all(x > 0 and np.isfinite(x) for x in v.values()) for v in matches), (
                block,
                family,
                matches,
            )
    return (
        trainer.model,
        trainer.ema.ema,
        {
            "status": "PASS",
            "device": str(device),
            "input_dtype": "torch.float32",
            "amp": amp,
            "parameter_dtype": str(next(trainer.model.parameters()).dtype),
            "batch": 2,
            "imgsz": 320,
            "accumulation": 1,
            "effective_updates": updates,
            "microbatches": microbatch + 1,
            "optimizer_groups": groups,
            "per_parameter": evidence,
            "O2O_detach_preserved": detach_checked,
            "note": "Synthetic nonempty labels; no dataset accuracy evidence. Softmax-common score/position biases can have zero task gradient.",
        },
    )


def check_lifecycle(model, ema, scratch):
    """Verify learned state survives deepcopy, EMA, native checkpoint loading, fresh process, and native fusion."""
    model = deepcopy(model).cpu().float().eval()
    model.criterion = None
    x = torch.rand(1, 3, 192, 288)
    result = {}
    prediction, result["unfused_calls"] = called_forward(model, x)
    copied = deepcopy(model)
    result["deepcopy"] = compare(prediction, raw_one2one(copied, x), atol=0, rtol=0)
    ema_model = deepcopy(ema).cpu().float().eval()
    ema_prediction, result["ema_calls"] = called_forward(ema_model, x)
    result["ema_deepcopy"] = compare(ema_prediction, raw_one2one(deepcopy(ema_model), x), atol=0, rtol=0)
    assert all(p.data_ptr() != q.data_ptr() for p, q in zip(model.parameters(), ema_model.parameters()))
    for i in range(2):
        for key in ("tokenizer.score_out.weight", "pos_mlp.2.weight", "context_gate.2.weight"):
            assert ema_model.state_dict()[f"{ADDED}{i}.{key}"].count_nonzero() > 0
    checkpoint = Path(scratch) / "learned.pt"
    # Mirror native training checkpoints' half-precision EMA serialization, then compare to its FP32 reload reference.
    serialized = deepcopy(ema_model).half()
    torch.save({"model": None, "ema": serialized, "train_args": vars(model.args)}, checkpoint)
    reload_reference = deepcopy(serialized).float().eval()
    serialized_prediction = raw_one2one(reload_reference, x)
    result["checkpoint_fp16_quantization"] = compare(ema_prediction, serialized_prediction, atol=0.05, rtol=0.05)
    # Relative paths avoid the locked upstream downloader stripping apostrophes from Windows home directories.
    loaded, _ = load_checkpoint(os.path.relpath(checkpoint, Path.cwd()))
    actual, result["reload_calls"] = called_forward(loaded, x)
    result["reload"] = compare(serialized_prediction, actual, atol=0, rtol=0)
    assert all(torch.equal(v, loaded.state_dict()[k]) for k, v in reload_reference.state_dict().items())
    resume = object.__new__(DTRTrainer)
    resume.args, resume.data = model.args, {"nc": 1, "names": {0: "crack"}, "channels": 3}
    resume.model = os.path.relpath(checkpoint, Path.cwd())
    resume.setup_model()  # args.pretrained still points to the original .pt; it must not override this checkpoint.
    result["checkpoint_setup_model"] = compare(actual, raw_one2one(resume.model.eval(), x), atol=0, rtol=0)
    torch.save(x, Path(scratch) / "input.pt")
    child = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "reload", "--directory", str(scratch)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    result["new_process_stdout"] = child.stdout.strip()
    fresh = torch.load(Path(scratch) / "output.pt", weights_only=True)
    result["new_process"] = compare(actual, fresh, atol=0, rtol=0)
    fused = deepcopy(loaded).fuse(verbose=False)
    fused_prediction, result["fused_calls"] = called_forward(fused, x)
    result["native_fuse"] = compare(actual, fused_prediction, atol=2e-4, rtol=2e-4)
    assert not any(isinstance(m, nn.BatchNorm2d) for m in fused.model[10].modules())
    assert sum(isinstance(m, nn.LayerNorm) for m in fused.model[10].modules()) == 4
    for name, module in fused.model[10].named_modules():
        if type(module) is nn.Conv2d:
            assert module.forward.__func__ is nn.Conv2d.forward, name
    for key, value in loaded.state_dict().items():
        if key.startswith(ADDED) and any(
            name in key for name in ("tokenizer.", "pos_mlp.", "type_bias", "context_gate.")
        ):
            assert torch.equal(value, fused.state_dict()[key]), key
    result["fused_rectangular"] = called_forward(fused, torch.rand(1, 3, 96, 160))[1]
    wrapper = YOLO(os.path.relpath(checkpoint, Path.cwd()))
    calls = []
    hooks = [b.register_forward_hook(lambda *args: calls.append(1)) for b in wrapper.model.model[10].dtr_blocks]
    wrapper.predict(x, imgsz=(192, 288), device="cpu", verbose=False, save=False)
    for hook in hooks:
        hook.remove()
    assert len(calls) >= 2 and len(calls) % 2 == 0
    result["default_predict_calls"] = len(calls)
    return result


def check_oom_policy():
    """Inject OOM at the real training-loop boundary and require propagation before any batch reduction."""

    class Loader(list):
        num_workers = 0

    class OOMModel(nn.Module):
        def forward(self, batch):
            raise torch.cuda.OutOfMemoryError("DTR verification injected OOM")

    trainer = object.__new__(DTRTrainer)
    trainer.world_size, trainer.start_epoch, trainer.epochs, trainer.batch_size = 0, 0, 200, 32
    trainer.args = SimpleNamespace(warmup_epochs=0, imgsz=640, time=None, close_mosaic=0, compile=False, batch=32)
    trainer.train_loader, trainer.save_dir = Loader([{}]), "temporary-injected-oom"
    trainer.optimizer = SimpleNamespace(zero_grad=lambda: None)
    trainer.scheduler = SimpleNamespace(step=lambda: None)
    trainer._setup_train = trainer._model_train = lambda: None
    trainer.run_callbacks = lambda event: None
    trainer.progress_string = lambda: "Injected OOM verification"
    trainer.preprocess_batch = lambda batch: batch
    trainer.model, trainer.amp = OOMModel(), False
    try:
        trainer._do_train()
    except torch.cuda.OutOfMemoryError:
        assert trainer.batch_size == trainer.args.batch == 32 and trainer._oom_retries == 0
    else:
        raise AssertionError("Native OOM was swallowed")
    assert DetectionTrainer.max_oom_retries == 3
    return {"experiment_retries": 0, "native_default_retries": 3, "batch_after_injected_oom": trainer.batch_size}


def initialization_probe(recipe, kind, output):
    """Capture the first Trainer initialization in a fresh process, including lazy integration imports."""
    LOGGER.setLevel(logging.WARNING)
    torch.set_num_threads(4)
    overrides = json.loads(Path(recipe).read_text(encoding="utf-8"))
    overrides.update(project=str(Path(output).parent), name=kind, save_dir=str(Path(output).parent / kind))
    cls = DTRTrainer if kind == "dtr" else DetectionTrainer
    if kind == "native":
        overrides["model"] = overrides["pretrained"]
    trainer = cls(overrides=overrides)
    trainer.setup_model()
    trainer.set_model_attributes()
    state = rng_state()
    fingerprints = {
        "cpu": hashlib.sha256(state["cpu"].numpy().tobytes()).hexdigest(),
        "cuda": [hashlib.sha256(s.cpu().numpy().tobytes()).hexdigest() for s in state["cuda"]],
        "python": hashlib.sha256(repr(state["python"]).encode()).hexdigest(),
        "numpy": hashlib.sha256(repr(state["numpy"]).encode()).hexdigest(),
    }
    write_report(output, fingerprints)


def run_checks(args):
    """Run only bounded implementation checks; never launch training or whole-dataset evaluation."""
    LOGGER.setLevel(logging.WARNING)
    torch.set_num_threads(4)
    resolved, report = preflight(args)
    report["status"] = "RUNNING"
    write_report(args.report, report)
    with tempfile.TemporaryDirectory(prefix="dtr-check-") as scratch:
        print("Checking region equations, coordinates, and initialization", flush=True)
        report["regions"] = check_regions()
        report["position"] = check_position_and_initialization()
        report["oom_policy"] = check_oom_policy()
        print("Checking the actual setup_model chain and native-reference RNG/state", flush=True)
        native_args = {
            **resolved,
            "model": resolved["pretrained"],
            "name": "native",
            "project": scratch,
            "save_dir": str(Path(scratch) / "native"),
        }
        native = DetectionTrainer(overrides=native_args)
        native.setup_model()
        native.set_model_attributes()
        trainer = setup_dry_run(resolved, scratch)
        report["migration"] = trainer.migration_report
        recipe_path = Path(scratch) / "recipe.json"
        write_report(recipe_path, resolved)
        states = {}
        for kind in ("native", "dtr"):
            output = Path(scratch) / f"{kind}_rng.json"
            subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "from experiments.dtr_c2psa.validate import initialization_probe; "
                    "import sys; initialization_probe(*sys.argv[1:])",
                    str(recipe_path),
                    kind,
                    str(output),
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            states[kind] = json.loads(output.read_text(encoding="utf-8"))
        report["initialization_rng_fingerprints"] = states
        report["native_path_rng_equality"] = {k: states["native"][k] == states["dtr"][k] for k in states["native"]}
        assert all(report["native_path_rng_equality"].values()), report["native_path_rng_equality"]
        native_state = native.model.state_dict()
        retained = {k: v for k, v in native_state.items() if not k.startswith(REMOVED)}
        assert all(torch.equal(value, trainer.model.state_dict()[key]) for key, value in retained.items())
        report["parameters"] = {
            "baseline_unfused": count(native.model),
            "dtr_unfused": count(trainer.model),
            "removed_native_m": count(native.model.model[10].m),
            "retained_cv1_cv2": count(trainer.model.model[10].cv1) + count(trainer.model.model[10].cv2),
            "new_dtr_blocks": count(trainer.model.model[10].dtr_blocks),
            "complete_C2PSA_DTR": count(trainer.model.model[10]),
            "baseline_fused": count(deepcopy(native.model).fuse(verbose=False)),
            "dtr_fused": count(deepcopy(trainer.model).fuse(verbose=False)),
        }
        report["parameters"]["net_unfused"] = (
            report["parameters"]["dtr_unfused"] - report["parameters"]["baseline_unfused"]
        )
        report["parameters"]["net_fused"] = report["parameters"]["dtr_fused"] - report["parameters"]["baseline_fused"]
        device = trainer.device
        print("Checking real 640/rectangular wiring and finite optimizer updates", flush=True)
        report["wiring"] = check_wiring(trainer.model, device)
        trained, ema, report["smoke_fp32"] = smoke_updates(trainer.model, device, amp=False)
        report["lifecycle"] = check_lifecycle(trained, ema, scratch)
        del trained, ema
        if device.type == "cuda":
            print("Checking native AMP preflight, actual CUDA AMP updates, and pure-half execution", flush=True)
            report["native_amp_check"] = check_amp(deepcopy(trainer.model).to(device))
            assert report["native_amp_check"]
            trained, ema, report["smoke_cuda_amp"] = smoke_updates(
                trainer.model, device, amp=True, budget=24 - report["smoke_fp32"]["microbatches"]
            )
            block = deepcopy(trained.model[10].dtr_blocks[0]).half().eval()
            half_input = torch.randn(2, 128, 19, 27, device=device, dtype=torch.float16, requires_grad=True)
            half_output = block(half_input)
            half_output.float().square().mean().backward()
            assert torch.isfinite(half_output).all() and torch.isfinite(half_input.grad).all()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in block.pos_mlp.parameters())
            half_model = deepcopy(trained).eval().half()
            half_prediction = raw_one2one(half_model, torch.rand(1, 3, 192, 288, device=device).half())
            assert torch.isfinite(half_prediction).all()
            report["pure_half"] = {"block_forward_backward": "PASS", "whole_model_forward": "PASS"}
        else:
            report["smoke_cuda_amp"] = report["native_amp_check"] = report["pure_half"] = (
                "UNVERIFIED (no CUDA selected)"
            )
        report["formal_batch32_imgsz640_server"] = "UNVERIFIED"
        report["full_dataset_val_test"] = "NOT_RUN"
        report["flops"] = (
            "NOT_PROFILED: native profilers do not fully count custom pooling, attention, reorders, or elementwise operations"
        )
        report["status"] = "PASS"
        write_report(args.report, report)
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "parameters": report["parameters"],
                    "formal_training": "NOT_STARTED",
                    "report": args.report,
                },
                indent=2,
            )
        )


def evaluate(args):
    """Evaluate an explicitly supplied checkpoint at the shared FP32 protocol, without threshold selection."""
    data = audit_data(args.data)
    output = Path(args.project).resolve() / args.name
    output.mkdir(parents=True, exist_ok=False)
    model, _ = load_checkpoint(args.weights)
    unfused_parameters = sum(p.numel() for p in model.parameters())
    model = model.float().eval().fuse(verbose=False)
    assert model.end2end and model.model[-1].nc == 1 and model.names == {0: "crack"}
    dtypes = {"model": str(next(model.parameters()).dtype), "inputs": set()}

    def input_dtype(module, inputs):
        dtypes["inputs"].add(str(inputs[0].dtype))

    hook = model.register_forward_pre_hook(input_dtype)
    configuration = dict(
        model=args.weights,
        data=args.data,
        split=args.split,
        device=args.device,
        imgsz=640,
        batch=32,
        conf=0.001,
        iou=0.7,
        max_det=300,
        rect=True,
        augment=False,
        quantize=None,
        end2end=True,
        plots=True,
        project=str(output.parent),
        name=output.name,
        save_dir=str(output),
        exist_ok=False,
    )
    validator = DetectionValidator(args=configuration)
    validator(model=model)
    hook.remove()
    box = validator.metrics.box
    index75 = torch.where(torch.isclose(validator.iouv.cpu(), torch.tensor(0.75)))[0].item()
    thresholds = {}
    for threshold in (0.25, 0.50):
        p = float(np.mean([np.interp(threshold, box.px, curve) for curve in box.p_curve])) if len(box.p) else 0.0
        r = float(np.mean([np.interp(threshold, box.px, curve) for curve in box.r_curve])) if len(box.r) else 0.0
        thresholds[str(threshold)] = {
            "precision": p,
            "recall": r,
            "f1": 2 * p * r / (p + r + 1e-16),
            "method": "PR-curve interpolation estimate at IoU=0.5; not exact TP/FP/FN",
        }
    report = {
        "configuration": configuration,
        "checkpoint_sha256": sha256(args.weights),
        "data": data,
        "environment": environment(),
        "precision": box.mp,
        "recall": box.mr,
        "f1": float(np.mean(box.f1)) if len(box.f1) else 0.0,
        "ap50": box.map50,
        "ap75": float(box.all_ap[:, index75].mean()) if len(box.all_ap) else 0.0,
        "map50_95": box.map,
        "native_operating_point": "Each model's own best smoothed-F1 working point",
        "fixed_threshold_estimates": thresholds,
        "iou_thresholds": validator.iouv.cpu().tolist(),
        "dtypes": {"model": dtypes["model"], "inputs": sorted(dtypes["inputs"])},
        "parameters": {"unfused": unfused_parameters, "fused": sum(p.numel() for p in model.parameters())},
        "latency": "Not a fair latency benchmark when sharing a GPU",
    }
    assert dtypes["inputs"] == {"torch.float32"}
    write_report(output / "dtr_evaluation.json", report)


def main():
    """Expose bounded checks separately from future full evaluation and the fresh-process reload probe."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check", help="Finite synthetic verification, no full dataset evaluation")
    check.add_argument("--data", required=True)
    check.add_argument("--weights", required=True)
    check.add_argument("--baseline-args", required=True)
    check.add_argument("--device", default="0")
    check.add_argument("--project", default=str(ROOT / "runs" / "detect"))
    check.add_argument("--name", default=RUN_NAME)
    check.add_argument("--report", required=True)
    evaluation = commands.add_parser("evaluate", help="Explicit future FP32 val/test evaluation")
    evaluation.add_argument("--data", required=True)
    evaluation.add_argument("--weights", required=True)
    evaluation.add_argument("--project", required=True)
    evaluation.add_argument("--name", required=True)
    evaluation.add_argument("--device", default="0")
    evaluation.add_argument("--split", choices=("val", "test"), default="val")
    child = commands.add_parser("reload", help=argparse.SUPPRESS)
    child.add_argument("--directory", required=True)
    args = parser.parse_args()
    if args.command == "check":
        run_checks(args)
    elif args.command == "evaluate":
        evaluate(args)
    else:
        torch.set_num_threads(4)
        directory = Path(args.directory)
        model, _ = load_checkpoint(os.path.relpath(directory / "learned.pt", Path.cwd()))
        image = torch.load(directory / "input.pt", weights_only=True)
        prediction, calls = called_forward(model, image)
        torch.save(prediction, directory / "output.pt")
        print(json.dumps(calls))


if __name__ == "__main__":
    main()
