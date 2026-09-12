"""Layered DCS audits; local synthetic checks never authorize the independent server preflight."""

# ruff: noqa: E402 -- Direct script entry must import this worktree.
import copy
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from tools.experiments import b19_common as common
from tools.experiments.run_b19_dcs_sppf import (
    AuditedTrainer,
    audit_arguments,
    audit_training_setup,
    options_parser,
    require_runtime,
)
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.modules import C2PSA, DCS_SPPF, SPPF
from ultralytics.nn.tasks import DetectionModel, load_checkpoint
from ultralytics.optim.muon import MuSGD
from ultralytics.utils import YAML
from ultralytics.utils.torch_utils import autocast, get_flops, init_seeds


@contextmanager
def check_phase(name):
    """Print completion only after a phase actually succeeds; keep exceptions fatal to preflight."""
    print(f"BEGIN preflight {name}", flush=True)
    try:
        yield
    except BaseException:
        print(f"END preflight {name}: FAIL", flush=True)
        raise
    else:
        print(f"END preflight {name}: PASS", flush=True)


def module_identity(native, candidate, x):
    """Compare aligned module outputs exactly and identify the first differing native-path tensor on failure."""
    stages = [{}, {}]
    handles = []
    for model, captured in zip((native, candidate), stages):
        for name in ("cv1", "m", "cv2"):

            def capture(module, inputs, output, name=name, captured=captured):
                captured.setdefault(name, []).append(output.detach().clone())

            handles.append(getattr(model, name).register_forward_hook(capture))
    try:
        with torch.no_grad():
            a, b = native(x), candidate(x)
        errors = []
        for name in ("cv1", "m", "cv2"):
            common.assert_close_tree(stages[0][name], stages[1][name], 0, 0, name, errors)
        common.assert_close_tree(a, b, 0, 0, "module_output", errors)
        return errors
    finally:
        for handle in handles:
            handle.remove()


def new_parameters(module):
    """Select the scalar, depthwise refinements and linear fuse, excluding the native path."""
    return {k: p for k, p in module.named_parameters() if k == "theta" or k.startswith(("refine.", "fuse."))}


def gradient_row(parameters, before):
    """Require finite connected gradients and observe actual tensor changes independently of weight decay."""
    result = {}
    for name, parameter in parameters.items():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        assert torch.isfinite(parameter).all(), name
        result[name] = {
            "gradient_max_abs": parameter.grad.detach().abs().max().item(),
            "update_max_abs": (parameter.detach() - before[name]).abs().max().item(),
        }
    return result


def module_checks(device="cpu", amp=False, block_type=DCS_SPPF, residual_transform=None):
    """Check exact identity, an independent nonzero formula, and the necessary delayed branch learning."""
    init_seeds(42, deterministic=True)
    native = SPPF(256, 256, 5, 3, True).to(device).eval()
    candidate = block_type(256, 256, 5, 3, True).to(device).eval()
    loaded = candidate.load_state_dict(native.state_dict(), strict=False)
    assert not loaded.unexpected_keys
    assert all(k.startswith(("refine.", "fuse.")) or k == "theta" for k in loaded.missing_keys)
    x = torch.randn(2, 256, 20, 27, device=device, requires_grad=True)
    with autocast(amp, device=torch.device(device).type):
        identity = module_identity(native, candidate, x)
    params = new_parameters(candidate)
    optimizer = torch.optim.SGD(params.values(), lr=0.01, weight_decay=0)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    target = torch.randn_like(x)
    updates = []
    for step in range(3):
        optimizer.zero_grad(set_to_none=True)
        before = {k: p.detach().clone() for k, p in params.items()}
        with autocast(amp, device=torch.device(device).type):
            output = candidate(x)
            loss = (output.float() - target).square().mean(dim=(2, 3)).sum()
        assert torch.isfinite(output).all() and torch.isfinite(loss)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if step == 0:
            assert params["theta"].grad.count_nonzero() > 0
            assert all(p.grad is not None and p.grad.count_nonzero() == 0 for k, p in params.items() if k != "theta")
        scaler.step(optimizer)
        scaler.update()
        updates.append(gradient_row(params, before))
        assert torch.isfinite(x.grad).all()
    for name in params:
        assert any(row[name]["gradient_max_abs"] > 0 and row[name]["update_max_abs"] > 0 for row in updates), name
    with torch.no_grad():
        candidate.theta.fill_(0.37)
        z = native.cv1(x)
        maxima = [torch.nn.functional.max_pool2d(z, k, 1, k // 2) for k in (5, 9, 13)]
        contrasts = [m - torch.nn.functional.avg_pool2d(z, k, 1, k // 2) for m, k in zip(maxima, (5, 9, 13))]
        refinements = [branch(c) for branch, c in zip(candidate.refine, contrasts)]
        y0 = native(x)
        raw = (0.10 * candidate.theta.tanh()) * candidate.fuse(torch.cat(refinements, 1))
        expected = y0 + (raw if residual_transform is None else residual_transform(raw, y0))
        formula = []
        common.assert_close_tree(expected, candidate(x), 0, 0, "independent_formula", formula)
        bounds = [float(0.10 * torch.tensor(t).tanh()) for t in (-100.0, 0.0, 100.0)]
        assert all(abs(a) <= float(torch.tensor(0.10)) for a in bounds)
    return {"identity": identity, "updates": updates, "formula": formula, "bounds": bounds, "amp": amp}


def build_pair(source=None, model=common.MODEL):
    """Build native and DCS with aligned RNG, then apply the same native COCO-to-crack loading."""
    init_seeds(42, deterministic=True)
    with torch.random.fork_rng(devices=[]):
        baseline = DetectionModel(common.baseline_architecture(), verbose=False)
        baseline_rng = torch.get_rng_state()
    candidate = DetectionModel(str(model), nc=1, verbose=False)
    assert torch.equal(baseline_rng, torch.get_rng_state()), "DCS consumed subsequent shared initialization RNG"
    for model in (baseline, candidate):
        model.names = {0: "crack"}
        if source is not None:
            model.load(source, verbose=False)
    return baseline, candidate


def topology_checks(baseline, candidate, model=common.MODEL, block_type=DCS_SPPF):
    """Require the complete b19 graph with exactly one layer-9 substitution."""
    cfg = copy.deepcopy(candidate.yaml)
    binding = common.model_binding(model, block_type, candidate.model[9])
    assert cfg["backbone"][9] == YAML.load(model)["backbone"][9]
    cfg["backbone"][9] = baseline.yaml["backbone"][9]
    assert common.architecture_signature(cfg) == common.architecture_signature(baseline.yaml)
    assert len(candidate.model) == len(baseline.model) == 24
    for i, (a, b) in enumerate(zip(baseline.model, candidate.model)):
        assert a.f == b.f and a.i == b.i
        if i != 9:
            assert str(a) == str(b)
    block, head = candidate.model[9], candidate.model[-1]
    assert type(block) is block_type and type(candidate.model[10]) is C2PSA
    assert block.theta.ndim == 0 and block.theta.item() == 0
    assert candidate.model[21].f == [-1, 10] and head.f == [16, 19, 22]
    assert head.nc == 1 and head.reg_max == 1 and candidate.end2end
    assert candidate.stride.tolist() == [8, 16, 32]
    assert block.cv1.conv.in_channels == block.cv2.conv.out_channels == 256
    assert block.cv1.conv.out_channels == 128 and block.add
    for d, branch in enumerate(block.refine, 1):
        assert branch.conv.groups == 128 and branch.conv.kernel_size == (3, 3)
        assert branch.conv.dilation == branch.conv.padding == (d, d) and branch.conv.stride == (1, 1)
    return {
        "changed_layers": [9],
        "layer9": block_type.__name__,
        "layer10": "C2PSA",
        "detect_inputs": head.f,
        "binding": binding,
    }


def save_reload_check(model, directory, x, model_yaml=common.MODEL):
    """Save one FP32 model snapshot and require exact same-instance fixed-input outputs after a fresh-process reload."""
    directory.mkdir(parents=True, exist_ok=True)
    model = copy.deepcopy(model).eval()
    with torch.no_grad():
        expected = model(x)
    snapshot = directory / "reload.pt"
    torch.save({"model": model, "train_args": common.REFERENCE["args"]}, snapshot)
    loaded, _ = load_checkpoint(snapshot, device=x.device)
    binding = common.model_binding(model_yaml, type(model.model[9]), model.model[9])
    assert common.model_binding(model_yaml, type(model.model[9]), loaded.model[9]) == binding
    rows = []
    common.assert_close_tree(model.state_dict(), loaded.state_dict(), 0, 0, "state", rows)
    with torch.no_grad():
        common.assert_close_tree(expected, loaded(x), 0, 0, "reload_output", rows)
    torch.save(
        {
            "x": x,
            "expected": expected,
            "backend": common.computation_conditions(),
            "model_yaml": str(model_yaml),
            "binding": binding,
        },
        directory / "expected.pt",
    )
    subprocess.run([sys.executable, str(Path(__file__).resolve()), "--reload", str(directory)], cwd=ROOT, check=True)
    return {
        "exact_state_tensors": len(model.state_dict()),
        "outputs": [r for r in rows if r["path"].startswith("reload")],
        "fresh_process": True,
        "binding": binding,
    }


def reload_in_process(directory):
    """Replay the saved input using the snapshot's recorded backend settings without changing precision policy."""
    payload = torch.load(directory / "expected.pt", weights_only=False)
    torch.set_num_threads(payload["backend"]["threads"])
    init_seeds(42, deterministic=True)
    assert common.computation_conditions() == payload["backend"], "Fresh-process numerical backend differs"
    model, _ = load_checkpoint(directory / "reload.pt", device=payload["x"].device)
    assert common.model_binding(payload["model_yaml"], type(model.model[9]), model.model[9]) == payload["binding"]
    errors = []
    with torch.no_grad():
        common.assert_close_tree(payload["expected"], model(payload["x"]), 0, 0, "fresh_reload", errors)
    common.write_json(directory / "reload_checks.json", errors)


def tensor_stats(value):
    """Use the CCA/NDP finite-gradient ledger convention, retaining FP64 diagnostic norms only."""
    if value is None:
        return dict(is_none=True, finite=None, norm=None, max_abs=None)
    value = value.detach()
    finite = bool(torch.isfinite(value).all())
    return dict(
        is_none=False,
        finite=finite,
        norm=value.double().norm().item() if finite else None,
        max_abs=value.abs().max().item() if finite else None,
    )


def parameter_snapshot(value):
    """Keep full scalar/BN values and compact convolution statistics without changing live tensor precision."""
    return {**tensor_stats(value), "values": value.detach().cpu().tolist() if value.ndim < 2 else None}


def optimizer_replay(optimizer, group, parameter, zero_task=False):
    """Reuse CCA/NDP's isolated native MuSGD replay to separate task gradients from decay and stale momentum."""
    replica = torch.nn.Parameter(parameter.detach().clone())
    replica.grad = torch.zeros_like(parameter) if zero_task else parameter.grad.detach().clone()
    replay = type(optimizer)([dict(group, params=[replica])], muon=optimizer.muon, sgd=optimizer.sgd)
    replay.state[replica] = copy.deepcopy(optimizer.state.get(parameter, {}))
    replay.step()
    return replica.detach(), replay.state[replica]


def observe_step(model, parameters, optimizer, scaler, step):
    """Observe the actual native step after unscale/clip; prove task updates with exact FP32 and zero-task replays."""
    assert type(optimizer) is MuSGD, "The fixed b19 audit requires native MuSGD"
    before = {k: p.detach().clone() for k, p in parameters.items()}
    all_before = [p.detach().clone() for p in model.parameters()]
    scale = scaler.get_scale()
    nonfinite = [
        name for name, p in model.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()
    ]
    rows, groups, expected, no_task = {}, {}, {}, {}
    for name, parameter in parameters.items():
        matches = [(i, g) for i, g in enumerate(optimizer.param_groups) for p in g["params"] if p is parameter]
        assert len(matches) == 1, f"{name}: missing/duplicate optimizer membership"
        assert parameter.dtype == torch.float32 and parameter.requires_grad, name
        index, groups[name] = matches[0]
        rows[name] = dict(
            registrations=len(matches),
            dtype=str(parameter.dtype),
            before=parameter_snapshot(parameter),
            gradient_before_clip=tensor_stats(
                None if parameter.grad is None else parameter.grad.detach().double() / scale
            ),
            optimizer_group={
                "index": index,
                **{
                    k: groups[name].get(k)
                    for k in ("param_group", "lr", "momentum", "weight_decay", "use_muon", "nesterov")
                },
            },
        )
        assert parameter.grad is not None, f"{name}: disconnected gradient"
    calls = []

    def before_step(opt, args, kwargs):
        calls.append("before")
        for name, p in parameters.items():
            info = rows[name]
            info["gradient_after_clip"] = tensor_stats(p.grad)
            assert info["gradient_after_clip"]["finite"], f"{name}: nonfinite accepted gradient"
            expected[name] = optimizer_replay(opt, groups[name], p)
            no_task[name] = optimizer_replay(opt, groups[name], p, zero_task=True)
            if name.endswith("bn.weight"):
                group = groups[name]
                assert group["param_group"] == "bn" and not group["use_muon"] and group["weight_decay"] == 0
                assert group["nesterov"]
                # Native MuSGD SGD arithmetic, including its FP32 momentum/Nesterov rounding.
                direction = p.grad.add(expected[name][1]["momentum_buffer"], alpha=group["momentum"])
                zero_direction = torch.zeros_like(p).add(no_task[name][1]["momentum_buffer"], alpha=group["momentum"])
                requested = -group["lr"] * direction.double()
                task_component = -group["lr"] * (direction.double() - zero_direction.double())
                neighbor = torch.nextafter(before[name], torch.where(requested > 0, float("inf"), -float("inf")))
                half_ulp = (neighbor.double() - before[name].double()).abs() / 2
                info["rounding"] = dict(
                    requested_update_max_abs=requested.abs().max().item(),
                    current_task_component_max_abs=task_component.abs().max().item(),
                    half_ulp_min=half_ulp.min().item(),
                    max_half_ulp_fraction=(requested.abs() / half_ulp).max().item(),
                    strictly_below_half_ulp=bool((requested.abs() < half_ulp).all()),
                    fp64_requested_step_rounds_to_before=torch.equal(
                        (before[name].double() + requested).float(), before[name]
                    ),
                    momentum_before=parameter_snapshot(
                        opt.state.get(p, {}).get("momentum_buffer", torch.zeros_like(p))
                    ),
                    momentum_after=parameter_snapshot(expected[name][1]["momentum_buffer"]),
                    task_changes_momentum=not torch.equal(
                        expected[name][1]["momentum_buffer"], no_task[name][1]["momentum_buffer"]
                    ),
                )

    handles = [
        optimizer.register_step_pre_hook(before_step),
        optimizer.register_step_post_hook(lambda *args: calls.append("after")),
    ]
    try:
        step()  # The trainer still owns unscale, clipping, native MuSGD, scaler, zero_grad and EMA.
    finally:
        for handle in handles:
            handle.remove()
    skipped = not calls
    if skipped:
        assert nonfinite and scaler.get_scale() < scale, "No optimizer step without an AMP overflow"
        assert all(torch.equal(a, b) for a, b in zip(all_before, model.parameters())), "Overflow changed parameters"
    else:
        assert calls == ["before", "after"], f"Unexpected optimizer invocation: {calls}"
        assert not nonfinite, f"Accepted nonfinite gradients: {nonfinite}"
    for name, p in parameters.items():
        info = rows[name]
        info.update(
            after=parameter_snapshot(p), update_max_abs=(p.detach().double() - before[name].double()).abs().max().item()
        )
        info["gradient_max_abs"] = info["gradient_before_clip"]["max_abs"]
        info["effective_update"] = False
        info["update_kind"] = "overflow_skip" if skipped else "no_task_update"
        if skipped:
            continue
        predicted, state = expected[name]
        assert torch.equal(predicted, p), f"{name}: actual parameter differs from native FP32 replay"
        assert set(state) == set(optimizer.state[p]), f"{name}: optimizer state keys differ"
        assert all(torch.equal(value, optimizer.state[p][key]) for key, value in state.items()), (
            f"{name}: optimizer did not apply expected task momentum"
        )
        info["native_replay_exact"] = True
        info["current_task_update_max_abs"] = (p.detach().double() - no_task[name][0].double()).abs().max().item()
        post = info["gradient_after_clip"]
        eligible = post["max_abs"] > 0 and info["gradient_max_abs"] > 0 and groups[name]["lr"] > 0
        if eligible and info["update_max_abs"] > 0 and info["current_task_update_max_abs"] > 0:
            info.update(effective_update=True, update_kind="observable_task_update")
        elif eligible and name.endswith("bn.weight") and info["update_max_abs"] == 0:
            rounding = info["rounding"]
            if (
                rounding["requested_update_max_abs"] > 0
                and rounding["current_task_component_max_abs"] > 0
                and rounding["task_changes_momentum"]
                and rounding["strictly_below_half_ulp"]
                and rounding["fp64_requested_step_rounds_to_before"]
            ):
                info.update(effective_update=True, update_kind="verified_sub_ulp_task_step")
    assert all(torch.isfinite(p).all() for p in model.parameters())
    return {
        "parameters": rows,
        "theta_before": before["theta"].item(),
        "theta_after": parameters["theta"].item(),
        "alpha_before": (0.10 * before["theta"].tanh()).item(),
        "alpha_after": (0.10 * parameters["theta"].detach().tanh()).item(),
        "scaler_before": scale,
        "scaler_after": scaler.get_scale(),
        "skipped": skipped,
        "nonfinite_scaled_gradient_keys": nonfinite,
    }


def staged_gradient_audit(rows, require_complete=True):
    """Require theta to unlock before branch learning; retain every parameter and separate rounded from physical updates."""
    summary = {
        name: dict(
            initial_value=info["before"],
            dtype=info["dtype"],
            optimizer_group=info["optimizer_group"],
            first_nonzero_gradient_batch=None,
            first_observable_update_batch=None,
            first_effective_update_batch=None,
            effective_update_kind=None,
        )
        for name, info in rows[0]["parameters"].items()
    }
    assert rows[0]["theta_before"] == rows[0]["alpha_before"] == 0, "Expected zero-init identity gate"
    for name, info in summary.items():
        if name.endswith("bn.weight"):
            assert all(x == 1 for x in info["initial_value"]["values"]), f"{name}: unexpected gamma initialization"
    first_unlock = None
    for row in rows:
        batch = row["batch"]
        for name, info in row["parameters"].items():
            entry = summary[name]
            grad = info["gradient_before_clip"]
            if grad["finite"] and grad["max_abs"] > 0 and entry["first_nonzero_gradient_batch"] is None:
                entry["first_nonzero_gradient_batch"] = batch
            if info["update_max_abs"] > 0 and entry["first_observable_update_batch"] is None:
                entry["first_observable_update_batch"] = batch
            if row["alpha_before"] == 0 and name != "theta" and grad["finite"]:
                assert grad["max_abs"] == 0, f"{name}: branch task gradient before gate unlock"
            if info["effective_update"] and entry["first_effective_update_batch"] is None:
                if name == "theta":
                    assert info["update_kind"] == "observable_task_update" and row["alpha_after"] != 0
                    first_unlock = batch
                else:
                    assert first_unlock is not None and batch > first_unlock and row["alpha_before"] != 0, name
                entry.update(first_effective_update_batch=batch, effective_update_kind=info["update_kind"])
    missing = [name for name, info in summary.items() if info["first_effective_update_batch"] is None]
    result = dict(parameters=summary, first_gate_unlock_batch=first_unlock, missing_effective_parameters=missing)
    if require_complete:
        assert not missing, f"No finite gradient-backed effective update within {len(rows)} batches: {missing}"
    return result


def synthetic_model_updates(model, device, amp, directory):
    """Exercise the unchanged end-to-end detection loss and native MuSGD on a small local synthetic batch."""
    model = copy.deepcopy(model).to(device).train()
    model.args = SimpleNamespace(**common.REFERENCE["args"])
    optimizer = DetectionTrainer.build_optimizer(None, model, "MuSGD", 0.01, 0.937, 0.0005)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    params = new_parameters(model.model[9])
    batch = {
        "img": torch.rand(2, 3, 640, 640, device=device),
        "batch_idx": torch.arange(2, device=device),
        "cls": torch.zeros(2, 1, device=device),
        "bboxes": torch.tensor([[0.5, 0.5, 0.08, 0.6], [0.3, 0.6, 0.6, 0.08]], device=device),
    }
    rows, observed = [], set()
    for i in range(64):
        optimizer.zero_grad(set_to_none=True)
        with autocast(amp, device=torch.device(device).type):
            loss, _ = model(batch)
            loss = loss.sum()
        assert torch.isfinite(loss)
        scaler.scale(loss).backward()

        def native_step():
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        row = observe_step(model, params, optimizer, scaler, native_step)
        row["batch"] = i
        row["loss"] = loss.item()
        rows.append(row)
        observed.update(k for k, v in row["parameters"].items() if v["effective_update"])
        if observed == set(params):
            break
    common.write_json(directory / f"synthetic_amp_{amp}.json", rows)
    assert observed == set(params), f"No gradient-backed update: {set(params) - observed}"
    return {
        "batch": 2,
        "imgsz": 640,
        "amp": amp,
        "optimizer": "MuSGD",
        "max_batches": 64,
        "staged_audit": staged_gradient_audit(rows),
        "steps": rows,
    }


def full_identity_checks(baseline, candidate, device="cpu"):
    """Compare every aligned layer and complete Detect tree in train/eval, including one2one and shared BN state."""
    rows = []
    modes = [("cpu", False)]
    if str(device).startswith("cuda"):
        modes += [(device, False), (device, True)]
    for location, amp in modes:
        for training in (False, True):
            a, b = (copy.deepcopy(m).to(location).train(training) for m in (baseline, candidate))
            x = torch.randn(2, 3, 128, 160, device=location)
            stages = [{}, {}]
            handles = []
            for model, captured in zip((a, b), stages):
                for layer in model.model:

                    def capture(module, inputs, output, captured=captured):
                        captured[module.i] = output

                    handles.append(layer.register_forward_hook(capture))
            errors = []
            try:
                with torch.no_grad(), autocast(amp, device=torch.device(location).type):
                    expected, actual = a(x), b(x)
                for i in stages[0]:
                    common.assert_close_tree(stages[0][i], stages[1][i], 0, 0, f"layer.{i}", errors)
                common.assert_close_tree(expected, actual, 0, 0, "complete_detect", errors)
                shared = a.state_dict()
                common.assert_close_tree(shared, {k: b.state_dict()[k] for k in shared}, 0, 0, "shared_state")
                rows.append({"device": location, "amp": amp, "train": training, "errors": errors})
            finally:
                for handle in handles:
                    handle.remove()
    return rows


def model_checks(source, directory, device="cpu", model=common.MODEL, block_type=DCS_SPPF):
    """Separate graph/weight audits from snapshot equality; never compare unrelated raw Detect outputs."""
    baseline, candidate = build_pair(source, model)
    report = {
        "topology": topology_checks(baseline, candidate, model, block_type),
        "weights": common.audit_weights(baseline, candidate, source),
    }
    assert report["weights"]["matched_tensors"] == common.REFERENCE["transferred_items"]
    assert len(YOLO(str(model)).model.model) == 24
    assert report["weights"]["common_keys"] == 708
    assert report["weights"]["added_parameters"] == 103041
    assert len(report["weights"]["new_parameters"]) == 13
    report["zero_init_model"] = full_identity_checks(baseline, candidate, device)
    report["complexity"] = {
        "parameters": [sum(p.numel() for p in m.parameters()) for m in (baseline, candidate)],
        "gflops": [get_flops(m, 640) for m in (baseline, candidate)],
    }
    assert min(report["complexity"]["gflops"]) > 0
    meta = copy.deepcopy(candidate).to("meta").train()
    output = meta(torch.empty(32, 3, 640, 640, device="meta"))
    assert output["one2many"]["boxes"].shape == (32, 4, 8400)
    assert output["one2one"]["scores"].shape == (32, 1, 8400)
    report["batch32_shape"] = {"input": [32, 3, 640, 640], "boxes": [32, 4, 8400], "meta_only": True}
    candidate = candidate.to(device).eval()
    with torch.no_grad():
        candidate.model[9].theta.fill_(0.37)
    x = torch.randn(1, 3, 640, 640, device=device)
    with check_phase("unfused nonzero snapshot and fresh-process reload"):
        report["save_reload"] = save_reload_check(candidate, directory / "snapshot", x, model)
    with torch.no_grad():
        candidate.model[9].theta.zero_()
    with check_phase("B2/640 FP32 staged MuSGD updates"):
        report["fp32"] = synthetic_model_updates(candidate, device, False, directory)
    if str(device).startswith("cuda"):
        with check_phase("B2/640 AMP GradScaler staged MuSGD updates"):
            report["amp"] = synthetic_model_updates(candidate, device, True, directory)
    return report


def native_preflight(config, directory, trainer_type=AuditedTrainer, evidence=None):
    """Observe bounded native B32 AMP optimizer steps with the original b19 warmup and full dataset loader."""
    trainer = trainer_type({**config, "project": str(directory / "native"), "name": "check"})
    binding = {"commit": common.git("rev-parse", "HEAD"), "model": trainer_type.block_type.__name__}
    if evidence is not None:
        binding.update(
            model_binding=evidence["model_binding"],
            args_sha256=evidence["args_sha256"],
            initial_sha256=evidence["initial_sha256"],
            data_manifest=evidence["dataset_manifest"],
            source_sha256=evidence["source_sha256"],
        )
    common.write_json(directory / "native_binding.json", binding)
    trainer._setup_train()
    audit_training_setup(trainer)
    trainer._model_train()
    trainer.epoch = 0
    trainer.optimizer.zero_grad(set_to_none=True)
    parameters = new_parameters(trainer.model.model[9])
    nb = len(trainer.train_loader)
    nw = max(round(trainer.args.warmup_epochs * nb), 100)
    rows = []
    last_step = -1
    for i, batch in enumerate(trainer.train_loader):
        if i % 16 == 0:
            print(f"Native B32/640 preflight batch {i}/64", flush=True)
        trainer.accumulate = max(1, int(np.interp(i, [0, nw], [1, trainer.args.nbs / trainer.batch_size]).round()))
        for group in trainer.optimizer.param_groups:
            group["lr"] = float(
                np.interp(
                    i,
                    [0, nw],
                    [
                        trainer.args.warmup_bias_lr if group.get("param_group") == "bias" else 0.0,
                        group["initial_lr"] * trainer.lf(0),
                    ],
                )
            )
            if "momentum" in group:
                group["momentum"] = float(np.interp(i, [0, nw], [trainer.args.warmup_momentum, trainer.args.momentum]))
        with autocast(trainer.amp):
            batch = trainer.preprocess_batch(batch)
            assert tuple(batch["img"].shape) == (32, 3, 640, 640)
            loss, _ = trainer.model(batch)
            loss = loss.sum()
        assert torch.isfinite(loss)
        trainer.scaler.scale(loss).backward()
        if i - last_step >= trainer.accumulate:
            row = observe_step(trainer.model, parameters, trainer.optimizer, trainer.scaler, trainer.optimizer_step)
            rows.append({"batch": i, "loss": loss.item(), **row})
            last_step = i
        common.write_json(directory / "native_steps.json", {"binding": binding, "steps": rows})
        summary = staged_gradient_audit(rows, require_complete=False)
        common.write_json(directory / "native_gradient_summary.json", {"binding": binding, **summary})
        if i >= 63:
            break
    assert i == 63 and len(rows) == 64, "Expected exactly 64 original warmup B32 optimizer attempts"
    summary = staged_gradient_audit(rows)
    print("Native B32/640 preflight batch 64/64 complete", flush=True)
    return {
        "binding": binding,
        "batch": 32,
        "imgsz": 640,
        "amp": trainer.amp,
        "max_batches": 64,
        "batches_observed": i + 1,
        "staged_audit": summary,
    }


def main(name=None, model=common.MODEL, trainer_type=AuditedTrainer, module_check=module_checks, extra_checks=None):
    """Issue PASS only after all checks finish in the current independent process and evidence directory."""
    if len(sys.argv) == 3 and sys.argv[1] == "--reload":
        reload_in_process(Path(sys.argv[2]))
        return
    parser = options_parser() if name is None else options_parser(name)
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.local:
        torch.set_num_threads(4)
    init_seeds(42, deterministic=True)
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"passed": False, "local_only": args.local, "commit": common.git("rev-parse", "HEAD")}
    try:
        if not args.local:
            common.require_clean_source()
            require_runtime()
        with check_phase("recipe, launcher, data and original weights"):
            raw, effective, evidence = common.resolve_recipe(args, model=model, block_type=trainer_type.block_type)
            audit_arguments(raw, effective)
            evidence["launcher"] = common.launcher_evidence(args, raw)
            evidence["source_sha256"] = common.source_hashes()
            report["recipe"] = evidence
            source, _ = load_checkpoint(evidence["initial_path"])
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        with check_phase("module CPU"):
            report["module_cpu"] = module_check("cpu")
        if device.startswith("cuda"):
            with check_phase("module CUDA FP32 and AMP"):
                report["module_cuda"] = module_check(device)
                report["module_amp"] = module_check(device, True)
        with check_phase("model identity, MuSGD updates and reload"):
            report.update(model_checks(source, args.output, device, model, trainer_type.block_type))
        if extra_checks is not None:
            with check_phase("controller, lifecycle, fuse profiles and benchmark"):
                report["additional_checks"] = extra_checks(source, args.output, device)
        if not args.local:
            with check_phase("real B32/640 native detection"):
                report["native_preflight"] = native_preflight(effective, args.output, trainer_type, evidence)
        else:
            print("NOT RUN: real B32/640 native detection (--local)", flush=True)
        report["passed"] = True
    except Exception as error:
        report["error"] = str(error)
        raise
    finally:
        common.write_json(args.output / "checks.json", report)
    print(f"PASS: {'local validation' if args.local else 'server preflight'} -> {args.output / 'checks.json'}")


if __name__ == "__main__":
    main()
