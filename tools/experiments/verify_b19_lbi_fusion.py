"""Layered LBI audits; local checks cannot issue a native server B32 preflight receipt."""

# ruff: noqa: E402 -- Direct entry must import this worktree.
import copy
import subprocess
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from tools.experiments import b19_common as common
from tools.experiments.run_b19_lbi_fusion import (
    AuditedTrainer,
    audit_arguments,
    audit_training_setup,
    options_parser,
    require_runtime,
)
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.modules import C2PSA, SPPF, Concat, Concat_LBI_Fusion
from ultralytics.nn.tasks import DetectionModel, load_checkpoint
from ultralytics.optim.muon import MuSGD
from ultralytics.utils.torch_utils import ModelEMA, autocast, get_flops, init_seeds

# Fixed before execution. Zero-init is exact; fuse has a separate FP32 numerical bound.
FUSE_ATOL, FUSE_RTOL = 1e-4, 1e-4
MAX_BATCHES = 64


def tensor_stats(value):
    """Summarize finite gradients without modifying live tensors or calling unscale a second time."""
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


def build_pair(source=None):
    """Apply native seed42/class adaptation with identical subsequent RNG and common state."""
    init_seeds(42, deterministic=True)
    with torch.random.fork_rng(devices=[]):
        baseline = DetectionModel(common.baseline_architecture(), verbose=False)
        baseline_rng = torch.get_rng_state()
    candidate = DetectionModel(str(common.MODEL), nc=1, verbose=False)
    assert torch.equal(baseline_rng, torch.get_rng_state()), "New initialization changed later native RNG"
    for model in (baseline, candidate):
        model.names = {0: "crack"}  # native cls_remap=True sets names before .load()
        if source is not None:
            model.load(source, verbose=False)
    return baseline, candidate


def topology_checks(baseline, candidate):
    """Require exactly the layer15 substitution, all ordered inputs and the full native channel table."""
    cfg = copy.deepcopy(candidate.yaml)
    assert cfg["head"][4] == [[-1, 4], 1, "Concat_LBI_Fusion", [16, 0.0001, 1]]
    cfg["head"][4] = baseline.yaml["head"][4]
    assert common.architecture_signature(cfg) == common.architecture_signature(baseline.yaml)
    assert len(candidate.model) == len(baseline.model) == 24
    table = []
    for i, (a, b) in enumerate(zip(baseline.model, candidate.model)):
        assert a.f == b.f and a.i == b.i
        if i != 15:
            assert str(a) == str(b), i
        table.append(dict(layer=i, inputs=b.f, native=type(a).__name__, candidate=type(b).__name__))
    assert type(candidate.model[9]) is SPPF and type(candidate.model[10]) is C2PSA
    assert all(type(candidate.model[i]) is Concat for i in (12, 18, 21))
    assert type(candidate.model[15]) is Concat_LBI_Fusion
    assert candidate.model[15].channels == (128, 128) and candidate.model[16].cv1.conv.in_channels == 256
    assert candidate.model[-1].f == [16, 19, 22]
    assert candidate.stride.tolist() == [8, 16, 32] and candidate.end2end
    assert candidate.model[-1].nc == 1 and candidate.model[-1].reg_max == 1
    return dict(changed_layers=[15], layers=table, fusion_input_channels=[128, 128], fusion_output_channels=256)


def module_checks(device="cpu", amp=False, spatial=(13, 17), batch=2):
    """Check ordered identity, hand-derived FP32 RMS, finite limits and sequential data-driven learning."""
    init_seeds(42, deterministic=True)
    module = Concat_LBI_Fusion([128, 128]).to(device)
    params = dict(module.named_parameters())
    assert list(params) == ["proj_l.weight", "proj_s.weight", "dw.weight", "out.weight"]
    assert sum(p.numel() for p in params.values()) == 6288 and not list(module.buffers())
    assert all(p.count_nonzero() > 0 for n, p in params.items() if n != "out.weight")
    h, w = spatial
    semantic = torch.randn(batch, 128, w, h, device=device).transpose(2, 3)
    detail = torch.randn_like(semantic) + 2
    before = [semantic.clone(), detail.clone()]
    identities, formulas = [], []
    with autocast(amp, device=torch.device(device).type):
        y = module([semantic, detail])
    common.assert_close_tree(torch.cat([semantic, detail], 1), y, 0, 0, "module.identity", identities)
    assert torch.equal(y[:, :128], semantic) and torch.equal(semantic, before[0]) and torch.equal(detail, before[1])
    with torch.no_grad():
        module.out.weight.fill_(0.015625)
    for magnitude in (0.0, 1e-8, 1.0):
        with autocast(amp, device=torch.device(device).type):
            u, v = module.proj_l(detail * magnitude), module.proj_s(semantic * magnitude)
            # Independent division/sqrt reference, including eps inside the root and dim=1.
            uf, vf = u.float(), v.float()
            b = (uf / (uf.square().sum(1, keepdim=True) / 16 + 1e-4).sqrt()) * (
                vf / (vf.square().sum(1, keepdim=True) / 16 + 1e-4).sqrt()
            )
            residual = module.out(torch.nn.functional.silu(module.dw(b.to(u.dtype)), inplace=False)).to(detail.dtype)
            expected = torch.cat([semantic * magnitude, detail * magnitude + residual], 1)
            actual = module([semantic * magnitude, detail * magnitude])
        # Division vs multiply-rsqrt has its own predeclared rounding allowance.
        common.assert_close_tree(
            expected,
            actual,
            1e-5 if not amp else 1e-3,
            1e-5 if not amp else 1e-3,
            f"RMS_reference.{magnitude}",
            formulas,
        )
    with torch.no_grad():
        module.out.weight.zero_()  # fresh module test fixture, never a production lifecycle action
    optimizer = torch.optim.SGD(params.values(), lr=0.01, weight_decay=0)
    target = torch.randn(batch, 256, h, w, device=device)
    updates = []
    for step in range(3):
        optimizer.zero_grad(set_to_none=True)
        before = {k: p.detach().clone() for k, p in params.items()}
        with autocast(amp, device=torch.device(device).type):
            loss = (module([semantic, detail]).float() * target).mean() * 100
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params.values())
        if step == 0:
            assert params["out.weight"].grad.count_nonzero() > 0
            assert all(p.grad.count_nonzero() == 0 for k, p in params.items() if k != "out.weight")
        optimizer.step()
        updates.append(
            {
                k: dict(gradient=tensor_stats(p.grad), delta=(p - before[k]).abs().max().item())
                for k, p in params.items()
            }
        )
    assert all(any(row[k]["gradient"]["max_abs"] > 0 and row[k]["delta"] > 0 for row in updates) for k in params)
    return dict(
        identity=identities,
        rms_reference=formulas,
        updates=updates,
        batch=batch,
        spatial=spatial,
        amp=amp,
        device=device,
        optimizer="ordinary SGD module test only",
        parameters=6288,
        initialization="Conv2d default reset_parameters; out.weight alone zero; isolated CPU RNG",
    )


def whole_identity(baseline, candidate, device, amp=False, size=160):
    """Locate the first differing native layer; compare both Detect branches and all shared buffers."""
    rows = []
    for training in (False, True):
        native, model = [copy.deepcopy(m).to(device).train(training) for m in (baseline, candidate)]
        captured = [{}, {}]
        handles = []
        for m, values in zip((native, model), captured):
            for i, layer in enumerate(m.model):

                def capture(module, inputs, output, i=i, values=values):
                    values[i] = output

                handles.append(layer.register_forward_hook(capture))
        try:
            x = torch.randn(2, 3, size, size, device=device)
            with torch.no_grad(), autocast(amp, device=torch.device(device).type):
                a, b = native(x), model(x)
            for i in range(24):
                common.assert_close_tree(captured[0][i], captured[1][i], 0, 0, f"train={training}.layer{i}", rows)
            common.assert_close_tree(a, b, 0, 0, f"train={training}.output", rows)
            shared = {k: v for k, v in model.state_dict().items() if k in native.state_dict()}
            common.assert_close_tree(native.state_dict(), shared, 0, 0, "shared_after_forward")
            attrs = common.inference_attributes(model)
            expected_attrs = common.inference_attributes(native)
            expected_attrs.pop("model.15")
            common.assert_close_tree(expected_attrs, {k: attrs[k] for k in expected_attrs}, 0, 0, "inference_attrs")
        finally:
            for handle in handles:
                handle.remove()
    return dict(device=device, amp=amp, size=size, rows=rows)


def reload_in_process(directory):
    """Verify a FP32 snapshot in a fresh Python process under the recorded execution conditions."""
    payload = torch.load(directory / "expected.pt", weights_only=False)
    torch.set_num_threads(payload["backend"]["threads"])
    init_seeds(42, deterministic=True)
    assert common.computation_conditions() == payload["backend"], "Fresh-process numerical backend differs"
    model, _ = load_checkpoint(directory / "reload.pt", device=payload["x"].device)
    rows = []
    with torch.no_grad():
        common.assert_close_tree(payload["expected"], model(payload["x"]), 0, 0, "fresh_reload", rows)
    common.write_json(directory / "reload_checks.json", rows)


def lifecycle_checks(baseline, candidate, directory, device="cpu"):
    """Separate FP32 round trips, native half-save quantization, EMA preservation and Conv-BN fusion."""
    directory.mkdir(parents=True, exist_ok=True)
    model = copy.deepcopy(candidate).to(device).eval()
    with torch.no_grad():
        model.model[15].out.weight.normal_(0, 0.01)
    x = torch.randn(1, 3, 160, 160, device=device)
    original_state = copy.deepcopy(model.state_dict())
    torch.save(original_state, directory / "state_dict.pt")
    restored = copy.deepcopy(candidate).to(device).eval()
    restored.load_state_dict(torch.load(directory / "state_dict.pt", map_location=device, weights_only=True))
    rows = []
    with torch.no_grad():
        expected = model(x)
        common.assert_close_tree(expected, restored(x), 0, 0, "state_dict_output", rows)
    common.assert_close_tree(original_state, restored.state_dict(), 0, 0, "state_dict")
    torch.save({"model": model, "train_args": common.REFERENCE["args"]}, directory / "reload.pt")
    loaded, _ = load_checkpoint(directory / "reload.pt", device=device)
    common.assert_close_tree(original_state, loaded.state_dict(), 0, 0, "checkpoint_state")
    with torch.no_grad():
        common.assert_close_tree(expected, loaded(x), 0, 0, "reload_output", rows)
    torch.save(dict(x=x, expected=expected, backend=common.computation_conditions()), directory / "expected.pt")
    subprocess.run(
        [sys.executable, "-u", str(Path(__file__).resolve()), "--reload", str(directory)], cwd=ROOT, check=True
    )
    ema = ModelEMA(model)
    ema.update(model)
    assert set(ema.ema.state_dict()) == set(model.state_dict())
    for name in dict(model.model[15].named_parameters()):
        common.assert_close_tree(
            model.model[15].state_dict()[name], ema.ema.model[15].state_dict()[name], 1e-7, 1e-7, f"ema.{name}", rows
        )
    assert ema.ema.model[15].out.weight.count_nonzero() > 0
    torch.save(
        {"model": None, "ema": ema.ema, "updates": ema.updates, "train_args": common.REFERENCE["args"]},
        directory / "ema.pt",
    )
    ema_loaded, checkpoint = load_checkpoint(directory / "ema.pt", device=device)
    common.assert_close_tree(ema.ema.state_dict(), ema_loaded.state_dict(), 0, 0, "ema_reload_state")
    assert checkpoint["updates"] == 1
    quantization = []
    for label, m in (("baseline", baseline), ("candidate_nonzero", model)):
        m = copy.deepcopy(m).to(device).eval()
        api = YOLO(str(common.MODEL))
        api.model, api.ckpt = m, {"train_args": common.REFERENCE["args"]}
        api.save(directory / f"{label}_native_half.pt")
        half_loaded, ckpt = load_checkpoint(directory / f"{label}_native_half.pt", device=device)
        control = copy.deepcopy(m).half().float()
        common.assert_close_tree(control.state_dict(), half_loaded.state_dict(), 0, 0, f"{label}.half_state")
        with torch.no_grad():
            common.assert_close_tree(control(x), half_loaded(x), 0, 0, f"{label}.same_quantization", rows)
            original_output, quantized_output = m(x), half_loaded(x)
            a, b = original_output[0], quantized_output[0]
            raw = {}
            for branch in ("one2many", "one2one"):
                for key in ("boxes", "scores"):
                    delta = (original_output[1][branch][key] - quantized_output[1][branch][key]).abs()
                    raw[f"{branch}.{key}"] = dict(max_abs=delta.max().item(), mean_abs=delta.mean().item())
        quantization.append(
            dict(
                model=label,
                max_abs=(a - b).abs().max().item(),
                mean_abs=(a - b).abs().mean().item(),
                comparison="FP32 vs native FP16-save then FP32-load",
                postprocessed_rows_are_order_sensitive=True,
                aligned_raw_heads=raw,
            )
        )
    fused = copy.deepcopy(model).fuse(verbose=False)
    assert torch.equal(fused.model[15].out.weight, model.model[15].out.weight)
    with torch.no_grad():
        # Native fuse intentionally removes one2many; compare retained one2one prediction tensors.
        common.assert_close_tree(expected[0], fused(x)[0], FUSE_ATOL, FUSE_RTOL, "fused_predictions", rows)
    return dict(
        rows=rows,
        fresh_process=True,
        quantization=quantization,
        ema_preserved=True,
        new_state_count=4,
        fuse_tolerance=dict(atol=FUSE_ATOL, rtol=FUSE_RTOL),
    )


def optimizer_replay(optimizer, group, parameter, zero_task=False):
    """Reuse DCS's isolated exact native MuSGD replay, including momentum and the zero-task counterfactual."""
    replica = torch.nn.Parameter(parameter.detach().clone())
    replica.grad = torch.zeros_like(parameter) if zero_task else parameter.grad.detach().clone()
    replay = type(optimizer)([dict(group, params=[replica])], muon=optimizer.muon, sgd=optimizer.sgd)
    replay.state[replica] = copy.deepcopy(optimizer.state.get(parameter, {}))
    replay.step()
    return replica.detach(), replay.state[replica]


def observe_step(model, optimizer, scaler, step):
    """Observe trainer-owned unscale/clip/step/EMA without replacing native MuSGD or calling unscale twice."""
    assert type(optimizer) is MuSGD
    parameters = dict(model.model[15].named_parameters())
    before = {k: p.detach().clone() for k, p in parameters.items()}
    scale = scaler.get_scale()
    rows, groups, expected, no_task, calls = {}, {}, {}, {}, []
    nonfinite = [k for k, p in model.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
    for name, p in parameters.items():
        matches = [(i, g) for i, g in enumerate(optimizer.param_groups) for v in g["params"] if v is p]
        assert len(matches) == 1 and p.requires_grad and p.dtype == torch.float32, name
        index, group = matches[0]
        groups[name] = group
        assert p.grad is not None, f"{name}: disconnected gradient"
        rows[name] = dict(
            name=name,
            shape=list(p.shape),
            dtype=str(p.dtype),
            requires_grad=p.requires_grad,
            optimizer_group={
                "index": index,
                **{k: group.get(k) for k in ("param_group", "lr", "momentum", "weight_decay", "use_muon", "nesterov")},
            },
            unscaled_before_clip=tensor_stats(p.grad.detach().double() / scale),
            before=tensor_stats(p),
            optimizer_state_before={k: tensor_stats(v) for k, v in optimizer.state.get(p, {}).items()},
        )

    def before_step(opt, args, kwargs):
        calls.append("before")
        for name, p in parameters.items():
            rows[name]["unscaled_after_clip"] = tensor_stats(p.grad)
            assert rows[name]["unscaled_after_clip"]["finite"], name
            expected[name] = optimizer_replay(opt, groups[name], p)
            no_task[name] = optimizer_replay(opt, groups[name], p, zero_task=True)

    handles = [
        optimizer.register_step_pre_hook(before_step),
        optimizer.register_step_post_hook(lambda *args: calls.append("after")),
    ]
    try:
        step()
    finally:
        for handle in handles:
            handle.remove()
    skipped = not calls
    if skipped:
        assert nonfinite and scaler.get_scale() < scale, "No optimizer step without an AMP overflow"
        assert all(torch.equal(before[k], p) for k, p in parameters.items())
    else:
        assert calls == ["before", "after"] and not nonfinite
    for name, p in parameters.items():
        info = rows[name]
        info.update(
            after=tensor_stats(p),
            delta=(p.detach().double() - before[name].double()).abs().max().item(),
            effective_update=False,
            update_kind="overflow_skip" if skipped else "no_task_update",
            optimizer_state_after={k: tensor_stats(v) for k, v in optimizer.state.get(p, {}).items()},
        )
        if skipped:
            continue
        predicted, state = expected[name]
        assert torch.equal(predicted, p), f"{name}: actual step differs from native replay"
        common.assert_close_tree(state, optimizer.state[p], 0, 0, f"{name}.optimizer_state")
        info["native_replay_exact"] = True
        info["task_counterfactual_delta"] = (p.detach().double() - no_task[name][0].double()).abs().max().item()
        grad = info["unscaled_after_clip"]
        if (
            grad["max_abs"] > 0
            and info["delta"] > 0
            and info["task_counterfactual_delta"] > 0
            and groups[name]["lr"] > 0
        ):
            info.update(effective_update=True, update_kind="observable_task_update")
        # Rounding-limited changes are kept as incomplete; never call decay-only motion a task update.
    return dict(
        parameters=rows,
        out_zero_before=not bool(before["out.weight"].count_nonzero()),
        out_zero_after=not bool(parameters["out.weight"].count_nonzero()),
        scaler_before=scale,
        scaler_after=scaler.get_scale(),
        skipped=skipped,
        nonfinite_scaled_gradient_keys=nonfinite,
    )


def staged_gradient_audit(rows, require_complete=True):
    """Require output projection learning to precede data-driven updates in every upstream tensor."""
    summary = {
        k: dict(first_gradient_batch=None, first_effective_update_batch=None)
        for k in ("proj_l.weight", "proj_s.weight", "dw.weight", "out.weight")
    }
    first_unlock = None
    for row in rows:
        for name, info in row["parameters"].items():
            grad = info["unscaled_before_clip"]
            if grad["finite"] and grad["max_abs"] > 0 and summary[name]["first_gradient_batch"] is None:
                summary[name]["first_gradient_batch"] = row["batch"]
            if row["out_zero_before"] and name != "out.weight" and grad["finite"]:
                assert grad["max_abs"] == 0, f"{name}: upstream data gradient before output projection unlock"
            if info["effective_update"] and summary[name]["first_effective_update_batch"] is None:
                if name == "out.weight":
                    assert not row["out_zero_after"]
                    first_unlock = row["batch"]
                else:
                    assert first_unlock is not None and row["batch"] > first_unlock, name
                summary[name]["first_effective_update_batch"] = row["batch"]
    missing = [name for name, info in summary.items() if info["first_effective_update_batch"] is None]
    if require_complete:
        assert rows and rows[0]["out_zero_before"]
        assert not missing, f"Missing gradient-backed task updates within fixed budget: {missing}"
    return dict(parameters=summary, missing=missing, first_output_update_batch=first_unlock)


def synthetic_model_updates(candidate, device, amp, directory):
    """Use native end-to-end detection loss and native MuSGD on local synthetic B2/160 only."""
    model = copy.deepcopy(candidate).to(device).train()
    model.args = SimpleNamespace(**common.REFERENCE["args"])
    optimizer = DetectionTrainer.build_optimizer(None, model, "MuSGD", 0.01, 0.937, 0.0005)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    batch = dict(
        img=torch.rand(2, 3, 160, 160, device=device),
        batch_idx=torch.arange(2, device=device),
        cls=torch.zeros(2, 1, device=device),
        bboxes=torch.tensor([[0.5, 0.5, 0.08, 0.6], [0.3, 0.6, 0.6, 0.08]], device=device),
    )
    rows = []
    for i in range(MAX_BATCHES):
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

        row = observe_step(model, optimizer, scaler, native_step)
        rows.append(dict(batch=i, loss=loss.item(), **row))
        common.write_json(directory / f"synthetic_{device.replace(':', '_')}_amp_{amp}.json", rows)
        if not staged_gradient_audit(rows, False)["missing"]:
            break
    return dict(
        batch=2,
        imgsz=160,
        amp=amp,
        device=device,
        source="synthetic; not server preflight",
        summary=staged_gradient_audit(rows),
        steps=rows,
    )


class PreflightComplete(Exception):
    """Terminate only the bounded verifier at the native batch-end callback, before epoch validation/save."""


class PreflightTrainer(AuditedTrainer):
    """Instrument the actual native training loop; all warmup/accumulation/scaler/EMA actions remain native."""

    def optimizer_step(self):
        """Record a real native attempt and its task-gradient counterfactual at the legal optimizer hook."""
        row = observe_step(self.model, self.optimizer, self.scaler, super().optimizer_step)
        self.audit_rows.append(dict(batch=self.audit_batch, **row))

    def preprocess_batch(self, batch):
        """Verify actual B32/640 and record exact train image IDs after the native batch preprocessing."""
        batch = super().preprocess_batch(batch)
        assert tuple(batch["img"].shape) == (32, 3, 640, 640)
        self.audit_images = batch["im_file"]
        return batch


def native_preflight(config, directory):
    """Run at most 64 real native B32 batches and discard the entire preflight process afterwards."""
    trainer = PreflightTrainer({**config, "project": str(directory / "native"), "name": "check"})
    trainer.audit_rows, trainer.audit_batches, trainer.audit_batch = [], [], -1
    trainer.add_callback("on_pretrain_routine_end", audit_training_setup)

    def start(t):
        t.audit_batch += 1
        assert t.audit_batch < MAX_BATCHES, "Preflight exceeded 64 batches"
        print(f"LBI native preflight batch {t.audit_batch + 1}/{MAX_BATCHES}", flush=True)

    def end(t):
        stepped = bool(t.audit_rows and t.audit_rows[-1]["batch"] == t.audit_batch)
        t.audit_batches.append(
            dict(
                batch=t.audit_batch,
                images=t.audit_images,
                loss=t.loss.item(),
                accumulate=t.accumulate,
                optimizer_attempt=stepped,
                status=("overflow_skip" if t.audit_rows[-1]["skipped"] else "step") if stepped else "accumulation",
            )
        )
        assert torch.isfinite(t.loss)
        summary = staged_gradient_audit(t.audit_rows, False)
        common.write_json(directory / "native_steps.json", t.audit_rows)
        common.write_json(directory / "native_batches.json", t.audit_batches)
        common.write_json(directory / "native_gradient_summary.json", summary)
        if not summary["missing"] or t.audit_batch + 1 == MAX_BATCHES:
            staged_gradient_audit(t.audit_rows)
            raise PreflightComplete

    trainer.add_callback("on_train_batch_start", start)
    trainer.add_callback("on_train_batch_end", end)
    try:
        trainer.train()
    except PreflightComplete:
        return dict(
            batch=32,
            imgsz=640,
            amp=trainer.amp,
            max_batches=MAX_BATCHES,
            batches_observed=trainer.audit_batch + 1,
            summary=staged_gradient_audit(trainer.audit_rows),
            lifecycle="native BaseTrainer._do_train; callback-only bound; state discarded",
        )
    raise AssertionError("Native trainer exited without completing staged preflight")


def model_checks(source, directory):
    """Run the same layered checks on CPU and available CUDA, keeping their scopes explicit."""
    baseline, candidate = build_pair(source)
    report = dict(
        topology=topology_checks(baseline, candidate), weights=common.audit_weights(baseline, candidate, source)
    )
    report["complexity"] = dict(
        parameters=[sum(p.numel() for p in m.parameters()) for m in (baseline, candidate)],
        gflops=[get_flops(m, 640) for m in (baseline, candidate)],
        added_conv_gflops=2 * 6288 * 80 * 80 / 1e9,
        convention="THOP whole-model; 2 operations/MAC; extra conv budget excludes RMS/elementwise/memory",
    )
    assert report["complexity"]["parameters"][1] - report["complexity"]["parameters"][0] == 6288
    report["identity"], report["lifecycle"], report["local_detection_updates"] = [], {}, []
    for device in ["cpu", "cuda:0"] if torch.cuda.is_available() else ["cpu"]:
        print(f"LBI shared-model and lifecycle checks: {device}", flush=True)
        report["identity"].append(whole_identity(baseline, candidate, device))
        report["lifecycle"][device] = lifecycle_checks(
            baseline, candidate, directory / device.replace(":", "_"), device
        )
        report["local_detection_updates"].append(synthetic_model_updates(candidate, device, False, directory))
        if device.startswith("cuda"):
            report["identity"].append(whole_identity(baseline, candidate, device, True))
            report["local_detection_updates"].append(synthetic_model_updates(candidate, device, True, directory))
    return report


def main():
    """Write failure evidence as well as successes; only the exact server runtime can issue native PASS."""
    if len(sys.argv) == 3 and sys.argv[1] == "--reload":
        reload_in_process(Path(sys.argv[2]))
        return
    parser = options_parser()
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.local:
        torch.set_num_threads(4)
    init_seeds(42, deterministic=True)
    args.output.mkdir(parents=True, exist_ok=True)
    report = dict(
        passed=False,
        local_only=args.local,
        commit=common.git("rev-parse", "HEAD"),
        native_server_b32="NOT RUN",
        formal_training="NOT STARTED",
        runtime=common.computation_conditions(),
    )
    started = time.perf_counter()
    try:
        if not args.local:
            common.require_clean_source()
            require_runtime()
        raw, effective, evidence = common.resolve_recipe(args)
        audit_arguments(raw, effective)
        evidence["launcher"] = common.launcher_evidence(args, raw)
        evidence["source_sha256"] = common.source_hashes()
        report["recipe"] = evidence
        source, _ = load_checkpoint(evidence["initial_path"])
        print("LBI module checks: CPU/CUDA/AMP", flush=True)
        report["module_cpu"] = module_checks()
        if torch.cuda.is_available():
            report["module_cuda"] = module_checks("cuda:0")
            report["module_amp"] = module_checks("cuda:0", True)
        report.update(model_checks(source, args.output))
        if not args.local:
            report["module_B32_P3"] = module_checks("cuda:0", True, spatial=(80, 80), batch=32)
            report["native_preflight"] = native_preflight(effective, args.output)
            report["native_server_b32"] = "PASSED"
        report["passed"] = True
    except BaseException as error:
        report["error"] = str(error)
        report["traceback"] = traceback.format_exc()
        if not args.local:
            report["native_server_b32"] = "FAILED"
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        common.write_json(args.output / "checks.json", report)
    print(
        f"PASS: {'local validation' if args.local else 'server preflight'} -> {args.output / 'checks.json'}", flush=True
    )


if __name__ == "__main__":
    main()
