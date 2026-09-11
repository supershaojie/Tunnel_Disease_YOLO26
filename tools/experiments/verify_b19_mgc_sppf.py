"""Layered MGC audits; local checks never authorize the native server B32 preflight."""

# ruff: noqa: E402 -- Direct script entry imports this worktree.
import copy
import json
import subprocess
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from tools.experiments import b19_common as common
from tools.experiments.run_b19_mgc_sppf import (
    AuditedTrainer,
    audit_arguments,
    audit_training_setup,
    options_parser,
    require_runtime,
)
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.nn.modules import C2PSA, MGC_SPPF, SPPF
from ultralytics.nn.tasks import DetectionModel, load_checkpoint
from ultralytics.optim.muon import MuSGD
from ultralytics.utils.torch_utils import ModelEMA, autocast, get_flops, init_seeds


def module_checks(device="cpu", amp=False):
    """Check shape, exact identity, one BN update and delayed gradients with a nondegenerate probe."""
    init_seeds(42, deterministic=True)
    morphology = []
    for k in (3, 5):
        for value in (0.0, 2.0, -3.0):
            u = torch.full((2, 16, 13, 17), value, device=device, requires_grad=True)
            g = MGC_SPPF.gap(u, k)
            assert torch.isfinite(g).all() and g.count_nonzero() == 0
            g.sum().backward()
            assert torch.isfinite(u.grad).all()
            morphology.append(
                dict(kernel=k, constant=value, gap_max_abs=g.abs().max().item(), tie_gradient_finite=True)
            )
    identities = []
    for batch in (1, 2):
        for h, w in ((20, 20), (13, 17)):
            for training in (False, True):
                native = SPPF(256, 256, 5, 3, True).to(device).train(training)
                candidate = MGC_SPPF(256, 256, 5, 3, True).to(device).train(training)
                missing = candidate.load_state_dict(native.state_dict(), strict=False)
                assert set(missing.missing_keys) == {"gap_in.weight", "gap_out.weight"}
                assert not missing.unexpected_keys
                x = torch.randn(batch, 256, h, w, device=device)
                before = candidate.cv1.bn.num_batches_tracked.item()
                with autocast(amp, device=torch.device(device).type):
                    errors = module_identity(native, candidate, x)
                assert candidate.cv1.bn.num_batches_tracked.item() - before == int(training)
                common.assert_close_tree(
                    native.state_dict(),
                    {k: candidate.state_dict()[k] for k in native.state_dict()},
                    0,
                    0,
                    "shared_after_forward",
                )
                identities.append(dict(batch=batch, hw=[h, w], train=training, comparisons=errors))
    candidate = MGC_SPPF(256, 256, 5, 3, True).to(device).eval()
    x = torch.randn(2, 256, 13, 17, device=device, requires_grad=True)
    params = new_parameters(candidate)
    optimizer = torch.optim.SGD(params.values(), lr=0.01, weight_decay=0)
    target = torch.randn_like(x)
    steps = []
    for i in range(2):
        optimizer.zero_grad(set_to_none=True)
        before = {k: p.detach().clone() for k, p in params.items()}
        with autocast(amp, device=torch.device(device).type):
            output = candidate(x)
            loss = (output.float() * target).mean(dim=(2, 3)).sum()
        assert torch.isfinite(output).all() and output.shape == x.shape
        loss.backward()
        gradients = {k: tensor_stats(p.grad) for k, p in params.items()}
        assert all(g["finite"] for g in gradients.values())
        assert gradients["gap_out.weight"]["max_abs"] > 0
        assert (gradients["gap_in.weight"]["max_abs"] == 0) == (i == 0)
        optimizer.step()
        steps.append(
            {k: dict(gradient=gradients[k], delta=(p - before[k]).abs().max().item()) for k, p in params.items()}
        )
    assert all(steps[1][k]["delta"] > 0 for k in params)
    assert torch.isfinite(x.grad).all()
    with torch.no_grad():
        z = candidate.cv1(x)
        levels = [z]
        levels.extend(candidate.m(levels[-1]) for _ in range(3))
        native = candidate.cv2(torch.cat(levels, 1)) + x
        u = candidate.gap_in(z)
        gaps = []
        for k in (3, 5):
            d = torch.nn.functional.max_pool2d(u, k, 1, k // 2)
            gaps.append(-torch.nn.functional.max_pool2d(-d, k, 1, k // 2) - u)
        expected = native + candidate.gap_out(torch.cat(gaps, 1))
        common.assert_close_tree(expected, candidate(x), 0, 0, "nonzero_formula")
    return dict(device=device, amp=amp, morphology=morphology, identities=identities, staged_sgd=steps, formula="PASS")


def topology_checks(baseline, candidate):
    """Verify all layer classes and connections, with the sole fixed layer-nine replacement."""
    cfg = copy.deepcopy(candidate.yaml)
    assert cfg["backbone"][9] == [-1, 1, "MGC_SPPF", [1024, 5, 3, True, 16]]
    cfg["backbone"][9] = baseline.yaml["backbone"][9]
    assert common.architecture_signature(cfg) == common.architecture_signature(baseline.yaml)
    assert len(candidate.model) == len(baseline.model) == 24
    table = []
    for i, (a, b) in enumerate(zip(baseline.model, candidate.model)):
        assert (a.f, a.i) == (b.f, b.i)
        if i != 9:
            assert str(a) == str(b)
        table.append(dict(index=i, source=a.f, native=type(a).__name__, candidate=type(b).__name__))
    block, head = candidate.model[9], candidate.model[-1]
    assert type(block) is MGC_SPPF and type(candidate.model[10]) is C2PSA
    assert tuple(block.gap_in.weight.shape) == (16, 128, 1, 1)
    assert tuple(block.gap_out.weight.shape) == (256, 32, 1, 1)
    assert block.gap_out.weight.count_nonzero() == 0 and block.gap_in.weight.count_nonzero() > 0
    assert block.cv1.conv.in_channels == block.cv2.conv.out_channels == 256
    assert block.cv1.conv.out_channels == 128 and block.add and block.n == 3
    assert candidate.yaml["scale"] == "n" and candidate.yaml["nc"] == 1
    assert head.f == [16, 19, 22] and head.nc == head.reg_max == 1 and candidate.end2end
    assert candidate.stride.tolist() == [8, 16, 32]
    return dict(changed_layers=[9], table=table, channels=[256, 128, 256], rank=16)


def model_checks(source, directory, device="cpu", report=None):
    """Audit aligned shared states, both Detect branches, native loss updates and deployment lifecycle."""
    baseline, candidate = build_pair(source)
    report = {} if report is None else report
    report.update(
        topology=topology_checks(baseline, candidate), weights=common.audit_weights(baseline, candidate, source)
    )
    assert report["weights"]["added_parameters"] == 10240
    assert len(YOLO(str(common.MODEL)).model.model) == 24
    report["complexity"] = {
        "parameters": [sum(p.numel() for p in m.parameters()) for m in (baseline, candidate)],
        "gflops": [get_flops(m, 640) for m in (baseline, candidate)],
        "pointwise_delta_gflops": 0.008192,
        "scope": "THOP multiply-add=2; excludes morphology comparisons and memory traffic",
    }
    assert min(report["complexity"]["gflops"]) > 0
    x = torch.randn(1, 3, 640, 640, device=device)
    baseline, candidate = baseline.to(device), candidate.to(device)
    report["identity"] = []
    for training in (False, True):
        baseline.train(training)
        candidate.train(training)
        shapes = []
        handle = candidate.model[9].register_forward_hook(lambda m, ins, out: shapes.append(list(out.shape)))
        try:
            with torch.no_grad():
                a, b = baseline(x), candidate(x)
        finally:
            handle.remove()
        assert shapes == [[1, 256, 20, 20]]
        common.assert_close_tree(a, b, 0, 0, f"train_{training}", report["identity"])
        common.assert_close_tree(
            baseline.state_dict(),
            {k: candidate.state_dict()[k] for k in baseline.state_dict()},
            0,
            0,
            "model_shared_bn",
        )
    if str(device).startswith("cuda"):
        with torch.no_grad(), autocast(True):
            a, b = baseline.eval()(x), candidate.eval()(x)
        report["amp_identity"] = []
        common.assert_close_tree(a, b, 1e-6, 1e-5, "amp_identity", report["amp_identity"])
    report["stride_exact"] = torch.equal(baseline.stride, candidate.stride)
    report["head_state_exact"] = torch.equal(baseline.model[-1].stride, candidate.model[-1].stride)
    del baseline
    report["fp32"] = synthetic_model_updates(candidate, device, False, directory)
    if str(device).startswith("cuda"):
        report["amp"] = synthetic_model_updates(candidate, device, True, directory)
    candidate.eval().zero_grad(set_to_none=True)
    candidate.args = SimpleNamespace(**common.REFERENCE["args"])
    with torch.no_grad():
        candidate.model[9].gap_out.weight.fill_(0.01)
    ema = ModelEMA(candidate)
    with torch.no_grad():
        candidate.model[9].gap_out.weight.fill_(0.02)
    ema.update(candidate)
    assert ema.ema.model[9].gap_out.weight.count_nonzero() > 0
    assert torch.all(ema.ema.model[9].gap_out.weight > 0.01)
    report["ema"] = {"updates": ema.updates, "output_projection": tensor_stats(ema.ema.model[9].gap_out.weight)}
    report["fp32_reload"] = fp32_reload_check(candidate, directory / "fp32_snapshot", x)
    report["reload"] = save_reload_check(candidate, directory, x)
    with torch.no_grad():
        restored, _ = load_checkpoint(report["reload"]["checkpoint"])
        backend = AutoBackend(model=restored, device=torch.device(device), fp16=False, verbose=False)
        before = backend(x)
        backend.model.model[9].gap_out.weight.zero_()
        after = backend(x)
        effect = {k: (before[1]["one2one"][k] - after[1]["one2one"][k]).abs().max().item() for k in ("boxes", "scores")}
        assert max(effect.values()) > 0, "Saved nonzero MGC residual must survive native deployment"
        report["deployment_nonzero_effect"] = effect
    return report


class PreflightComplete(Exception):
    """End the independent verifier before epoch evaluation or formal training."""


class PreflightTrainer(AuditedTrainer):
    """Observe the real native training loop without replacing its loss, warmup, AMP or optimizer schedule."""

    def preprocess_batch(self, batch):
        """Check real batch dimensions and record original training sample order."""
        batch = super().preprocess_batch(batch)
        assert tuple(batch["img"].shape) == (32, 3, 640, 640)
        self.audit_samples = [str(p) for p in batch["im_file"]]
        return batch

    def optimizer_step(self):
        """Observe exactly one native scaler/clip/step/EMA lifecycle when the trainer requests a step."""
        updates = self.ema.updates
        row = observe_step(
            self.model, new_parameters(self.model.model[9]), self.optimizer, self.scaler, super().optimizer_step
        )
        row.update(batch=self.audit_batch, loss=self.loss.item(), ema_before=updates, ema_after=self.ema.updates)
        self.audit_steps.append(row)


def native_preflight(config, directory):
    """Stop after both projections learn or at 64 original B32 batches, preserving failure evidence."""
    print(
        "MGC preflight ONLY: native recipe retains epochs=200, audit stops within 64 B32 batches; formal NOT STARTED",
        flush=True,
    )
    trainer = PreflightTrainer({**config, "project": str(directory / "native"), "name": "check"})
    trainer.audit_batch, trainer.audit_steps = -1, []
    batches, shapes, handles = [], [], []

    def setup(t):
        audit_training_setup(t)
        handles.append(
            t.model.model[9].register_forward_hook(
                lambda m, ins, out: shapes.append([list(ins[0].shape), list(out.shape)])
            )
        )

    def start(t):
        t.audit_batch += 1
        assert t.audit_batch < 64, "Preflight cannot consume a 65th training batch"
        shapes.clear()
        print(f"MGC native preflight batch={t.audit_batch + 1}/64", flush=True)

    def end(t):
        assert torch.isfinite(t.loss), "Nonfinite native detection loss"
        assert shapes == [[[32, 256, 20, 20], [32, 256, 20, 20]]]
        stepped = bool(t.audit_steps and t.audit_steps[-1]["batch"] == t.audit_batch)
        batches.append(
            dict(
                batch=t.audit_batch,
                samples=t.audit_samples,
                layer9=shapes.copy(),
                accumulate=t.accumulate,
                optimizer_requested=stepped,
                event=("amp_overflow_skip" if t.audit_steps[-1]["skipped"] else "optimizer_step")
                if stepped
                else "gradient_accumulation",
            )
        )
        common.write_json(directory / "native_batches.json", batches)
        common.write_json(directory / "native_steps.json", t.audit_steps)
        if t.audit_steps:
            summary = staged_gradient_audit(t.audit_steps, require_complete=False)
            common.write_json(directory / "native_gradient_summary.json", summary)
            if not summary["missing_effective_parameters"]:
                raise PreflightComplete
        if t.audit_batch == 63:
            raise AssertionError("No full staged native gradient/update evidence within 64 B32 batches")

    trainer.add_callback("on_pretrain_routine_end", setup)
    trainer.add_callback("on_train_batch_start", start)
    trainer.add_callback("on_train_batch_end", end)
    try:
        trainer.train()
        raise AssertionError("Native trainer returned before the bounded audit completed")
    except PreflightComplete:
        return dict(
            batch=32,
            imgsz=640,
            amp=True,
            max_batches=64,
            batches_observed=len(batches),
            lifecycle="native BaseTrainer._do_train",
            staged_audit=staged_gradient_audit(trainer.audit_steps),
        )
    finally:
        for handle in handles:
            handle.remove()
        common.write_json(directory / "native_steps.json", trainer.audit_steps)
        common.write_json(directory / "native_batches.json", batches)


def main():
    """Persist partial failures and distinguish local evidence from an independent server receipt."""
    if len(sys.argv) == 3 and sys.argv[1] == "--reload-fp32":
        reload_fp32_in_process(Path(sys.argv[2]))
        return
    parser = options_parser()
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--local-device", choices=["cpu", "cuda:0"], default="cpu")
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
        native_server_preflight="NOT RUN",
        formal_training="NOT STARTED",
    )
    try:
        if not args.local:
            common.require_clean_source()
            require_runtime()
        print("MGC source, original weights, all args and dataset manifest audit", flush=True)
        raw, effective, evidence = common.resolve_recipe(args)
        audit_arguments(raw, effective)
        evidence["launcher"] = common.launcher_evidence(args, raw)
        evidence["source_sha256"] = common.source_hashes()
        report["recipe"] = evidence
        source, _ = load_checkpoint(evidence["initial_path"])
        device = args.local_device if args.local else "cuda:0"
        print(f"MGC module checks: {device}", flush=True)
        report["module"] = module_checks(device)
        if device.startswith("cuda"):
            report["module_amp"] = module_checks(device, True)
        print(f"MGC model/loss/reload/fuse checks: {device}", flush=True)
        model_checks(source, args.output, device, report)
        if not args.local:
            report["native_server_preflight"] = "FAILED"
            report["native_preflight"] = native_preflight(effective, args.output)
            report["native_server_preflight"] = "PASSED"
        report["passed"] = True
    except Exception as error:
        report.update(error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        common.write_json(args.output / "checks.json", report)
    print(f"PASS: {'local validation' if args.local else 'native server preflight'} -> {args.output}", flush=True)


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
    """Select exactly the two new projection weights."""
    return {k: p for k, p in module.named_parameters() if k in {"gap_in.weight", "gap_out.weight"}}


def build_pair(source=None):
    """Build native and MGC with aligned RNG, then apply the same native COCO-to-crack loading."""
    init_seeds(42, deterministic=True)
    with torch.random.fork_rng(devices=[]):
        baseline = DetectionModel(common.baseline_architecture(), verbose=False)
        baseline_rng = torch.get_rng_state()
    candidate = DetectionModel(str(common.MODEL), nc=1, verbose=False)
    assert torch.equal(baseline_rng, torch.get_rng_state()), "MGC consumed subsequent shared initialization RNG"
    for model in (baseline, candidate):
        model.names = {0: "crack"}
        if source is not None:
            model.load(source, verbose=False)
    return baseline, candidate


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
            name=name,
            shape=list(parameter.shape),
            requires_grad=parameter.requires_grad,
            registrations=len(matches),
            dtype=str(parameter.dtype),
            before=parameter_snapshot(parameter),
            optimizer_state_before={
                key: tensor_stats(value) for key, value in optimizer.state.get(parameter, {}).items()
            },
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
        info["optimizer_state_after"] = {key: tensor_stats(value) for key, value in state.items()}
        info["current_task_update_max_abs"] = (p.detach().double() - no_task[name][0].double()).abs().max().item()
        post = info["gradient_after_clip"]
        eligible = post["max_abs"] > 0 and info["gradient_max_abs"] > 0 and groups[name]["lr"] > 0
        if eligible and info["update_max_abs"] > 0 and info["current_task_update_max_abs"] > 0:
            info.update(effective_update=True, update_kind="observable_task_update")
    assert all(torch.isfinite(p).all() for p in model.parameters())
    return {
        "parameters": rows,
        "output_zero_before": before["gap_out.weight"].count_nonzero().item() == 0,
        "output_zero_after": parameters["gap_out.weight"].count_nonzero().item() == 0,
        "scaler_before": scale,
        "scaler_after": scaler.get_scale(),
        "skipped": skipped,
        "nonfinite_scaled_gradient_keys": nonfinite,
    }


def staged_gradient_audit(rows, require_complete=True):
    """Require an effective output step before a later gradient-backed upstream update."""
    assert rows and rows[0]["output_zero_before"], "Missing step-zero identity"
    summary = {
        name: dict(
            shape=info["shape"],
            dtype=info["dtype"],
            requires_grad=info["requires_grad"],
            optimizer_group=info["optimizer_group"],
            first_nonzero_gradient_batch=None,
            first_effective_update_batch=None,
        )
        for name, info in rows[0]["parameters"].items()
    }
    for row in rows:
        for name, info in row["parameters"].items():
            entry = summary[name]
            grad = info["gradient_before_clip"]
            if grad["finite"] and grad["max_abs"] > 0 and entry["first_nonzero_gradient_batch"] is None:
                entry["first_nonzero_gradient_batch"] = row["batch"]
            if row["output_zero_before"] and name == "gap_in.weight" and grad["finite"]:
                assert grad["max_abs"] == 0, "Upstream task gradient must initially be zero"
            if info["effective_update"] and entry["first_effective_update_batch"] is None:
                entry["first_effective_update_batch"] = row["batch"]
    out = summary["gap_out.weight"]["first_effective_update_batch"]
    upstream = summary["gap_in.weight"]["first_effective_update_batch"]
    if upstream is not None:
        assert out is not None and upstream > out
    missing = [k for k, v in summary.items() if v["first_effective_update_batch"] is None]
    if require_complete:
        assert not missing, f"No observable task update within the fixed budget: {missing}"
    return dict(parameters=summary, missing_effective_parameters=missing)


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
        common.write_json(directory / f"synthetic_amp_{amp}.json", rows)
        print(f"MGC local synthetic AMP={amp} batch={i + 1}/64 skipped={row['skipped']}", flush=True)
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


def fp32_reload_check(model, directory, x):
    """Save one FP32 model snapshot and require exact same-instance fixed-input outputs after a fresh-process reload."""
    directory.mkdir(parents=True, exist_ok=True)
    model = copy.deepcopy(model).eval()
    with torch.no_grad():
        expected = model(x)
    snapshot = directory / "reload.pt"
    torch.save({"model": model, "train_args": common.REFERENCE["args"]}, snapshot)
    loaded, _ = load_checkpoint(snapshot, device=x.device)
    rows = []
    common.assert_close_tree(model.state_dict(), loaded.state_dict(), 0, 0, "state", rows)
    with torch.no_grad():
        common.assert_close_tree(expected, loaded(x), 0, 0, "reload_output", rows)
    torch.save({"x": x, "expected": expected, "backend": common.computation_conditions()}, directory / "expected.pt")
    subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--reload-fp32", str(directory)],
        cwd=ROOT,
        check=True,
        timeout=180,
    )
    return {
        "exact_state_tensors": len(model.state_dict()),
        "outputs": [r for r in rows if r["path"].startswith("reload")],
        "fresh_process": True,
    }


def reload_fp32_in_process(directory):
    """Replay the saved input using the snapshot's recorded backend settings without changing precision policy."""
    payload = torch.load(directory / "expected.pt", weights_only=False)
    torch.set_num_threads(payload["backend"]["threads"])
    init_seeds(42, deterministic=True)
    assert common.computation_conditions() == payload["backend"], "Fresh-process numerical backend differs"
    model, _ = load_checkpoint(directory / "reload.pt", device=payload["x"].device)
    errors = []
    with torch.no_grad():
        common.assert_close_tree(payload["expected"], model(payload["x"]), 0, 0, "fresh_reload", errors)
    common.write_json(directory / "reload_checks.json", errors)


def save_reload_check(model, directory, x):
    """Reuse NDP's FP16 EMA snapshot/FP32 reference lifecycle, preserving the exact same model and input."""
    directory = Path(directory) / "reload"
    directory.mkdir(parents=True, exist_ok=True)
    snapshot = copy.deepcopy(model).cpu().half().eval()
    snapshot.criterion = None
    assert snapshot.model[9].gap_out.weight.count_nonzero() > 0
    args = snapshot.args if isinstance(snapshot.args, dict) else vars(snapshot.args)
    path = directory / "preflight.pt"
    torch.save({"model": None, "ema": snapshot, "train_args": args}, path)
    snapshot.float()  # Compare the saved quantized snapshot, never the original pre-quantization model.
    with torch.no_grad():
        x = x.detach().cpu().float()
        reference = {
            "x": x,
            "state": {k: v.clone() for k, v in snapshot.state_dict().items()},
            "attributes_before": common.inference_attributes(snapshot),
            "threads": torch.get_num_threads(),
            "conditions": common.computation_conditions(),
        }
        reference["raw"] = snapshot(x)
        before_quantization = copy.deepcopy(model).cpu().float().eval()(x)
        quantization = []
        for branch in ("one2many", "one2one"):
            for key in ("boxes", "scores"):
                a, b = before_quantization[1][branch][key], reference["raw"][1][branch][key]
                delta = (a.double() - b.double()).abs()
                quantization.append(
                    dict(
                        branch=branch,
                        tensor=key,
                        shape=list(a.shape),
                        dtype=str(a.dtype),
                        max_abs=delta.max().item(),
                        mean_abs=delta.mean().item(),
                    )
                )
        common.assert_close_tree(reference["state"], snapshot.state_dict(), 0, 0, path="snapshot_after_forward")
        reference["attributes_after"] = common.inference_attributes(snapshot)
        torch.save(reference, directory / "reload_reference.pt")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from tools.experiments.verify_b19_mgc_sppf import reload_in_process; "
            "import sys; reload_in_process(sys.argv[1])",
            str(path),
        ],
        cwd=common.ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=180,
    )
    (directory / "reload.log").write_text(result.stdout + result.stderr, encoding="utf-8")
    result.check_returncode()
    return {
        "checkpoint": str(path),
        "sha256": common.sha256(path),
        "fresh_process": True,
        "quantization_effect_not_identity_error": quantization,
        **json.loads((directory / "reload_check.json").read_text(encoding="utf-8")),
    }


def fusion_check(model, x):
    """Retain the complete one2one assertion on the native CPU checkpoint fusion path."""
    assert x.device.type == next(model.parameters()).device.type == "cpu"
    before = model(x)
    fused = copy.deepcopy(model).fuse(verbose=False)
    after = fused(x)
    assert after[1]["one2many"] == {}
    rows = []
    common.assert_close_tree(before[1]["one2one"], after[1]["one2one"], atol=1e-4, rtol=1e-4, report=rows)
    # NDP compares decoded anchors before top-k: tiny score roundoff can reorder otherwise identical boxes.
    decoded_before = fused.model[-1]._inference(before[1]["one2one"])
    decoded_after = fused.model[-1]._inference(after[1]["one2one"])
    assert decoded_before.shape == decoded_after.shape == (x.shape[0], 5, before[1]["one2one"]["boxes"].shape[-1])
    # Preserve the original pixel-unit bound: decoding multiplies raw distances by stride, up to 32.
    torch.testing.assert_close(decoded_after[:, :4], decoded_before[:, :4], rtol=1e-4, atol=32e-4)
    torch.testing.assert_close(decoded_after[:, 4:], decoded_before[:, 4:], rtol=1e-4, atol=1e-6)
    return {
        "raw_one2one_errors": rows,
        "coordinate_max_abs_pixels": (decoded_after[:, :4] - decoded_before[:, :4]).abs().max().item(),
        "decoded_order": "all anchors before top-k",
    }


def reload_in_process(path):
    """Audit exact state, Detect caches and all raw outputs before native CPU fusion, as in SIR/NDP."""
    path = Path(path)
    reference = torch.load(path.with_name("reload_reference.pt"), map_location="cpu", weights_only=False)
    torch.set_num_threads(reference["threads"])
    report = {
        "passed": False,
        "state_exact": False,
        "raw": [],
        "checkpoint_precision": "FP16 EMA, FP32 reload",
        "reference_conditions": reference["conditions"],
        "conditions": common.computation_conditions(),
    }
    try:
        model = YOLO(path).model
        assert type(model.model[9]) is MGC_SPPF and model.model[9].gap_out.weight.count_nonzero() > 0
        common.assert_close_tree(reference["state"], model.state_dict(), 0, 0, path="reload.state")
        common.assert_close_tree(
            reference["attributes_before"], common.inference_attributes(model), 0, 0, path="reload.attributes_before"
        )
        with torch.no_grad():
            before = model(reference["x"])
            common.assert_close_tree(reference["state"], model.state_dict(), 0, 0, path="reload.state_after_forward")
            common.assert_close_tree(
                reference["attributes_after"], common.inference_attributes(model), 0, 0, path="reload.attributes_after"
            )
            common.assert_close_tree(reference["raw"], before, 0, 0, path="reload.raw", report=report["raw"])
            report.update(state_exact=True, state_keys=len(reference["state"]), attributes_exact=True)
            # Native control inherits all shared tensors from this same snapshot, including adapted Detect weights.
            native = DetectionModel(common.baseline_architecture(), verbose=False).eval()
            native.load_state_dict({k: reference["state"][k] for k in native.state_dict()}, strict=True)
            report["native_fusion"] = fusion_check(native, reference["x"])
            report["mgc_fusion"] = fusion_check(model, reference["x"])
        report["passed"] = True
    finally:
        common.write_json(path.with_name("reload_check.json"), report)
    print("PASS: exact snapshot state/raw/Detect caches; native and MGC CPU one2one fusion")


if __name__ == "__main__":
    main()
