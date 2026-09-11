"""Layered DCS audits; local synthetic checks never authorize the independent server preflight."""

# ruff: noqa: E402 -- Direct script entry must import this worktree.
import copy
import subprocess
import sys
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
from ultralytics.utils.torch_utils import autocast, get_flops, init_seeds


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


def module_checks(device="cpu", amp=False):
    """Check exact identity, an independent nonzero formula, and the necessary delayed branch learning."""
    init_seeds(42, deterministic=True)
    native = SPPF(256, 256, 5, 3, True).to(device).eval()
    candidate = DCS_SPPF(256, 256, 5, 3, True).to(device).eval()
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
        expected = native(x) + (0.10 * candidate.theta.tanh()) * candidate.fuse(torch.cat(refinements, 1))
        formula = []
        common.assert_close_tree(expected, candidate(x), 0, 0, "independent_formula", formula)
        bounds = [float(0.10 * torch.tensor(t).tanh()) for t in (-100.0, 0.0, 100.0)]
        assert all(abs(a) <= float(torch.tensor(0.10)) for a in bounds)
    return {"identity": identity, "updates": updates, "formula": formula, "bounds": bounds, "amp": amp}


def build_pair(source=None):
    """Build native and DCS with aligned RNG, then apply the same native COCO-to-crack loading."""
    init_seeds(42, deterministic=True)
    with torch.random.fork_rng(devices=[]):
        baseline = DetectionModel(common.baseline_architecture(), verbose=False)
        baseline_rng = torch.get_rng_state()
    candidate = DetectionModel(str(common.MODEL), nc=1, verbose=False)
    assert torch.equal(baseline_rng, torch.get_rng_state()), "DCS consumed subsequent shared initialization RNG"
    for model in (baseline, candidate):
        model.names = {0: "crack"}
        if source is not None:
            model.load(source, verbose=False)
    return baseline, candidate


def topology_checks(baseline, candidate):
    """Require the complete b19 graph with exactly one layer-9 substitution."""
    cfg = copy.deepcopy(candidate.yaml)
    assert cfg["backbone"][9] == [-1, 1, "DCS_SPPF", [1024, 5, 3, True]]
    cfg["backbone"][9] = baseline.yaml["backbone"][9]
    assert common.architecture_signature(cfg) == common.architecture_signature(baseline.yaml)
    assert len(candidate.model) == len(baseline.model) == 24
    for i, (a, b) in enumerate(zip(baseline.model, candidate.model)):
        assert a.f == b.f and a.i == b.i
        if i != 9:
            assert str(a) == str(b)
    block, head = candidate.model[9], candidate.model[-1]
    assert type(block) is DCS_SPPF and type(candidate.model[10]) is C2PSA
    assert block.theta.ndim == 0 and block.theta.item() == 0
    assert candidate.model[21].f == [-1, 10] and head.f == [16, 19, 22]
    assert head.nc == 1 and head.reg_max == 1 and candidate.end2end
    assert candidate.stride.tolist() == [8, 16, 32]
    assert block.cv1.conv.in_channels == block.cv2.conv.out_channels == 256
    assert block.cv1.conv.out_channels == 128 and block.add
    for d, branch in enumerate(block.refine, 1):
        assert branch.conv.groups == 128 and branch.conv.kernel_size == (3, 3)
        assert branch.conv.dilation == branch.conv.padding == (d, d) and branch.conv.stride == (1, 1)
    return {"changed_layers": [9], "layer9": "DCS_SPPF", "layer10": "C2PSA", "detect_inputs": head.f}


def save_reload_check(model, directory, x):
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
    subprocess.run([sys.executable, str(Path(__file__).resolve()), "--reload", str(directory)], cwd=ROOT, check=True)
    return {
        "exact_state_tensors": len(model.state_dict()),
        "outputs": [r for r in rows if r["path"].startswith("reload")],
        "fresh_process": True,
    }


def reload_in_process(directory):
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


def observe_step(model, parameters, scaler, step):
    """Observe native scaler acceptance: overflows must skip all updates and lower scale; accepted gradients must be finite."""
    before = {k: p.detach().clone() for k, p in parameters.items()}
    all_before = [p.detach().clone() for p in model.parameters()]
    scale = scaler.get_scale()
    nonfinite = [
        name for name, p in model.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()
    ]
    gradients = {}
    for name, parameter in parameters.items():
        assert parameter.grad is not None, name
        gradients[name] = (
            parameter.grad.detach().float().abs().max().item() / scale if torch.isfinite(parameter.grad).all() else None
        )
    step()
    skipped = scaler.get_scale() < scale
    if skipped:
        assert nonfinite, "Scaler skipped despite finite gradients"
        assert all(torch.equal(a, b) for a, b in zip(all_before, model.parameters())), "Overflow changed parameters"
    else:
        assert not nonfinite, f"Accepted nonfinite gradients: {nonfinite}"
    assert all(torch.isfinite(p).all() for p in model.parameters())
    return {
        "parameters": {
            name: {
                "gradient_max_abs": gradients[name],
                "update_max_abs": (p.detach() - before[name]).abs().max().item(),
            }
            for name, p in parameters.items()
        },
        "scaler_before": scale,
        "scaler_after": scaler.get_scale(),
        "skipped": skipped,
        "nonfinite_scaled_gradient_keys": nonfinite,
    }


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
    for _ in range(64):
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

        row = observe_step(model, params, scaler, native_step)
        row["loss"] = loss.item()
        rows.append(row)
        observed.update(k for k, v in row["parameters"].items() if v["gradient_max_abs"] and v["update_max_abs"] > 0)
        if observed == set(params):
            break
    common.write_json(directory / f"synthetic_amp_{amp}.json", rows)
    assert observed == set(params), f"No gradient-backed update: {set(params) - observed}"
    return {"batch": 2, "imgsz": 640, "amp": amp, "optimizer": "MuSGD", "max_batches": 64, "steps": rows}


def model_checks(source, directory, device="cpu"):
    """Separate graph/weight audits from snapshot equality; never compare unrelated raw Detect outputs."""
    baseline, candidate = build_pair(source)
    report = {
        "topology": topology_checks(baseline, candidate),
        "weights": common.audit_weights(baseline, candidate, source),
    }
    assert report["weights"]["matched_tensors"] == common.REFERENCE["transferred_items"]
    assert len(YOLO(str(common.MODEL)).model.model) == 24
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
    report["save_reload"] = save_reload_check(candidate, directory / "snapshot", x)
    with torch.no_grad():
        candidate.model[9].theta.zero_()
    report["fp32"] = synthetic_model_updates(candidate, device, False, directory)
    if str(device).startswith("cuda"):
        report["amp"] = synthetic_model_updates(candidate, device, True, directory)
    return report


def native_preflight(config, directory):
    """Observe bounded native B32 AMP optimizer steps with the original b19 warmup and full dataset loader."""
    trainer = AuditedTrainer({**config, "project": str(directory / "native"), "name": "check"})
    trainer._setup_train()
    audit_training_setup(trainer)
    trainer._model_train()
    trainer.epoch = 0
    trainer.optimizer.zero_grad(set_to_none=True)
    parameters = new_parameters(trainer.model.model[9])
    nb = len(trainer.train_loader)
    nw = max(round(trainer.args.warmup_epochs * nb), 100)
    rows, observed = [], set()
    last_step = -1
    for i, batch in enumerate(trainer.train_loader):
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
            row = observe_step(trainer.model, parameters, trainer.scaler, trainer.optimizer_step)
            observed.update(
                name
                for name, value in row["parameters"].items()
                if value["gradient_max_abs"] and value["update_max_abs"] > 0
            )
            rows.append({"batch": i, "loss": loss.item(), **row})
            last_step = i
        common.write_json(directory / "native_steps.json", rows)
        if observed == set(parameters):
            break
        if i >= 63:
            raise AssertionError(
                f"No finite gradient-backed update within 64 native B32 batches: {set(parameters) - observed}"
            )
    assert observed == set(parameters)
    return {
        "batch": 32,
        "imgsz": 640,
        "amp": trainer.amp,
        "max_batches": 64,
        "batches_observed": i + 1,
        "updated_parameters": sorted(observed),
    }


def main():
    """Issue PASS only after all checks finish in the current independent process and evidence directory."""
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
    report = {"passed": False, "local_only": args.local, "commit": common.git("rev-parse", "HEAD")}
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
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        report["module_cpu"] = module_checks("cpu")
        if device.startswith("cuda"):
            report["module_cuda"] = module_checks(device)
            report["module_amp"] = module_checks(device, True)
        report.update(model_checks(source, args.output, device))
        if not args.local:
            report["native_preflight"] = native_preflight(effective, args.output)
        report["passed"] = True
    except Exception as error:
        report["error"] = str(error)
        raise
    finally:
        common.write_json(args.output / "checks.json", report)
    print(f"PASS: {'local validation' if args.local else 'server preflight'} -> {args.output / 'checks.json'}")


if __name__ == "__main__":
    main()
