"""Independent FDV zero-init, shared-weight, update and checkpoint audits; never start formal training."""

# Direct entry must resolve this worktree before package imports.
# ruff: noqa: E402

import copy
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from tools.experiments import b19_common as common
from tools.experiments.run_b19_fdv_c2psa import AuditedTrainer, audit_arguments, options_parser, require_runtime
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.modules import C2PSA, FDV_C2PSA, SPPF, FDVAttention
from ultralytics.nn.modules.block import Attention, PSABlock
from ultralytics.nn.tasks import DetectionModel, load_checkpoint
from ultralytics.utils.torch_utils import get_flops, init_seeds


def module_checks(device="cpu"):
    """Compare identical inputs/weights at Attention and C2PSA, including odd and rectangular grids."""
    reports = []
    for h, w in ((20, 20), (20, 30), (7, 9), (1, 1)):
        for kind in ("Attention", "C2PSA"):
            native = Attention(128, 2) if kind == "Attention" else C2PSA(256, 256)
            candidate = FDVAttention(copy.deepcopy(native)) if kind == "Attention" else FDV_C2PSA(256, 256)
            result = candidate.load_state_dict(native.state_dict(), strict=False)
            expected = "theta_c" if kind == "Attention" else "m.0.attn.theta_c"
            assert result.missing_keys == [expected] and not result.unexpected_keys
            native, candidate = native.to(device).eval(), candidate.to(device).eval()
            x = torch.randn(2, 128 if kind == "Attention" else 256, h, w, device=device)
            errors = []
            with torch.no_grad():
                common.assert_close_tree(native(x), candidate(x), 0, 0, report=errors)
            reports.append(dict(module=kind, grid=[h, w], allclose=True, errors=errors))
    return reports


def build_pair(source=None):
    """Align constructor RNG and native nc=1 loading, auditing every shared tensor before any forward comparison."""
    init_seeds(42, deterministic=True)
    with torch.random.fork_rng(devices=[]):
        baseline = DetectionModel(common.baseline_architecture(), verbose=False)
        expected_rng = torch.get_rng_state()
    candidate = DetectionModel(str(common.MODEL), nc=1, verbose=False)
    assert torch.equal(expected_rng, torch.get_rng_state())
    for model in (baseline, candidate):
        model.names = {0: "crack"}
        if source is not None:
            model.load(source, verbose=False)
    return baseline, candidate


def topology_checks(baseline, candidate):
    """Require layer 10 Attention to be the sole semantic graph change, including all Detect paths."""
    bcfg, ccfg = copy.deepcopy(baseline.yaml), copy.deepcopy(candidate.yaml)
    assert ccfg["backbone"][10] == [-1, 2, "FDV_C2PSA", [1024]]
    ccfg["backbone"][10] = bcfg["backbone"][10]
    assert common.architecture_signature(bcfg) == common.architecture_signature(ccfg)
    assert len(baseline.model) == len(candidate.model) == 24
    for i, (a, b) in enumerate(zip(baseline.model, candidate.model)):
        assert a.f == b.f
        if i != 10:
            assert str(a) == str(b)
    block, head = candidate.model[10], candidate.model[-1]
    assert type(block) is FDV_C2PSA and type(candidate.model[9]) is SPPF
    assert FDV_C2PSA.forward is C2PSA.forward and len(block.m) == 1
    for a, b in zip(baseline.model[10].m, block.m):
        assert type(b) is PSABlock and b.add == a.add and str(a.ffn) == str(b.ffn)
        assert type(b.attn) is FDVAttention and b.attn.theta_c.shape == (128,)
    assert candidate.model[21].f == [-1, 10] and head.f == [16, 19, 22]
    assert candidate.yaml["scale"] == "n" and candidate.yaml["nc"] == head.nc == 1
    assert head.reg_max == 1 and candidate.end2end and candidate.stride.tolist() == [8, 16, 32]
    return dict(
        only_changed_layer=10,
        layer9="SPPF",
        layer10="FDV_C2PSA",
        detect_inputs=head.f,
        strides=candidate.stride.tolist(),
        hidden_channels=128,
        new_parameters=128,
    )


def optimizer_for(model):
    """Use b19's native MuSGD grouping and coefficients for isolated update checks."""
    return DetectionTrainer.build_optimizer(object.__new__(DetectionTrainer), model, "MuSGD", 0.01, 0.937, 0.0005)


def update_check(model, device, amp=False, batch=1):
    """Exercise real detection loss and native optimizer; GradScaler skips do not count as parameter updates."""
    model = model.to(device).train()
    model.args = common.get_cfg(overrides={k: v for k, v in common.REFERENCE["args"].items() if k != "save_dir"})
    inputs = dict(
        img=torch.rand(batch, 3, 640, 640, device=device),
        batch_idx=torch.arange(batch, device=device),
        cls=torch.zeros(batch, 1, device=device),
        bboxes=torch.tensor([[0.5, 0.5, 0.1, 0.4]], device=device).repeat(batch, 1),
    )
    parameters = {
        k: p
        for k, p in model.named_parameters()
        if k
        in (
            "model.10.m.0.attn.theta_c",
            "model.10.m.0.attn.qkv.conv.weight",
            "model.10.m.0.attn.proj.conv.weight",
            "model.10.m.0.attn.pe.conv.weight",
            "model.10.m.0.ffn.0.conv.weight",
            "model.10.m.0.ffn.1.conv.weight",
        )
    }
    assert len(parameters) == 6
    optimizer = optimizer_for(model)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    updated, gradients, attempts = set(), {}, []
    for step in range(16):
        optimizer.zero_grad(set_to_none=True)
        before = {k: p.detach().clone() for k, p in parameters.items()}
        with torch.autocast(device_type=torch.device(device).type, enabled=amp):
            loss, _ = model(inputs)
            loss = loss.sum()
        assert torch.isfinite(loss)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        finite = all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        scale_before = scaler.get_scale()
        if finite:
            for k, p in parameters.items():
                assert p.grad is not None
                gradients[k] = p.grad.float().norm().item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        elif not amp:
            raise AssertionError("Nonfinite FP32 gradients")
        scaler.step(optimizer)
        scaler.update()
        if not finite:
            assert scaler.get_scale() < scale_before
            assert all(torch.equal(before[k], p) for k, p in parameters.items())
        else:
            updated.update(k for k, p in parameters.items() if gradients[k] > 0 and not torch.equal(before[k], p))
        assert all(torch.isfinite(v).all() for v in model.state_dict().values())
        attempts.append(
            dict(
                attempt=step + 1,
                loss=float(loss.detach()),
                finite_gradients=bool(finite),
                scale_before=scale_before,
                scale_after=scaler.get_scale(),
                updated=sorted(updated),
            )
        )
        if len(updated) == len(parameters):
            break
    assert set(parameters) == updated, f"No effective data-gradient update: {set(parameters) - updated}"
    return dict(
        device=device,
        amp=amp,
        batch=batch,
        imgsz=640,
        gradients=gradients,
        attempts=attempts,
        updated=sorted(updated),
        finite_parameters=True,
    )


def model_checks(source, device="cpu", directory=None):
    """Audit full shared weights, model construction, isolated updates and the same saved FDV checkpoint."""
    baseline, candidate = build_pair(source)
    weights = common.audit_weights(baseline, candidate, source)
    assert weights["matched_tensors"] == 606 and weights["common_keys"] == 708
    topology = topology_checks(baseline, candidate)
    assert len(YOLO(str(common.MODEL)).model.model) == 24
    counts = [sum(p.numel() for p in m.parameters()) for m in (baseline, candidate)]
    flops = [get_flops(m.eval(), 640) for m in (baseline, candidate)]
    assert min(flops) > 0 and counts == [2504190, 2504318]
    # THOP omits functional matmuls/pointwise work. Report its convention and the explicit FDV branch budget.
    complexity = dict(
        parameters=counts,
        thop_gflops=flops,
        fdv_extra_scalar_ops_640=128 * 20 * 20 * (8 + 1 + 1 + 1 + 1) + 128 * 2,
        convention="THOP estimate; branch budget counts 8 adds + divide for pool, subtract, multiply, add; "
        "tanh counted as one scalar op; not a hardware FLOP guarantee",
    )
    del baseline
    fp32 = update_check(copy.deepcopy(candidate), device)
    amp = update_check(copy.deepcopy(candidate), device, True) if device.startswith("cuda") else None
    candidate.eval()
    with torch.no_grad():
        candidate.model[10].m[0].attn.theta_c.copy_(torch.linspace(-0.3, 0.3, 128))
    candidate.args = common.get_cfg()
    reload = save_reload_check(candidate, directory, torch.randn(1, 3, 640, 640))
    return dict(topology=topology, weights=weights, complexity=complexity, fp32=fp32, amp=amp, reload=reload)


class PreflightComplete(Exception):
    """Terminate only the disposable preflight at the native batch callback."""


def native_preflight(config, directory):
    """Observe fixed b19 real batches through the unmodified native AMP/warmup/clip/step/EMA lifecycle."""
    trainer = AuditedTrainer(dict(config, project=str(directory), name="native_preflight"))
    report = dict(passed=False, batches=[], steps=0, updates=[], max_batches=32)
    handles, state, updates = [], {}, set()

    def setup(t):
        from tools.experiments.run_b19_fdv_c2psa import audit_training_setup

        audit_training_setup(t)
        assert len(t.train_loader.dataset) == 8414 and len(t.test_loader.dataset) == 2404
        assert t.amp and t.scaler.is_enabled() and t.ema.updates == 0
        state["parameters"] = {
            k: p
            for k, p in t.model.model[10].m[0].attn.named_parameters()
            if k in {"theta_c", "qkv.conv.weight", "proj.conv.weight", "pe.conv.weight"}
        }
        assert len(state["parameters"]) == 4

        def before_step(optimizer, args, kwargs):
            state["before"] = {k: p.detach().clone() for k, p in state["parameters"].items()}
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in t.model.parameters())
            state["gradients"] = {k: float(p.grad.float().norm()) for k, p in state["parameters"].items()}

        def after_step(optimizer, args, kwargs):
            report["steps"] += 1
            updates.update(
                k
                for k, p in state["parameters"].items()
                if state["gradients"][k] > 0 and not torch.equal(state["before"][k], p)
            )

        def observe(module, inputs):
            assert inputs[0]["img"].shape == (32, 3, 640, 640)

        handles.extend(
            [
                t.optimizer.register_step_pre_hook(before_step),
                t.optimizer.register_step_post_hook(after_step),
                t.model.register_forward_pre_hook(observe),
            ]
        )

    def batch_end(t):
        assert t.batch_size == 32 and torch.isfinite(t.loss)
        assert all(torch.isfinite(v).all() for v in t.model.state_dict().values())
        report["batches"].append(
            dict(
                loss=float(t.loss),
                scale=t.scaler.get_scale(),
                steps=report["steps"],
                gradients=state.get("gradients"),
                ema_updates=t.ema.updates,
            )
        )
        common.write_json(directory / "native_steps.json", report)
        if (report["steps"] >= 2 and len(updates) == 4) or len(report["batches"]) >= report["max_batches"]:
            raise PreflightComplete

    trainer.add_callback("on_pretrain_routine_end", setup)
    trainer.add_callback("on_train_batch_end", batch_end)
    try:
        try:
            trainer.train()
        except PreflightComplete:
            pass
        assert report["steps"] >= 2 and len(updates) == 4
        assert trainer.ema.ema.model[10].m[0].attn.theta_c.count_nonzero() > 0
        report.update(passed=True, updates=sorted(updates), ema_updates=trainer.ema.updates)
    finally:
        for h in handles:
            h.remove()
        for name in ("train_loader", "test_loader"):
            loader = getattr(trainer, name, None)
            if loader is not None:
                loader.close()
        common.write_json(directory / "native_steps.json", report)
    return report


def main():
    """Separate local evidence from an independent server PASS that includes real batch32 native updates."""
    parser = options_parser()
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--local-device", choices=["cpu", "cuda:0"], default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    report = dict(passed=False, local_only=args.local, commit=common.git("rev-parse", "HEAD"))
    common.write_json(args.output / "checks.json", report)
    try:
        if not args.local:
            common.require_clean_source()
            require_runtime()
        raw, config, evidence = common.resolve_recipe(args)
        audit_arguments(raw, config)
        evidence["launcher"] = common.launcher_evidence(args, raw)
        source, _ = load_checkpoint(evidence["initial_path"])
        device = args.local_device if args.local else "cuda:0"
        report.update(module=module_checks(device), **model_checks(source, device, args.output), recipe=evidence)
        if not args.local:
            report["native_preflight"] = native_preflight(config, args.output)
        report["passed"] = True
    finally:
        common.write_json(args.output / "checks.json", report)
    print(f"PASS: {'local checks' if args.local else 'independent server preflight'} -> {args.output / 'checks.json'}")


def save_reload_check(model, directory, x):
    """Reuse NDP's FP16 EMA snapshot/FP32 reference lifecycle, preserving the exact same model and input."""
    directory = Path(directory) / "reload"
    directory.mkdir(parents=True, exist_ok=True)
    snapshot = copy.deepcopy(model).cpu().half().eval()
    snapshot.criterion = None
    assert snapshot.model[10].m[0].attn.theta_c.count_nonzero() == 128
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
        common.assert_close_tree(reference["state"], snapshot.state_dict(), 0, 0, path="snapshot_after_forward")
        reference["attributes_after"] = common.inference_attributes(snapshot)
        torch.save(reference, directory / "reload_reference.pt")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from tools.experiments.verify_b19_fdv_c2psa import reload_in_process; "
            "import sys; reload_in_process(sys.argv[1])",
            str(path),
        ],
        cwd=common.ROOT,
        check=False,
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
        assert type(model.model[10]) is FDV_C2PSA and model.model[10].m[0].attn.theta_c.count_nonzero() == 128
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
            report["fdv_fusion"] = fusion_check(model, reference["x"])
        report["passed"] = True
    finally:
        common.write_json(path.with_name("reload_check.json"), report)
    print("PASS: exact snapshot state/raw/Detect caches; native and FDV CPU one2one fusion")


if __name__ == "__main__":
    main()
