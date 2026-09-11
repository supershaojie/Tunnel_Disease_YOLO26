"""Verify the single fixed SICR design using synthetic tensors, never a reduced-batch formal experiment."""

# ruff: noqa: E402 - Direct script entry must prioritize this worktree before importing Ultralytics.

import copy
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from tools.experiments import b19_common as common
from tools.experiments.run_b19_sicr_sppf import audit_arguments, options_parser, require_runtime
from ultralytics import YOLO
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.nn.modules import C2PSA, SICRSPPF, SPPF
from ultralytics.nn.tasks import DetectionModel, load_checkpoint
from ultralytics.utils.torch_utils import get_flops, init_seeds


def module_identity(native, candidate, x):
    """Locate the first unequal shared state/input/pooling stage using the actual module forwards."""
    rows = []
    assert candidate.theta.count_nonzero() == 0
    alpha = candidate.alpha_max * candidate.theta.tanh()
    common.assert_close_tree(torch.zeros_like(alpha), alpha, 0, 0, path="alpha", report=rows)
    for name in ("cv1", "cv2"):
        a, b = getattr(native, name), getattr(candidate, name)
        common.assert_close_tree(a.conv.bias, b.conv.bias, 0, 0, path=f"{name}.conv.bias", report=rows)
        common.assert_close_tree(a.state_dict(), b.state_dict(), 0, 0, path=name, report=rows)
        assert (a.bn.eps, a.bn.momentum) == (b.bn.eps, b.bn.momentum)
    traces = []
    for block in (native, candidate):
        trace = {"input": x.detach().clone()}
        stages = []
        handles = [
            block.cv1.register_forward_hook(lambda m, ins, out: stages.append(out.detach().clone())),
            block.m.register_forward_hook(lambda m, ins, out: stages.append(out.detach().clone())),
            block.cv2.register_forward_pre_hook(lambda m, ins: trace.update(cv2_input=ins[0].detach().clone())),
        ]
        try:
            with torch.no_grad():
                result = block(x)
        finally:
            for handle in handles:
                handle.remove()
        assert len(stages) == 4
        trace.update({f"Z{i}": z for i, z in enumerate(stages)})
        trace["output"] = result
        traces.append(trace)
    for key in ("input", "Z0", "Z1", "Z2", "Z3", "cv2_input", "output"):
        common.assert_close_tree(traces[0][key], traces[1][key], 0, 0, path=key, report=rows)
    return {
        "comparisons": rows,
        "first_difference": None,
        "conv_bias_present": {name: getattr(native, name).conv.bias is not None for name in ("cv1", "cv2")},
    }


def module_checks(device="cpu"):
    """Check independent formula, native identity, bounds and delayed task gradients in all three branches."""
    torch.manual_seed(42)
    native = SPPF(32, 32, 5, 3, True).to(device).eval()
    candidate = SICRSPPF(32, 32, 5, 3, True).to(device).eval()
    candidate.load_state_dict(native.state_dict(), strict=False)
    x = torch.randn(2, 32, 20, 27, device=device, requires_grad=True)
    identity = module_identity(native, candidate, x)
    expected, actual = native(x), candidate(x)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    error = (actual - expected).abs().max().item()
    target = torch.randn_like(actual)
    (actual - target).square().mean().backward()
    assert torch.isfinite(x.grad).all() and torch.isfinite(actual).all()
    assert candidate.theta.grad is not None and candidate.theta.grad.count_nonzero() == 3
    assert all(p.grad is not None and p.grad.count_nonzero() == 0 for p in candidate.refine.parameters())
    with torch.no_grad():
        candidate.theta.add_(candidate.theta.grad, alpha=-0.01)
    candidate.zero_grad(set_to_none=True)
    (candidate(x) - target).square().mean().backward()
    branch_gradients = []
    for branch in candidate.refine:
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in branch.parameters())
        norm = sum(p.grad.abs().sum().item() for p in branch.parameters())
        assert norm > 0
        branch_gradients.append(norm)
    with torch.no_grad():
        candidate.theta.copy_(torch.tensor([-100.0, 0.3, 100.0], device=device))
        alpha = 0.10 * candidate.theta.tanh()
        assert alpha.abs().max() <= 0.10
        z0 = candidate.cv1(x)
        z1, z2 = candidate.m(z0), candidate.m(candidate.m(z0))
        z3 = candidate.m(z2)
        result = (
            candidate.cv2(
                torch.cat(
                    (
                        z0,
                        z1 + alpha[0] * candidate.refine[0](z1 - z0),
                        z2 + alpha[1] * candidate.refine[1](z2 - z1),
                        z3 + alpha[2] * candidate.refine[2](z3 - z2),
                    ),
                    1,
                )
            )
            + x
        )
        torch.testing.assert_close(candidate(x), result, rtol=1e-5, atol=1e-6)
    return {
        "zero_init_audit": identity,
        "native_max_abs_error": error,
        "theta_gradient": True,
        "refine_gradient_after_theta_update": branch_gradients,
        "bounded_alpha": alpha.tolist(),
        "formula": "PASS",
        "forward_backward_finite": True,
    }


def build_pair(source=None):
    """Follow native single-class loading with identical RNG for all common parameters."""
    init_seeds(42, deterministic=True)
    with torch.random.fork_rng(devices=[]):
        baseline = DetectionModel(common.baseline_architecture(), verbose=False)
        baseline_rng = torch.get_rng_state()
    candidate = DetectionModel(str(common.MODEL), nc=1, verbose=False)
    assert torch.equal(baseline_rng, torch.get_rng_state()), "Constructor changed shared initialization RNG"
    for model in (baseline, candidate):
        model.names = {0: "crack"}
        if source is not None:
            model.load(source, verbose=False)
    return baseline, candidate


def topology_checks(baseline, candidate):
    """Require the native graph, nano scale and every non-layer-9 module to remain unchanged."""
    bcfg, ccfg = copy.deepcopy(baseline.yaml), copy.deepcopy(candidate.yaml)
    assert ccfg["backbone"][9] == [-1, 1, "SICRSPPF", [1024, 5, 3, True, 0.10]]
    ccfg["backbone"][9] = bcfg["backbone"][9]
    assert common.architecture_signature(bcfg) == common.architecture_signature(ccfg)
    for i, (a, b) in enumerate(zip(baseline.model, candidate.model)):
        assert a.f == b.f
        if i != 9:
            assert str(a) == str(b)
    block, head = candidate.model[9], candidate.model[-1]
    assert type(block) is SICRSPPF and type(candidate.model[10]) is C2PSA
    assert candidate.model[21].f == [-1, 10] and head.f == [16, 19, 22]
    assert candidate.yaml["scale"] == "n" and candidate.yaml["nc"] == 1
    assert head.nc == 1 and head.reg_max == 1 and candidate.end2end
    assert candidate.stride.tolist() == [8, 16, 32]
    assert block.cv1.conv.in_channels == block.cv2.conv.out_channels == 256
    assert block.cv1.conv.out_channels == 128 and block.add
    return {
        "scale": "n",
        "layer9": "SICRSPPF",
        "layer10": "C2PSA",
        "detect_inputs": head.f,
        "strides": candidate.stride.tolist(),
        "channels_in_out": [256, 256],
        "nc": 1,
        "end2end": True,
        "reg_max": 1,
    }


def model_checks(source, device="cpu", batch=1, directory=None):
    """Audit 640 forward/backward, pretrained coverage, complexity and checkpoint/fusion compatibility."""
    baseline, candidate = build_pair(source)
    report = {
        "topology": topology_checks(baseline, candidate),
        "weights": common.audit_weights(baseline, candidate, source),
    }
    assert report["weights"]["matched_tensors"] == common.REFERENCE["transferred_items"]
    assert len(YOLO(str(common.MODEL)).model.model) == 24
    baseline.eval()
    candidate.eval()
    counts = [sum(p.numel() for p in model.parameters()) for model in (baseline, candidate)]
    flops = [get_flops(model, imgsz=640) for model in (baseline, candidate)]
    assert min(flops) > 0, "Install the project's ultralytics-thop dependency for the GFLOPs audit"
    report["complexity"] = {
        "parameters": counts,
        "gflops": flops,
        "params_delta_percent": 100 * (counts[1] / counts[0] - 1),
        "gflops_delta_percent": 100 * (flops[1] / flops[0] - 1),
    }
    assert report["complexity"]["params_delta_percent"] < 3 and report["complexity"]["gflops_delta_percent"] < 5
    baseline, candidate = baseline.to(device), candidate.to(device)
    layer9_inputs = []
    hooks = [
        m.model[9].register_forward_pre_hook(lambda m, args, values=layer9_inputs: values.append(args[0].clone()))
        for m in (baseline, candidate)
    ]
    x = torch.randn(1, 3, 640, 640, device=device)
    with torch.no_grad():
        native_output, candidate_output = baseline(x), candidate(x)
    for hook in hooks:
        hook.remove()
    common.assert_close_tree(*layer9_inputs, 0, 0, path="layer9.input")
    assert list(layer9_inputs[0].shape) == [1, 256, 20, 20]
    report["module_at_layer9"] = module_identity(baseline.model[9], candidate.model[9], layer9_inputs[0])
    errors = []
    common.assert_close_tree(native_output, candidate_output, atol=1e-6, rtol=1e-5, report=errors)
    report["topology"]["layer9_shapes_640"] = [[1, 256, 20, 20]] * 2
    report["weights"]["constructor_rng_equal"] = True
    report["initial_equivalence"] = errors
    del baseline, native_output, candidate_output, layer9_inputs
    candidate.train()
    candidate.args = common.get_cfg(overrides={k: v for k, v in common.REFERENCE["args"].items() if k != "save_dir"})
    inputs = {
        "img": torch.rand(batch, 3, 640, 640, device=device),
        "batch_idx": torch.arange(batch, device=device),
        "cls": torch.zeros(batch, 1, device=device),
        "bboxes": torch.tensor([[0.5, 0.5, 0.1, 0.4]], device=device).repeat(batch, 1),
    }
    loss, _ = candidate(inputs)
    loss.sum().backward()
    assert torch.isfinite(loss).all() and torch.isfinite(candidate.model[9].theta.grad).all()
    assert candidate.model[9].theta.grad.count_nonzero() == 3
    if device.startswith("cuda"):
        candidate.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda"):
            amp_loss, _ = candidate(inputs)
        amp_loss.sum().backward()
        assert torch.isfinite(amp_loss).all() and torch.isfinite(candidate.model[9].theta.grad).all()
        report["amp_backward"] = True
    report["synthetic_detection_backward"] = {"batch": batch, "imgsz": 640, "device": device, "finite": True}
    candidate.eval().zero_grad(set_to_none=True)
    with torch.no_grad():
        candidate.model[9].theta.copy_(torch.tensor([0.1, -0.2, 0.3], device=device))
    assert directory is not None, "A persistent checkpoint audit directory is required"
    report["reload"] = save_reload_check(candidate, directory, x)
    with torch.no_grad():
        # Match native evaluation: load/fuse the checkpoint on CPU before moving it to the inference device.
        restored, _ = load_checkpoint(report["reload"]["checkpoint"])
        fused = copy.deepcopy(restored).fuse(verbose=False).to(device)
        backend = AutoBackend(model=restored, device=torch.device(device), fp16=False, verbose=False)
        deployed = backend(x)
        deployment_errors = []
        common.assert_close_tree(list(fused(x)), deployed, 1e-6, 1e-5, report=deployment_errors)
        assert deployed[1]["one2many"] == {} and deployed[1]["one2one"]["boxes"].shape == (1, 4, 8400)
        backend.model.model[9].theta.zero_()
        bypassed = backend(x)
        effect = {
            key: (deployed[1]["one2one"][key] - bypassed[1]["one2one"][key]).abs().max().item()
            for key in ("boxes", "scores")
        }
        assert max(effect.values()) > 0, "Nonzero SICR must affect deployed one2one inference"
        report["deployment"] = {
            "device": device,
            "cpu_fusion": True,
            "comparisons": deployment_errors,
            "nonzero_alpha_effect": effect,
        }
    return report


def save_reload_check(model, directory, x):
    """Reuse NDP's FP16 EMA snapshot/FP32 reference lifecycle, preserving the exact same model and input."""
    directory = Path(directory) / "reload"
    directory.mkdir(parents=True, exist_ok=True)
    snapshot = copy.deepcopy(model).cpu().half().eval()
    snapshot.criterion = None
    assert snapshot.model[9].theta.count_nonzero() == 3
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
            "from tools.experiments.verify_b19_sicr_sppf import reload_in_process; "
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
        assert type(model.model[9]) is SICRSPPF and model.model[9].theta.count_nonzero() == 3
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
            report["sicr_fusion"] = fusion_check(model, reference["x"])
        report["passed"] = True
    finally:
        common.write_json(path.with_name("reload_check.json"), report)
    print("PASS: exact snapshot state/raw/Detect caches; native and SICR CPU one2one fusion")


def main():
    """Record local structural checks separately from the full-recipe server preflight."""
    parser = options_parser()
    parser.add_argument(
        "--local", action="store_true", help="CPU synthetic B1/640 audit; not a formal training receipt"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    if not args.local:
        common.require_clean_source()
        require_runtime()
    raw, effective, evidence = common.resolve_recipe(args)
    audit_arguments(raw, effective)
    evidence["launcher"] = common.launcher_evidence(args, raw)
    source, _ = load_checkpoint(evidence["initial_path"])
    device = "cpu" if args.local else "cuda:0"
    report = {
        "module": module_checks(device),
        **model_checks(source, device, 1 if args.local else 32, args.output),
        "recipe": evidence,
        "local_only": args.local,
        "passed": True,
        "commit": common.git("rev-parse", "HEAD"),
    }
    common.write_json(args.output / "checks.json", report)
    print(report["complexity"])
    print(f"PASS: {'local structural audit' if args.local else 'server preflight'} -> {args.output / 'checks.json'}")


if __name__ == "__main__":
    main()
