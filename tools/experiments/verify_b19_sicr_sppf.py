"""Verify the single fixed SICR design using synthetic tensors, never a reduced-batch formal experiment."""

# ruff: noqa: E402 - Direct script entry must prioritize this worktree before importing Ultralytics.

import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from tools.experiments import b19_common as common
from tools.experiments.run_b19_sicr_sppf import audit_arguments, options_parser, require_runtime
from ultralytics import YOLO
from ultralytics.nn.modules import C2PSA, SICRSPPF, SPPF
from ultralytics.nn.tasks import DetectionModel, load_checkpoint
from ultralytics.utils.torch_utils import get_flops, init_seeds


def module_checks(device="cpu"):
    """Check independent formula, native identity, bounds and delayed task gradients in all three branches."""
    torch.manual_seed(42)
    native = SPPF(32, 32, 5, 3, True).to(device).eval()
    candidate = SICRSPPF(32, 32, 5, 3, True).to(device).eval()
    candidate.load_state_dict(native.state_dict(), strict=False)
    x = torch.randn(2, 32, 20, 27, device=device, requires_grad=True)
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
    candidate = DetectionModel(str(common.MODEL), nc=1, verbose=False)
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


def model_checks(source, device="cpu", batch=1):
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
    shapes = []
    hook = candidate.model[9].register_forward_hook(
        lambda m, args, out: shapes.append([list(args[0].shape), list(out.shape)])
    )
    x = torch.randn(1, 3, 640, 640, device=device)
    with torch.no_grad():
        errors = []
        common.assert_close_tree(baseline(x), candidate(x), atol=1e-6, rtol=1e-5, report=errors)
    hook.remove()
    assert shapes == [[[1, 256, 20, 20], [1, 256, 20, 20]]]
    report["topology"]["layer9_shapes_640"] = shapes[0]
    report["initial_equivalence"] = errors
    del baseline
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
        before = candidate(x)
        fused = copy.deepcopy(candidate).fuse(verbose=False)
        after = fused(x)
        raw_errors = []
        common.assert_close_tree(before[1]["one2one"], after[1]["one2one"], atol=1e-4, rtol=1e-4, report=raw_errors)
        # Decoding multiplies raw distances by stride (up to 32) and subtracts anchors.
        # Scale the raw 1e-4 absolute tolerance by the largest output stride, in pixel units.
        torch.testing.assert_close(after[0][..., :4], before[0][..., :4], rtol=1e-4, atol=32e-4)
        torch.testing.assert_close(after[0][..., 4:], before[0][..., 4:], rtol=1e-4, atol=1e-6)
        report["fusion"] = {
            "raw_one2one_errors": raw_errors,
            "coordinate_max_abs_pixels": (after[0][..., :4] - before[0][..., :4]).abs().max().item(),
        }
    report["fused_nonzero_alpha"] = True
    return report


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
        **model_checks(source, device, 1 if args.local else 32),
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
