"""Audit the fixed residual controller and reuse all verified native b19 preflight machinery."""

# ruff: noqa: E402 -- Direct entry must import this worktree.
import copy
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from tools.experiments import b19_common as common
from tools.experiments import verify_b19_dcs_sppf as shared
from tools.experiments.b19_detect_fuse_audit import fuse_precision_checks
from tools.experiments.run_b19_dcs_sppf_v2 import MODEL, NAME, AuditedTrainer
from ultralytics.nn.modules import DCS_SPPF_V2, SPPF
from ultralytics.nn.modules.dcs_sppf_v2 import relative_residual
from ultralytics.utils.torch_utils import ModelEMA
from ultralytics.models.yolo.detect import DetectionTrainer


def module_checks(device="cpu", amp=False):
    """Keep v1 identity/gradient checks while applying the v2 residual arithmetic to the independent pool formula."""
    return shared.module_checks(
        device, amp, DCS_SPPF_V2, lambda raw, y0: relative_residual(raw.float(), y0.float())[0].to(y0.dtype)
    )


def controller_checks(device="cpu"):
    """Record every numerical boundary, gradients, per-image bound, precision and observable rounding error."""
    torch.manual_seed(42)
    rows = []
    dtypes = [torch.float32, torch.float16, torch.bfloat16]
    # These are tensor/controller checks; CPU BF16 convolutions are checked separately below.
    for dtype in dtypes:
        for height, width in ((1, 1), (4, 7), (20, 20), (20, 27)):
            for reference_scale, raw_scale in ((1, 0), (0, 0), (0, 10), (1e-12, 1), (1, 1e-8), (1, -1), (1, 1e4)):
                reference = (torch.randn(3, 256, height, width, device=device) * reference_scale).to(dtype)
                reference.requires_grad_()
                raw = (torch.randn_like(reference.float()) * raw_scale).requires_grad_()
                with torch.autocast(device_type=torch.device(device).type, enabled=False):
                    injected, q = relative_residual(raw, reference.float())
                output = reference + injected.to(dtype)
                grad_raw, grad_reference = torch.autograd.grad(output.float().sum(), (raw, reference))
                assert torch.equal(grad_reference, torch.ones_like(reference)), "Budget leaked a reference gradient"
                assert torch.isfinite(grad_raw).all() and torch.isfinite(output).all()
                assert injected.dtype == q.dtype == torch.float32 and q.shape == (3, 1, 1, 1)
                assert ((q >= 0) & (q <= 1)).all()

                def rms(value):
                    return value.double().square().mean((1, 2, 3)).sqrt()

                denominator = rms(reference)
                observed = output.float() - reference.float()
                rounding = observed - injected
                ratios = rms(injected) / denominator.clamp_min(1e-30)
                observed_ratios = rms(observed) / denominator.clamp_min(1e-30)
                rounding_ratios = rms(rounding) / denominator.clamp_min(1e-30)
                assert (ratios <= 0.05 + 2e-8).all(), ratios
                assert (observed_ratios <= ratios + rounding_ratios + 1e-12).all()
                assert (injected[denominator == 0] == 0).all()
                b = 0.05 * rms(reference).reshape(-1, 1, 1, 1)
                expected = (
                    raw.double() * b / (b.square() + raw.double().square().mean((1, 2, 3), keepdim=True) + 1e-12).sqrt()
                )
                torch.testing.assert_close(injected.double(), expected, rtol=3e-6, atol=1e-7)
                torch.testing.assert_close(injected, q * raw, rtol=0, atol=0)
                rows.append(
                    {
                        "device": device,
                        "dtype": str(dtype),
                        "shape": list(reference.shape),
                        "reference_scale": reference_scale,
                        "raw_scale": raw_scale,
                        "max_abs": injected.abs().max().item(),
                        "reference_error_max_abs": (injected.double() - expected).abs().max().item(),
                        "injected_ratio": ratios.tolist(),
                        "observed_ratio": observed_ratios.tolist(),
                        "rounding_ratio": rounding_ratios.tolist(),
                        "q": q.flatten().tolist(),
                        "finite": True,
                        "gradient_max_abs": grad_raw.abs().max().item(),
                    }
                )
    raw = torch.randn(3, 256, 4, 7, device=device)
    reference = torch.randn_like(raw)
    expected = relative_residual(raw, reference)[0]
    raw[1:] *= 1000
    reference[1:] *= 1e-8
    actual = relative_residual(raw, reference)[0]
    torch.testing.assert_close(expected[0], actual[0], rtol=0, atol=0)
    fixed = torch.randn(2, 3, 4, 5, dtype=torch.float64)
    variable = torch.randn_like(fixed, requires_grad=True)
    assert torch.autograd.gradcheck(lambda r: relative_residual(r, fixed)[0], (variable,))
    ratios = []
    fixed = torch.ones(2, 256, 4, 7, device=device)
    for magnitude in (1e-6, 0.01, 1, 1000):
        injection, _ = relative_residual(fixed * magnitude, fixed)
        ratios.append(injection.double().square().mean().sqrt().item())
    assert all(a < b for a, b in zip(ratios, ratios[1:])) and ratios[-1] <= 0.05 + 2e-8
    return {
        "cases": rows,
        "same_shape_other_samples_independent": True,
        "raw_fp64_gradcheck": True,
        "amplification_ratios": ratios,
        "bound_tolerance_fp32": 2e-8,
        "rounding_policy": "Observed norm is checked against FP32 injection plus measured cast/add rounding norm",
    }


def state_checks(source, directory, device):
    """Check actual train BN updates, AMP dtypes, EMA, nonzero snapshot binding and Conv-BN fusion."""
    report = {"passed": False, "module_states": [], "fuse": {}}
    try:
        _state_checks(source, directory, device, report)
        report["passed"] = True
    except BaseException:
        report["traceback"] = traceback.format_exc()
        raise
    finally:
        common.write_json(directory / "state_checks.json", report)
    return report


def updated_fixture(candidate, device):
    """Produce nonzero theta by one real native-loss/MuSGD diagnostic update; never used to initialize training."""
    model = copy.deepcopy(candidate).to(device).train()
    model.args = SimpleNamespace(**common.REFERENCE["args"])
    optimizer = DetectionTrainer.build_optimizer(None, model, "MuSGD", 0.01, 0.937, 0.0005)
    batch = dict(
        img=torch.rand(2, 3, 128, 160, device=device),
        batch_idx=torch.arange(2, device=device),
        cls=torch.zeros(2, 1, device=device),
        bboxes=torch.tensor([[0.5, 0.5, 0.08, 0.6], [0.3, 0.6, 0.6, 0.08]], device=device),
    )
    theta_before = model.model[9].theta.item()
    loss, _ = model(batch)
    loss = loss.sum()
    assert torch.isfinite(loss)
    loss.backward()
    gradient = model.model[9].theta.grad.item()
    assert gradient != 0 and torch.isfinite(model.model[9].theta.grad)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    assert model.model[9].theta.item() != theta_before
    return model.eval(), dict(
        loss=loss.item(),
        theta_before=theta_before,
        gradient=gradient,
        theta_after=model.model[9].theta.item(),
        optimizer="MuSGD",
        batch=2,
        shape=[128, 160],
        scope="One diagnostic update, separate from native B32 and existing staged gamma audits",
    )


def _state_checks(source, directory, device, report):
    """Accumulate lifecycle results in the receipt owner before any failing assertion can discard them."""
    rows = report["module_states"]
    modes = [("cpu", False, torch.bfloat16), (device, False, torch.float16)]
    modes += [("cpu", True, torch.bfloat16)]
    if str(device).startswith("cuda"):
        modes += [(device, True, torch.float16)]
    for location, amp, dtype in modes:
        for training in (False, True):
            native = SPPF(256, 256, shortcut=True).to(location).train(training)
            block = DCS_SPPF_V2(256, 256, shortcut=True).to(location).train(training)
            block.load_state_dict(native.state_dict(), strict=False)
            x = torch.randn(2, 256, 4, 7, device=location)
            captured = {}
            handle = block.fuse.register_forward_hook(lambda m, a, o: captured.update(features=str(o.dtype)))
            try:
                with torch.autocast(device_type=torch.device(location).type, enabled=amp, dtype=dtype):
                    errors = shared.module_identity(native, block, x)
            finally:
                handle.remove()
            bn = {}
            for name in ("cv1", "cv2"):
                a, b = getattr(native, name).bn, getattr(block, name).bn
                for field in ("running_mean", "running_var", "num_batches_tracked"):
                    assert torch.equal(getattr(a, field), getattr(b, field))
                assert b.num_batches_tracked.item() == int(training)
                bn[name] = {"updates": b.num_batches_tracked.item(), "state_equal": True}
            if amp:
                assert captured["features"] == str(dtype)
            rows.append({"device": location, "amp": amp, "training": training, "bn": bn, **captured, "errors": errors})
    baseline, v1 = shared.build_pair(source)
    _, candidate = shared.build_pair(source, MODEL)
    candidate = candidate.to(device).eval()
    with shared.check_phase("actual MuSGD state fixture B2/128x160"):
        updated, report["optimizer_fixture"] = updated_fixture(candidate, device)
    x = torch.randn(1, 3, 128, 160, device=device)
    diagnostic = copy.deepcopy(candidate)
    with torch.no_grad():
        diagnostic.model[9].theta.fill_(-0.7)
    fuse_precision_checks(
        dict(native=baseline, v1=v1, v2_zero=candidate, v2_updated=updated, v2_diagnostic=diagnostic),
        x,
        directory / "fuse_precision",
        report["fuse"],
    )
    candidate = updated
    binding = common.model_binding(MODEL, DCS_SPPF_V2, candidate.model[9])
    ema = ModelEMA(candidate)
    ema.update(candidate)
    assert common.model_binding(MODEL, DCS_SPPF_V2, ema.ema.model[9]) == binding
    assert torch.equal(ema.ema.model[9].theta, candidate.model[9].theta)
    fused = copy.deepcopy(candidate).fuse(verbose=False)
    assert common.model_binding(MODEL, DCS_SPPF_V2, fused.model[9]) == binding
    assert torch.equal(candidate.model[9].theta, fused.model[9].theta)
    with shared.check_phase("fused nonzero snapshot and fresh-process reload"):
        reload = shared.save_reload_check(fused, directory / "fused_snapshot", x, MODEL)
    report.update(binding=binding, ema=True, fused_reload=reload)


def benchmark(source, device):
    """Time sequential fused FP32 inference; report controller arithmetic separately from THOP."""
    baseline, v1 = shared.build_pair(source)
    _, v2 = shared.build_pair(source, MODEL)
    assert sum(p.numel() for p in v1.parameters()) == sum(p.numel() for p in v2.parameters()) == 2607231
    rows = []
    x = torch.randn(1, 3, 640, 640, device=device)
    for name, model in (("b19", baseline), ("DCS_v1", v1), ("DCS_v2", v2)):
        model = model.to(device).eval().fuse(verbose=False)
        with torch.no_grad():
            for _ in range(5):
                model(x)
            if str(device).startswith("cuda"):
                torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(20):
                model(x)
            if str(device).startswith("cuda"):
                torch.cuda.synchronize()
        rows.append({"model": name, "milliseconds": (time.perf_counter() - start) * 1000 / 20})
        model.cpu()
    return {
        "timings": rows,
        "device": device,
        "hardware": torch.cuda.get_device_name(0) if str(device).startswith("cuda") else "CPU",
        "dtype": "FP32",
        "fused": True,
        "batch": 1,
        "imgsz": 640,
        "warmup": 5,
        "repeats": 20,
        "concurrency": "Sequential within this audit; other GPU processes are not controlled",
        "parameter_delta_v1": 0,
        "controller_arithmetic": "Per image N=C*H*W: two square+sum reductions (2N multiplies, 2(N-1) adds), N scale multiplies, two sqrt, two mean divisions, scalar budget/denominator arithmetic; FP32 casts and memory traffic also cost time. At layer9/640 N=102400. Raw alpha multiply and final add already exist in v1. THOP may omit these operations.",
    }


def extra_checks(source, directory, device):
    """Attach controller/state/latency evidence to the same preflight receipt without changing native stepping."""
    report = {"passed": False}
    try:
        with shared.check_phase("controller CPU"):
            report["controller_cpu"] = controller_checks("cpu")
        if str(device).startswith("cuda"):
            with shared.check_phase("controller CUDA"):
                report["controller_cuda"] = controller_checks(device)
        with shared.check_phase("module BN lifecycle, updated fixture, fuse, EMA and fused reload"):
            report["states"] = state_checks(source, directory, device)
        with shared.check_phase("sequential inference benchmark"):
            report["benchmark"] = benchmark(source, device)
        report["passed"] = True
    except BaseException:
        report["traceback"] = traceback.format_exc()
        raise
    finally:
        common.write_json(directory / "extra_checks.json", report)
    return report


if __name__ == "__main__":
    shared.main(NAME, MODEL, AuditedTrainer, module_checks, extra_checks)
