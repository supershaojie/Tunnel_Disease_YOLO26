"""Module SGD scaling and failure receipts; separate from native detection preflight and formal training."""

import json
import os
from pathlib import Path

import pytest
import torch

from tools.experiments import b19_common as common
from tools.experiments import verify_b19_lbi_fusion as verifier
from tools.experiments.lbi_fuse_audit import fuse_precision, snapshot
from tools.experiments.verify_b19_lbi_fusion import module_checks, module_updates
from ultralytics.nn.modules import Concat_LBI_Fusion
from ultralytics.utils.torch_utils import init_seeds


@pytest.fixture(autouse=True)
def cpu_threads():
    """Bound local test overhead without changing the fixture's loss or observation budget."""
    previous = torch.get_num_threads()
    torch.set_num_threads(4)
    yield
    torch.set_num_threads(previous)


def fixture(device="cpu"):
    """Use a fixed local SGD fixture with zero out; full B32 evidence is tested separately below."""
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA NOT RUN: unavailable")
    init_seeds(42, deterministic=True)
    module = Concat_LBI_Fusion([128, 128]).to(device)
    semantic = torch.randn(2, 128, 17, 13, device=device).transpose(2, 3)
    return module, semantic, torch.randn_like(semantic) + 2, torch.randn(2, 256, 13, 17, device=device)


@pytest.mark.parametrize("device,amp", [("cpu", False), ("cuda:0", False), ("cuda:0", True)])
def test_module_scaler_is_persistent_and_unscales_once(tmp_path, monkeypatch, device, amp):
    """The actual CUDA AMP path uses one scaler, one unscale per attempt and real successful SGD steps."""
    inputs = fixture(device)
    original = torch.amp.GradScaler
    scalers = []

    class TrackedScaler(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.unscale_calls = 0
            scalers.append(self)

        def unscale_(self, optimizer):
            self.unscale_calls += 1
            return super().unscale_(optimizer)

    monkeypatch.setattr(torch.amp, "GradScaler", TrackedScaler)
    report = module_updates(*inputs, amp, tmp_path / "module")
    assert report["passed"] and len(scalers) == 1
    assert report["scaler_enabled"] == amp
    assert scalers[0].unscale_calls == (3 if amp else 0)
    assert len(report["updates"]) == report["max_attempts"] == 3
    assert [r["successful_step_count"] for r in report["updates"]] == [1, 2, 3]
    assert not report["missing_gradient"] and not report["missing_update"]
    summary = report["summary"]["parameters"]
    assert summary["out.weight"]["first_effective_update_batch"] == 0
    assert all(v["first_gradient_batch"] > 0 for k, v in summary.items() if k != "out.weight")
    assert report["backend_before"] == report["backend_after"]
    assert json.loads((tmp_path / "module/summary.json").read_text())["passed"]


@pytest.mark.parametrize("broken", ["detach", "zero_gradient", "zero_target", "omitted_optimizer", "lost_update"])
def test_module_failure_names_and_preserves_fixture(tmp_path, monkeypatch, broken):
    """Disconnected/zero task gradients and missing optimizer membership cannot be hidden by AMP or finite values."""
    module, semantic, detail, target = fixture()
    initial = snapshot(module.state_dict())
    if broken == "detach":
        module.proj_s.register_forward_hook(lambda m, x, y: y.detach())
    elif broken == "zero_gradient":
        module.proj_s.weight.register_hook(torch.zeros_like)
    elif broken == "zero_target":
        target.zero_()
    elif broken == "omitted_optimizer":
        original = torch.optim.SGD
        monkeypatch.setattr(
            torch.optim,
            "SGD",
            lambda params, **kwargs: original([p for p in params if p is not module.proj_s.weight], **kwargs),
        )
    else:
        original = torch.optim.SGD

        def lost_update(params, **kwargs):
            optimizer = original(params, **kwargs)

            def restore(*args):
                with torch.no_grad():
                    module.proj_s.weight.copy_(initial["proj_s.weight"])

            optimizer.register_step_post_hook(restore)
            return optimizer

        monkeypatch.setattr(torch.optim, "SGD", lost_update)
    directory = tmp_path / "failure"
    with pytest.raises(AssertionError, match="out.weight|proj_s.weight"):
        module_updates(module, semantic, detail, target, False, directory)
    report = json.loads((directory / "summary.json").read_text())
    updates = json.loads((directory / "updates.json").read_text())
    assert not report["passed"] and "traceback" in report
    payload = torch.load(directory / "failure.pt", weights_only=True, map_location="cpu")
    common.assert_close_tree(payload["initial_state"], initial, 0, 0)
    assert torch.equal(payload["target"], target)
    assert "optimizer" in payload and "scaler" in payload and "rng_before" in payload and "rng_after" in payload
    assert report["backend_before"] == report["backend_after"]
    if broken == "omitted_optimizer":
        assert report["parameters"]["proj_s.weight"]["optimizer_memberships"] == 0
    elif broken == "zero_target":
        assert len(updates) == 1 and "out.weight" in report["missing_gradient"]
    elif broken == "lost_update":
        assert len(updates) == 3 and "proj_s.weight" not in report["missing_gradient"]
        assert report["missing_update"] == ["proj_s.weight"]
    else:
        assert len(updates) == 3 and "proj_s.weight" in report["missing_gradient"]
        assert "proj_s.weight" in report["missing_update"]


@pytest.mark.parametrize("overflow_attempts", [1, 2])
def test_overflow_is_not_an_update_and_budget_is_fixed(tmp_path, overflow_attempts):
    """Injected Inf consumes attempts with normal scale backoff; two skips exhaust upstream learning time."""
    inputs = fixture("cuda:0")
    module = inputs[0]
    calls = []

    def overflow(gradient):
        calls.append(True)
        return torch.full_like(gradient, float("inf")) if len(calls) <= overflow_attempts else gradient

    module.out.weight.register_hook(overflow)
    directory = tmp_path / "overflow"
    if overflow_attempts == 1:
        assert module_updates(*inputs, True, directory)["passed"]
    else:
        with pytest.raises(AssertionError, match="proj_l.weight.*proj_s.weight.*dw.weight"):
            module_updates(*inputs, True, directory)
    report = json.loads((directory / "summary.json").read_text())
    rows = report["updates"]
    assert len(rows) == len(calls) == 3
    for i, row in enumerate(rows):
        assert row["scaler_before"] == 65536 / 2 ** min(i, overflow_attempts)
        assert row["skipped"] == (i < overflow_attempts)
        if row["skipped"]:
            assert not row["successful_optimizer_step"] and row["successful_step_count"] == 0
            assert all(not p["effective_update"] and p["changed_elements"] == 0 for p in row["parameters"].values())
            assert row["nonfinite_gradient_parameters"] == ["out.weight"]


def test_fuse_precision_returns_to_module_amp(tmp_path):
    """A completed fuse diagnostic context must not disable the following fixture's autocast/scaler."""
    inputs = fixture("cuda:0")
    before = common.computation_conditions()
    with fuse_precision("cuda:0", False):
        torch.randn(8, 8, device="cuda").square()
    assert common.computation_conditions() == before
    report = module_updates(*inputs, True, tmp_path / "after_fuse")
    assert report["passed"] and report["scaler_enabled"] and report["autocast_dtype"] == "torch.float16"
    assert report["backend_after"] == before


@pytest.mark.parametrize("mode", ["fp32", "fp16_unscaled", "fp16_scaled"])
def test_saved_full_B32_same_fixture(tmp_path, monkeypatch, mode):
    """Opt into the captured local B32 fixture; an unscaled control is observed, never required to fail on all GPUs."""
    location = os.environ.get("LBI_B32_FIXTURE")
    if not location or not torch.cuda.is_available():
        pytest.skip("Saved CUDA B32/80x80 fixture NOT RUN: require LBI_B32_FIXTURE and CUDA")
    saved = torch.load(Path(location), weights_only=True, map_location="cpu")
    module = Concat_LBI_Fusion([128, 128]).cuda()
    module.load_state_dict(saved["state"], strict=True)
    common.assert_close_tree(snapshot(module.state_dict()), saved["state"], 0, 0)
    assert saved["target"].shape == (32, 256, 80, 80)
    inputs = [saved[k].to("cuda:0") for k in ("semantic", "detail", "target")]
    if mode == "fp16_unscaled":
        original = torch.amp.GradScaler
        monkeypatch.setattr(torch.amp, "GradScaler", lambda *args, **kwargs: original("cuda", enabled=False))
    directory = tmp_path / mode
    try:
        module_updates(module, *inputs, mode != "fp32", directory)
    except AssertionError:
        if mode != "fp16_unscaled":
            raise
    report = json.loads((directory / "summary.json").read_text())
    assert len(report["updates"]) == 3 and report["inputs"]["target"]["shape"] == [32, 256, 80, 80]
    if mode != "fp16_unscaled":
        assert report["passed"] and not report["missing_update"]


def test_complete_B32_module_checks(tmp_path, monkeypatch):
    """Full original identity/formula checks plus the fixed update path retain the B32/P3 dimensions."""
    if not os.environ.get("LBI_B32_FIXTURE") or not torch.cuda.is_available():
        pytest.skip("Full CUDA B32/P3 module checks NOT RUN: opt in with LBI_B32_FIXTURE")
    saved = torch.load(Path(os.environ["LBI_B32_FIXTURE"]), weights_only=True, map_location="cpu")
    actual_updates = verifier.module_updates

    def same_fixture(module, semantic, detail, target, amp, directory):
        common.assert_close_tree(snapshot(module.state_dict()), saved["state"], 0, 0)
        for name, value in dict(semantic=semantic, detail=detail, target=target).items():
            assert torch.equal(value.cpu(), saved[name]), f"Full module fixture changed: {name}"
        return actual_updates(module, semantic, detail, target, amp, directory)

    monkeypatch.setattr(verifier, "module_updates", same_fixture)
    result = module_checks("cuda:0", True, spatial=(80, 80), batch=32, directory=tmp_path / "B32")
    assert result["passed"] and result["batch"] == 32 and result["scaler_enabled"]
    assert result["identity"][0]["max_abs"] == 0
