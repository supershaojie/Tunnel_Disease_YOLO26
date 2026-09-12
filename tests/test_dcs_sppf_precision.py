"""Scoped precision, real incident fixtures and corruption controls; local passes never certify server B32."""

import json
import os
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from tools.experiments import b19_common as common
from tools.experiments import b19_detect_fuse_audit as audit
from tools.experiments import verify_b19_dcs_sppf as shared
from tools.experiments.b19_detect_fuse_audit import (
    attributes,
    fuse_audit,
    fuse_precision,
    fuse_precision_checks,
    snapshot,
)
from tools.experiments.replay_b19_dcs_sppf_v2_precision import VARIANTS, first_block_replay, restore_fixture
from ultralytics.nn.tasks import DetectionModel, load_checkpoint


@pytest.mark.parametrize("precision", ["highest", "high", "medium"])
@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_backend_autocast_rng_restore(precision, fail, device):
    """Both exits restore nondefault grouped matmul policy, ambient autocast, Python/NumPy/CPU/CUDA RNG."""
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA NOT RUN: unavailable")
    original = common.computation_conditions()
    try:
        torch.set_float32_matmul_precision(precision)
        torch.backends.cudnn.allow_tf32 = True
        before = common.computation_conditions()
        cpu_rng, python_rng, numpy_rng = torch.get_rng_state(), random.getstate(), np.random.get_state()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        with torch.autocast(torch.device(device).type, enabled=True):
            try:
                with fuse_precision(device, "strict_fp32_equivalence") as receipt:
                    assert not torch.is_autocast_enabled(torch.device(device).type)
                    assert not torch.backends.cudnn.allow_tf32 and not torch.backends.cuda.matmul.allow_tf32
                    assert torch.get_float32_matmul_precision() == "highest"
                    random.random()
                    np.random.random(7)
                    torch.randn(7)
                    for i in range(len(cuda_rng)):
                        torch.randn(7, device=f"cuda:{i}")
                    if fail:
                        raise RuntimeError("injected scope failure")
            except RuntimeError as error:
                assert fail and str(error) == "injected scope failure"
            assert torch.is_autocast_enabled(torch.device(device).type)
        assert common.computation_conditions() == before == receipt["restored"]
        assert receipt["rng_restored"] and torch.equal(cpu_rng, torch.get_rng_state())
        assert python_rng == random.getstate()
        assert numpy_rng[0] == np.random.get_state()[0] and np.array_equal(numpy_rng[1], np.random.get_state()[1])
        assert numpy_rng[2:] == np.random.get_state()[2:]
        assert all(torch.equal(state, torch.cuda.get_rng_state(i)) for i, state in enumerate(cuda_rng))
    finally:
        torch.backends.cudnn.allow_tf32 = original["cudnn_allow_tf32"]
        torch.backends.cuda.matmul.allow_tf32 = original["matmul_allow_tf32"]
        torch.set_float32_matmul_precision(original["float32_matmul_precision"])


@pytest.mark.parametrize("fail", [False, True])
def test_source_and_input_untouched(tmp_path, monkeypatch, fail):
    """Even a failed fusion that changes its own copy cannot mutate caller state, BN mode, caches or input."""
    model, _ = shared.build_pair()
    model.train()
    x = torch.randn(1, 3, 128, 160)
    state, attrs, original_x, rng = snapshot(model.state_dict()), attributes(model), x.clone(), torch.get_rng_state()
    if fail:

        def broken_fuse(candidate, verbose=True):
            with torch.no_grad():
                candidate.model[0].conv.weight.add_(1)
            torch.rand(7)
            raise RuntimeError("injected fusion failure")

        monkeypatch.setattr(DetectionModel, "fuse", broken_fuse)
        with pytest.raises(RuntimeError, match="injected"):
            fuse_audit(model, x, tmp_path)
    else:
        assert fuse_audit(model, x, tmp_path)["passed"]
    common.assert_close_tree(state, snapshot(model.state_dict()), 0, 0)
    common.assert_close_tree(attrs, attributes(model), 0, 0)
    assert torch.equal(x, original_x) and torch.equal(rng, torch.get_rng_state())
    report = json.loads(next(tmp_path.glob("fuse-*/audit.json")).read_text())
    assert report["precision"]["rng_restored"] and report["precision"]["before"] == report["precision"]["restored"]


@pytest.mark.parametrize("corruption", ["conv_weight", "bn_running_var"])
def test_strict_rejects_fold_corruption(tmp_path, monkeypatch, corruption):
    """Actual first-block Conv weight or pre-fold BN corruption remains fatal under explicit FP32."""
    checkpoint = common.ROOT / "yolo26n.pt"
    if not checkpoint.is_file():
        pytest.skip("Original checkpoint NOT RUN: locally supplied weights required, no download")
    assert common.sha256(checkpoint) == common.PRETRAINED_SHA256
    source, _ = load_checkpoint(checkpoint, device="cpu", fuse=False)
    model, _ = shared.build_pair(source)
    native_fuse = DetectionModel.fuse

    def corrupted(candidate, verbose=True):
        if corruption == "bn_running_var":
            candidate.model[2].cv1.bn.running_var.mul_(2)
        result = native_fuse(candidate, verbose=verbose)
        if corruption == "conv_weight":
            with torch.no_grad():
                result.model[2].cv1.conv.weight.mul_(2)
        return result

    monkeypatch.setattr(DetectionModel, "fuse", corrupted)
    with pytest.raises(AssertionError, match="raw.one2one.boxes"):
        fuse_audit(model, torch.randn(1, 3, 128, 160), tmp_path)
    report = json.loads(next(tmp_path.glob("fuse-*/audit.json")).read_text())
    assert report["status"] == "FAIL" and report["required"]
    assert report["first_layer_outside_tolerance"] == "model.2.cv1"


@pytest.fixture
def server_directories():
    """Use opt-in authorized evidence, loaded only as tensors; absent real fixtures are explicitly skipped."""
    location = os.environ.get("DCS_SERVER_EVIDENCE")
    if not location:
        pytest.skip("Server fixture NOT RUN: set DCS_SERVER_EVIDENCE to validated extraction root")
    root = Path(location)
    result = {}
    for variant in VARIANTS:
        matches = list((root / "preflight" / variant).glob("fuse-evidence-*"))
        assert len(matches) == 1
        result[variant] = matches[0]
    return result


@pytest.mark.parametrize("variant", ["native", "v2_updated"])
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_server_first_block(server_directories, tmp_path, variant, device):
    """Native and whole-model-updated first-block fixtures share their actual saved input across precision arms."""
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA NOT RUN: unavailable")
    result = first_block_replay(server_directories[variant], device, tmp_path / "block")
    assert result["strict_gate_passed"]
    assert result["profiles"]["native_precision_diagnostic"]["precision"]["during"]["cudnn_allow_tf32"]
    assert not result["profiles"]["strict_fp32_equivalence"]["precision"]["during"]["cudnn_allow_tf32"]


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_five_server_states_all_profiles(server_directories, tmp_path, device):
    """All five complete states cover strict raw/feats/decode/top-k, finite low precision FAIL and exact same folds."""
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA NOT RUN: unavailable")
    models, inputs = {}, []
    for variant in VARIANTS:
        models[variant], x = restore_fixture(server_directories[variant], variant)
        assert models[variant].names == {0: "crack"}
        inputs.append(x)
    assert models["v2_zero"].model[9].theta.item() == 0
    assert all(models[v].model[9].theta.item() != 0 for v in ("v2_updated", "v2_diagnostic"))
    assert not torch.equal(models["native"].model[2].cv1.conv.weight, models["v2_updated"].model[2].cv1.conv.weight)
    assert all(torch.equal(inputs[0], x) for x in inputs[1:])
    result = fuse_precision_checks(models, inputs[0].to(device), tmp_path)
    assert result["strict_gate_passed"] and result["same_source_and_fold_across_profiles"]
    for variant in VARIANTS:
        strict = result["profiles"]["strict_fp32_equivalence"][variant]
        assert strict["status"] == "PASS" and strict["stage"] == "complete" and strict["batches"]
        assert any(row["path"].startswith("raw.one2one.feats") for row in strict["candidate_rows"])
        assert result["profiles"]["amp_diagnostic"][variant]["status"] == "FAIL"


def test_diagnostic_exception_is_blocking(tmp_path, monkeypatch):
    """An arbitrary native diagnostic exception cannot be silently reclassified as finite numeric mismatch."""
    model, _ = shared.build_pair()
    native_fuse = DetectionModel.fuse

    def broken(candidate, verbose=True):
        if torch.backends.cudnn.allow_tf32:
            raise RuntimeError("injected native-only implementation error")
        return native_fuse(candidate, verbose=verbose)

    monkeypatch.setattr(DetectionModel, "fuse", broken)
    with pytest.raises(AssertionError, match="Required fuse checks failed"):
        fuse_precision_checks({"native": model}, torch.randn(1, 3, 128, 160), tmp_path)
    report = json.loads((tmp_path / "precision_checks.json").read_text())
    assert not report["strict_gate_passed"]
    assert "injected native-only" in report["profiles"]["native_precision_diagnostic"]["native"]["traceback"]


@pytest.mark.parametrize("fault", ["finite", "nan", "shape", "dtype", "keys"])
def test_mismatch_type_belongs_to_finite_comparison(fault):
    """Only a finite, shape/dtype-compatible numeric difference receives the opt-in cross-path exception type."""
    a, b = torch.zeros(2), torch.ones(2)
    if fault == "nan":
        b[0] = float("nan")
    elif fault == "shape":
        b = b[:1]
    elif fault == "dtype":
        b = b.double()
    elif fault == "keys":
        a, b = {"a": a}, {"b": b}
    with pytest.raises(AssertionError) as caught:
        common.assert_close_tree({"nested": [a]}, {"nested": [b]}, mismatch_error=audit.FuseEquivalenceMismatch)
    assert isinstance(caught.value, audit.FuseEquivalenceMismatch) == (fault == "finite")


def test_late_plain_numeric_assertion_still_blocks(tmp_path, monkeypatch):
    """A later state assertion stays mandatory even after successful raw/decode/top-k comparisons populated rows."""
    model, _ = shared.build_pair()
    original = audit._fuse_audit

    def late_failure(model, x, directory, report):
        original(model, x, directory, report)
        if report["profile"] == "native_precision_diagnostic":
            common.assert_close_tree(torch.zeros(1), torch.ones(1), path="state", report=report["candidate_rows"])

    monkeypatch.setattr(audit, "_fuse_audit", late_failure)
    with pytest.raises(AssertionError, match="Required fuse checks failed"):
        fuse_precision_checks({"native": model}, torch.randn(1, 3, 128, 160), tmp_path)
    report = json.loads((tmp_path / "precision_checks.json").read_text())
    assert not report["strict_gate_passed"] and "native_precision_diagnostic/native" in report["blocking_failures"]
