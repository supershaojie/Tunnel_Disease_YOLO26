"""Precision protocol, real server fixture replays and failures; local devices are never server certification."""

import copy
import json
import os
from pathlib import Path

import pytest
import torch

from tools.experiments import b19_common as common
from tools.experiments.lbi_fuse_audit import fuse_precision, fuse_precision_checks, snapshot
from tools.experiments.replay_lbi_fuse_precision import first_block_replay, restore_fixture
from tools.experiments.verify_b19_lbi_fusion import FUSE_ATOL, FUSE_RTOL, build_pair
from ultralytics.nn.tasks import DetectionModel


@pytest.fixture(autouse=True)
def local_threads():
    """Limit test overhead, restoring the caller's CPU thread setting."""
    previous = torch.get_num_threads()
    torch.set_num_threads(4)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("precision", ["highest", "high", "medium"])
@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_grouped_backend_restore(precision, fail, device):
    """Non-default matmul policies, autocast and exceptions cannot leak out of the precision scope."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA NOT RUN: unavailable")
    original = common.computation_conditions()
    try:
        torch.set_float32_matmul_precision(precision)
        torch.backends.cudnn.allow_tf32 = True
        before = common.computation_conditions()
        with torch.autocast(device, enabled=True):
            try:
                with fuse_precision(device, False) as receipt:
                    assert not torch.is_autocast_enabled(device)
                    assert not torch.backends.cudnn.allow_tf32 and not torch.backends.cuda.matmul.allow_tf32
                    assert torch.get_float32_matmul_precision() == "highest"
                    torch.randn(8, 8, device=device).square().sum()
                    if fail:
                        raise RuntimeError("injected failure")
            except RuntimeError as error:
                assert fail and str(error) == "injected failure"
            assert torch.is_autocast_enabled(device)
        assert common.computation_conditions() == before == receipt["restored"]
    finally:
        torch.backends.cudnn.allow_tf32 = original["cudnn_allow_tf32"]
        torch.backends.cuda.matmul.allow_tf32 = original["matmul_allow_tf32"]
        torch.set_float32_matmul_precision(original["float32_matmul_precision"])


def family():
    """Construct native, zero and nonzero fixtures with identical shared state."""
    native, zero = build_pair()
    nonzero = copy.deepcopy(zero)
    with torch.no_grad():
        nonzero.model[15].out.weight.normal_(0, 0.01)
    return dict(native=native, lbi_zero=zero, lbi_nonzero=nonzero)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_full_precision_protocol(tmp_path, device):
    """All three real networks pass explicit FP32 while native diagnostics and backend receipts remain separate."""
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA NOT RUN: unavailable")
    models = family()
    original = common.computation_conditions()
    report = fuse_precision_checks(
        models, torch.randn(1, 3, 160, 160, device=device), tmp_path / "precision", FUSE_ATOL, FUSE_RTOL
    )
    assert report["passed"] and common.computation_conditions() == original
    for arm in ("native", "explicit_fp32"):
        for variant in models:
            result = report["arms"][arm]["models"][variant]
            assert result["postprocess_valid"]
            assert (tmp_path / "precision" / arm / variant / "layer_outputs.pt").is_file()
    assert all(v["passed"] for v in report["arms"]["explicit_fp32"]["models"].values())


@pytest.mark.parametrize("native_only", [False, True])
def test_fused_corruption_cannot_be_precision_classified(tmp_path, monkeypatch, native_only):
    """Bad folded bias must fail strict checks, or the exact cross-arm state gate, even if native is finite."""
    models = family()
    original = common.computation_conditions()
    original_fuse = DetectionModel.fuse

    def corrupted(model, verbose=True):
        model = original_fuse(model, verbose=verbose)
        if not native_only or torch.backends.cudnn.allow_tf32:
            with torch.no_grad():
                model.model[-1].one2one_cv3[0][-1].bias.add_(0.5)
        return model

    monkeypatch.setattr(DetectionModel, "fuse", corrupted)
    directory = tmp_path / "precision"
    with pytest.raises(AssertionError, match="same_fold|Explicit FP32"):
        fuse_precision_checks(models, torch.randn(1, 3, 160, 160), directory, FUSE_ATOL, FUSE_RTOL)
    report = json.loads((directory / "precision_checks.json").read_text())
    assert not report["passed"] and common.computation_conditions() == original
    for variant in models:
        native = report["arms"]["native"]["models"][variant]
        assert not native["passed"] and not native["raw_close"] and native["postprocess_valid"]
        assert (directory / "native" / variant / "before.pt").is_file()
        assert "traceback" in native


def test_nonzero_only_native_anomaly_is_rejected(tmp_path, monkeypatch):
    """A nonzero-LBI-only precision-dependent forward error has no native control and cannot be waived."""
    models = family()
    original_fuse = DetectionModel.fuse

    def corrupted(model, verbose=True):
        model = original_fuse(model, verbose=verbose)
        block = model.model[15]
        if hasattr(block, "out") and block.out.weight.count_nonzero():
            layer = model.model[-1].one2one_cv3[0][-1]
            original_forward = layer.forward

            def forward(x):
                result = original_forward(x)
                return result + 0.5 if torch.backends.cudnn.allow_tf32 else result

            layer.forward = forward
        return model

    monkeypatch.setattr(DetectionModel, "fuse", corrupted)
    with pytest.raises(AssertionError, match="LBI-only native anomaly"):
        fuse_precision_checks(models, torch.randn(1, 3, 160, 160), tmp_path / "precision", FUSE_ATOL, FUSE_RTOL)


@pytest.fixture
def server_fixture():
    """Opt into authorized external evidence; never download or fabricate a server fixture for CI."""
    location = os.environ.get("LBI_SERVER_EVIDENCE")
    if not location:
        pytest.skip("Real server fixture NOT RUN: set LBI_SERVER_EVIDENCE to validated extraction root")
    return Path(location) / "preflight/cuda_0"


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_real_server_first_block(server_fixture, tmp_path, device):
    """Use the exact server model.1 tensor and saved folded parameters on the explicitly reported device."""
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA NOT RUN: unavailable")
    original = common.computation_conditions()
    report = first_block_replay(server_fixture / "fuse_native", device, tmp_path / "first_block")
    assert report["arms"]["False"]["comparison"]["passed"]
    assert common.computation_conditions() == original


def test_restore_complete_nonzero_server_fixture(server_fixture):
    """Loading the complete saved state replaces constructor initialization, including nonzero LBI out."""
    directory = server_fixture / "fuse_lbi_nonzero"
    source = torch.load(directory / "source.pt", weights_only=True, map_location="cpu")
    model, x = restore_fixture(directory)
    common.assert_close_tree(source["state"], snapshot(model.state_dict()), 0, 0)
    assert torch.equal(x, source["input"]) and model.model[15].out.weight.count_nonzero() > 0
