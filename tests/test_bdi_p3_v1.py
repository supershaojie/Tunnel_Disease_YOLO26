"""Regression tests for BDI's fixed formula, graph and diagnostic matching contract."""

import copy
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from tools.experiments import b19_common as common
from tools.experiments.bdi_experiment import V1
from tools.experiments.finish_b19_bdi_p3_v1 import match_objects, package
from tools.experiments.run_b19_bdi_p3_v1 import execute_preflight
from tools.experiments.verify_b19_bdi_p3_v1 import branch_checks, structural_checks
from ultralytics.engine.validator import BaseValidator
from ultralytics.nn.modules import Concat_BDI_P3
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.metrics import box_iou


def test_fixed_filters():
    """Check independent boundary reference, odd downsample and actual autocast operator dtypes."""
    branch_checks()


def test_peak_gain_and_grid_error():
    """The normalized passband has unity gain at q=1/2, and a wrong grid raises rather than resizes."""
    module = Concat_BDI_P3(128, 128, 64)
    x = torch.cos(torch.arange(41).double() * np.pi / 2).float()[None, None, None].expand(1, 16, 9, 41)
    torch.testing.assert_close(module.bandpass(x)[:, :, 2:-2, 2:-2], x[:, :, 2:-2, 2:-2], atol=1e-6, rtol=1e-6)
    with pytest.raises(ValueError, match="BDI P2 down2 grid"):
        module([torch.randn(1, 128, 16, 24), torch.randn(1, 128, 16, 24), torch.randn(1, 64, 33, 47)])


def test_full_graph(tmp_path):
    """Retain native shared states, RNG, Detect and complete train/eval outputs."""
    structural_checks(tmp_path)


def test_matching_retains_native_semantics():
    """Ground-truth identity tracking must yield the native matching result, including duplicate detections."""
    pred = torch.tensor(
        [[0, 0, 10, 10, 0.9, 0], [1, 0, 11, 10, 0.8, 0], [20, 20, 30, 30, 0.7, 0], [0, 0, 10, 10, 0.6, 1]]
    )
    gt = torch.tensor([[0, 0, 0, 10, 10], [0, 20, 20, 30, 30]])
    validator = object.__new__(BaseValidator)
    validator.iouv = torch.tensor([0.5, 0.75])
    expected = validator.match_predictions(pred[:, 5], gt[:, 0], box_iou(gt[:, 1:], pred[:, :4]))
    for index, threshold in enumerate((0.5, 0.75)):
        report = match_objects(pred, gt, threshold)
        actual = {v["prediction"] for v in report["matches"].values()}
        assert actual == set(expected[:, index].nonzero().flatten().tolist())
    assert match_objects(pred[:0], gt, 0.5)["fn"] == 2
    assert match_objects(pred, gt[:0], 0.5)["fp"] == 4


def test_incomplete_package_refused(tmp_path, monkeypatch):
    """A missing lifecycle cannot produce a success archive."""
    monkeypatch.setattr(common, "require_clean_source", lambda: None)
    run = tmp_path / V1.name
    run.mkdir()
    with pytest.raises(FileNotFoundError, match="Incomplete result"):
        package(run, None)


def test_failed_preflight_preserves_exit_and_attempt(tmp_path):
    """A failing independent child produces a failed receipt and raises before formal training can launch."""
    audit = tmp_path / "launch"
    audit.mkdir()
    with pytest.raises(subprocess.CalledProcessError) as error:
        execute_preflight(
            [sys.executable, "-c", "print('intentional test failure'); raise SystemExit(17)"], tmp_path, audit
        )
    assert error.value.returncode == 17
    attempt = Path((tmp_path / f"{V1.name}_preflight.current_attempt").read_text().strip())
    assert (attempt / "exit_status").read_text().strip() == "17"
    assert json.loads((attempt / "process_status.json").read_text())["state"] == "exited"
    assert "intentional test failure" in (attempt / "console.log").read_text()
    assert not (attempt / "passed.json").exists()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_initial_equivalence(tmp_path):
    """Compare independent native/candidate training and inference in both FP32 and AMP at batch1/640."""
    torch.manual_seed(42)
    baseline = DetectionModel(common.baseline_architecture(), verbose=False).cuda()
    torch.manual_seed(42)
    candidate = DetectionModel(str(V1.model), nc=1, verbose=False).cuda()
    x = torch.rand(1, 3, 640, 640, device="cuda")
    rows = []
    for train in (True, False):
        for amp in (False, True):
            a, b = copy.deepcopy(baseline).train(train), copy.deepcopy(candidate).train(train)
            comparisons = []
            with torch.no_grad(), torch.autocast("cuda", enabled=amp):
                common.assert_close_tree(a(x), b(x), 1e-3 if amp else 1e-6, 1e-3 if amp else 1e-5, report=comparisons)
            rows.append({"training": train, "amp": amp, "comparisons": comparisons})
            del a, b
    common.write_json(Path(tmp_path) / "cuda_initial_equivalence.json", rows)
