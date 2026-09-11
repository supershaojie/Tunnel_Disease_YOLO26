"""Regression tests for v2 ownership, fixed budget, initialization, source binding and v1 compatibility."""

import copy
import inspect
from pathlib import Path

import pytest
import torch

from tools.experiments import b19_common as common
from tools.experiments import run_b19_dcs_sppf as v1_run
from tools.experiments import finish_b19_dcs_sppf as finish
from tools.experiments import verify_b19_dcs_sppf as shared
from tools.experiments.run_b19_dcs_sppf_v2 import MODEL, AuditedTrainer
from tools.experiments.verify_b19_dcs_sppf_v2 import controller_checks, module_checks
from ultralytics.nn.modules import DCS_SPPF, DCS_SPPF_V2, SPPF
from ultralytics.nn.modules.dcs_sppf_v2 import relative_residual
from ultralytics.utils import YAML


def test_controller_boundary_ledger(tmp_path):
    """Exercise per-image bounds, actual rounding, detached budgets and non-detached raw gradients."""
    report = controller_checks()
    common.write_json(tmp_path / "controller.json", report)
    assert len(report["cases"]) == 84


def test_zero_and_delayed_learning():
    """Theta must unlock through a live task gradient before every original branch can learn."""
    assert module_checks()["identity"][-1]["max_abs"] == 0


@pytest.mark.parametrize("constant,value", [("residual_budget", 0.03), ("residual_budget", 0.08), ("residual_eps", 0)])
def test_fixed_architecture_constants(constant, value):
    """Reject accidental sweeps before constructing the formal variant."""
    with pytest.raises(ValueError, match="constants are fixed"):
        DCS_SPPF_V2(256, 256, **{constant: value})


@pytest.mark.parametrize(
    "shape,shortcut", [((32, 48, 4, 7), True), ((32, 32, 7, 4), False), ((256, 256, 20, 27), True)]
)
@pytest.mark.parametrize("training", [False, True])
def test_main_path_bn_and_dtype(shape, shortcut, training):
    """Compare each original BN update once for rectangular and unequal-channel native shortcuts."""
    c1, c2, height, width = shape
    native = SPPF(c1, c2, shortcut=shortcut).train(training)
    block = DCS_SPPF_V2(c1, c2, shortcut=shortcut).train(training)
    block.load_state_dict(native.state_dict(), strict=False)
    x = torch.randn(2, c1, height, width)
    shared.module_identity(native, block, x)
    for name in ("cv1", "cv2"):
        a, b = getattr(native, name).bn, getattr(block, name).bn
        assert b.num_batches_tracked.item() == int(training)
        for field in ("running_mean", "running_var", "num_batches_tracked"):
            torch.testing.assert_close(getattr(a, field), getattr(b, field), rtol=0, atol=0)
    assert all(pool.count_include_pad for pool in block.avg)


def test_shared_rng_topology_and_zero_init_model():
    """The complete one2one/one2many tree and every native state tensor agree before any update."""
    baseline, candidate = shared.build_pair(model=MODEL)
    shared.topology_checks(baseline, candidate, MODEL, DCS_SPPF_V2)
    coverage = common.audit_weights(baseline, candidate, None)
    assert coverage["common_keys"] == 708 and coverage["added_parameters"] == 103041
    assert len(coverage["new_parameters"]) == 13
    _, v1 = shared.build_pair()
    common.assert_close_tree(v1.state_dict(), candidate.state_dict(), 0, 0)
    assert sum(p.numel() for p in v1.parameters()) == sum(p.numel() for p in candidate.parameters()) == 2607231
    shared.full_identity_checks(baseline, candidate)
    candidate.yaml = copy.deepcopy(candidate.yaml)
    candidate.yaml["head"][10][0] = [-1, 9]
    with pytest.raises(AssertionError):
        shared.topology_checks(baseline, candidate, MODEL, DCS_SPPF_V2)


def test_explicit_source_binding_and_v1_defaults():
    """Definition-time defaults remain v1; the thin v2 trainer and all source hashes resolve explicitly."""
    assert inspect.signature(common.resolve_recipe).parameters["model"].default == common.MODEL
    assert v1_run.options_parser().parse_args([]).name == "yolo26n_b19_dcs_sppf_v1"
    assert v1_run.AuditedTrainer.block_type is DCS_SPPF
    assert AuditedTrainer.block_type is DCS_SPPF_V2 and AuditedTrainer.model_yaml == MODEL
    block = DCS_SPPF_V2(256, 256)
    binding = common.model_binding(MODEL, DCS_SPPF_V2, block)
    assert Path(binding["module_path"]).name == "dcs_sppf_v2.py"
    assert binding["residual_budget"] == 0.05 and binding["residual_eps"] == 1e-6
    with pytest.raises(AssertionError):
        common.model_binding(MODEL, DCS_SPPF_V2, DCS_SPPF(256, 256))
    block.residual_budget = 0.08
    with pytest.raises(AssertionError):
        common.model_binding(MODEL, DCS_SPPF_V2, block)
    assert "tests/test_dcs_sppf_v2.py" in {k.replace("\\", "/") for k in common.source_hashes()}


def test_raw_gradient_is_not_a_detached_scale():
    """A radial derivative includes the scale's raw-energy derivative, which vanishes toward saturation."""
    raw = torch.full((1, 256, 4, 7), 2.0, requires_grad=True)
    reference = torch.ones_like(raw, requires_grad=True)
    injected, q = relative_residual(raw, reference)
    grad, budget_grad = torch.autograd.grad(injected.sum(), (raw, reference), allow_unused=True)
    assert budget_grad is None and grad.abs().max() > 0
    assert grad.max() < q.item() * 0.01


@pytest.mark.parametrize(
    "model,block_type,name",
    [(common.MODEL, DCS_SPPF, "yolo26n_b19_dcs_sppf_v1"), (MODEL, DCS_SPPF_V2, "yolo26n_b19_dcs_sppf_v2")],
)
def test_completed_provenance_identity_and_receipt(tmp_path, monkeypatch, model, block_type, name):
    """Accept a complete canonical identity and reject a swapped preflight recipe before evaluating any weights."""
    run = tmp_path / name
    run.mkdir()
    data = tmp_path / "data.yaml"
    data.write_text("names: [crack]\n")
    args = {"project": str(tmp_path), "name": name}
    YAML.save(run / "args.yaml", args)
    (run / "results.csv").write_text("fixture only\n")
    binding = common.model_binding(model, block_type)
    evidence = {
        "source_sha256": {"fixture": "hash"},
        "dataset_manifest": {"fixture": 1},
        "model_binding": binding,
        "data_sha256": common.sha256(data),
    }
    common.write_json(run / "provenance/resolved.json", {"config": args, "evidence": evidence})
    common.write_json(
        run / "completed.json",
        {
            "commit": "fixture-sha",
            "files": {"weights/best.pt": "weight-hash", "results.csv": common.sha256(run / "results.csv")},
        },
    )
    receipt = {"passed": True, "local_only": False, "commit": "fixture-sha", "recipe": evidence, "complexity": {}}
    common.write_json(run / "provenance/preflight/checks.json", receipt)
    monkeypatch.setattr(common, "require_clean_source", lambda: None)
    monkeypatch.setattr(common, "git", lambda *args: "fixture-sha")
    monkeypatch.setattr(common, "source_hashes", lambda: {"fixture": "hash"})
    monkeypatch.setattr(common, "dataset_manifest", lambda data: {"fixture": 1})
    sha256 = common.sha256
    monkeypatch.setattr(common, "sha256", lambda path: "weight-hash" if str(path).endswith("best.pt") else sha256(path))
    assert finish.provenance(run, tmp_path / "data.yaml", name, model, block_type)["model_binding"] == binding
    receipt["recipe"] = {"wrong": True}
    common.write_json(run / "provenance/preflight/checks.json", receipt)
    with pytest.raises(AssertionError, match="Preflight receipt"):
        finish.provenance(run, tmp_path / "data.yaml", name, model, block_type)
    data.write_text("names: [different]\n")
    with pytest.raises(ValueError, match="data YAML"):
        finish.provenance(run, data, name, model, block_type)
