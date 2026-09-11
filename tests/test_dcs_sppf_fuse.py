"""Positive and adversarial candidate-identity audits; local lifecycle tests never certify native B32."""

import copy
import json
import tarfile

import pytest
import torch

from tools.experiments import b19_common as common
from tools.experiments import verify_b19_dcs_sppf as shared
from tools.experiments import verify_b19_dcs_sppf_v2 as verifier
from tools.experiments.b19_detect_fuse_audit import audit_candidates, capture, fuse_audit, snapshot
from tools.experiments.finish_b19_dcs_sppf import archive_package
from ultralytics.nn.tasks import DetectionModel, load_checkpoint


def candidate_fixture(scores, ids, second=None):
    """Represent actual stage1 and stage2 identities, including tied scores and off-screen coordinates."""
    scores = torch.tensor(scores, dtype=torch.float32).view(1, -1, 1)
    boxes = (torch.arange(scores.numel() * 4).float() * 100).reshape(1, -1, 4)
    decoded = torch.cat((boxes, scores), -1)
    first = torch.tensor(ids, dtype=torch.int64).view(1, -1)
    second = torch.tensor(second if second is not None else list(range(len(ids))), dtype=torch.int64).view(1, -1)
    indices = first.gather(1, second).unsqueeze(-1)
    classes = torch.zeros_like(indices, dtype=torch.float32)
    selected = decoded.gather(1, indices.expand(-1, -1, 5))
    stage1_input = scores[..., 0]
    stage2_input = stage1_input.gather(1, first)
    stages = [
        dict(input=values, values=values.gather(1, index), indices=index, k=len(ids), dim=-1, largest=True, sorted=True)
        for values, index in ((stage1_input, first), (stage2_input, second))
    ]
    return dict(
        raw=dict(
            boxes=boxes.transpose(1, 2),
            scores=scores.logit().transpose(1, 2),
            feats=[boxes.transpose(1, 2).unsqueeze(-1).clone()],
        ),
        decoded=decoded,
        indices=indices,
        classes=classes,
        final=torch.cat((selected, classes), -1),
        selected_scores=selected[..., 4:5].clone(),
        topk_stages=stages,
    )


@pytest.mark.parametrize("boundary,tied", [(False, False), (False, True), (True, False), (True, True)])
def test_candidate_permutation_and_boundary(boundary, tied):
    """Accept a changed row only after every raw candidate, selected identity and boundary is proven."""
    before = candidate_fixture([0.8, 0.5, 0.5 if tied else 0.500001, 0.1], [0, 2] if boundary else [0, 2, 1])
    after = candidate_fixture([0.8, 0.5 if tied else 0.500002, 0.5, 0.1], [0, 1] if boundary else [0, 1, 2])
    report = {}
    audit_candidates(before, after, 2 if boundary else 3, 1e-4, 1e-4, report)
    assert not report["positional"]["passed"] and report["stage"] == "complete"
    batch = report["batches"][0]
    assert batch["same_set"] != boundary and batch["compared_union"] == 3
    if boundary:
        assert batch["dropped"] == [2] and batch["added"] == [1]
        assert batch["max_gap_minus_measured_budget"] <= 0 and len(batch["boundary"]) == 2


def test_second_stage_actual_tied_permutation():
    """Stage2 can permute ties without changing stage1; audit the actual composed original identities."""
    before = candidate_fixture([0.8, 0.5, 0.5, 0.1], [0, 1, 2])
    after = candidate_fixture([0.8, 0.5, 0.5, 0.1], [0, 1, 2], [0, 2, 1])
    report = {}
    audit_candidates(before, after, 3, 1e-4, 1e-4, report)
    assert report["topk_stages"]["after"]["stage2_is_full_permutation"]
    assert report["batches"][0]["changed_rank_positions"] == 2


@pytest.mark.parametrize(
    "corruption",
    [
        "raw_box",
        "features",
        "decoded_box",
        "logit",
        "score",
        "omit_higher",
        "tied_omit_highest",
        "gather",
        "class",
        "duplicate",
        "stage1",
        "stage2",
        "count",
        "unresolved",
    ],
)
def test_candidate_negative_controls(corruption):
    """Reject even common-mode bad selection and corruption outside the final intersection."""
    before = candidate_fixture([0.9, 0.8, 0.7, 0.1], [0, 1])
    after = copy.deepcopy(before)
    k = 2
    if corruption == "raw_box":
        after["raw"]["boxes"][0, 0, 2] += 100
    elif corruption == "features":
        after["raw"]["feats"][0][0, 0, 0, 0] += 100
    elif corruption == "decoded_box":
        after["decoded"][0, 2, 0] += 100
    elif corruption == "logit":
        after["raw"]["scores"][0, 0, 2] += 0.1
    elif corruption == "score":
        after = candidate_fixture([0.9, 0.8, 0.75, 0.1], [0, 1])
    elif corruption == "omit_higher":
        before = candidate_fixture([0.9, 0.8, 0.7, 0.1], [0, 2])
        after = copy.deepcopy(before)
    elif corruption == "tied_omit_highest":
        # The same kth score and unique, sorted IDs do NOT prove the whole top-k is correct.
        before = candidate_fixture([0.9, 0.8, 0.7, 0.7, 0.1], [1, 2, 3])
        after = copy.deepcopy(before)
        k = 3
    elif corruption == "gather":
        after["final"][0, 0, 0] += 100
    elif corruption == "class":
        after["classes"][0, 0, 0] = 1
    elif corruption == "duplicate":
        before = candidate_fixture([0.9, 0.8, 0.7, 0.1], [0, 0])
        after = copy.deepcopy(before)
    elif corruption == "stage1":
        after["topk_stages"][0]["input"][0, 2] = 0.75
    elif corruption == "stage2":
        after["topk_stages"][1]["indices"][0, 0] = 1
    elif corruption == "count":
        after["indices"] = after["indices"][:, :1]
    else:
        after.pop("topk_stages")
    pattern = {"unresolved": "UNRESOLVED", "features": "raw.one2one.feats"}.get(corruption)
    with pytest.raises(AssertionError, match=pattern):
        audit_candidates(before, after, k, 1e-4, 1e-4, {})


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("variant", ["native", "v1", "v2_zero", "v2_updated", "v2_diagnostic"])
def test_real_fuse(tmp_path, device, variant):
    """Same-device native Conv-BN fuse, with both an optimizer-unlocked theta and the original diagnostic state."""
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable; no server PASS implied")
    baseline, v1 = shared.build_pair()
    _, v2 = shared.build_pair(model=verifier.MODEL)
    model = {"native": baseline, "v1": v1}.get(variant, v2)
    if variant == "v2_updated":
        model, update = verifier.updated_fixture(model, device)
        assert update["theta_before"] == 0 and update["theta_after"] != 0
    elif variant == "v2_diagnostic":
        with torch.no_grad():
            model.model[9].theta.fill_(-0.7)
    original = snapshot(model.state_dict())
    report = fuse_audit(model, torch.randn(1, 3, 128, 160, device=device), tmp_path)
    assert report["passed"]
    common.assert_close_tree(original, snapshot(model.state_dict()), 0, 0)
    if variant.startswith("v2"):
        assert report["binding"]["residual_budget"] == 0.05 and report["binding"]["residual_eps"] == 1e-6
        assert (report["residual_norms"]["before"]["theta"] == 0) == (variant == "v2_zero")


def test_failure_evidence_and_first_layer(tmp_path, monkeypatch):
    """An actual fused-head regression fails with original inputs, two stages, full trace and first-layer evidence."""
    model, _ = shared.build_pair()
    native_fuse = DetectionModel.fuse

    def broken_fuse(model, verbose=True):
        model = native_fuse(model, verbose=verbose)
        with torch.no_grad():
            model.model[-1].one2one_cv3[0][-1].bias.add_(0.5)
        return model

    monkeypatch.setattr(DetectionModel, "fuse", broken_fuse)
    with pytest.raises(AssertionError, match="raw.one2one.scores"):
        fuse_audit(model, torch.randn(1, 3, 128, 160), tmp_path)
    directory = next(tmp_path.glob("fuse-evidence-*"))
    report = json.loads((directory / "audit.json").read_text())
    assert not report["passed"] and "AssertionError" in report["traceback"]
    assert report["candidate_rows"][-1]["outside_tolerance"] > 0
    assert report["first_layer_outside_tolerance"] == "model.23.one2one_cv3.0.2"
    for name in ("input_rng.pt", "source.pt", "fused_state.pt", "before.pt", "after.pt", "layer_outputs.pt"):
        assert (directory / name).is_file()


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("variant", ["native", "v1", "v2_zero", "v2_updated"])
def test_strict_amp_fuse_rejects_raw_error(tmp_path, device, variant):
    """Regression for measured FP16/BF16 raw errors: test PASS means rejection, never AMP fuse equivalence."""
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    checkpoint = common.ROOT / "yolo26n.pt"
    if not checkpoint.is_file():
        pytest.skip("Requires locally supplied original checkpoint; no download")
    assert common.sha256(checkpoint) == common.PRETRAINED_SHA256
    source, _ = load_checkpoint(checkpoint, device="cpu", fuse=False)
    native, v1 = shared.build_pair(source)
    _, v2 = shared.build_pair(source, verifier.MODEL)
    model = {"native": native, "v1": v1}.get(variant, v2)
    if variant == "v2_updated":
        model, _ = verifier.updated_fixture(model, device)
    x = torch.randn(1, 3, 128, 160, device=device)
    with pytest.raises(AssertionError, match="raw.one2one.boxes"):
        fuse_audit(model, x, tmp_path, forward_amp=True)
    directory = next(tmp_path.glob("fuse-evidence-*"))
    report = json.loads((directory / "audit.json").read_text())
    assert not report["passed"] and report["stage"] == "raw_and_decode"
    assert report["first_layer_outside_tolerance"] == "model.0"
    assert report["candidate_rows"][-1]["outside_tolerance"] > 0
    for label in ("before", "after"):
        data = torch.load(directory / f"{label}.pt", weights_only=False)
        assert data["attributes"]["parameter_dtype"] == "torch.float32"  # fold was outside autocast
        assert data["autocast"]["enabled"]
        assert data["raw"]["boxes"].dtype == (torch.float16 if x.is_cuda else torch.bfloat16)


def test_source_config_survives_fuse_exception(tmp_path, monkeypatch):
    """Non-state_dict BN and residual configuration must be saved before native fuse can fail."""
    _, model = shared.build_pair(model=verifier.MODEL)
    model.model[0].bn.eps = 0.002

    def broken_fuse(model, verbose=True):
        raise RuntimeError("injected fusion failure")

    monkeypatch.setattr(DetectionModel, "fuse", broken_fuse)
    with pytest.raises(RuntimeError, match="injected"):
        fuse_audit(model, torch.randn(1, 3, 128, 160), tmp_path)
    directory = next(tmp_path.glob("fuse-evidence-*"))
    data = torch.load(directory / "source.pt", weights_only=False)
    assert data["attributes"]["bn"]["model.0.bn"]["eps"] == 0.002
    assert data["attributes"]["block_config"]["residual_budget"] == 0.05
    assert data["attributes"]["block_config"]["residual_eps"] == 1e-6


def test_partial_capture_failure_and_unique_evidence(tmp_path, monkeypatch):
    """An error inside postprocess still preserves raw/decode/top-k; attempts must never overwrite each other."""
    model, _ = shared.build_pair()
    head_type = type(model.model[-1])
    native_topk = head_type.get_topk_index

    def broken_topk(self, scores, k):
        native_topk(self, scores, k)
        raise RuntimeError("injected after topk")

    monkeypatch.setattr(head_type, "get_topk_index", broken_topk)
    for _ in range(2):
        with pytest.raises(RuntimeError, match="injected"):
            fuse_audit(model, torch.randn(1, 3, 128, 160), tmp_path)
    directories = list(tmp_path.glob("fuse-evidence-*"))
    assert len(directories) == 2
    for directory in directories:
        data = torch.load(directory / "before.pt", weights_only=False)
        assert "raw" in data and "decoded" in data and len(data["topk_stages"]) == 2
        assert "injected after topk" in (directory / "audit.json").read_text()
    assert all(not module._forward_hooks for module in model.modules())
    assert "_inference" not in model.model[-1].__dict__


def test_capture_restores_and_clones():
    """A later inference cannot mutate already recorded tensors or retain diagnostic hooks/methods."""
    _, model = shared.build_pair(model=verifier.MODEL)
    model.eval()
    result = {}
    capture(model, torch.randn(1, 3, 128, 160), result)
    saved = copy.deepcopy(result)
    capture(model, torch.randn(1, 3, 128, 160), {})
    common.assert_close_tree(saved, result, 0, 0)
    assert all(not module._forward_hooks and not module._forward_pre_hooks for module in model.modules())
    assert "_inference" not in model.model[-1].__dict__ and "get_topk_index" not in model.model[-1].__dict__


def test_partial_lifecycle_receipts(tmp_path, monkeypatch):
    """Results already collected survive exceptions in state_checks and in its extra_checks caller."""

    def fail(source, directory, device, report):
        report["module_states"].append({"fixture": "saved"})
        raise AssertionError("injected lifecycle failure")

    monkeypatch.setattr(verifier, "_state_checks", fail)
    monkeypatch.setattr(verifier, "controller_checks", lambda device: {"fixture": "saved"})
    with pytest.raises(AssertionError, match="injected"):
        verifier.extra_checks(None, tmp_path, "cpu")
    for filename in ("state_checks.json", "extra_checks.json"):
        report = json.loads((tmp_path / filename).read_text())
        assert not report["passed"] and "injected lifecycle failure" in report["traceback"]
    assert json.loads((tmp_path / "state_checks.json").read_text())["module_states"] == [{"fixture": "saved"}]


def test_archive_keeps_canonical_fuse_evidence(tmp_path):
    """Canonical provenance tensors survive packaging while unrelated transient attempts remain excluded."""
    run = tmp_path / "run"
    evidence = run / "provenance/preflight/v2_zero/fuse-evidence-fixture/input_rng.pt"
    evidence.parent.mkdir(parents=True)
    torch.save({"input": torch.zeros(1), "fixture_only": True}, evidence)
    transient = run / "diagnose.attempt.old"
    transient.mkdir()
    (transient / "discard.txt").write_text("transient fixture")
    output = tmp_path / "fixture.tar.gz"
    relative = evidence.relative_to(run).as_posix()
    archive_package(run, output, [relative])
    with tarfile.open(output) as archive:
        assert archive.extractfile("run/" + relative).read() == evidence.read_bytes()
        assert not any(".attempt." in name for name in archive.getnames())
