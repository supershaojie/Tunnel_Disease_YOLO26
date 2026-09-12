"""Regressions for ordered LBI interaction, native parser/loading and auditable experiment lifecycle."""

import io
import copy
import json
import subprocess
import tarfile
from pathlib import Path

import pytest
import torch

from tools.experiments import b19_common as common
from tools.experiments.finish_b19_lbi_fusion import archive_package, package
from tools.experiments.lbi_fuse_audit import audit_candidates, capture, fuse_audit, snapshot
from tools.experiments.run_b19_lbi_fusion import audit_arguments, require_runtime
from tools.experiments.verify_b19_lbi_fusion import (
    FUSE_ATOL,
    FUSE_RTOL,
    build_pair,
    module_checks,
    staged_gradient_audit,
    topology_checks,
    whole_identity,
)
from ultralytics.nn.modules import Concat_LBI_Fusion
from ultralytics.nn.tasks import DetectionModel


@pytest.fixture(autouse=True)
def small_cpu_pool():
    """Bound CPU test overhead without changing production threading or precision settings."""
    previous = torch.get_num_threads()
    torch.set_num_threads(4)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("device,amp", [("cpu", False), ("cuda:0", False), ("cuda:0", True)])
def test_module_formula_and_staged_updates(device, amp):
    """Check hand-derived channel RMS and nonzero task updates after output projection unlock."""
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    result = module_checks(device, amp)
    assert result["identity"][0]["max_abs"] == 0


def test_order_noncontiguous_and_shared_inputs():
    """S stays first and untouched; L is never mutated even with a nonzero residual and shared views."""
    block = Concat_LBI_Fusion([3, 5])
    semantic = torch.full((1, 3, 17, 13), 3.0).transpose(2, 3)
    detail = torch.full((1, 5, 17, 13), 7.0).transpose(2, 3)
    assert not semantic.is_contiguous()
    assert torch.equal(block([semantic, detail]), torch.cat([semantic, detail], 1))
    with torch.no_grad():
        block.out.weight.fill_(0.2)
    before = [semantic.clone(), detail.clone()]
    output = block([semantic, detail])
    assert torch.equal(output[:, :3], semantic) and torch.equal(semantic, before[0]) and torch.equal(detail, before[1])
    assert output.shape == (1, 8, 13, 17) and torch.isfinite(output).all()
    with pytest.raises(ValueError, match="channels"):
        block([detail, semantic])
    # Equal channel counts cannot reveal semantic origins from tensor metadata; graph f=[-1,4] is authoritative.
    equal = Concat_LBI_Fusion([3, 3])
    assert not torch.equal(equal([semantic, semantic + 1]), equal([semantic + 1, semantic]))


@pytest.mark.parametrize(
    "inputs,message",
    [
        ([], "exactly two"),
        ([torch.zeros(1, 3, 4, 5)], "exactly two"),
        ([torch.zeros(1, 3, 4, 5)] * 3, "exactly two"),
        ([torch.zeros(1, 3, 4), torch.zeros(1, 5, 4, 5)], "NCHW"),
        ([torch.zeros(2, 3, 4, 5), torch.zeros(1, 5, 4, 5)], "batch and spatial"),
        ([torch.zeros(1, 3, 4, 5), torch.zeros(1, 5, 5, 4)], "batch and spatial"),
        ([torch.zeros(1, 5, 4, 5), torch.zeros(1, 3, 4, 5)], "channels"),
        ([torch.zeros(1, 3, 4, 5, dtype=torch.int64), torch.zeros(1, 5, 4, 5, dtype=torch.int64)], "floating dtype"),
    ],
)
def test_invalid_inputs(inputs, message):
    """Reject metadata errors at the module interface rather than interpolating, cropping or reordering."""
    with pytest.raises(ValueError, match=message):
        Concat_LBI_Fusion([3, 5])(inputs)


def test_invalid_device():
    """Report differing input/module devices before any convolution starts."""
    with pytest.raises(ValueError, match="devices"):
        Concat_LBI_Fusion([3, 5])([torch.empty(1, 3, 4, 5, device="meta"), torch.zeros(1, 5, 4, 5)])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_native_amp_mixed_input_dtypes():
    """Native AMP can produce FP32 S and FP16 L; preserve native cat promotion and never coerce S."""
    module = Concat_LBI_Fusion([128, 128]).cuda()
    semantic = torch.randn(2, 128, 13, 17, device="cuda", dtype=torch.float32)
    detail = torch.randn(2, 128, 13, 17, device="cuda", dtype=torch.float16)
    with torch.autocast("cuda"):
        y = module([semantic, detail])
        expected = torch.cat([semantic, detail], 1)
    assert y.dtype == expected.dtype == torch.float32 and torch.equal(y, expected)


@pytest.mark.parametrize("kwargs", [dict(rank=8), dict(eps=1e-5), dict(dim=0)])
def test_fixed_design(kwargs):
    """The v1 interface cannot silently enable alternative ranks, normalization or channel layouts."""
    with pytest.raises(ValueError, match="fixes"):
        Concat_LBI_Fusion([128, 128], **kwargs)


def test_parameter_budget_rng_and_nonzero_reload():
    """Only four ordinary bias-free conv weights are added, with no subsequent RNG drift or reset on reload."""
    torch.manual_seed(42)
    state = torch.get_rng_state().clone()
    block = Concat_LBI_Fusion([128, 128])
    assert torch.equal(state, torch.get_rng_state())
    assert len(list(block.parameters())) == 4 and sum(p.numel() for p in block.parameters()) == 6288
    assert not list(block.buffers())
    for layer in (block.proj_l, block.proj_s, block.dw, block.out):
        assert type(layer) is torch.nn.Conv2d and layer.bias is None
    assert block.dw.groups == 16 and block.dw.kernel_size == (3, 3)
    assert block.dw.padding == block.dw.stride == block.dw.dilation == (1, 1)
    assert block.dw.padding_mode == "zeros"
    assert not block.out.weight.count_nonzero()
    with torch.no_grad():
        block.out.weight.normal_()
    clone = Concat_LBI_Fusion([128, 128])
    clone.load_state_dict(block.state_dict())
    assert torch.equal(clone.out.weight, block.out.weight)


def test_parser_topology_and_native_shared_identity():
    """All 24 parsed layers and both raw Detect branches remain b19-equivalent at step zero."""
    baseline, candidate = build_pair()
    assert topology_checks(baseline, candidate)["changed_layers"] == [15]
    audit = common.audit_weights(baseline, candidate, None)
    assert audit["exact_matches"] == audit["common_keys"]
    whole_identity(baseline, candidate, "cpu", size=96)


def test_recipe_rejects_nonidentity_overrides():
    """No tweak to patience, batch, AMP, optimizer or augmentation is accepted as an identity difference."""
    raw = common.REFERENCE["args"]
    for name, value in dict(patience=0, batch=16, amp=False, optimizer="SGD", mosaic=0.0).items():
        with pytest.raises(ValueError, match="Non-identity"):
            audit_arguments(raw, {**raw, name: value})
    assert set(audit_arguments(raw, {**raw, "name": "lbi"})) == {"name"}


def test_local_environment_cannot_authorize_server():
    """Local runtime results cannot substitute for the exact b19 server environment."""
    if torch.__version__ != common.REFERENCE["environment"]["torch"]:
        with pytest.raises(RuntimeError, match="environment mismatch"):
            require_runtime()


def test_staged_audit_does_not_accept_decay_or_grad_objects():
    """Finite grad objects and weight decay motion alone never count as task-driven effective updates."""
    with pytest.raises(AssertionError):
        staged_gradient_audit([])
    rows = [
        dict(
            batch=0,
            out_zero_before=True,
            out_zero_after=True,
            parameters={
                k: dict(unscaled_before_clip=dict(finite=True, max_abs=0), effective_update=False)
                for k in ("out.weight", "proj_l.weight", "proj_s.weight", "dw.weight")
            },
        )
    ]
    assert len(staged_gradient_audit(rows, False)["missing"]) == 4
    with pytest.raises(AssertionError, match="Missing gradient-backed"):
        staged_gradient_audit(rows)


def test_archive_dry_run(tmp_path):
    """Round-trip a minimal fixture archive including the committed LBI implementation and its dependencies."""
    run = tmp_path / "fixture"
    (run / "weights").mkdir(parents=True)
    for name in ("weights/best.pt", "weights/last.pt", "args.yaml"):
        (run / name).write_bytes(b"LOCAL FIXTURE ONLY; no training results")
    evidence = run / "provenance/preflight/cuda_0/fuse_lbi_nonzero/before.pt"
    evidence.parent.mkdir(parents=True)
    torch.save(candidate_fixture([0.9, 0.8, 0.7, 0.1], [0, 1]), evidence)
    (run / "args_hard.txt").hardlink_to(run / "args.yaml")
    (run / "failed.attempt.1").mkdir()
    (run / "failed.attempt.1/console.log").write_text("excluded")
    result = archive_package(run, tmp_path / "dry_run.tar.gz", ["weights/best.pt", "weights/last.pt", "args.yaml"])
    assert result["gzip_crc_verified"]
    with tarfile.open(result["path"]) as archive:
        assert not any(".attempt." in n for n in archive.getnames())
        assert archive.getmember("run/args_hard.txt").isfile()
        assert archive.extractfile("run/args_hard.txt").read() == (run / "args.yaml").read_bytes()
        assert archive.extractfile("run/" + evidence.relative_to(run).as_posix()).read() == evidence.read_bytes()
        with tarfile.open(fileobj=io.BytesIO(archive.extractfile("run/source.tar").read())) as source:
            required = [
                "ultralytics/nn/modules/lbi_fusion.py",
                "tools/experiments/b19_common.py",
                "tools/experiments/run_b19_lbi_fusion.py",
                "tools/experiments/verify_b19_lbi_fusion.py",
                "tools/experiments/lbi_fuse_audit.py",
                "tools/experiments/replay_lbi_fuse_precision.py",
                "tools/experiments/diagnose_b19_lbi_fusion.py",
                "tools/experiments/finish_b19_lbi_fusion.py",
            ]
            assert all(n in source.getnames() for n in required), "Package dry-run requires committed LBI sources"
            shell = source.extractfile("tools/experiments/server_b19_lbi_fusion_v1.sh").read()
            assert b"\r\n" not in shell
    common.write_json(common.ROOT / "runs/lbi_local/package_dry_run.json", {**result, "fixture_only": True})
    with pytest.raises(ValueError):
        archive_package(run, Path(result["path"]), [])
    with pytest.raises(FileNotFoundError):
        archive_package(run, tmp_path / "missing.tar.gz", ["results.csv"])


def test_archive_real_links(tmp_path):
    """Exercise actual link handling where supported; never substitute a mock Linux symlink claim."""
    run = tmp_path / "run"
    run.mkdir()
    (run / "regular.txt").write_text("data")
    (run / "hard.txt").hardlink_to(run / "regular.txt")
    try:
        (run / "dangling.log").symlink_to(run / "absent.attempt.1/console.log")
    except OSError:
        pytest.skip("Windows does not grant real symlink creation; Linux symlink integration NOT RUN")
    result = archive_package(run, tmp_path / "links.tar.gz", ["regular.txt", "hard.txt"])
    with tarfile.open(result["path"]) as archive:
        assert archive.getmember("run/hard.txt").isfile()
        assert "run/dangling.log" not in archive.getnames()
        excluded = json.load(archive.extractfile("run/package_exclusions.json"))
        assert any(x["path"] == "dangling.log" for x in excluded["excluded"])


def test_package_rejects_incomplete_formal_results(tmp_path):
    """No final experiment package may be produced from fixture weights or missing evaluation evidence."""
    with pytest.raises(FileNotFoundError):
        package(tmp_path, tmp_path / "invalid.tar.gz", {})


def test_source_core_unchanged():
    """Training math, losses, dependencies and all native block implementations remain canonical b19."""
    paths = [
        "ultralytics/engine/trainer.py",
        "ultralytics/utils/loss.py",
        "ultralytics/optim/muon.py",
        "ultralytics/nn/modules/block.py",
        "ultralytics/nn/modules/conv.py",
        "ultralytics/nn/modules/head.py",
        "ultralytics/utils/torch_utils.py",
        "ultralytics/cfg/default.yaml",
        "pyproject.toml",
    ]
    for name in paths:
        original = subprocess.check_output(
            [
                "git",
                "-c",
                f"safe.directory={common.ROOT.as_posix()}",
                "show",
                f"{common.REFERENCE['source_commit']}:{name}",
            ],
            cwd=common.ROOT,
        )
        assert original.replace(b"\r\n", b"\n") == (common.ROOT / name).read_bytes().replace(b"\r\n", b"\n")


@pytest.mark.parametrize("nonfinite,expected_batches", [(True, 1), (False, 64)])
def test_preflight_failure_receipts_and_fixed_bound(tmp_path, monkeypatch, nonfinite, expected_batches):
    """Test only callback failure ownership with a stub; this is not native trainer or server validation."""
    from tools.experiments import verify_b19_lbi_fusion as verifier

    class CallbackDriver:
        def __init__(self, config):
            self.callbacks = {}
            self.loss = torch.tensor(float("nan") if nonfinite else 1.0)
            self.audit_images, self.accumulate = ["fixture-only"], 1

        def add_callback(self, name, callback):
            self.callbacks[name] = callback

        def train(self):
            for _ in range(65):
                self.callbacks["on_train_batch_start"](self)
                self.callbacks["on_train_batch_end"](self)

    monkeypatch.setattr(verifier, "PreflightTrainer", CallbackDriver)
    with pytest.raises(AssertionError):
        verifier.native_preflight({}, tmp_path)
    batches = json.loads((tmp_path / "native_batches.json").read_text())
    assert len(batches) == expected_batches and batches[-1]["batch"] == expected_batches - 1
    assert batches[-1]["loss"] == ("nan" if nonfinite else 1.0)
    assert (tmp_path / "native_steps.json").is_file() and (tmp_path / "native_gradient_summary.json").is_file()


def candidate_fixture(scores, ids):
    """Represent a single-class postprocess with explicit grid identities, including off-screen boxes."""
    scores = torch.tensor(scores, dtype=torch.float32).view(1, -1, 1)
    boxes = (torch.arange(scores.numel() * 4).float() * 100).reshape(1, -1, 4)
    decoded = torch.cat((boxes, scores), -1)
    indices = torch.tensor(ids, dtype=torch.int64).view(1, -1, 1)
    classes = torch.zeros_like(indices, dtype=torch.float32)
    selected = decoded.gather(1, indices.expand(-1, -1, 5))
    return dict(
        raw=dict(boxes=boxes.transpose(1, 2), scores=scores.logit().transpose(1, 2)),
        decoded=decoded,
        indices=indices,
        classes=classes,
        final=torch.cat((selected, classes), -1),
        selected_scores=selected[..., 4:5].clone(),
    )


@pytest.mark.parametrize("boundary,tied", [(False, False), (False, True), (True, False), (True, True)])
def test_fuse_candidate_permutation_and_boundary(boundary, tied):
    """Large positional box errors may pass only after all candidates and actual selection remain valid."""
    before = candidate_fixture([0.8, 0.5, 0.5 if tied else 0.500001, 0.1], [0, 2] if boundary else [0, 2, 1])
    after = candidate_fixture([0.8, 0.5 if tied else 0.500002, 0.5, 0.1], [0, 1] if boundary else [0, 1, 2])
    report = {}
    audit_candidates(before, after, 2 if boundary else 3, FUSE_ATOL, FUSE_RTOL, report)
    assert not report["positional"]["passed"] and report["stage"] == "complete"
    batch = report["batches"][0]
    assert batch["same_set"] != boundary
    assert batch["compared_union"] == 3
    if boundary:
        assert batch["dropped"] == [2] and batch["added"] == [1]
        assert batch["max_gap_minus_measured_budget"] <= 0
        assert len(batch["boundary"]) == 2


@pytest.mark.parametrize(
    "corruption", ["raw_box", "decoded_box", "score", "omit_higher", "gather", "class", "duplicate"]
)
def test_fuse_candidate_negative_controls(corruption):
    """Reject real same-grid errors and broken selection even when both final tables look identical."""
    before = candidate_fixture([0.9, 0.8, 0.7, 0.1], [0, 1])
    after = copy.deepcopy(before)
    if corruption == "raw_box":
        after["raw"]["boxes"][0, 0, 2] += 100  # outside the selected intersection, still audited
    elif corruption == "decoded_box":
        after["decoded"][0, 2, 0] += 100
    elif corruption == "score":
        after = candidate_fixture([0.9, 0.8, 0.75, 0.1], [0, 1])
    elif corruption == "omit_higher":
        before = candidate_fixture([0.9, 0.8, 0.7, 0.1], [0, 2])
        after = copy.deepcopy(before)
    elif corruption == "gather":
        after["final"][0, 0, 0] += 100
    elif corruption == "class":
        after["classes"][0, 0, 0] = 1
    else:
        before = candidate_fixture([0.9, 0.8, 0.7, 0.1], [0, 0])
        after = copy.deepcopy(before)
    with pytest.raises(AssertionError):
        audit_candidates(before, after, 2, FUSE_ATOL, FUSE_RTOL, {})


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("variant", ["native", "zero", "nonzero"])
def test_real_fuse_candidates(tmp_path, device, variant):
    """Run native Conv-BN fuse with both zero and learned-path LBI projections and preserve CPU evidence."""
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    baseline, candidate = build_pair()
    if variant == "nonzero":
        with torch.no_grad():
            candidate.model[15].out.weight.normal_(0, 0.01)
    model = baseline if variant == "native" else candidate
    original = snapshot(model.state_dict())
    result = fuse_audit(model, torch.randn(1, 3, 160, 160, device=device), tmp_path / "audit", FUSE_ATOL, FUSE_RTOL)
    assert result["passed"] and result["stage"] == "complete"
    common.assert_close_tree(original, snapshot(model.state_dict()), 0, 0)
    if variant != "native":
        assert bool(result["lbi_out_nonzero"]) == (variant == "nonzero")
    saved = torch.load(tmp_path / "audit/after.pt", weights_only=False)
    assert saved["indices"].device.type == saved["raw"]["boxes"].device.type == "cpu"
    assert not saved["one2many_keys"]


def test_fuse_failure_evidence_and_localization(tmp_path, monkeypatch):
    """A deliberate real head error must fail with pre-assert tensors, traceback and per-layer localization."""
    baseline, _ = build_pair()
    native_fuse = DetectionModel.fuse

    def broken_fuse(model, verbose=True):
        model = native_fuse(model, verbose=verbose)
        with torch.no_grad():
            model.model[-1].one2one_cv3[0][-1].bias.add_(0.5)
        return model

    monkeypatch.setattr(DetectionModel, "fuse", broken_fuse)
    directory = tmp_path / "broken"
    with pytest.raises(AssertionError, match="raw.one2one.scores"):
        fuse_audit(baseline, torch.randn(1, 3, 160, 160), directory, FUSE_ATOL, FUSE_RTOL)
    report = json.loads((directory / "audit.json").read_text())
    assert not report["passed"] and "AssertionError" in report["traceback"]
    assert report["candidate_rows"][-1]["outside_tolerance"] > 0
    assert report["first_layer_outside_tolerance"] == "model.23.one2one_cv3.0.2"
    for name in ("source.pt", "fused_state.pt", "before.pt", "after.pt", "layer_outputs.pt"):
        assert (directory / name).is_file()


def test_diagnostic_capture_restores_methods_and_clones():
    """Observers neither persist on models nor retain output aliases that a later inference could mutate."""
    baseline, _ = build_pair()
    baseline.eval()
    head = baseline.model[-1]
    result = capture(baseline, torch.randn(1, 3, 160, 160))
    saved = copy.deepcopy(result)
    assert "_inference" not in head.__dict__ and "get_topk_index" not in head.__dict__
    capture(baseline, torch.randn(1, 3, 160, 160))
    common.assert_close_tree(saved, result, 0, 0)


def test_diagnostic_capture_restores_on_failure(monkeypatch):
    """An exception inside production postprocess cannot leave diagnostic methods or hooks installed."""
    baseline, _ = build_pair()
    baseline.eval()
    head = baseline.model[-1]

    def failed_postprocess(predictions):
        raise RuntimeError("injected postprocess failure")

    monkeypatch.setattr(head, "postprocess", failed_postprocess)
    with pytest.raises(RuntimeError, match="injected"):
        capture(baseline, torch.randn(1, 3, 160, 160), {})
    assert "_inference" not in head.__dict__ and "get_topk_index" not in head.__dict__
    assert all(not m._forward_hooks for m in baseline.modules())


def test_lifecycle_preserves_partial_receipt(tmp_path, monkeypatch):
    """Test receipt ownership alone: accumulated results survive an exception before lifecycle return."""
    from tools.experiments import verify_b19_lbi_fusion as verifier

    def failed_lifecycle(baseline, candidate, directory, device, report):
        report["rows"].append(dict(path="fixture_only", outside_tolerance=1))
        raise AssertionError("injected lifecycle failure")

    monkeypatch.setattr(verifier, "_lifecycle_checks", failed_lifecycle)
    with pytest.raises(AssertionError, match="injected"):
        verifier.lifecycle_checks(None, None, tmp_path)
    report = json.loads((tmp_path / "lifecycle_checks.json").read_text())
    assert not report["passed"] and report["rows"][0]["outside_tolerance"] == 1
    assert "injected lifecycle failure" in report["traceback"]
