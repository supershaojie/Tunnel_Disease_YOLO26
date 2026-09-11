"""Regression tests for the fixed DCS formula, native graph inheritance and archive lifecycle."""

import copy
import json
import subprocess
import tarfile
from pathlib import Path

import pytest
import torch

from tools.experiments import b19_common as common
from tools.experiments.finish_b19_dcs_sppf import archive_package, package
from tools.experiments.run_b19_dcs_sppf import audit_arguments
from tools.experiments.verify_b19_dcs_sppf import build_pair, module_checks, module_identity, topology_checks
from ultralytics.nn.modules import DCS_SPPF, SPPF


def test_zero_init_and_delayed_updates():
    """Require exact native identity and nonzero gradient-backed updates, including the zero-first-step invariant."""
    report = module_checks()
    assert report["identity"][-1]["max_abs"] == report["identity"][-1]["mean_abs"] == 0


@pytest.mark.parametrize("channels,shortcut", [((32, 32), True), ((32, 48), True), ((32, 32), False)])
def test_native_path_and_small_boundaries(channels, shortcut):
    """Cover unequal channels, shortcut disabled and windows larger than the input with exact default AvgPool padding."""
    native = SPPF(*channels, shortcut=shortcut).eval()
    model = DCS_SPPF(*channels, shortcut=shortcut).eval()
    model.load_state_dict(native.state_dict(), strict=False)
    x = torch.randn(2, channels[0], 4, 7)
    module_identity(native, model, x)
    with torch.no_grad():
        model.theta.fill_(-0.4)
        z = native.cv1(x)
        refinements = []
        for k, average, branch in zip((5, 9, 13), model.avg, model.refine):
            assert average.count_include_pad
            contrast = torch.nn.functional.max_pool2d(z, k, 1, k // 2) - torch.nn.functional.avg_pool2d(z, k, 1, k // 2)
            refinements.append(branch(contrast))
        expected = native(x) + (0.1 * model.theta.tanh()) * model.fuse(torch.cat(refinements, 1))
        torch.testing.assert_close(model(x), expected, rtol=0, atol=0)


@pytest.mark.parametrize("k,n", [(3, 3), (5, 2), (7, 4)])
def test_fixed_design(k, n):
    """Reject unsupported pooling variants at construction, before any partial model is registered."""
    with pytest.raises(ValueError, match="k=5 and n=3"):
        DCS_SPPF(32, 32, k, n)


def test_graph_shared_initialization_and_reproducibility():
    """Require all shared state to match and detect a changed neck connection without a pretrained download."""
    baseline, candidate = build_pair()
    topology_checks(baseline, candidate)
    report = common.audit_weights(baseline, candidate, None)
    assert report["common_keys"] == 708 and report["added_parameters"] == 103041
    _, rebuilt = build_pair()
    common.assert_close_tree(candidate.state_dict(), rebuilt.state_dict(), 0, 0)
    candidate.yaml = copy.deepcopy(candidate.yaml)
    candidate.yaml["head"][10][0] = [-1, 9]
    with pytest.raises(AssertionError):
        topology_checks(baseline, candidate)


@pytest.mark.parametrize(
    "key", ["batch", "amp", "lr0", "warmup_epochs", "hsv_h", "mosaic", "mixup", "cutmix", "close_mosaic"]
)
def test_recipe_mutation_is_rejected(key):
    """Prevent training or augmentation edits from being classified as portable path changes."""
    original = common.REFERENCE["args"]
    changed = {**original, key: None}
    with pytest.raises(ValueError, match="Non-identity"):
        audit_arguments(original, changed)


def test_archive_ignores_attempts_and_console_aliases(tmp_path, monkeypatch):
    """Exercise the real tar/hash/CRC path with missing console.log, transient attempts and a dangling link."""
    run = tmp_path / "run"
    (run / "weights").mkdir(parents=True)
    for name in ("best.pt", "last.pt"):
        (run / "weights" / name).write_bytes(b"dry-run fixture, not trained weights\n")
    (run / "args.yaml").write_text("epochs: 200\n")
    (run / "failed.attempt.123").mkdir()
    (run / "failed.attempt.123/console.log").write_text("must not be archived")
    alias = run / "console.log"
    try:
        alias.symlink_to(run / "absent.attempt.1/console.log")
        real_link = True
    except OSError:
        # Windows without symlink privilege: exercise exactly the archive's symlink predicate.
        alias.write_text("emulated dangling console alias")
        original = Path.is_symlink
        monkeypatch.setattr(Path, "is_symlink", lambda path: path == alias or original(path))
        real_link = False
    output = tmp_path / "package.tar.gz"
    result = archive_package(run, output, ["weights/best.pt", "weights/last.pt", "args.yaml"])
    assert result["gzip_crc_verified"]
    assert (run / "source.patch").read_bytes() == subprocess.check_output(
        [
            "git",
            "-c",
            f"safe.directory={common.ROOT.as_posix()}",
            "diff",
            "--binary",
            common.REFERENCE["source_commit"],
            "HEAD",
        ],
        cwd=common.ROOT,
    )
    with tarfile.open(output) as archive:
        names = archive.getnames()
        assert "run/weights/best.pt" in names and "run/weights/last.pt" in names
        assert not any(".attempt." in n or n.endswith("console.log") for n in names)
    (tmp_path / "dry_run_result.json").write_text(json.dumps({**result, "real_dangling_symlink": real_link}))
    with pytest.raises(ValueError, match="new file"):
        archive_package(run, output, [])


def test_package_requires_completed_evidence(tmp_path):
    """Never publish a final experiment archive from only checkpoints or an unsuccessful stage."""
    with pytest.raises(FileNotFoundError):
        package(tmp_path, tmp_path.parent / "not-created.tar.gz", {})


def test_package_rejects_mismatched_evaluation(tmp_path):
    """Do not archive a val receipt from different weights even when comparison and diagnosis identify the current run."""
    evidence = {"commit": "current", "weight_sha256": "current-weight"}
    common.write_json(tmp_path / "baseline_comparison/comparison.json", {"evidence": evidence})
    common.write_json(
        tmp_path / "baseline_comparison/diagnostics/dcs_sppf_diagnostics.json",
        {**evidence, "evidence": evidence, "artifacts": {}},
    )
    common.write_json(
        tmp_path / "baseline_comparison/dcs_val/metrics.json",
        {"evidence": {**evidence, "weight_sha256": "wrong-weight", "split": "val"}},
    )
    with pytest.raises(AssertionError, match="Stale evaluation receipt"):
        package(tmp_path, tmp_path.parent / "not-created.tar.gz", evidence)
