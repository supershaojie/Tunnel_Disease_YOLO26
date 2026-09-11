"""Regression tests for the fixed DCS formula, native graph inheritance and archive lifecycle."""

import copy
import io
import json
import subprocess
import tarfile
from pathlib import Path

import pytest
import torch

from tools.experiments import b19_common as common
from tools.experiments.finish_b19_dcs_sppf import archive_package, package
from tools.experiments.run_b19_dcs_sppf import audit_arguments
from tools.experiments.verify_b19_dcs_sppf import (
    build_pair,
    module_checks,
    module_identity,
    new_parameters,
    observe_step,
    staged_gradient_audit,
    topology_checks,
)
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.modules import DCS_SPPF, SPPF
from ultralytics.utils.torch_utils import init_seeds


def test_zero_init_and_delayed_updates():
    """Require exact native identity and nonzero gradient-backed updates, including the zero-first-step invariant."""
    report = module_checks()
    assert report["identity"][-1]["max_abs"] == report["identity"][-1]["mean_abs"] == 0


def test_zero_init_gated_residual_branch_staged_update(tmp_path):
    """Expose FP32 gamma rounding at native warmup LR while requiring every branch's task-attributable update."""
    init_seeds(42, deterministic=True)
    native, model = SPPF(32, 32, shortcut=True).train(), DCS_SPPF(32, 32, shortcut=True).train()
    model.load_state_dict(native.state_dict(), strict=False)
    x = torch.randn(32, 32, 8, 13)
    identity = module_identity(native, model, x)
    params = new_parameters(model)
    optimizer = DetectionTrainer.build_optimizer(None, model, "MuSGD", 0.01, 0.937, 0.0005)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    target = torch.randn_like(x)
    rows = []
    for i in range(64):
        for group in optimizer.param_groups:
            start = 0.1 if group["param_group"] == "bias" else 0.0
            group["lr"] = start + (0.01 - start) * i / 792
            group["momentum"] = 0.8 + (0.937 - 0.8) * i / 792
        optimizer.zero_grad(set_to_none=True)
        (model(x) - target).square().mean().backward()

        def step():
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        rows.append({"batch": i, **observe_step(model, params, optimizer, scaler, step)})
    summary = staged_gradient_audit(rows)
    assert rows[0]["parameters"]["theta"]["gradient_max_abs"] > 0
    assert summary["first_gate_unlock_batch"] == 1  # All non-bias groups have LR=0 on batch 0.
    for name, info in summary["parameters"].items():
        if name != "theta":
            assert info["first_nonzero_gradient_batch"] == 2
            assert info["first_effective_update_batch"] >= 2
        if name.endswith("bn.weight"):
            assert info["first_observable_update_batch"] is None
            assert info["effective_update_kind"] == "verified_sub_ulp_task_step"
    common.write_json(
        tmp_path / "staged_module_b32.json",
        dict(scope="local B32 module fixture, not server data", identity=identity, summary=summary, rows=rows),
    )


@pytest.mark.parametrize(
    "fault",
    [
        "disconnected",
        "zero_gradient",
        "zero_lr",
        "missing_member",
        "noop",
        "zero_gamma",
        "stale_momentum",
        "lost_state",
    ],
)
def test_staged_audit_rejects_dead_or_unapplied_parameters(fault):
    """Tiny-gradient acceptance must still reject missing chains, no-op optimizers and double-zero initialization."""
    model = DCS_SPPF(8, 8)
    params = new_parameters(model)
    optimizer = DetectionTrainer.build_optimizer(None, model, "MuSGD", 0.01, 0.937, 0.0005)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    gamma = params["fuse.bn.weight"]
    if fault == "zero_gamma":
        gamma.data.zero_()
    for p in model.parameters():
        p.grad = torch.zeros_like(p)
    params["theta"].grad.fill_(0.01)
    if fault == "missing_member":
        for group in optimizer.param_groups:
            group["params"] = [p for p in group["params"] if p is not gamma]
    if fault == "disconnected":
        gamma.grad = None
    if fault == "stale_momentum":
        optimizer.state[gamma]["momentum_buffer"] = torch.full_like(gamma, 0.1)
    if fault == "zero_lr":
        for group in optimizer.param_groups:
            group["lr"] = 0
    if fault == "lost_state":

        def lose_state():
            optimizer.step()
            optimizer.state[params["theta"]]["momentum_buffer"].zero_()

        with pytest.raises(AssertionError, match="expected task momentum"):
            observe_step(model, params, optimizer, scaler, lose_state)
        return
    if fault in {"disconnected", "missing_member", "noop"}:
        with pytest.raises(AssertionError, match="disconnected|membership|No optimizer step"):
            observe_step(model, params, optimizer, scaler, (lambda: None) if fault == "noop" else optimizer.step)
        return
    row = observe_step(model, params, optimizer, scaler, optimizer.step)
    with pytest.raises(AssertionError, match="No finite gradient|gamma initialization"):
        staged_gradient_audit([{"batch": 0, **row}])


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
        with tarfile.open(fileobj=io.BytesIO(archive.extractfile("run/source.tar").read())) as source:
            # Git archive must not apply the Windows checkout's CRLF conversion to portable shell scripts.
            name = "tools/experiments/server_b19_dcs_sppf_v1.sh"
            assert source.extractfile(name).read() == subprocess.check_output(
                ["git", "-c", f"safe.directory={common.ROOT.as_posix()}", "show", f"HEAD:{name}"], cwd=common.ROOT
            )
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
