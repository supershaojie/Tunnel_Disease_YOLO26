"""Fixed SICR mathematics, native lifecycle compatibility and fail-closed experiment boundaries."""

import copy
import os
import subprocess
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tools.experiments import b19_common as common
from tools.experiments.diagnose_b19_sicr_sppf import stage_statistics
from tools.experiments.finish_b19_sicr_sppf import archive_package
from tools.experiments.run_b19_sicr_sppf import AuditedTrainer, audit_arguments, audit_training_setup
from tools.experiments.verify_b19_sicr_sppf import (
    build_pair,
    model_checks,
    module_checks,
    module_identity,
    reload_in_process,
    save_reload_check,
    topology_checks,
)
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.nn.modules import SICRSPPF, SPPF
from ultralytics.nn.tasks import load_checkpoint
from ultralytics.utils.torch_utils import ModelEMA


@pytest.fixture(autouse=True)
def cpu_threads():
    """Avoid excessive CPU threading for small synthetic tensors."""
    previous = torch.get_num_threads()
    torch.set_num_threads(4)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("shape,shortcut", [((2, 32, 20, 20), True), ((2, 32, 9, 27), False), ((2, 32, 1, 1), True)])
def test_native_identity(shape, shortcut):
    """Preserve forward and input gradient at zero alpha, including spatial boundaries and shortcut off."""
    torch.manual_seed(42)
    baseline = SPPF(32, 32, 5, 3, shortcut).eval()
    candidate = SICRSPPF(32, 32, 5, 3, shortcut).eval()
    candidate.load_state_dict(baseline.state_dict(), strict=False)
    x = torch.randn(shape, requires_grad=True)
    native, new = baseline(x), candidate(x)
    torch.testing.assert_close(new, native, rtol=1e-5, atol=1e-6)
    a = torch.autograd.grad(native.sum(), x)[0]
    b = torch.autograd.grad(new.sum(), x)[0]
    torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)


def test_formula_gradients_and_bound():
    """Check the prescribed formula, signed bounds and delayed branch gradients."""
    assert module_checks()["formula"] == "PASS"
    for kwargs in ({"n": 2}, {"k": 3}, {"alpha_max": 0.2}):
        with pytest.raises(ValueError, match="requires"):
            SICRSPPF(32, 32, **kwargs)
    for k, branch in zip((3, 5, 7), SICRSPPF(32, 32).refine):
        assert branch[0].conv.kernel_size == (1, k) and branch[0].conv.groups == 16
        assert branch[1].conv.kernel_size == (k, 1) and branch[1].conv.groups == 16
        assert isinstance(branch[2].act, torch.nn.Identity)


def test_topology_rng_and_pretrained():
    """Require full common-state equality, native class remapping and no additional graph innovations."""
    source, _ = load_checkpoint(common.ROOT / "yolo26n.pt")
    baseline, candidate = build_pair(source)
    topology_checks(baseline, candidate)
    report = common.audit_weights(baseline, candidate, source)
    assert report["matched_tensors"] == 606
    assert len(report["shared_tensors"]) == 708 and len(report["expected_new_state"]) == 55
    assert report["missing_shared_keys"] == report["unexpected_keys"] == []
    assert report["layer9_cv1_loaded"] and report["layer9_cv2_loaded"]
    candidate.model[10].cv1.conv.weight.data.add_(1)
    with pytest.raises(AssertionError, match="Common baseline initialization"):
        common.audit_weights(baseline, candidate, source)


def test_recipe_rejects_every_nonidentity_field():
    """Changing any training value, including defaults and introducing new fields, must fail."""
    raw = copy.deepcopy(common.REFERENCE["args"])
    allowed = {"model", "pretrained", "data", "project", "name", "save_dir"}
    for key in raw.keys() - allowed:
        with pytest.raises(ValueError, match="Non-identity"):
            audit_arguments(raw, {**raw, key: "changed"})
    with pytest.raises(ValueError, match="Non-identity"):
        audit_arguments(raw, {**raw, "unknown_training_option": True})


def test_native_musgd_and_ema():
    """Confirm theta and every branch join the native optimizer and can update before EMA/checkpoint use."""
    torch.manual_seed(42)
    model = SICRSPPF(32, 32, shortcut=True).train()
    optimizer = BaseTrainer.build_optimizer(SimpleNamespace(), model, "MuSGD", 0.01, 0.937, 0.0005)
    ids = [id(p) for g in optimizer.param_groups for p in g["params"]]
    assert len(ids) == len(set(ids)) == len(list(model.parameters()))
    ema = ModelEMA(model)
    x, target = torch.randn(2, 32, 20, 20), torch.randn(2, 32, 20, 20)
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        (model(x) - target).square().mean().backward()
        assert torch.isfinite(model.theta.grad).all() and model.theta.grad.count_nonzero() == 3
        if step:
            assert all(sum(p.grad.abs().sum() for p in branch.parameters()) > 0 for branch in model.refine)
        optimizer.step()
        ema.update(model)
    assert ema.ema.theta.count_nonzero() == 3
    assert (0.10 * ema.ema.theta.tanh()).abs().max() <= 0.10


def test_checkpoint_fresh_process(tmp_path):
    """Require exact nonzero snapshot restoration, and prove state/cache/raw corruption fails closed."""
    _, model = build_pair()
    model.model[9].theta.data.copy_(torch.tensor([0.1, -0.2, 0.3]))
    model.args = common.REFERENCE["args"].copy()
    state = copy.deepcopy(model.state_dict())
    report = save_reload_check(model, tmp_path, torch.randn(1, 3, 64, 96))
    assert report["passed"] and report["state_exact"] and report["attributes_exact"]
    assert all(row["max_abs"] == 0 for row in report["raw"])
    common.assert_close_tree(state, model.state_dict(), 0, 0)
    path = Path(report["checkpoint"])
    reference_path = path.with_name("reload_reference.pt")
    reference = torch.load(reference_path, weights_only=False)
    for field, key in (("state", "model.10.cv1.conv.weight"), ("attributes_before", "model.23"), ("raw", None)):
        corrupt = copy.deepcopy(reference)
        if field == "state":
            corrupt[field][key].add_(1)
        elif field == "attributes_before":
            corrupt[field][key]["stride"].add_(1)
        else:
            corrupt[field][1]["one2one"]["boxes"].add_(1)
        torch.save(corrupt, reference_path)
        with pytest.raises(AssertionError, match=f"reload.{field}"):
            reload_in_process(path)


def test_model_checks_640(tmp_path):
    """Run the production preflight's complete model/lifecycle path, including native fusion control."""
    source, _ = load_checkpoint(common.ROOT / "yolo26n.pt")
    report = model_checks(source, directory=tmp_path)
    assert report["module_at_layer9"]["first_difference"] is None
    assert report["reload"]["passed"] and report["deployment"]["cpu_fusion"]
    for name in ("native_fusion", "sicr_fusion"):
        assert all(row["atol"] == row["rtol"] == 1e-4 for row in report["reload"][name]["raw_one2one_errors"])


def test_module_audit_locates_first_difference():
    """Catch altered shared state and a faulty pooling stage before downstream Detect can hide them."""
    native, candidate = SPPF(32, 32).eval(), SICRSPPF(32, 32).eval()
    candidate.load_state_dict(native.state_dict(), strict=False)
    x = torch.randn(1, 32, 20, 20)
    candidate.cv1.bn.running_mean.add_(1)
    with pytest.raises(AssertionError, match="cv1.bn.running_mean"):
        module_identity(native, candidate, x)
    candidate.load_state_dict(native.state_dict(), strict=False)
    handle = candidate.m.register_forward_hook(lambda m, ins, out: out + 1)
    try:
        with pytest.raises(AssertionError, match="Z1"):
            module_identity(native, candidate, x)
    finally:
        handle.remove()


def test_oom_refuses_batch_mutation():
    """Re-raise at the native retry boundary before the native loop halves its batch."""
    trainer = object.__new__(AuditedTrainer)
    trainer.batch_size = 32
    trainer._oom_retries = 0
    error = torch.cuda.OutOfMemoryError("synthetic OOM")
    with pytest.raises(torch.cuda.OutOfMemoryError) as caught:
        try:
            raise error
        except torch.cuda.OutOfMemoryError:
            trainer._oom_retries += 1
            trainer.batch_size //= 2
    assert caught.value is error and trainer.batch_size == 32


def test_failed_preflight_never_constructs_trainer(tmp_path, monkeypatch):
    """A failed child audit must stop the actual train entry before any training model or run is claimed."""
    from tools.experiments import run_b19_sicr_sppf as run

    stages = []
    monkeypatch.setattr(sys, "argv", ["run_b19_sicr_sppf.py", "--project", str(tmp_path)])
    monkeypatch.setattr(common, "require_clean_source", lambda: None)
    monkeypatch.setattr(run, "require_runtime", lambda: stages.append("runtime"))
    recipe = common.REFERENCE["args"].copy()
    monkeypatch.setattr(
        common, "resolve_recipe", lambda args: (recipe, recipe, {"args_path": "b19.yaml", "initial_path": "yolo26n.pt"})
    )
    monkeypatch.setattr(common, "launcher_evidence", lambda *args: {})
    monkeypatch.setattr(common, "source_hashes", lambda: {})

    def fail_preflight(command, **kwargs):
        stages.append("preflight")
        assert kwargs["check"] is True and command[1].endswith("verify_b19_sicr_sppf.py")
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(run.subprocess, "run", fail_preflight)
    monkeypatch.setattr(run, "AuditedTrainer", lambda *args: pytest.fail("Training started after failed preflight"))
    with pytest.raises(subprocess.CalledProcessError):
        run.main()
    assert stages == ["runtime", "preflight"]
    assert not list(tmp_path.iterdir())


def test_actual_trainer_reconstruction_and_setup_callback(tmp_path):
    """Exercise production get_model and setup callback using the real native optimizer builder."""
    trainer = object.__new__(AuditedTrainer)
    trainer.data = {"nc": 1, "names": {0: "crack"}, "channels": 3}
    trainer.args = common.get_cfg(overrides={k: v for k, v in common.REFERENCE["args"].items() if k != "save_dir"})
    trainer.expected_args = vars(trainer.args).copy()
    trainer.save_dir = tmp_path
    trainer.batch_size, trainer.amp = 32, True
    source, _ = load_checkpoint(common.ROOT / "yolo26n.pt")
    trainer.model = trainer.get_model(str(common.MODEL), source, verbose=False)
    trainer.optimizer = trainer.build_optimizer(trainer.model, "MuSGD", 0.01, 0.937, 0.0005)
    audit_training_setup(trainer)
    assert (tmp_path / "provenance/optimizer.json").is_file()


def test_diagnostic_statistics():
    """Check analytically known magnitude ratios and negative-alpha magnitude semantics."""
    z = [torch.full((2, 4, 5, 5), 4.0)] * 3
    d = [torch.full_like(z[0], 2.0)] * 3
    r = [torch.full_like(z[0], 6.0)] * 3
    stats = stage_statistics(z, d, r, torch.tensor([-0.1, 0, 0.1]))
    assert stats[0]["increment_over_stage"] == [0.5, 0.5]
    assert stats[0]["refine_over_increment"] == [3, 3]
    assert stats[1]["correction_rms"] == [0, 0]
    assert stats[0]["correction_over_stage"] == pytest.approx([0.15, 0.15])


@pytest.mark.parametrize("real_links", [False, True])
def test_package_transient_links_and_hardlinks(tmp_path, monkeypatch, real_links):
    """Transient dangling aliases cannot break canonical packaging, and hardlinks materialize as regular files."""
    run = tmp_path / "run"
    (run / "weights").mkdir(parents=True)
    (run / "weights/best.pt").write_bytes(b"checkpoint fixture, not a model")
    os.link(run / "weights/best.pt", run / "weights/last.pt")
    attempt = run / "obsolete.attempt.fixture"
    attempt.mkdir()
    links = [attempt / "console.log", run / "optional_dangling.log"]
    if real_links:
        try:
            for link in links:
                link.symlink_to(tmp_path / "nonexistent.log")
        except OSError as error:
            if getattr(error, "winerror", None) == 1314:
                pytest.skip("Windows account cannot create symlinks; portable fault injection runs separately")
            raise
    else:
        for link in links:
            link.touch()
        is_link, is_file = Path.is_symlink, Path.is_file
        monkeypatch.setattr(Path, "is_symlink", lambda path: path in links or is_link(path))

        def reject_dereference(path):
            if path in links:
                raise FileNotFoundError("Injected dangling console alias")
            return is_file(path)

        monkeypatch.setattr(Path, "is_file", reject_dereference)
    (run / "diagnostics.md").write_text("canonical summary")
    output = tmp_path / "result.tar.gz"
    result = archive_package(run, output, ["weights/best.pt", "weights/last.pt", "diagnostics.md"])
    with tarfile.open(output) as archive:
        assert not any("attempt" in n or "dangling" in n for n in archive.getnames())
        assert archive.getmember("run/weights/last.pt").isfile()
        assert archive.extractfile("run/weights/last.pt").read() == b"checkpoint fixture, not a model"
    assert result["gzip_crc_verified"] and output.with_name(output.name + ".sha256").is_file()
    with pytest.raises(FileNotFoundError):
        archive_package(run, tmp_path / "missing.tar.gz", ["missing.pt"])


@pytest.mark.parametrize("entry", ["run", "verify", "diagnose", "finish"])
def test_entry_help(entry):
    """All entry points can be imported in isolation and describe their actual command interfaces."""
    result = subprocess.run(
        [sys.executable, f"tools/experiments/{entry}_b19_sicr_sppf.py", "--help"],
        cwd=common.ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


def test_package_failure_keeps_test_success(tmp_path, monkeypatch):
    """A failed archive operation reports package failure without rewriting successful evaluation status."""
    from tools.experiments import finish_b19_sicr_sppf as finish

    test_receipt = tmp_path / "test_status.json"
    test_receipt.write_text('{"stage":"test","passed":true}')
    original = test_receipt.read_bytes()
    monkeypatch.setattr(finish, "provenance", lambda run, data: {"commit": "a" * 40})

    def failed_package(*args):
        raise OSError("simulated archive storage failure")

    monkeypatch.setattr(finish, "package", failed_package)
    with pytest.raises(OSError, match="storage failure"):
        finish.main(["--stage", "package", "--run", str(tmp_path)])
    assert test_receipt.read_bytes() == original
    assert '"passed": false' in (tmp_path / "package_status.json").read_text()
