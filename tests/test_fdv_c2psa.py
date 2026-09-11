"""FDV specification, native inheritance, training and robust archive regression checks."""

import copy
import os
import tarfile
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from tools.experiments import b19_common as common
from tools.experiments import finish_b19_fdv_c2psa as finish
from tools.experiments.diagnose_b19_fdv_c2psa import diagnose
from tools.experiments.run_b19_fdv_c2psa import AuditedTrainer, audit_arguments, audit_training_setup, options_parser
from tools.experiments.verify_b19_fdv_c2psa import build_pair, module_checks, topology_checks
from ultralytics.nn.modules import FDVAttention
from ultralytics.nn.modules.block import Attention
from ultralytics.nn.tasks import load_checkpoint
from ultralytics.utils import YAML


@pytest.fixture(scope="module", autouse=True)
def threads():
    """Keep CPU tests bounded without changing numerical backend policy."""
    previous = torch.get_num_threads()
    torch.set_num_threads(4)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_zero_init(device):
    """Exact module equivalence must hold at standard, rectangular, odd and singleton sizes."""
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    assert len(module_checks(device)) == 8


@pytest.mark.parametrize("coefficient", [-100.0, -0.3, 0.0, 0.3, 100.0])
def test_only_value_highpass_changes(coefficient):
    """Compare to native pre-projection output plus an independent fixed-convolution high-pass oracle."""
    native = Attention(128, 2).eval()
    candidate = FDVAttention(copy.deepcopy(native)).eval()
    x = torch.randn(2, 128, 7, 9)
    captured = {}
    hooks = [
        native.qkv.register_forward_hook(lambda m, a, y: captured.update(qkv=y)),
        native.proj.register_forward_pre_hook(lambda m, a: captured.update(native=a[0])),
    ]
    with torch.no_grad():
        theta = torch.linspace(-1, 1, 128) * coefficient
        candidate.theta_c.copy_(theta)
        native(x)
        qkv = captured["qkv"].reshape(2, 2, 128, 7, 9)
        v = qkv[:, :, 64:].reshape(2, 128, 7, 9)
        low = F.conv2d(v, torch.ones(128, 1, 3, 3) / 9, padding=1, groups=128)
        gamma = 0.1 * theta.tanh()
        expected = native.proj(captured["native"] + gamma[None, :, None, None] * (v - low))
        torch.testing.assert_close(candidate(x), expected, atol=1e-6, rtol=1e-5)
        assert candidate.gamma_c.shape == (128,) and (candidate.gamma_c.abs() <= 0.1).all()
        assert torch.equal(native.qkv(x), candidate.qkv(x))
    for hook in hooks:
        hook.remove()
    assert set(candidate.state_dict()) - set(native.state_dict()) == {"theta_c"}


def test_constructor_and_weights():
    """Only theta may be missing; all shared state including randomly adapted Detect heads must match."""
    weights = Path(os.environ.get("FDV_PRETRAINED", common.ROOT / "yolo26n.pt"))
    assert common.sha256(weights) == common.PRETRAINED_SHA256
    source, _ = load_checkpoint(weights)
    baseline, candidate = build_pair(source)
    topology_checks(baseline, candidate)
    report = common.audit_weights(baseline, candidate, source)
    assert report["common_keys"] == 708 and report["matched_tensors"] == 606
    assert report["layer10_shared_loaded"] and report["added_parameters"] == 128
    assert report["all_common_tensors_equal"]


def test_config_cannot_change_recipe():
    """Reject every non-identity override, while preserving all 112 archived b19 fields."""
    raw = common.REFERENCE["args"]
    for key in ("epochs", "imgsz", "batch", "seed", "optimizer", "hsv_h", "warmup_epochs", "amp"):
        with pytest.raises(ValueError, match="Non-identity"):
            audit_arguments(raw, {**raw, key: "changed"})
    assert audit_arguments(raw, {**raw, "name": "yolo26n_b19_fdv_c2psa_v1"})
    with pytest.raises(SystemExit):
        options_parser().parse_args(["--batch", "16"])
    trainer = object.__new__(AuditedTrainer)
    with pytest.raises(RuntimeError, match="batch reduction"):
        trainer._oom_retries = 1


def test_trainer_rebuild_and_optimizer(tmp_path):
    """Exercise the actual trainer reconstruction and final MuSGD parameter-group audit without a training run."""
    trainer = object.__new__(AuditedTrainer)
    trainer.data = {"nc": 1, "channels": 3, "names": {0: "crack"}}
    trainer.args = common.get_cfg(overrides={k: v for k, v in common.REFERENCE["args"].items() if k != "save_dir"})
    source, _ = load_checkpoint(common.ROOT / "yolo26n.pt")
    trainer.model = trainer.get_model(str(common.MODEL), source, verbose=False)
    trainer.expected_args = vars(trainer.args).copy()
    trainer.batch_size, trainer.amp, trainer.save_dir = 32, True, tmp_path
    trainer.optimizer = trainer.build_optimizer(trainer.model, "MuSGD", 0.01, 0.937, 0.0005)
    audit_training_setup(trainer)
    assert (tmp_path / "provenance/weights.json").is_file()


def test_fixed_diagnostics(tmp_path):
    """Exercise the real 16-image hook path and its JSON/CSV output from a nonzero FDV checkpoint."""
    import cv2
    import numpy as np

    _, model = build_pair()
    with torch.no_grad():
        model.model[10].m[0].attn.theta_c.copy_(torch.linspace(-0.5, 0.5, 128))
    weights = tmp_path / "fdv.pt"
    torch.save({"model": model, "train_args": {}}, weights)
    images = tmp_path / "images/val"
    images.mkdir(parents=True)
    rng = np.random.default_rng(42)
    for i in range(16):
        assert cv2.imwrite(str(images / f"{i:02}.png"), rng.integers(0, 256, (64, 96, 3), dtype=np.uint8))
    data = tmp_path / "data.yaml"
    YAML.save(data, {"path": str(tmp_path), "train": "images/val", "val": "images/val", "names": {0: "crack"}})
    report = diagnose(weights, data, tmp_path / "diagnostics", "cpu")
    assert report["state_unchanged"] and len(report["samples"]) == 16
    assert report["top_positive_gamma_channels"] and report["top_negative_gamma_channels"]
    assert all(s["high_frequency_ratio"] > 0 for s in report["samples"])
    assert common.sha256(tmp_path / "diagnostics/fdv_c2psa_diagnostics.csv") == report["csv_sha256"]


def package_fixture(run):
    """Create explicit synthetic result receipts for the production package gate, never a training PASS."""
    evidence = {"commit": "synthetic", "weight_sha256": "synthetic"}
    required = [
        "weights/best.pt",
        "weights/last.pt",
        "args.yaml",
        "results.csv",
        "results.png",
        "completed.json",
        "baseline_comparison/comparison.md",
        "baseline_comparison/diagnostics/fdv_c2psa_diagnostics.csv",
    ]
    for name in required:
        p = run / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("SYNTHETIC PACKAGE DRY RUN\n", encoding="utf-8")
    common.write_json(run / "baseline_comparison/comparison.json", {"evidence": evidence})
    common.write_json(
        run / "baseline_comparison/diagnostics/fdv_c2psa_diagnostics.json",
        {
            "commit": "synthetic",
            "weight_sha256": "synthetic",
            "csv_sha256": common.sha256(run / "baseline_comparison/diagnostics/fdv_c2psa_diagnostics.csv"),
        },
    )
    for stage in ("fdv_val", "fdv_test", "b19_val", "b19_test"):
        folder = run / "baseline_comparison" / stage
        folder.mkdir()
        for name in finish.CURVES:
            (folder / name).write_text("synthetic", encoding="utf-8")
        common.write_json(folder / "metrics.json", {"artifacts": {n: common.sha256(folder / n) for n in finish.CURVES}})
    return evidence


def test_package_dry_run(tmp_path):
    """Run the full required-artifact gate, excluding attempts and accepting absent console.log and hard links."""
    run = tmp_path / "run"
    evidence = package_fixture(run)
    attempt = run / "logs/old.attempt.123"
    attempt.mkdir(parents=True)
    (attempt / "console.log").write_text("transient", encoding="utf-8")
    os.link(run / "weights/best.pt", run / "best-copy.txt")
    result = finish.package(run, tmp_path / "dry-run.tar.gz", evidence)
    with tarfile.open(result["path"]) as archive:
        assert not any(".attempt." in n for n in archive.getnames())
        assert archive.getmember("run/weights/last.pt").isfile()
        assert archive.getmember("run/best-copy.txt").isfile()
    assert result["gzip_crc_verified"]


def test_dangling_symlink_archive(tmp_path):
    """On supported hosts, a real dangling console alias must not break the archive."""
    run = tmp_path / "run"
    run.mkdir()
    try:
        (run / "console.log").symlink_to(run / "missing.log")
    except OSError as error:
        pytest.skip(f"Host disallows symlink creation: {error}")
    (run / "args.yaml").write_text("synthetic", encoding="utf-8")
    finish.archive_package(run, tmp_path / "links.tar.gz", ["args.yaml"])


def test_package_rejects_incomplete_or_changed_outputs(tmp_path):
    """No archive may turn missing last weights or a changed evaluation artifact into a successful package."""
    run = tmp_path / "run"
    evidence = package_fixture(run)
    (run / "baseline_comparison/fdv_test/BoxPR_curve.png").write_text("changed", encoding="utf-8")
    with pytest.raises(AssertionError):
        finish.package(run, tmp_path / "invalid.tar.gz", evidence)
    with pytest.raises(FileNotFoundError):
        finish.archive_package(tmp_path, tmp_path.parent / "missing.tar.gz", ["absent-last.pt"])
