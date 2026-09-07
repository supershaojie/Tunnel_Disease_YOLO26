"""SIR formula, native initialization, fixed-budget failure, and inference contracts."""

import copy
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from tools.experiments import run_b19_sir_sppf as run
from ultralytics.cfg import get_cfg
from ultralytics.data.dataset import YOLODataset
from ultralytics.nn.modules import SPPF, SPPF_SIR
from ultralytics.nn.tasks import load_checkpoint
from ultralytics.utils import YAML
from ultralytics.utils.checks import check_version
from ultralytics.utils.torch_utils import ModelEMA, init_seeds


@pytest.fixture(autouse=True)
def cpu_threads():
    """Bound local verification threads without changing the experiment entry's training configuration."""
    previous = torch.get_num_threads()
    torch.set_num_threads(4)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize(
    "shape,shortcut,n", [((2, 32, 20, 20), True, 3), ((2, 32, 17, 23), False, 3), ((1, 32, 9, 13), True, 2)]
)
def test_identity_rng_and_nonzero_formula(shape, shortcut, n):
    """An independent stacked cumsum reference distinguishes raw pooling, cumulative routing and identity gates."""
    init_seeds(42)
    original = SPPF(32, 32, 5, n, shortcut).eval()
    after_original = torch.get_rng_state()
    init_seeds(42)
    candidate = SPPF_SIR(32, 32, 5, n, shortcut).eval()
    assert torch.equal(after_original, torch.get_rng_state())
    for key, tensor in original.state_dict().items():
        assert torch.equal(tensor, candidate.state_dict()[key])
    assert isinstance(candidate.cv1.act, nn.Identity)
    x = torch.randn(shape)
    with torch.no_grad():
        assert torch.equal(original(x), candidate(x))
        nn.init.normal_(candidate.router[-1].weight, std=0.2)
        nn.init.normal_(candidate.router[-1].bias, std=0.2)
        raw = [candidate.cv1(x)]
        for _ in range(n):
            raw.append(candidate.m(raw[-1]))
        increments = torch.stack(raw[1:]) - torch.stack(raw[:-1])
        assert increments.min() >= 0
        gate = candidate.router(torch.cat([raw[0], *increments.unbind()], 1)).reshape(shape[0], n, 16, *shape[2:])
        gate = gate.permute(1, 0, 2, 3, 4).tanh()
        correction = (0.5 * gate * increments).cumsum(0)
        calibrated = torch.stack(raw[1:]) + correction
        previous = torch.cat([raw[0].unsqueeze(0), calibrated[:-1]])
        torch.testing.assert_close(calibrated - previous, (1 + 0.5 * gate) * increments, atol=2e-6, rtol=2e-5)
        assert calibrated.sub(previous).min() >= -2e-6
        assert gate.min() >= -1 and gate.max() <= 1
        expected = candidate.cv2(torch.cat([raw[0], *calibrated.unbind()], 1))
        if shortcut:
            expected = expected + x
        torch.testing.assert_close(candidate(x), expected)
        assert not torch.allclose(candidate(x), original(x))


def test_actual_nano_graph(tmp_path):
    """Only the ninth layer changes, with the required parameter budget and exact full raw output equality."""
    report = run.structural_checks(tmp_path)
    assert report["added_parameters"] == 14896
    assert report["candidate_parameters"] == 2519086


def baseline_root():
    """Use the explicit accessible b19 files; no downloads or synthetic target substitution."""
    root = os.environ.get("SIR_BASELINE_ROOT")
    if not root:
        pytest.skip("Set SIR_BASELINE_ROOT for real weights/data checks")
    return Path(root)


def sample_images(tmp_path):
    """Copy two actual image/label pairs so disposable subset label caches never touch b19/v2 data."""
    source = baseline_root() / "datasets/Tunnel_Crack_AugFirst_Diverse5x_RandomSplit_7_2_1_seed42"
    images = sorted((source / "images/train").glob("*.jpg"))[:2]
    assert len(images) == 2
    for image in images:
        for original in (image, source / "labels/train" / (image.stem + ".txt")):
            target = tmp_path / original.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(original, target)
    return sorted((tmp_path / "images/train").glob("*.jpg"))


def native_probe(device="cpu", *, trainer_type=run.AuditedTrainer, model=run.MODEL):
    """Build through the same native DetectionTrainer.get_model path used by formal train."""
    root = baseline_root()
    assert run.sha256(root / "yolo26n.pt") == run.PRETRAINED_SHA256
    trainer = object.__new__(trainer_type)
    trainer.args = get_cfg(overrides={k: v for k, v in run.REFERENCE["args"].items() if k != "save_dir"})
    trainer.data = dict(nc=1, channels=3, names={0: "crack"})
    weights, _ = load_checkpoint(root / "yolo26n.pt")
    init_seeds(42, deterministic=True)
    trainer.model = trainer.get_model(str(model), weights, verbose=False).to(device)
    trainer.device = torch.device(device)
    trainer.stride = 32
    trainer.set_model_attributes()
    trainer.optimizer = trainer.build_optimizer(trainer.model, "MuSGD", 0.01, 0.937, 0.0005)
    trainer.ema = ModelEMA(trainer.model)
    return trainer


def test_original_weight_trainer_and_fresh_process(tmp_path):
    """Compatible tensors and the nc-adapted head match; a new process can load, fuse and predict."""
    torch.set_num_threads(1)  # The old fixture's 4/4 threads masked the entrypoint's 1/4 mismatch.
    trainer = native_probe()
    assert len(trainer.weight_audit["loaded_keys"]) == 606
    assert trainer.weight_audit["common_keys"] == 708
    assert trainer.weight_audit["all_common_tensors_equal"]
    original = {k: v.clone() for k, v in trainer.model.state_dict().items()}
    rng = torch.get_rng_state()
    run.save_reload_check(trainer.model, tmp_path)
    assert torch.get_num_threads() == 1 and torch.equal(rng, torch.get_rng_state())
    run.assert_close_tree(original, trainer.model.state_dict(), 0, 0, path="caller_state")
    report = json.loads((tmp_path / "reload_check.json").read_text())
    assert report["passed"] and report["state_exact"] and report["state_keys"] == 714
    assert max(row["max_abs"] for row in report["raw"]) == 0
    assert report["conditions"]["threads"] == 1
    assert report["conditions"]["model_float_dtypes"] == ["torch.float32"]

    # Tamper the serialized router and a BN counter independently; the original snapshot reference stays intact.
    checkpoint = torch.load(tmp_path / "preflight.pt", weights_only=False)
    for key in ("model.9.router.4.weight", "model.9.cv1.bn.num_batches_tracked"):
        directory = tmp_path / key
        directory.mkdir()
        shutil.copy2(tmp_path / "reload_reference.pt", directory / "reload_reference.pt")
        damaged = copy.deepcopy(checkpoint)
        damaged["ema"].state_dict()[key].view(-1)[0] += 1
        torch.save(damaged, directory / "preflight.pt")
        with pytest.raises(AssertionError, match=rf"state\.{key}.*outside_tolerance"):
            run.reload_in_process(directory / "preflight.pt")


@pytest.mark.skipif(
    not check_version(torch.__version__, "2.7.0"), reason="SIR reload audit uses torch>=2.7 backend APIs"
)
def test_reload_context_restores_caller_on_failure():
    """A failing isolated check must not change training threads, RNG, AMP or backend policy."""
    rng = torch.get_rng_state()
    with torch.autocast("cpu", enabled=True):
        before = (
            torch.get_num_threads(),
            torch.get_float32_matmul_precision(),
            torch.are_deterministic_algorithms_enabled(),
            torch.is_deterministic_algorithms_warn_only_enabled(),
            torch.backends.mkldnn.enabled,
            torch.backends.mkldnn.deterministic,
            torch.backends.mkldnn.allow_tf32,
            torch.is_autocast_enabled("cpu"),
        )
        with pytest.raises(RuntimeError, match="intentional probe failure"), run.reload_context():
            assert torch.get_num_threads() == 1 and not torch.is_autocast_enabled("cpu")
            torch.randn(3)
            raise RuntimeError("intentional probe failure")
        after = (
            torch.get_num_threads(),
            torch.get_float32_matmul_precision(),
            torch.are_deterministic_algorithms_enabled(),
            torch.is_deterministic_algorithms_warn_only_enabled(),
            torch.backends.mkldnn.enabled,
            torch.backends.mkldnn.deterministic,
            torch.backends.mkldnn.allow_tf32,
            torch.is_autocast_enabled("cpu"),
        )
        assert before == after and torch.equal(rng, torch.get_rng_state())


def test_raw_error_diagnostics_reject_out_of_tolerance():
    """A near-zero output error still fails, with a stable scale diagnostic and the exact tensor path."""
    report = []
    with pytest.raises(AssertionError, match=r"raw.one2one\[0\].*outside_tolerance"):
        run.assert_close_tree(
            {"one2one": [torch.tensor([0.0, 1.0])]}, {"one2one": [torch.tensor([1e-3, 1.0])]}, 1e-6, 1e-5, report=report
        )
    assert report[0]["outside_tolerance"] == 1
    assert report[0]["max_error_over_scale"] == pytest.approx(1e-3)
    assert report[0]["mean_abs"] == pytest.approx(5e-4)
    with pytest.raises(AssertionError, match='"finite": false'):
        run.assert_close_tree(torch.tensor([float("inf")]), torch.tensor([float("inf")]))


def test_real_crack_gradients(tmp_path):
    """Use three small disposable updates, explicitly distinct from the required server batch=32 preflight."""
    trainer = native_probe("cuda:0" if torch.cuda.is_available() else "cpu")
    images = sample_images(tmp_path)
    listing = tmp_path / "images.txt"
    listing.write_text("\n".join(map(str, images)), encoding="utf-8")
    data = YOLODataset(
        img_path=str(listing), imgsz=640, batch_size=2, augment=True, hyp=trainer.args, data=trainer.data, cache=False
    )
    rows = []
    for _ in range(3):
        batch = data.collate_fn([data[0], data[1]])
        rows.append(run.gradient_check(trainer, batch, trainer.device.type == "cuda"))
    first = rows[0]["new_gradient_norms"]
    assert all(v == 0 for k, v in first.items() if ".router.4." not in k)
    assert all(v > 0 for v in rows[-1]["new_gradient_norms"].values())
    assert isinstance(trainer.ema.ema.model[9], SPPF_SIR)
    run.save_reload_check(trainer.ema.ema, tmp_path)
    run.write_json(
        run.ROOT / "runs/sir_development/local_real_gradient.json",
        dict(formal_preflight=False, batch=2, device=str(trainer.device), rows=rows),
    )


def test_native_setup_oom_and_audit_rng(tmp_path, *, trainer_type=run.AuditedTrainer, model=run.MODEL):
    """Inject OOM at the real catch boundary; verify no batch mutation, recovery pipeline or output suffix."""
    root = baseline_root()
    images = sample_images(tmp_path)
    listing = tmp_path / "images.txt"
    listing.write_text("\n".join(map(str, images)), encoding="utf-8")
    data = tmp_path / "data.yaml"
    YAML.save(data, dict(path=str(tmp_path), train=str(listing), val=str(listing), names={0: "crack"}))
    config = {k: v for k, v in run.REFERENCE["args"].items() if k != "save_dir"}
    config.update(
        model=str(model),
        pretrained=str(root / "yolo26n.pt"),
        data=str(data),
        project=str(tmp_path),
        name="native_setup",
        device="cpu",
        workers=0,
        amp=False,
        batch=2,
        imgsz=64,
        plots=False,
    )
    trainer = trainer_type(overrides=config)

    def audit(observed):
        rng = torch.get_rng_state()
        run.final_model_audit(observed)
        assert torch.equal(rng, torch.get_rng_state())

    trainer.add_callback("on_pretrain_routine_end", audit)
    error = torch.cuda.OutOfMemoryError("injected SIR OOM")
    with patch.object(trainer, "preprocess_batch", side_effect=error), patch.object(
        trainer, "_build_train_pipeline", wraps=trainer._build_train_pipeline
    ) as pipeline:
        with pytest.raises(torch.cuda.OutOfMemoryError, match="injected SIR") as caught:
            trainer.train()
        assert caught.value is error and pipeline.call_count == 1
    assert trainer.args.batch == trainer.batch_size == 2 and trainer._oom_retries == 0
    assert trainer.save_dir == tmp_path / "native_setup"
    assert not trainer.csv.exists() and not trainer.best.exists()
    report = json.loads((trainer.save_dir / "provenance/final_optimizer.json").read_text())
    assert report["name"] == "MuSGD"
    assert sum(len(g["parameters"]) for g in report["router_groups"]) == 6
    run.write_json(run.ROOT / "runs/sir_development/local_final_optimizer.json", report)
    with pytest.raises(FileExistsError):
        trainer_type(overrides=config)
    assert not (tmp_path / "native_setup2").exists()
    for loader in (trainer.train_loader, trainer.test_loader):
        loader.close()


def test_snapshot_and_launcher_fallback(tmp_path):
    """Missing history uses the attributed provided snapshot; it never turns into guessed defaults."""
    path, raw = run.find_baseline(tmp_path, tmp_path / "missing/args.yaml")
    assert path is None and raw == run.REFERENCE["args"]
    assert run.launcher_evidence(SimpleNamespace(baseline_launcher=None), raw)["verified"]
    changed = copy.deepcopy(raw)
    changed["batch"] = 16
    with pytest.raises(ValueError, match="differs"):
        run.launcher_evidence(SimpleNamespace(baseline_launcher=None), changed)


def test_completed_test_is_one_call_and_keeps_empty_json(tmp_path):
    """Validate exact FP32 settings and single-call JSON output without accessing or tuning the test dataset."""
    import numpy as np

    from tools.experiments import finish_b19_sir_sppf as finish
    from ultralytics import YOLO
    from ultralytics.cfg import get_save_dir

    data = tmp_path / "data.yaml"
    data.write_text("test: images/test\n", encoding="utf-8")
    metrics = SimpleNamespace(
        results_dict={"metrics/precision(B)": 0.123456789}, box=SimpleNamespace(all_ap=np.zeros((1, 10)))
    )

    class Validator:
        def __init__(self, args, _callbacks):
            self.args = get_cfg(overrides=args)
            self.save_dir = get_save_dir(self.args)
            self.callbacks = _callbacks
            self.metrics = metrics
            self.metrics.nt_per_class = np.array([1477])
            self.jdict, self.speed, self.seen = [], {}, 1202

        def __call__(self, model):
            assert self.args.split == "test" and not self.args.exist_ok and self.args.quantize is None
            for callback in self.callbacks["on_val_end"]:
                callback(self)

    # Exercise the actual Model.val orchestration. Neither the model nor validator owns the
    # other's local object; only inference itself is stubbed to avoid consuming held-out test data.
    model = YOLO(str(run.MODEL))

    with patch.object(finish, "provenance", return_value={"weight": "fixture.pt"}), patch.object(
        finish, "YOLO", return_value=model
    ) as factory, patch.object(model, "_smart_load", return_value=Validator):
        finish.test_best(tmp_path, data)
        finish.test_best(tmp_path, data)
        assert factory.call_count == 1
    assert json.loads((tmp_path / "test/predictions.json").read_text()) == []


def test_cli_import_preserves_native_thread_initialization():
    """A clean subprocess initializes the same default CPU thread budget as the original CLI."""
    env = {k: v for k, v in os.environ.items() if k not in {"OMP_NUM_THREADS", "MKL_NUM_THREADS"}}
    env["PYTHONPATH"] = str(run.ROOT)
    values = []
    for entry in ("ultralytics", "tools.experiments.run_b19_sir_sppf"):
        code = f"import {entry}; import os,torch; print(os.environ['OMP_NUM_THREADS'],torch.get_num_threads())"
        result = subprocess.check_output([sys.executable, "-c", code], cwd=run.ROOT, env=env, text=True)
        values.append(result.strip().splitlines()[-1])
    assert values == ["1 1", "1 1"]


def test_package_preserves_applicable_patch_and_excludes_history(tmp_path, monkeypatch):
    """Check archive integrity and apply the raw patch in check-only mode against the committed index."""
    from tools.experiments import finish_b19_sir_sppf as finish

    # A subtree from HEAD provides a real, nonempty tree diff even in a one-commit CI checkout.
    # Only this synthetic archive fixture changes its base; production still requires recorded b19 history.
    monkeypatch.setitem(run.REFERENCE, "source_commit", run.git("rev-parse", "HEAD:ultralytics/nn/modules"))
    fixture = tmp_path / "synthetic_package"
    for name in (
        "args.yaml",
        "results.csv",
        "train.log",
        "weights/best.pt",
        "weights/last.pt",
        "weights/epoch20.pt",
        "test/metrics.json",
        "test/predictions.json",
        "diagnostics.json",
        "val_metrics.json",
        "completed.json",
    ):
        path = fixture / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"Synthetic archive contract fixture, not a training result\n")
    output = tmp_path / "synthetic.tar.gz"
    finish.package(fixture, output)
    with tarfile.open(output) as archive:
        names = archive.getnames()
        patch = archive.extractfile("source.patch").read()
    assert "run/weights/best.pt" in names
    assert all(not name.endswith(("last.pt", "epoch20.pt", ".tar.gz")) for name in names)
    assert output.with_name(output.name + ".sha256").read_text().split()[0] == run.sha256(output)
    # The source patch is for HEAD. --cached checks that exact committed/staged state even while
    # developing a subsequent worktree edit; --check performs no application or index mutation.
    subprocess.run(
        ["git", "-c", f"safe.directory={run.ROOT.as_posix()}", "apply", "--cached", "--check", "--reverse", "-"],
        input=patch,
        cwd=run.ROOT,
        check=True,
    )
