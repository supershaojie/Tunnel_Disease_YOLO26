"""DSD v1 numerical, routing, initialization, serialization and native training contracts."""

import copy
import json
import os
import tarfile
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch

from tools.experiments import b19_common as shared
from tools.experiments.dsd_preflight import preflight_batches, routing_checks
from tools.experiments.dsd_validator import model_identity, raw_comparison, validator_check
from tools.experiments.run_b19_dsd_head import MODEL, AuditedTrainer, structural_checks
from ultralytics.cfg import get_cfg
from ultralytics.nn.modules import DSDAdapter
from ultralytics.nn.tasks import DetectionModel, load_checkpoint
from ultralytics.utils import YAML


@pytest.fixture(autouse=True)
def threads():
    """Bound small CPU checks and restore the surrounding process settings."""
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("h,w", [(80, 80), (48, 80), (7, 11), (3, 3), (1, 7), (8, 2)])
def test_formula_groups_directions_boundary(h, w):
    """Compare against a scalar pixel oracle independent of the vectorized implementation."""
    adapter = DSDAdapter()
    torch.manual_seed(7)
    x = torch.randint(-10, 10, (1, 64, h, w)).float()
    z = torch.linspace(-2, 2, 32)
    with torch.no_grad():
        adapter.coeff[-1].bias.copy_(z)
        expected = x.clone()
        for c in range(64):
            for y, col in ((1, 1), (h // 2, w // 2), (h - 2, w - 2)):
                if not (1 <= y < h - 1 and 1 <= col < w - 1):
                    continue
                correction = 0.0
                for d, (dy, dx) in enumerate(((0, 1), (1, 0), (1, 1), (1, -1))):
                    difference = (x[0, c, y + dy, col + dx] + x[0, c, y - dy, col - dx] - 2 * x[0, c, y, col]) / (
                        dy * dy + dx * dx
                    )
                    correction += z[(c // 8) * 4 + d].tanh() * difference
                expected[0, c, y, col] = x[0, c, y, col] + 0.025 * correction
        actual = adapter(x)
        assert actual.shape == x.shape and actual.dtype == x.dtype
        assert torch.equal(actual[..., (0, -1), :], x[..., (0, -1), :])
        assert torch.equal(actual[..., :, (0, -1)], x[..., :, (0, -1)])
        if h < 3 or w < 3:
            assert torch.equal(actual, x)
        else:
            for y, col in ((1, 1), (h // 2, w // 2), (h - 2, w - 2)):
                torch.testing.assert_close(actual[..., y, col], expected[..., y, col], rtol=0, atol=1e-6)


def test_structural_and_routing(tmp_path):
    """Exercise whole-network 640/rectangular equivalence and the native detach boundary."""
    report = structural_checks(tmp_path)
    assert report["added_parameters"] == 3488
    assert routing_checks()["one2one_backbone_detached"]


def test_pretrained_inheritance():
    """Compare all 708 nc-adapted state items and all 606 transferred original COCO items."""
    path = Path(os.environ.get("B19_TEST_WEIGHT", shared.ROOT.parents[1] / "yolo26n.pt"))
    if not path.is_file():
        pytest.skip("Original b19 yolo26n.pt is not available")
    assert shared.sha256(path) == shared.PRETRAINED_SHA256
    weights, _ = load_checkpoint(path)
    probe = object.__new__(AuditedTrainer)
    probe.args = get_cfg()
    probe.data = dict(nc=1, channels=3, names={0: "crack"})
    probe.get_model(str(MODEL), weights, False)
    assert probe.weight_audit["common_keys"] == 708
    assert len(probe.weight_audit["loaded_keys"]) == 606


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA AMP requires a GPU")
def test_fp32_difference_under_amp():
    """Half inputs use FP32 differences even when native half arithmetic would overflow."""
    adapter = DSDAdapter().cuda().half()
    with torch.no_grad():
        adapter.coeff[0].weight.zero_()
        adapter.coeff[0].bias.zero_()
        adapter.coeff[-1].bias.fill_(0.2)
    x = torch.full((1, 64, 5, 7), 50000.0, device="cuda", dtype=torch.float16, requires_grad=True)
    with torch.autocast("cuda", dtype=torch.float16):
        result = adapter(x)
    assert torch.equal(result, x) and torch.isfinite(result).all()
    result.float().sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all() and x.grad.count_nonzero()


def test_save_reload_fuse_fresh_process(tmp_path):
    """Round-trip a nonzero FP16 EMA snapshot and execute the fused inference adapter."""
    model = DetectionModel(str(MODEL), verbose=False)
    model.args = get_cfg()
    with torch.no_grad():
        model.model[-1].one2one_reg_adapter.coeff[-1].weight.normal_(0, 0.02)
        model.model[-1].one2one_reg_adapter.coeff[-1].bias.fill_(0.1)
    result = shared.save_reload_check(model, tmp_path)
    assert result["fresh_process"] and result["reload_raw_equal"]
    fused = copy.deepcopy(model).eval().fuse(verbose=False)
    assert fused.model[-1].reg_adapter is None
    assert sum(p.numel() for k, p in fused.named_parameters() if "reg_adapter." in k) == 1744


def test_reject_recipe_changes_and_missing_record(tmp_path):
    """Missing b19 records never become successful inferred evidence."""
    with pytest.raises(FileNotFoundError):
        shared.find_baseline(tmp_path)
    recipe = dict(shared.REFERENCE["args"], batch=16)
    path = tmp_path / "args.yaml"
    YAML.save(path, recipe)
    with pytest.raises(ValueError, match="recipe differences"):
        shared.resolve_recipe(SimpleNamespace(baseline_root=tmp_path, baseline_args=path))


def test_comparison_precision_restored_on_error():
    """The diagnostic owns fusion/forward precision and restores even non-default caller policy after failure."""
    original = shared.computation_conditions()
    try:
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.allow_tf32 = True
        ambient = shared.computation_conditions()
        with pytest.raises(RuntimeError, match="diagnostic failure"):
            with shared.reload_context(device="cuda" if torch.cuda.is_available() else "cpu"):
                assert torch.get_float32_matmul_precision() == "highest"
                assert not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32
                raise RuntimeError("diagnostic failure")
        assert shared.computation_conditions() == ambient
    finally:
        torch.set_float32_matmul_precision(original["float32_matmul_precision"])
        torch.backends.cudnn.allow_tf32 = original["cudnn_allow_tf32"]


def test_all_scales_reported_after_first_mismatch():
    """A P3 failure must not hide P4/P5 or scores; use the reported server anchor counts."""
    before = dict(
        boxes=torch.zeros(32, 4, 5292),
        scores=torch.zeros(32, 1, 5292),
        feats=[torch.zeros(32, 1, h, w) for h, w in ((56, 72), (28, 36), (14, 18))],
    )
    after = copy.deepcopy(before)
    after["boxes"][..., :4032] = 1
    after["boxes"][0, 0, 4032] = 0.4
    after["scores"][0, 0, -1] = 0.1
    rows = []
    raw_comparison(before, after, rows, "injected")
    by_path = {r["path"]: r for r in rows}
    assert by_path["injected.P3.boxes"]["outside_tolerance"] == 516096
    assert by_path["injected.P4.boxes"]["outside_tolerance"] == 1
    assert by_path["injected.P5.boxes"]["outside_tolerance"] == 0
    assert by_path["injected.P5.scores"]["outside_tolerance"] == 1


@pytest.mark.skipif(os.environ.get("DSD_RUN_NATIVE_SMOKE") != "1", reason="Explicit local native CUDA smoke")
def test_native_amp_updates_and_validator(tmp_path, monkeypatch):
    """Development-only batch=2/128 smoke; never creates a server batch=32 preflight receipt."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    images, labels = tmp_path / "images", tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    rng = np.random.default_rng(42)
    for i in range(128):
        cv2.imwrite(str(images / f"{i:03d}.jpg"), rng.integers(0, 255, (128, 160, 3), dtype=np.uint8))
        (labels / f"{i:03d}.txt").write_text("0 0.5 0.5 0.3 0.5\n", encoding="utf-8")
    data = tmp_path / "data.yaml"
    YAML.save(data, dict(path=str(tmp_path), train="images", val="images", names={0: "crack"}))
    config = {k: v for k, v in shared.REFERENCE["args"].items() if k not in {"save_dir", "cfg"}}
    config.update(
        model=str(MODEL),
        pretrained=str(shared.ROOT / "yolo26n.pt"),
        data=str(data),
        project=str(tmp_path),
        name="native_smoke",
        batch=2,
        imgsz=128,
        workers=0,
        plots=False,
        device="0",
    )
    trainer = AuditedTrainer(overrides=config)
    trainer.add_callback("on_pretrain_routine_end", shared.final_model_audit)
    try:
        trainer._setup_train()
        report = dict(
            scope="LOCAL DEVELOPMENT ONLY: batch=2/128 synthetic images; not formal preflight", real_batches=[]
        )
        updates = preflight_batches(trainer, report, tmp_path)
        assert report["adapter_preflight"]["status"] == "passed"
        assert trainer.ema.updates == updates
        assert len(report["adapter_preflight"]["first_effective_gradient"]) == 12
        assert set(trainer.model.state_dict()) == set(trainer.ema.ema.state_dict())
        # The Validator itself still receives an actual batch=32 in FP32 through its native loader/backend.
        identity = model_identity(trainer.ema.ema)
        conditions = shared.computation_conditions()
        result = validator_check(trainer.ema.ema, trainer, tmp_path)
        assert result["passed"] and result["settings_restored"] and result["source_unchanged"]
        assert set(result["cases"]) == {"ambient_native", "ambient_dsd", "strict_native", "strict_dsd"}
        assert all(c["equivalent_at_1e_4"] for k, c in result["cases"].items() if k.startswith("strict"))
        for name, case in result["cases"].items():
            assert case["images"] == 32 and case["input"]["dtype"] == "torch.float32"
            assert case["same_state_and_independent_storage"] and case["reference_unchanged"]
            assert len(case["scale_shapes"]) == 3
            if "dsd" in name:
                assert len(case["adapter_calls"]) == 2 and all(c["changed"] > 0 for c in case["adapter_calls"])
        assert model_identity(trainer.ema.ema) == identity and shared.computation_conditions() == conditions

        # Deliberate P4 implementation error MUST fail strict validation and retain the complete diagnostic.
        from ultralytics.nn.modules import DSDDetect

        native_fuse = DSDDetect.fuse

        def broken_fuse(head):
            native_fuse(head)
            with torch.no_grad():
                head.one2one_cv2[1][-1].bias.add_(0.1)

        monkeypatch.setattr(DSDDetect, "fuse", broken_fuse)
        failure_dir = tmp_path / "injected_failure"
        failure_dir.mkdir()
        with pytest.raises(AssertionError, match="Validator numerical contract failed"):
            validator_check(trainer.ema.ema, trainer, failure_dir)
        failed = json.loads((failure_dir / "validator_check.json").read_text())
        assert not failed["passed"] and failed["source_unchanged"] and failed["settings_restored"]
        assert failed["cases"]["strict_native"]["equivalent_at_1e_4"]
        assert any(
            r["path"] == "gpu_fused.P4.boxes" and r["outside_tolerance"] > 0
            for r in failed["cases"]["strict_dsd"]["comparisons"]
        )
    finally:
        for name in ("train_loader", "test_loader"):
            loader = getattr(trainer, name, None)
            if loader is not None:
                loader.close()


@pytest.mark.skipif(os.environ.get("DSD_RUN_FINISH_SMOKE") != "1", reason="Explicit FP32 artifact integration smoke")
def test_evaluation_diagnostics_package(tmp_path, monkeypatch):
    """Exercise actual FP32 reports and archive verification on an explicitly synthetic 32-image fixture."""
    from tools.experiments import finish_b19_dsd_head as finish

    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    # Scope counts to this fixture; no production count, CLI recipe or output directory is modified.
    monkeypatch.setattr(finish, "COUNTS", {"val": (32, 32), "test": (32, 32)})
    run = tmp_path / "synthetic_finish"
    (run / "weights").mkdir(parents=True)
    (tmp_path / "images").mkdir()
    (tmp_path / "labels").mkdir()
    rng = np.random.default_rng(42)
    for i in range(32):
        cv2.imwrite(str(tmp_path / f"images/{i:03d}.jpg"), rng.integers(0, 255, (64, 96, 3), dtype=np.uint8))
        (tmp_path / f"labels/{i:03d}.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    data = tmp_path / "data.yaml"
    YAML.save(data, dict(path=str(tmp_path), train="images", val="images", test="images", names={0: "crack"}))
    model = DetectionModel(str(MODEL), verbose=False).eval()
    model.names = {0: "crack"}
    with torch.no_grad():
        for conv in model.model[-1].one2one_cv3:
            conv[-1].bias.fill_(1)
        for adapter in (model.model[-1].reg_adapter, model.model[-1].one2one_reg_adapter):
            adapter.coeff[-1].bias.fill_(0.3)
    torch.save(dict(ema=model.half(), model=None, train_args=vars(get_cfg())), run / "weights/best.pt")
    (run / "weights/last.pt").write_bytes((run / "weights/best.pt").read_bytes())
    YAML.save(run / "args.yaml", dict(data=str(data)))
    shared.write_json(
        run / "completed.json",
        dict(
            completed=True,
            run=str(run),
            best_sha256=shared.sha256(run / "weights/best.pt"),
            commit=shared.git("rev-parse", "HEAD"),
            scope="SYNTHETIC FIXTURE",
        ),
    )
    (run.parent / f"{run.name}_train.exit_status").write_text("0\n", encoding="utf-8")
    (run / "results.csv").write_text("epoch,metric\n1,0\n", encoding="utf-8")
    (run / "train.log").write_text("SYNTHETIC ARTIFACT TEST; NO TRAINING CLAIM\n", encoding="utf-8")
    cv2.imwrite(str(run / "results.png"), np.zeros((10, 10, 3), dtype=np.uint8))
    shared.write_json(run / "val_metrics.json", dict(scope="synthetic fixture"))
    for split in finish.COUNTS:
        finish.evaluate_split(run, data, split)
    finish.diagnose(run, data)
    _, report = finish.load_report(run, json.loads((run / "diagnostics.json").read_text()))
    assert len(report["images"]) == 16 and len(report["fused_inference"]) == 16
    assert all(row["one2one_calls"] == 1 for row in report["fused_inference"])
    assert all("one2one_reg_adapter" in row and "reg_adapter" in row for row in report["images"])
    output = tmp_path / "synthetic_finish.tar.gz"
    finish.package(run, data, output)
    with tarfile.open(output) as archive:
        names = archive.getnames()
    assert "run/weights/best.pt" in names and "run/weights/last.pt" in names
    assert "source.tar" in names and "package_manifest.json" in names
    assert shared.sha256(output) in output.with_name(output.name + ".sha256").read_text()
    with pytest.raises(FileExistsError):
        finish.package(run, data, output)
