"""Offline checks for directional geometry, native YOLO loss, initialization, and checkpoint portability."""

import copy
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from tools.experiments.run_b19_dcrstrip import (
    MODEL,
    REFERENCE,
    AuditedTrainer,
    assert_close_tree,
    find_baseline,
    final_model_audit,
    gradient_check,
    runtime_issues,
    save_reload_check,
    structural_checks,
)
from ultralytics.cfg import get_cfg
from ultralytics.data.dataset import YOLODataset
from ultralytics.nn.modules import C3k2, C3k2_DCRStrip, DCRStrip
from ultralytics.nn.modules.dcr_strip import DiagonalStrip
from ultralytics.nn.tasks import load_checkpoint
from ultralytics.utils import YAML
from ultralytics.utils.torch_utils import ModelEMA, init_seeds


@pytest.fixture(autouse=True)
def bounded_cpu_threads():
    """Keep tiny deterministic checks fast on many-core hosts."""
    previous = torch.get_num_threads()
    torch.set_num_threads(4)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("anti", [False, True])
def test_diagonal_kernel_support_and_gradient(anti):
    """Only the seven chosen positions may affect a kernel, including after an optimizer update."""
    layer = DiagonalStrip(8, anti)
    expected = torch.eye(7).flip(1) if anti else torch.eye(7)
    kernel = layer.kernel()[:, 0]
    assert torch.count_nonzero(kernel * (1 - expected)) == 0
    assert torch.count_nonzero(kernel) == 8 * 7
    x = torch.arange(99, dtype=torch.float32).view(1, 1, 9, 11).expand(1, 8, 9, 11)
    layer(x).square().mean().backward()
    assert layer.weight.grad is not None and torch.count_nonzero(layer.weight.grad) == 8 * 7
    assert "mask" in dict(layer.named_buffers()) and "mask" not in dict(layer.named_parameters())


@pytest.mark.parametrize("normal", DCRStrip.normals)
def test_normal_sampling_replicates_edges(normal):
    """Explicit clamped-grid reference checks signs, axes, diagonal normals, and all boundaries."""
    t = torch.arange(35, dtype=torch.float32).view(1, 1, 5, 7).square()
    c, d = DCRStrip.contrast(t, normal)
    dy, dx = normal
    for y in range(5):
        for x in range(7):
            plus = t[..., min(4, max(0, y + dy)), min(6, max(0, x + dx))]
            minus = t[..., min(4, max(0, y - dy)), min(6, max(0, x - dx))]
            assert c[..., y, x] == t[..., y, x] - 0.5 * (plus + minus)
            assert d[..., y, x] == (plus - minus).abs()
    constant_c, constant_d = DCRStrip.contrast(torch.ones(2, 8, 5, 7), normal)
    assert torch.count_nonzero(constant_c) == torch.count_nonzero(constant_d) == 0


def test_four_strip_orientations():
    """A unit impulse exposes all four spatial supports without Reduce/BN obscuring the geometry."""
    dcr = DCRStrip(32)
    assert dcr.horizontal.kernel_size == (1, 7) and dcr.vertical.kernel_size == (7, 1)
    for layer in (dcr.horizontal, dcr.vertical):
        assert layer.groups == 8 and layer.bias is None and layer.padding == (0, 0)
    impulse = torch.zeros(1, 8, 9, 9)
    impulse[..., 4, 4] = 1
    for layer, expected in ((dcr.diagonal, torch.eye(7)), (dcr.antidiagonal, torch.eye(7).flip(1))):
        actual = layer(impulse)[0, 0, 1:8, 1:8]
        assert torch.equal(actual.ne(0), expected.bool())


@pytest.mark.parametrize("shape", [(2, 64, 17, 23), (1, 64, 1, 3), (1, 64, 20, 13)])
def test_bypass_split_and_initialization_stream(shape):
    """Preserve original C3k2 tensors/RNG when c1 differs from c2, and honor both parent entry points."""
    torch.manual_seed(12)
    baseline = C3k2(64, 128, n=1, e=0.25).eval()
    after_baseline = torch.rand(4)
    torch.manual_seed(12)
    candidate = C3k2_DCRStrip(64, 128, n=1, e=0.25).eval()
    assert torch.equal(after_baseline, torch.rand(4))
    assert all(torch.equal(v, candidate.state_dict()[k]) for k, v in baseline.state_dict().items())
    x = torch.randn(shape)
    with torch.no_grad():
        assert_close_tree(candidate(x), candidate.forward_split(x))
        candidate.dcr.enabled = False
        with patch.object(candidate.dcr, "residual", side_effect=AssertionError):
            torch.testing.assert_close(baseline(x), candidate(x))
            torch.testing.assert_close(baseline.forward_split(x), candidate.forward_split(x))


def test_controls_remove_the_intended_mechanism():
    """No-contrast cannot sample sides; fixed fusion cannot run the learned gate."""
    module = DCRStrip(32).eval()
    x = torch.randn(2, 32, 9, 13)
    with torch.no_grad():
        delta, gates = module.residual(x)
        assert torch.equal(gates, torch.full_like(gates, 0.25))
        module.use_contrast = False
        with patch.object(module, "contrast", side_effect=AssertionError):
            module(x)
        module.use_contrast = True
        module.adaptive_fusion = False
        with patch.object(module.gate, "forward", side_effect=AssertionError):
            fixed, fixed_gates = module.residual(x)
        torch.testing.assert_close(delta, fixed)
        torch.testing.assert_close(gates, fixed_gates)
    module(x).square().mean().backward()
    assert module.gate.weight.grad is None
    assert module.alpha.grad is not None


def test_model_structure_and_bypass(tmp_path):
    """Verify the full 640-square and 640x960 detection graph, not just the standalone block."""
    report = structural_checks(tmp_path)
    assert report["added_parameters"] == 9155
    assert report["baseline_parameters"] == REFERENCE["model_parameters_nc1"]


def native_probe(device, weights=None):
    """Construct through Trainer.get_model and native optimizer logic for disposable diagnostics."""
    trainer = object.__new__(AuditedTrainer)
    trainer.args = get_cfg(overrides={k: v for k, v in REFERENCE["args"].items() if k != "save_dir"})
    trainer.args.imgsz = 96  # Diagnostic size only, never persisted as a formal b19/A1 recipe.
    trainer.data = {"nc": 1, "channels": 3, "names": {0: "crack"}}
    trainer.device = torch.device(device)
    init_seeds(42)
    trainer.model = trainer.get_model(str(MODEL), weights, verbose=False).to(device)
    trainer.set_model_attributes()
    trainer.optimizer = trainer.build_optimizer(trainer.model, name="MuSGD", lr=0.01, momentum=0.937, decay=0.0005)
    optimizer_ids = {id(p) for group in trainer.optimizer.param_groups for p in group["params"]}
    assert all(id(p) in optimizer_ids for k, p in trainer.model.named_parameters() if ".dcr." in k)
    return trainer


@pytest.mark.parametrize("device,amp", [("cpu", False), ("cuda:0", True)])
def test_native_detection_loss_optimizer_and_ema(device, amp):
    """Test real YOLO26 o2m/o2o loss using synthetic labeled images, with CUDA AMP when available."""
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA/AMP unavailable")
    trainer = native_probe(device)
    batch = dict(
        img=torch.randint(0, 256, (2, 3, 64, 96), dtype=torch.uint8),
        batch_idx=torch.tensor([0, 1]),
        cls=torch.tensor([[0.0], [0.0]]),
        bboxes=torch.tensor([[0.5, 0.5, 0.25, 0.35], [0.4, 0.6, 0.1, 0.5]]),
    )
    report = gradient_check(trainer, batch, amp)
    assert report["optimizer_step"]
    ema = ModelEMA(trainer.model)
    ema.update(trainer.model)
    assert isinstance(ema.ema.model[4], C3k2_DCRStrip)


def test_real_data_batch_when_available(tmp_path):
    """Use two original training images and their labels; this is not the full b19 batch-32 preflight."""
    root = os.environ.get("DCR_BASELINE_ROOT")
    if not root:
        pytest.skip("Set DCR_BASELINE_ROOT to run a disposable real-data diagnostic")
    root = Path(root)
    weights, _ = load_checkpoint(root / "yolo26n.pt")
    trainer = native_probe("cuda:0" if torch.cuda.is_available() else "cpu", weights)
    trainer.args.imgsz = 640
    data = root / "datasets/Tunnel_Crack_AugFirst_Diverse5x_RandomSplit_7_2_1_seed42"
    images = sorted((data / "images/train").glob("*.jpg"))[:2]
    assert len(images) == 2
    listing = tmp_path / "images.txt"
    listing.write_text("\n".join(str(p) for p in images), encoding="utf-8")
    dataset = YOLODataset(
        img_path=str(listing),
        imgsz=640,
        batch_size=2,
        augment=True,
        hyp=trainer.args,
        data={**trainer.data, "train": str(data / "images/train")},
        cache=False,
    )
    batch = dataset.collate_fn([dataset[0], dataset[1]])
    assert batch["cls"].numel() > 0
    report = gradient_check(trainer, batch, trainer.device.type == "cuda")
    assert report["image_shape"] == [2, 3, 640, 640]
    save_reload_check(trainer.model, tmp_path)


def test_fresh_process_reload_and_fuse(tmp_path):
    """A checkpoint is portable without importing the experiment launcher in the child process."""
    trainer = native_probe("cpu")
    save_reload_check(trainer.model, tmp_path)


def test_discovery_dedup_and_ambiguity(tmp_path):
    """Reject b190, deduplicate identical b19 records, and require explicit paths for different recipes."""
    for name in ("b19_original", "b19_copy", "b190"):
        folder = tmp_path / "runs" / name
        folder.mkdir(parents=True)
        args = copy.deepcopy(REFERENCE["args"])
        if name == "b190":
            args["name"] = "b190_test"
        YAML.save(folder / "args.yaml", args)
        (folder / "results.csv").write_text("epoch,metrics/mAP50(B)\n1,0.1\n", encoding="utf-8")
    find_baseline(tmp_path)
    path = tmp_path / "runs/b19_copy/args.yaml"
    args = YAML.load(path)
    args["epochs"] = 2
    YAML.save(path, args)
    with pytest.raises(ValueError, match="found 2"):
        find_baseline(tmp_path)
    assert find_baseline(tmp_path, path)[1]["epochs"] == 2


def test_formal_directory_is_owned_before_trainer_setup(tmp_path):
    """A competing invocation fails before model/dataset work and never creates a name2 output."""
    target = tmp_path / "A1"
    target.mkdir()
    marker = target / "existing-result.txt"
    marker.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        AuditedTrainer(overrides=dict(project=str(tmp_path), name="A1"))
    assert marker.read_text(encoding="utf-8") == "keep"
    assert not (tmp_path / "A12").exists()


def test_native_setup_and_oom_never_changes_recipe(tmp_path):
    """Exercise native final setup and its real catch boundary, with a disposable two-image CPU diagnostic."""
    root = os.environ.get("DCR_BASELINE_ROOT")
    if not root:
        pytest.skip("Set DCR_BASELINE_ROOT for native final-trainer setup evidence")
    root = Path(root)
    image_dir = root / "datasets/Tunnel_Crack_AugFirst_Diverse5x_RandomSplit_7_2_1_seed42/images/train"
    images = sorted(image_dir.glob("*.jpg"))[:2]
    assert len(images) == 2
    listing = tmp_path / "train.txt"
    listing.write_text("\n".join(str(p) for p in images), encoding="utf-8")
    data = tmp_path / "data.yaml"
    YAML.save(data, dict(path=str(tmp_path), train=str(listing), val=str(listing), names={0: "crack"}))
    cfg = {k: v for k, v in REFERENCE["args"].items() if k != "save_dir"}
    cfg.update(
        model=str(MODEL),
        pretrained=str(root / "yolo26n.pt"),
        data=str(data),
        project=str(tmp_path),
        name="setup_check",
        device="cpu",
        workers=0,
        amp=False,
        batch=2,
        imgsz=64,
        plots=False,
    )
    trainer = AuditedTrainer(overrides=cfg)
    trainer.add_callback("on_pretrain_routine_end", final_model_audit)
    error = torch.cuda.OutOfMemoryError("injected preflight OOM; no full training epoch runs")
    with patch.object(trainer, "preprocess_batch", side_effect=error), patch.object(
        trainer, "_build_train_pipeline", wraps=trainer._build_train_pipeline
    ) as build:
        with pytest.raises(torch.cuda.OutOfMemoryError, match="injected preflight"):
            trainer.train()
        assert build.call_count == 1
    assert trainer.batch_size == trainer.args.batch == 2
    assert trainer._oom_retries == 0
    assert trainer.save_dir == tmp_path / "setup_check"
    assert (trainer.save_dir / "provenance/final_optimizer.json").is_file()
    assert isinstance(trainer.model.model[4], C3k2_DCRStrip)
    assert not trainer.csv.exists() and not trainer.best.exists()
    for loader in (trainer.train_loader, trainer.test_loader):
        loader.close()


def test_gpu_occupancy_uses_selected_uuid(monkeypatch):
    """A remapped logical GPU must query its own physical UUID, not physical index zero."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    with patch("torch.cuda.is_available", return_value=True), patch(
        "torch.cuda.mem_get_info", return_value=(20 * 1024**3, 24 * 1024**3)
    ), patch("torch.cuda.get_device_properties", return_value=SimpleNamespace(uuid="selected-device")), patch(
        "tools.experiments.run_b19_dcrstrip.subprocess.check_output", return_value="999999, 1024"
    ) as query:
        issues = runtime_issues(dict(device="0"))
    command = query.call_args.args[0]
    assert command[command.index("-i") + 1] == "GPU-selected-device"
    assert any("999999" in item for item in issues)
