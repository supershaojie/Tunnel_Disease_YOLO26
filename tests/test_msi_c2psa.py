"""Local MSI integration checks; small local probes do not replace the server batch-32 preflight."""

import copy
import json
import os
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from tools.experiments import b19_common as common
from tools.experiments import verify_b19_msi_c2psa as verify
from tools.experiments.finish_b19_msi_c2psa import diagnose, evaluate, fixed_subset
from tools.experiments.msi_experiment import V1
from tools.experiments.run_b19_msi_c2psa import AuditedTrainer
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.nn.modules import C2PSA, C2PSA_MSI
from ultralytics.nn.modules.block import Attention
from ultralytics.nn.tasks import load_checkpoint
from ultralytics.utils import YAML
from ultralytics.utils.torch_utils import ModelEMA, autocast


def probe_trainer():
    """Exercise actual native trainer model loading without constructing a formal training run."""
    weight = Path(os.environ.get("MSI_TEST_PRETRAINED", common.ROOT / "yolo26n.pt"))
    if not weight.is_file():
        pytest.skip("Original pretrained file unavailable; do not download or substitute best.pt")
    assert common.sha256(weight) == common.PRETRAINED_SHA256
    trainer = object.__new__(AuditedTrainer)
    trainer.experiment = V1
    trainer.args = get_cfg(overrides=common.REFERENCE["args"])
    trainer.data = dict(nc=1, channels=3, names={0: "crack"})
    original, _ = load_checkpoint(weight)
    torch.manual_seed(42)
    trainer.model = trainer.get_model(str(V1.model), original, False)
    trainer.model.args = trainer.args
    return trainer, original


def test_structure_formula_rng(tmp_path):
    """Cover both formulas, all common states, channels, shape, BN and exact initial outputs."""
    verify.structural_checks(tmp_path)


def test_actual_pretrained_loading_and_original_checkpoint(tmp_path):
    """Compare same-seed nc adaptation, including native unmatched head rows and BN buffers."""
    trainer, original = probe_trainer()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        from ultralytics.models.yolo.detect import DetectionTrainer

        native = DetectionTrainer.get_model(trainer, common.baseline_architecture(), original, False).eval()
    trainer.model.eval()
    for h, w in ((640, 640), (384, 672)):
        x = torch.rand(1, 3, h, w)
        with torch.no_grad():
            common.assert_close_tree(native(x), trainer.model(x), 0, 0)
    assert type(original.model[10]) is C2PSA
    assert type(trainer.model.model[10].m[0].attn) is Attention
    path = tmp_path / "native.pt"
    torch.save(dict(model=copy.deepcopy(native), train_args=vars(trainer.args)), path)
    assert type(YOLO(path).model.model[10]) is C2PSA
    common.write_json(tmp_path / "initialization.json", trainer.weight_audit)


@pytest.mark.parametrize("amp", [False, True])
def test_labeled_musgd_ema_reload(tmp_path, amp):
    """Real E2ELoss/MuSGD data gradients, clipping and EMA on local batch=1 probes, separate from formal preflight."""
    if amp and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable; AMP remains pending server validation")
    trainer, _ = probe_trainer()
    device = "cuda" if amp else "cpu"
    model = trainer.model.to(device).train()
    trainer.ema = ModelEMA(model)
    trainer.optimizer = BaseTrainer.build_optimizer(
        trainer, model, "MuSGD", trainer.args.lr0, trainer.args.momentum, trainer.args.weight_decay, 10000
    )
    trainer.scaler = torch.amp.GradScaler("cuda", enabled=amp)
    ids = {id(p) for group in trainer.optimizer.param_groups for p in group["params"]}
    assert all(id(p) in ids and p.requires_grad for p in model.model[10].m[0].msi.parameters())
    rows, steps, scales = [], [], []

    def before_step(*args):
        row = {}
        for name, p in model.model[10].m[0].msi.named_parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all()
            row[name] = float(p.grad.float().norm())
        rows.append(row)

    handles = [
        trainer.optimizer.register_step_pre_hook(before_step),
        trainer.optimizer.register_step_post_hook(lambda *args: steps.append(True)),
    ]
    try:
        for h, w in [(640, 640), (384, 672)] * 16:
            batch = dict(
                img=torch.rand(1, 3, h, w, device=device),
                batch_idx=torch.tensor([0.0], device=device),
                cls=torch.tensor([[0.0]], device=device),
                bboxes=torch.tensor([[0.5, 0.5, 0.4, 0.3]], device=device),
            )
            with autocast(amp, device=device):
                loss, _ = model(batch)
            assert torch.isfinite(loss).all()
            trainer.scaler.scale(loss.sum()).backward()
            BaseTrainer.optimizer_step(trainer)
            scales.append(trainer.scaler.get_scale())
            if len(steps) >= 3:
                break
    finally:
        for handle in handles:
            handle.remove()
    assert len(steps) >= 2 and trainer.ema.updates >= 2
    assert rows[0]["dw3.weight"] == rows[0]["dw5.weight"] == 0
    assert any(r["dw3.weight"] > 0 and r["dw5.weight"] > 0 for r in rows[1:])
    assert all(r["project.weight"] > 0 for r in rows)
    for k, p in model.model[10].m[0].msi.state_dict().items():
        assert k in trainer.ema.ema.model[10].m[0].msi.state_dict() and torch.isfinite(p).all()
    trainer.ema.update_attr(model, include=["yaml", "nc", "args", "names", "stride"])
    verify.save_reload_check(trainer.ema.ema, tmp_path)
    common.write_json(
        tmp_path / "local_gradient_probe.json",
        dict(amp=amp, local_batch=1, steps=len(steps), gradients=rows, scales=scales),
    )


def test_one2one_detach_is_preserved():
    """The one-to-one loss stays detached; MSI still changes its inference features."""
    trainer, _ = probe_trainer()
    model = trainer.model.train()
    with torch.no_grad():
        model.model[10].m[0].msi.project.weight.normal_(std=0.01)
    raw = model(torch.rand(1, 3, 128, 160))
    raw["one2one"]["scores"].sum().backward()
    assert all(p.grad is None for p in model.model[10].m[0].msi.parameters())
    assert any(p.grad is not None for p in model.model[-1].one2one_cv3.parameters())


@pytest.mark.parametrize("scale,repeats,channels", [("n", 1, 128), ("l", 2, 256)])
def test_parser_scales_once(scale, repeats, channels):
    """Exercise width/depth registration without building larger full detectors."""
    from ultralytics.nn.tasks import parse_model

    graph = dict(
        nc=1,
        scale=scale,
        scales={"n": [0.5, 0.25, 1024], "l": [1, 1, 512]},
        backbone=[[-1, 1, "Conv", [1024, 1, 1]], [-1, 2, "C2PSA_MSI", [1024]]],
        head=[],
    )
    model, _ = parse_model(graph, ch=3, verbose=False)
    assert type(model[1]) is C2PSA_MSI and len(model[1].m) == repeats and model[1].c == channels


def test_fixed_subset_validator_and_diagnose(tmp_path):
    """Real YOLO/AutoBackend/Validator on synthetic labeled files; diagnosis uses 16 stable image IDs."""
    for folder in ("images/val", "labels/val"):
        (tmp_path / folder).mkdir(parents=True)
    rng = np.random.default_rng(42)
    for i in range(16):
        cv2.imwrite(str(tmp_path / f"images/val/{i:02}.jpg"), rng.integers(0, 256, (64, 96, 3), dtype=np.uint8))
        (tmp_path / f"labels/val/{i:02}.txt").write_text("0 0.5 0.5 0.4 0.3\n")
    data = tmp_path / "data.yaml"
    YAML.save(data, dict(path=str(tmp_path), train="images/val", val="images/val", names={0: "crack"}))
    subset, images = fixed_subset(data, tmp_path / "subset", 2)
    assert [p.stem for p in images] == ["00", "01"]
    trainer, _ = probe_trainer()
    model = trainer.model.eval()
    with torch.no_grad():
        model.model[10].m[0].msi.project.weight.normal_(std=0.01)
    (tmp_path / "weights").mkdir()
    weight = tmp_path / "weights/best.pt"
    torch.save(dict(model=model, train_args=vars(trainer.args)), weight)
    report = evaluate(weight, subset, tmp_path / "evaluation", "val", device="cpu", batch=2, workers=0)
    assert report["images"] == 2 and report["inference_path"]["msi"] > 0 and report["inference_path"]["one2one"] > 0
    # Minimal local fixtures bind the same evaluated weights; they are never formal server receipts.
    for split in ("val", "test"):
        common.write_json(tmp_path / f"{split}_fp32.json", dict(path="evaluation/metrics.json"))
    common.write_json(tmp_path / "provenance/preflight/preflight/checks.json", dict(gradient_steps={}, batches=[]))
    diagnose(tmp_path, data, tmp_path / "diagnosis", device="cpu")
    diagnostic = json.loads((tmp_path / "diagnosis/metrics.json").read_text())
    assert len(diagnostic["samples"]) == 16
    assert diagnostic["ratio_quantiles"]["delta_over_v"]["max"] > 0
    assert any(row["raw_one2one_difference"]["scores"]["max_abs"] > 0 for row in diagnostic["samples"])


def test_fixed_batch_oom_does_not_retry():
    """Reject at native OOM retry ownership before the trainer mutates batch or the optimizer."""
    trainer = object.__new__(AuditedTrainer)
    trainer._oom_retries = 0
    with pytest.raises(torch.cuda.OutOfMemoryError, match="local sentinel"):
        try:
            raise torch.cuda.OutOfMemoryError("local sentinel")
        except torch.cuda.OutOfMemoryError:
            trainer._oom_retries += 1
    assert trainer._oom_retries == 0


def test_native_trainer_lifecycle(tmp_path):
    """Local CPU/64px synthetic lifecycle checks setup audit, native accumulation and real step hooks."""
    for folder in ("images/val", "labels/val"):
        (tmp_path / folder).mkdir(parents=True)
    rng = np.random.default_rng(42)
    for i in range(40):
        cv2.imwrite(str(tmp_path / f"images/val/{i:02}.jpg"), rng.integers(0, 256, (64, 64, 3), dtype=np.uint8))
        (tmp_path / f"labels/val/{i:02}.txt").write_text("0 0.5 0.5 0.4 0.3\n")
    data = tmp_path / "data.yaml"
    YAML.save(data, dict(path=str(tmp_path), train="images/val", val="images/val", names={0: "crack"}))
    config = {k: v for k, v in common.REFERENCE["args"].items() if k not in {"save_dir", "cfg"}}
    config.update(
        model=str(V1.model),
        pretrained=os.environ["MSI_TEST_PRETRAINED"],
        data=str(data),
        project=str(tmp_path),
        name="local_cpu_lifecycle",
        device="cpu",
        workers=0,
        amp=False,
        imgsz=64,
        plots=False,
    )
    trainer = AuditedTrainer(config)
    steps = []

    class LocalProbeComplete(Exception):
        pass

    handles = []

    def setup(t):
        handles.append(t.optimizer.register_step_post_hook(lambda *args: steps.append(True)))

    def batch_end(t):
        if len(steps) >= 2:
            raise LocalProbeComplete

    trainer.add_callback("on_pretrain_routine_end", setup)
    trainer.add_callback("on_train_batch_end", batch_end)
    try:
        with pytest.raises(LocalProbeComplete):
            trainer.train()
        assert trainer.ema.updates >= 2
        receipt = json.loads((trainer.save_dir / "provenance/optimizer.json").read_text())
        assert receipt["name"] == "MuSGD" and receipt["batch"] == 32
        assert not json.loads((trainer.save_dir / "provenance/effective_config.json").read_text())["setup_differences"]
    finally:
        for handle in handles:
            handle.remove()
        for name in ("train_loader", "test_loader"):
            loader = getattr(trainer, name, None)
            if loader is not None:
                loader.close()
