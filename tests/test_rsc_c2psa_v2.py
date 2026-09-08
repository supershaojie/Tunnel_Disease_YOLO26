"""Local v2 integration probes; none substitute for the fixed batch-32 server preflight."""

import copy
import json
import os
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from tools.experiments import b19_common as common
from tools.experiments import verify_b19_rsc_c2psa as verify
from tools.experiments.finish_b19_rsc_c2psa import diagnose
from tools.experiments.rsc_experiment import V2
from tools.experiments.run_b19_rsc_c2psa import AuditedTrainer, main
from tools.experiments.verify_b19_rsc_c2psa_v2 import probability_checks
from ultralytics.cfg import get_cfg
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.nn.modules.block import Attention
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.rsc_c2psa_v2 import Attention_RSC_V2, reciprocal_logit_correction
from ultralytics.nn.tasks import load_checkpoint
from ultralytics.utils import YAML
from ultralytics.utils.torch_utils import ModelEMA, autocast, fuse_conv_and_bn


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_v2_probabilities(device):
    """Exercise the formula oracle and invariants with separate batch/head indexing."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable; pending server validation")
    probability_checks(device)


@pytest.mark.parametrize("amp", [False, True])
def test_fused_native_bypass_and_precision(amp):
    """Compare equally fused native weights at the same device and precision; verify the correction stays FP32."""
    if amp and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable; pending server validation")
    device = "cuda" if amp else "cpu"
    torch.manual_seed(42)
    native = Attention(128, num_heads=2).eval().to(device)
    candidate = Attention_RSC_V2(copy.deepcopy(native)).eval()
    for model in (native, candidate):
        for module in model.modules():
            if isinstance(module, Conv):
                module.conv = fuse_conv_and_bn(module.conv, module.bn)
                delattr(module, "bn")
                module.forward = module.forward_fuse
    x = torch.randn(1, 128, 20, 16, device=device)
    with torch.no_grad(), autocast(amp, device=device):
        with verify.bypass(candidate):
            common.assert_close_tree(native(x), candidate(x), 0, 0)
        q, k, v = candidate.qkv(x).view(1, 2, 128, 320).split([32, 32, 64], dim=2)
        scores = (q * candidate.scale).transpose(-2, -1) @ k
        corrected, imbalance, delta = reciprocal_logit_correction(scores, candidate.theta)
        assert scores.dtype == (torch.float16 if amp else torch.float32)
        assert all(p.dtype == torch.float32 for p in (corrected, imbalance, delta))
        expected = candidate.proj(
            (v @ corrected.to(v.dtype).transpose(-2, -1)).view(1, 128, 20, 16) + candidate.pe(v.reshape(1, 128, 20, 16))
        )
        common.assert_close_tree(expected, candidate(x), 0, 0)


@pytest.mark.parametrize("amp", [False, True])
def test_labeled_task_update_ema_and_reload(tmp_path, amp):
    """Use the native labeled E2ELoss, MuSGD, GradScaler, clipping and EMA on local single-image integration probes.

    This is not the server warmup/accumulation/memory preflight. That runs the unchanged trainer at batch=32.
    """
    if amp and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable; pending server validation")
    weight = Path(os.environ.get("RSC_TEST_PRETRAINED", common.ROOT / "yolo26n.pt"))
    if not weight.is_file():
        pytest.skip("Original yolo26n.pt unavailable; no replacement download")
    assert common.sha256(weight) == common.PRETRAINED_SHA256
    device = "cuda" if amp else "cpu"
    torch.manual_seed(42)
    original, _ = load_checkpoint(weight)
    trainer = object.__new__(AuditedTrainer)
    trainer.experiment = V2
    trainer.args = get_cfg(overrides=common.REFERENCE["args"])
    trainer.data = dict(nc=1, channels=3, names={0: "crack"})
    model = trainer.get_model(str(V2.model), original, False).to(device).train()
    model.args = trainer.args
    trainer.model, trainer.ema = model, ModelEMA(model)
    trainer.optimizer = BaseTrainer.build_optimizer(
        trainer,
        model,
        trainer.args.optimizer,
        trainer.args.lr0,
        trainer.args.momentum,
        trainer.args.weight_decay,
        10000,
    )
    trainer.scaler = torch.amp.GradScaler("cuda", enabled=amp)
    attn = model.model[10].m[0].attn
    initial_theta, initial_ema = attn.theta.detach().clone(), trainer.ema.ema.model[10].m[0].attn.theta.clone()
    steps, gradients = [], []

    def before_step(optimizer, args, kwargs):
        assert attn.theta.grad is not None and torch.isfinite(attn.theta.grad).all()
        gradients.append(attn.theta.grad.detach().cpu().tolist())

    handles = [
        trainer.optimizer.register_step_pre_hook(before_step),
        trainer.optimizer.register_step_post_hook(lambda *args: steps.append(True)),
    ]
    try:
        for h, w in [(640, 640), (640, 512)] * 4:
            batch = dict(
                img=torch.rand(1, 3, h, w, device=device),
                batch_idx=torch.tensor([0.0], device=device),
                cls=torch.tensor([[0.0]], device=device),
                bboxes=torch.tensor([[0.5, 0.5, 0.4, 0.3]], device=device),
            )
            with autocast(amp, device=device):
                loss, _ = model(batch)
                loss = loss.sum()
            assert torch.isfinite(loss)
            trainer.scaler.scale(loss).backward()
            BaseTrainer.optimizer_step(trainer)
        assert len(steps) >= 2 and torch.tensor(gradients).count_nonzero()
        assert not torch.equal(initial_theta, attn.theta)
        assert trainer.ema.updates == 8 and not torch.equal(initial_ema, trainer.ema.ema.model[10].m[0].attn.theta)
    finally:
        for handle in handles:
            handle.remove()
    common.write_json(
        tmp_path / "task_updates.json",
        dict(
            amp=amp,
            local_batch=1,
            steps=len(steps),
            gradients=gradients,
            theta=attn.theta.detach().cpu().tolist(),
            ema_theta=trainer.ema.ema.model[10].m[0].attn.theta.cpu().tolist(),
        ),
    )
    # Save the UPDATED complete EMA and compare independent copies, never compare it to a fresh baseline.
    verify.save_reload_check(trainer.ema.ema.eval(), tmp_path, V2)
    probe = trainer.ema.ema.float().eval()
    with torch.no_grad():
        enabled = probe(batch["img"])
        with verify.bypass(probe):
            disabled = probe(batch["img"])
    assert (enabled[1]["one2one"]["boxes"] - disabled[1]["one2one"]["boxes"]).abs().max() > 0


def test_fixed_16_diagnostics(tmp_path):
    """Run the real 16-image diagnostic preprocessing and compact report on synthetic images and v2 weights."""
    torch.manual_seed(42)
    from ultralytics.nn.tasks import DetectionModel

    model = DetectionModel(str(V2.model), nc=1, verbose=False)
    model.args = vars(get_cfg(overrides=common.REFERENCE["args"]))
    run = tmp_path / V2.name
    (run / "weights").mkdir(parents=True)
    torch.save(dict(model=model, train_args=model.args), run / "weights/best.pt")
    for kind in ("images", "labels"):
        (tmp_path / f"data/{kind}/val").mkdir(parents=True)
    for i in range(16):
        image = np.zeros((100 + i, 180, 3), dtype=np.uint8)
        cv2.line(image, (10 + i, 10), (150, 90), (255, 255, 255), 2)
        cv2.imwrite(str(tmp_path / f"data/images/val/{i:02}.png"), image)
        (tmp_path / f"data/labels/val/{i:02}.txt").write_text("0 0.5 0.5 0.5 0.5\n")
    data = tmp_path / "data/data.yaml"
    YAML.save(data, dict(path=str(data.parent), train="images/val", val="images/val", names={0: "crack"}))
    diagnose(run, data, run / "diagnostics", V2, device="cpu")
    path = run / "diagnostics/metrics.json"
    report = json.loads(path.read_text())
    assert len(report["samples"]) == 16 and report["version"] == 2
    assert path.stat().st_size < 150000
    for row in report["samples"]:
        assert row["dense_one2one_difference"].keys() == {"boxes_pixels", "scores_probability"}
        for head in row["blocks"][0]["heads"]:
            assert head["D"]["max_abs"] <= 1 and head["delta_logits"]["max_abs"] <= head["beta"]


def test_version_name_isolation():
    """Reject a mismatched experiment name before constructing models or writing an attempt."""
    with pytest.raises(SystemExit) as error:
        main(
            [
                "--version",
                "2",
                "--name",
                "yolo26n_b19_rsc_c2psa_v1",
                "--stage",
                "train",
                "--baseline-root",
                ".",
                "--baseline-args",
                "args.yaml",
            ]
        )
    assert error.value.code == 2
