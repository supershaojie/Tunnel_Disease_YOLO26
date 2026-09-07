"""RSC mechanism and local integration tests; these do not issue a server batch-32 preflight receipt."""

import copy
import math
import os
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch

from tools.experiments import b19_common as common
from tools.experiments import verify_b19_rsc_c2psa as verify
from tools.experiments.finish_b19_rsc_c2psa import evaluate
from tools.experiments.run_b19_rsc_c2psa import AuditedTrainer
from ultralytics.cfg import get_cfg
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.nn.modules.block import Attention
from ultralytics.nn.modules.rsc_c2psa import Attention_RSC, reciprocal_probabilities
from ultralytics.nn.tasks import load_checkpoint
from ultralytics.utils import YAML
from ultralytics.utils.torch_utils import ModelEMA, autocast


def test_full_graph(tmp_path):
    """Check actual 640/rectangular outputs, backward, all common initialization and exact native bypass."""
    report = verify.structural_checks(tmp_path)
    assert report["added_parameters"] == 2


@pytest.mark.parametrize(
    "device",
    ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))],
)
def test_probabilities(device):
    """Check the prescribed operator independently across batch/head axes and extreme half scores."""
    verify.probability_checks(device)


@pytest.mark.parametrize("amp", [False, True])
def test_attention_native_bypass_and_optimizer(amp):
    """Use the real native MuSGD/GradScaler/clipping/EMA step on an attention-only CUDA probe."""
    if amp and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    device = "cuda" if amp else "cpu"
    torch.manual_seed(42)
    native = Attention(128, num_heads=2).to(device).eval()
    rng = torch.get_rng_state()
    candidate = Attention_RSC(copy.deepcopy(native)).eval()
    assert torch.equal(rng, torch.get_rng_state())
    torch.testing.assert_close(candidate.beta, torch.full((2,), 0.01, device=device))
    x = torch.randn(2, 128, 20, 20, device=device, requires_grad=True)
    with autocast(amp, device=device), verify.bypass(candidate):
        common.assert_close_tree(native(x), candidate(x), 0, 0)
    probe = SimpleNamespace(model=candidate, ema=ModelEMA(candidate), args=SimpleNamespace(warmup_bias_lr=0.1))
    probe.optimizer = BaseTrainer.build_optimizer(probe, candidate, "MuSGD", 0.01, 0.937, 0.0005, 10000)
    probe.scaler = torch.amp.GradScaler("cuda", enabled=amp)
    group = next(g for g in probe.optimizer.param_groups if any(p is candidate.theta for p in g["params"]))
    assert group["param_group"] == "weight" and group["weight_decay"] == 0.0005
    before = candidate.theta.detach().clone()
    nonzero = False
    steps = []
    handle = probe.optimizer.register_step_post_hook(lambda *args: steps.append(True))
    try:
        for i in range(10):
            with autocast(amp, device=device):
                # Large enough task signal to distinguish real gradients from a decay-only smoke test.
                loss = (candidate(x) * torch.randn_like(x)).sum() / x.shape[0]
            probe.scaler.scale(loss).backward()
            if candidate.theta.grad is not None and torch.isfinite(candidate.theta.grad).all():
                nonzero |= bool(candidate.theta.grad.count_nonzero())
            BaseTrainer.optimizer_step(probe)
        assert steps and nonzero and not torch.equal(before, candidate.theta)
        assert probe.ema.updates == 10 and torch.isfinite(probe.ema.ema.theta).all()
    finally:
        handle.remove()


def test_zero_gradient_fixed_point():
    """A uniform symmetric attention is allowed to have an exactly zero theta gradient."""
    theta = torch.tensor([math.log(0.05 / 0.95)], requires_grad=True)
    scores = torch.zeros(1, 1, 5, 5, requires_grad=True)
    a = scores.softmax(-1)
    r = reciprocal_probabilities(scores)
    beta = 0.2 * theta.sigmoid()
    mixed = (1 - beta) * a + beta * r
    (mixed * torch.randn_like(mixed)).sum().backward()
    assert theta.grad is not None and torch.isfinite(theta.grad).all() and theta.grad.item() == 0


def test_original_weight_reload_and_validator(tmp_path):
    """Audit the real original yolo26n.pt, serialization/fuse and a small local real Validator run."""
    weight = Path(os.environ.get("RSC_TEST_PRETRAINED", common.ROOT / "yolo26n.pt"))
    if not weight.is_file():
        pytest.skip("Set RSC_TEST_PRETRAINED to the original b19 yolo26n.pt; no download or replacement")
    assert common.sha256(weight) == common.PRETRAINED_SHA256
    weights, _ = load_checkpoint(weight)
    probe = object.__new__(AuditedTrainer)
    probe.args = get_cfg()
    probe.data = dict(nc=1, channels=3, names={0: "crack"})
    torch.manual_seed(42)
    model = probe.get_model(str(common.MODEL), weights, False)
    assert len(probe.weight_audit["loaded_keys"]) == 606
    assert probe.weight_audit["all_common_tensors_equal"]
    common.write_json(tmp_path / "weight_audit.json", probe.weight_audit)
    model.args = vars(probe.args)
    verify.save_reload_check(model.eval(), tmp_path)
    # Synthetic labeled images exercise the actual Validator API; no detection-quality claim is made.
    data_root = tmp_path / "dataset"
    for kind in ("images", "labels"):
        (data_root / kind / "val").mkdir(parents=True)
    for i in range(2):
        image = np.zeros((128, 192, 3), dtype=np.uint8)
        cv2.line(image, (20, 20 + i), (150, 90), (255, 255, 255), 3)
        cv2.imwrite(str(data_root / f"images/val/{i}.png"), image)
        (data_root / f"labels/val/{i}.txt").write_text("0 0.45 0.43 0.75 0.6\n")
    data = data_root / "data.yaml"
    YAML.save(data, dict(path=str(data_root), train="images/val", val="images/val", names={0: "crack"}))
    report = evaluate(tmp_path / "preflight.pt", data, tmp_path / "validator", "val", device="cpu", batch=2, workers=0)
    assert report["images"] == 2 and report["targets"] == 2 and report["precision"] == "FP32"


def test_fixed_batch_oom_ownership():
    """The experiment re-raises the original error at the native retry request, before batch mutation."""
    trainer = object.__new__(AuditedTrainer)
    trainer.args = SimpleNamespace(batch=32)
    trainer.batch_size = 32
    original = torch.cuda.OutOfMemoryError("probe")
    with pytest.raises(torch.cuda.OutOfMemoryError) as caught:
        try:
            raise original
        except torch.cuda.OutOfMemoryError:
            trainer._oom_retries += 1
    assert caught.value is original and trainer.args.batch == trainer.batch_size == 32
