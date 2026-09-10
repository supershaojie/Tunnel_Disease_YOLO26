"""Independent local reference tests for CCA's spatial indexing, arithmetic and gradients."""

import copy

import pytest
import torch
import torch.nn.functional as F

from ultralytics.nn.modules import Concat_CCA_Fusion


def reference(module, up, low, high):
    """Deliberately enumerate pixels and valid coarse candidates, without unfold or phase reshaping."""
    q = F.normalize(module.Wq(low).float(), dim=1, eps=1e-6)
    k = F.normalize(module.Wk(high).float(), dim=1, eps=1e-6)
    v = module.Wv(high).float()
    rows = []
    for y in range(low.shape[2]):
        columns = []
        for x in range(low.shape[3]):
            cy, cx = y // 2, x // 2
            neighbors = [
                (j, i)
                for j in range(cy - 1, cy + 2)
                for i in range(cx - 1, cx + 2)
                if 0 <= j < high.shape[2] and 0 <= i < high.shape[3]
            ]
            score = torch.stack([4 * (q[:, :, y, x] * k[:, :, j, i]).sum(1) for j, i in neighbors], 1)
            weights = score.softmax(1)
            delta = torch.stack([v[:, :, j, i] - v[:, :, cy, cx] for j, i in neighbors], 2)
            columns.append((weights[:, None] * delta).sum(2))
        rows.append(torch.stack(columns, -1))
    delta = torch.stack(rows, -2)
    return torch.cat((up + module.Wo(delta), low), 1)


@pytest.mark.parametrize("shape", [(1, 1), (1, 3), (3, 1), (3, 4)])
def test_reference_forward_gradient(shape):
    """Cover all boundary types and all task-gradient paths with a nonzero output projection."""
    torch.manual_seed(3)
    module = Concat_CCA_Fusion(6, 5, 6)
    torch.nn.init.normal_(module.Wo.weight, std=0.2)
    other = copy.deepcopy(module)
    h, w = shape
    values = [torch.randn(2, c, h * s, w * s, requires_grad=True) for c, s in ((6, 2), (5, 2), (6, 1))]
    clones = [v.detach().clone().requires_grad_() for v in values]
    actual, expected = module(values), reference(other, *clones)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
    probe = torch.randn_like(actual)
    ga = torch.autograd.grad((actual * probe).sum(), [*module.parameters(), *values])
    gb = torch.autograd.grad((expected * probe).sum(), [*other.parameters(), *clones])
    for a, b in zip(ga, gb):
        assert torch.isfinite(a).all()
        torch.testing.assert_close(a, b, atol=1e-5, rtol=2e-5)


def test_mask_parent_constant_and_queries():
    """Check exact exclusion, nearest parent mapping, constant-value cancellation and query dependence."""
    module = Concat_CCA_Fusion(6, 5, 6)
    torch.nn.init.normal_(module.Wo.weight)
    low, high = torch.randn(2, 5, 6, 8), torch.randn(2, 6, 3, 4)
    _, weights, _, _ = module.correspondence(low, high)
    torch.testing.assert_close(weights.sum(1), torch.ones(2, 6, 8))
    for y in range(6):
        for x in range(8):
            for j, (dy, dx) in enumerate(((dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1))):
                if not (0 <= y // 2 + dy < 3 and 0 <= x // 2 + dx < 4):
                    assert torch.count_nonzero(weights[:, j, y, x]) == 0
    parent = torch.arange(12).reshape(1, 1, 3, 4).float()
    assert F.interpolate(parent, scale_factor=2, mode="nearest")[0, 0, 3, 5] == parent[0, 0, 1, 2]
    assert (weights[:, :, 2, 2] - weights[:, :, 3, 3]).abs().max() > 0
    constant = torch.randn(2, 6, 1, 1).expand(-1, -1, 3, 4).contiguous()
    up = F.interpolate(constant, scale_factor=2, mode="nearest")
    assert torch.equal(module([up, low, constant]), torch.cat((up, low), 1))
    _, single, _, _ = module.correspondence(low[:, :, :2, :2], constant[:, :, :1, :1])
    assert torch.equal(single[:, 4], torch.ones_like(single[:, 4]))
    assert single.sum().item() == single[:, 4].sum().item()


@pytest.mark.parametrize("scale", [0.0, 1e-12])
def test_tiny_norms_and_rng(scale):
    """Zero and tiny normalized vectors remain finite; construction preserves CPU and CUDA RNG streams."""
    state = torch.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    module = Concat_CCA_Fusion(6, 5, 6)
    assert torch.equal(state, torch.get_rng_state())
    assert all(torch.equal(a, b) for a, b in zip(cuda, torch.cuda.get_rng_state_all()))
    torch.nn.init.normal_(module.Wo.weight)
    values = [(torch.randn(2, c, s * 2, s * 3) * scale).requires_grad_() for c, s in ((6, 2), (5, 2), (6, 1))]
    result = module(values)
    result.sum().backward()
    assert torch.isfinite(result).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in [*module.parameters(), *values])


def test_observed_step_matches_native_and_excludes_decay(tmp_path):
    from tools.experiments import run_b19_cca_fusion as run
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.models.yolo.detect import DetectionTrainer
    from ultralytics.utils.torch_utils import ModelEMA
    from ultralytics.cfg import get_cfg

    torch.set_num_threads(2)
    model = DetectionModel(str(run.MODEL), nc=1, verbose=False).train()
    model.args = get_cfg(overrides={k: v for k, v in run.REFERENCE["args"].items() if k != "save_dir"})
    trainer = object.__new__(DetectionTrainer)
    trainer.model = model
    trainer.optimizer = trainer.build_optimizer(model, "MuSGD", 0.01, 0.937, 0.0005, 26400)
    trainer.scaler = torch.amp.GradScaler("cpu", enabled=False)
    trainer.ema = ModelEMA(model)
    report = run.observation_report()
    params = dict(model.model[12].named_parameters())
    for step in range(3):
        batch = dict(
            img=torch.rand(2, 3, 64, 96),
            batch_idx=torch.arange(2),
            cls=torch.zeros(2, 1),
            bboxes=torch.tensor([[0.5, 0.5, 0.35, 0.3]]).repeat(2, 1),
        )
        loss, _ = model(batch)
        trainer.scaler.scale(loss.sum()).backward()
        native = copy.deepcopy(trainer)
        # PyTorch optimizer deepcopy omits custom MuSGD scalar attributes.
        native.optimizer.muon, native.optimizer.sgd = trainer.optimizer.muon, trainer.optimizer.sgd
        # deepcopy(Parameter) deliberately omits grad; copy task gradients explicitly.
        for original, replica in zip(model.parameters(), native.model.parameters()):
            replica.grad = original.grad.clone() if original.grad is not None else None
        native.optimizer_step()
        row = {}
        run.observed_optimizer_step(trainer, params, report, row)
        run.assert_close_tree(native.model.state_dict(), model.state_dict(), 0, 0)
        run.assert_close_tree(native.ema.ema.state_dict(), trainer.ema.ema.state_dict(), 0, 0)
        assert row["optimizer_step"]
        if step == 0:
            assert row["parameters"]["Wq.weight"]["changed"]  # weight decay alone is excluded
            assert not row["parameters"]["Wq.weight"]["different_from_zero_task_replay"]
    assert report["successful_steps"] == 3 and not report["missing_effective_parameters"]
    run.write_json(tmp_path / "observed_step.json", report)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_full_resolution_loss(tmp_path):
    """Exercise native FP32 and AMP loss at square/rectangular resolution; batch1 is not real preflight."""
    from tools.experiments import run_b19_cca_fusion as run
    from ultralytics.nn.tasks import DetectionModel

    model = DetectionModel(str(run.MODEL), nc=1, verbose=False).cuda().train()
    model.args = run.get_cfg(overrides={k: v for k, v in run.REFERENCE["args"].items() if k != "save_dir"})
    torch.nn.init.normal_(model.model[12].Wo.weight, std=0.01)
    rows = []
    for h, w in ((640, 640), (640, 960)):
        for amp in (False, True):
            model.zero_grad(set_to_none=True)
            torch.cuda.reset_peak_memory_stats()
            batch = dict(
                img=torch.rand(1, 3, h, w, device="cuda"),
                batch_idx=torch.zeros(1, device="cuda"),
                cls=torch.zeros(1, 1, device="cuda"),
                bboxes=torch.tensor([[0.5, 0.5, 0.2, 0.3]], device="cuda"),
            )
            with torch.autocast("cuda", enabled=amp):
                loss, _ = model(batch)
            loss.sum().backward()
            assert torch.isfinite(loss).all()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.model[12].parameters())
            rows.append(
                dict(
                    shape=[1, 3, h, w],
                    amp=amp,
                    loss=loss.detach().tolist(),
                    peak_bytes=torch.cuda.max_memory_allocated(),
                )
            )
    run.write_json(
        run.ROOT / "artifacts/cuda_resolution_checks.json",
        dict(scope="synthetic batch1; not batch32 preflight", gpu=torch.cuda.get_device_name(), cases=rows),
    )


def test_fixed_recipe_and_package_gate(tmp_path):
    """Reject an incomplete run and verify every archived recipe field, including non-default augmentations."""
    from types import SimpleNamespace
    from tools.experiments import run_b19_cca_fusion as run
    from tools.experiments import finish_b19_cca_fusion as finish

    raw = run.YAML.load(run.ROOT / "tools/experiments/b19_archived_args.yaml")
    assert raw == run.REFERENCE["args"] and len(raw) == 112
    assert run.launcher_evidence(
        SimpleNamespace(baseline_launcher=run.ROOT / "tools/experiments/b19_launcher_expanded.txt"), raw
    )["verified"]
    with pytest.raises(FileNotFoundError):
        finish.completed_run(tmp_path)
    trainer = object.__new__(run.AuditedTrainer)
    with pytest.raises(RuntimeError, match="fixed-batch"):
        trainer._oom_retries = 1
    assert raw["batch"] == 32


def test_curve_ties_and_empty_predictions():
    from unittest.mock import patch
    from types import SimpleNamespace
    import tempfile
    import numpy as np
    from pathlib import Path
    from tools.experiments.finish_b19_cca_fusion import CurveValidator

    for confidence in (np.array([0.9, 0.9, 0.3]), np.array([])):
        with tempfile.TemporaryDirectory() as tmp:
            obj = object.__new__(CurveValidator)
            obj.save_dir, obj.seen = Path(tmp), 2
            flags = np.zeros((len(confidence), 10), bool)
            if len(flags):
                flags[0] = True
            obj.metrics = SimpleNamespace(stats=dict(tp=[flags], conf=[confidence], target_cls=[np.array([0, 0])]))
            with patch("ultralytics.models.yolo.detect.DetectionValidator.get_stats", return_value={}):
                obj.get_stats()
            import json

            curve = json.loads((Path(tmp) / "operating_curves.json").read_text())["curves"]["IoU50"]
            assert curve["confidence"] == ([0.9, 0.3] if len(confidence) else [])
            if len(confidence):
                assert curve["precision"] == [0.5, 1 / 3]
                assert curve["fppi"] == [0.5, 1.0]
