"""Offline CPU contracts for SGK; synthetic tests do not substitute for server batch32 preflight."""

import copy
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("YOLO_AUTOINSTALL", "false")
os.environ.setdefault("YOLO_OFFLINE", "true")

from tools.experiments import run_b19_sgk_p3 as run
from tools.experiments.finish_b19_sgk_p3 import CurveValidator

import numpy as np
import torch
from torch import nn
from ultralytics.cfg import get_cfg
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.modules import C3k2_SGK_P3
from ultralytics.nn.tasks import DetectionModel, load_checkpoint
from ultralytics.utils.torch_utils import ModelEMA, initialize_weights


class SGKContracts(unittest.TestCase):
    """Exercise formula, initialization, native task gradients, serialization and experiment ownership."""

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_asymmetric_index_reference(self):
        from ultralytics.nn.modules.sgk_p3 import SGK
        from torch.nn import functional as F

        b = torch.arange(32 * 3 * 5, dtype=torch.float32).reshape(1, 32, 3, 5) / 17
        logits = torch.sin(torch.arange(4 * 9 * 3 * 5, dtype=torch.float32)).reshape(1, 4, 9, 3, 5)
        kernel = logits.softmax(2)
        # Independent scalar-clamped addressing, no production pad/slice helpers.
        expected = torch.empty_like(b)
        for c in range(32):
            for y in range(3):
                for x in range(5):
                    expected[0, c, y, x] = sum(
                        kernel[0, c // 8, j, y, x]
                        * b[0, c, min(2, max(0, y + j // 3 - 1)), min(4, max(0, x + j % 3 - 1))]
                        for j in range(9)
                    )
        torch.testing.assert_close(SGK.local_sum(kernel, b), expected, atol=0, rtol=0)
        for j in range(9):
            onehot = torch.zeros_like(kernel)
            onehot[:, :, j] = 1
            reference = F.unfold(F.pad(b, (1, 1, 1, 1), mode="replicate"), 3).reshape(1, 32, 9, 3, 5)[:, :, j]
            torch.testing.assert_close(SGK.local_sum(onehot, b), reference, atol=0, rtol=0)
        module = SGK()
        feature, guide = torch.randn(2, 64, 7, 11), torch.randn(2, 128, 4, 6)
        assert sum(p.numel() for p in module.parameters()) == 4596
        assert not any(isinstance(m, nn.BatchNorm2d) for m in module.modules())
        assert torch.count_nonzero(module.Wk.weight) and not torch.count_nonzero(module.Wk.bias)
        assert all(torch.count_nonzero(getattr(module, k).weight) for k in ("Ps", "DW5", "Pb"))
        initialize_weights(module)
        torch.testing.assert_close(module(feature, guide), feature, atol=0, rtol=0)
        with torch.no_grad():
            module.Po.weight.copy_(torch.eye(32).reshape(32, 32, 1, 1) * 0.25)
        kernel = module.kernels(feature[:, 32:], guide)
        assert kernel.shape == (2, 4, 9, 7, 11) and torch.isfinite(kernel).all()
        torch.testing.assert_close(kernel.sum(2), torch.ones(2, 4, 7, 11))
        output = module(feature, guide)
        torch.testing.assert_close(output[:, :32], feature[:, :32], atol=0, rtol=0)
        assert not torch.equal(output, module(feature, guide + 2))
        assert not torch.equal(kernel, module.kernels(feature[:, 32:], guide + 2))
        # Holding P3 values constant makes all replicate neighbors identical, irrespective of P4.
        constant = torch.ones_like(feature)
        torch.testing.assert_close(module(constant, guide), constant, atol=1e-6, rtol=1e-6)

    def test_split_path(self):
        block = C3k2_SGK_P3(256, 64, 128, c3k=True).eval()
        nn.init.normal_(block.sgk.Po.weight, std=0.02)
        inputs = [torch.randn(2, 256, 8, 12), torch.randn(2, 128, 4, 6)]
        torch.testing.assert_close(block(inputs), block.forward_split(inputs), atol=0, rtol=0)
        for scale in ("s", "m", "l", "x"):
            cfg = run.YAML.load(run.MODEL)
            cfg["scale"] = scale
            with self.assertRaisesRegex(ValueError, "nano"):
                DetectionModel(cfg, verbose=False)

    def test_rng_and_all_common_initialization(self):
        torch.manual_seed(42)
        native = DetectionModel(run.baseline_architecture(), verbose=False).eval()
        native_rng = torch.get_rng_state()
        torch.manual_seed(42)
        model = DetectionModel(str(run.MODEL), nc=1, verbose=False).eval()
        assert torch.equal(native_rng, torch.get_rng_state())
        audit = run.audit_weights(native, model, None)
        assert audit["common_keys"] == 708
        assert audit["added_parameters"] == 4596
        with torch.no_grad():
            x = torch.randn(1, 3, 64, 96)
            run.assert_close_tree(native(x), model(x), 0, 0)
        # Training-mode outputs/BN states must also remain equal before projection updates.
        native.train()
        model.train()
        with torch.no_grad():
            x = torch.randn(32, 3, 64, 96)
            run.assert_close_tree(native(x), model(x), 0, 0)
        run.audit_weights(native, model, None)

    def test_real_pretrained_trainer_initialization(self):
        weight = Path(os.environ.get("SGK_TEST_PRETRAINED", str(run.ROOT / "yolo26n.pt")))
        if not weight.is_file():
            self.skipTest("Set SGK_TEST_PRETRAINED to original yolo26n.pt for offline initialization audit")
        assert run.sha256(weight) == run.PRETRAINED_SHA256
        source, _ = load_checkpoint(weight)
        trainer = object.__new__(run.AuditedTrainer)
        trainer.args = get_cfg(overrides={k: v for k, v in run.REFERENCE["args"].items() if k != "save_dir"})
        trainer.data = dict(nc=1, channels=3, names={0: "crack"})
        torch.manual_seed(42)
        trainer.get_model(str(run.MODEL), source, verbose=False)
        assert len(trainer.weight_audit["loaded_keys"]) == 606
        run.write_json(run.ROOT / "artifacts/sgk_local/pretrained.json", trainer.weight_audit)

    def test_native_musgd_task_gradients_ema_reload(self):
        torch.manual_seed(42)
        model = DetectionModel(str(run.MODEL), nc=1, verbose=False).train()
        model.args = get_cfg(overrides={k: v for k, v in run.REFERENCE["args"].items() if k != "save_dir"})
        model.names = {0: "crack"}
        trainer = object.__new__(DetectionTrainer)
        optimizer = trainer.build_optimizer(model, "MuSGD", 0.01, 0.937, 0.0005, 26400)
        assert type(optimizer).__name__ == "MuSGD"
        params = dict(model.model[16].sgk.named_parameters())
        ema = ModelEMA(model)
        first, last, changed = {}, {}, {}
        for step in range(2):
            batch = dict(
                img=torch.rand(32, 3, 64, 96),
                batch_idx=torch.arange(32),
                cls=torch.zeros(32, 1),
                bboxes=torch.tensor([[0.5, 0.5, 0.35, 0.3]]).repeat(32, 1),
            )
            optimizer.zero_grad(set_to_none=True)
            loss, _ = model(batch)
            assert torch.isfinite(loss).all()
            loss.sum().backward()
            norms = {k: p.grad.norm().item() for k, p in params.items()}
            assert all(torch.isfinite(p.grad).all() for p in params.values())
            if step == 0:
                first = norms
                assert norms["Po.weight"] > 0
                assert all(v == 0 for k, v in norms.items() if k != "Po.weight")
            else:
                last = norms
                assert all(v > 0 for v in norms.values())
            before = {k: p.detach().clone() for k, p in params.items()}
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            ema.update(model)
            changed = {k: not torch.equal(before[k], p) for k, p in params.items()}
        assert all(changed.values())
        self.updated_model = model
        assert torch.count_nonzero(ema.ema.model[16].sgk.Po.weight) > 0
        with tempfile.TemporaryDirectory(dir=run.ROOT / "artifacts/sgk_local") as tmp:
            reload = run.save_reload_check(ema.ema, Path(tmp))
        run.write_json(
            run.ROOT / "artifacts/sgk_local/gradients_reload.json",
            dict(
                first=first,
                second=last,
                second_updated=changed,
                ema_updates=ema.updates,
                reload=reload,
                scope="CPU synthetic labeled batch32 64x96; not server preflight",
            ),
        )

    def test_observed_step_matches_native_and_excludes_decay(self):
        model = DetectionModel(str(run.MODEL), nc=1, verbose=False).train()
        model.args = get_cfg(overrides={k: v for k, v in run.REFERENCE["args"].items() if k != "save_dir"})
        trainer = object.__new__(DetectionTrainer)
        trainer.model = model
        trainer.optimizer = trainer.build_optimizer(model, "MuSGD", 0.01, 0.937, 0.0005, 26400)
        trainer.scaler = torch.amp.GradScaler("cpu", enabled=False)
        trainer.ema = ModelEMA(model)
        report = run.observation_report()
        params = dict(model.model[16].sgk.named_parameters())
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
                assert row["parameters"]["Ps.weight"]["changed"]  # weight decay alone is excluded
                assert not row["parameters"]["Ps.weight"]["different_from_zero_task_replay"]
        assert report["successful_steps"] == 3 and not report["missing_effective_parameters"]
        run.write_json(run.ROOT / "artifacts/sgk_local/observed_step.json", report)

    def test_cuda_large_precision(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        model = DetectionModel(str(run.MODEL), nc=1, verbose=False).cuda().train()
        model.args = get_cfg(overrides={k: v for k, v in run.REFERENCE["args"].items() if k != "save_dir"})
        nn.init.normal_(model.model[16].sgk.Po.weight, std=0.01)
        rows = []
        for h, w in ((640, 640), (640, 960)):
            for amp in (False, True):
                model.zero_grad(set_to_none=True)
                torch.cuda.reset_peak_memory_stats()
                batch = dict(
                    img=torch.rand(1, 3, h, w, device="cuda"),
                    batch_idx=torch.zeros(1, device="cuda"),
                    cls=torch.zeros(1, 1, device="cuda"),
                    bboxes=torch.tensor([[0.5, 0.5, 0.35, 0.3]], device="cuda"),
                )
                with torch.autocast("cuda", enabled=amp):
                    loss, _ = model(batch)
                loss.sum().backward()
                assert torch.isfinite(loss).all()
                assert all(
                    p.grad is not None and torch.isfinite(p.grad).all() for p in model.model[16].sgk.parameters()
                )
                rows.append(
                    dict(
                        shape=[1, 3, h, w],
                        amp=amp,
                        loss=loss.detach().tolist(),
                        peak_bytes=torch.cuda.max_memory_allocated(),
                    )
                )
        run.write_json(
            run.ROOT / "artifacts/sgk_local/cuda.json",
            dict(
                gpu=torch.cuda.get_device_name(),
                cases=rows,
                scope="local development batch1, not server batch32 preflight",
            ),
        )

    def test_recipe_launcher_and_fixed_batch_oom(self):
        raw = copy.deepcopy(run.REFERENCE["args"])
        assert run.launcher_evidence(
            SimpleNamespace(baseline_launcher=run.ROOT / "tools/experiments/b19_launcher_expanded.txt"), raw
        )["verified"]
        assert raw["batch"] == 32 and raw["epochs"] == 200 and raw["optimizer"] == "MuSGD"
        trainer = object.__new__(run.AuditedTrainer)
        trainer._oom_retries = 0
        with self.assertRaises(RuntimeError):
            trainer._oom_retries = 1
        assert trainer._oom_retries == 0
        with self.assertRaisesRegex(RuntimeError, "original OOM"):
            try:
                raise RuntimeError("original OOM")
            except RuntimeError:
                trainer._oom_retries += 1
        assert raw["batch"] == 32

    def test_curve_ties_and_empty_predictions(self):
        from unittest.mock import patch

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


if __name__ == "__main__":
    unittest.main()
