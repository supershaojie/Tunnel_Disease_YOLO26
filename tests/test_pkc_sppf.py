"""Offline CPU contracts for PKC; synthetic tests do not substitute for server batch32 preflight."""

import copy
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("YOLO_AUTOINSTALL", "false")
os.environ.setdefault("YOLO_OFFLINE", "true")

from tools.experiments import run_b19_pkc_sppf as run
from tools.experiments.finish_b19_pkc_sppf import CurveValidator

import numpy as np
import torch
from torch import nn
from ultralytics.cfg import get_cfg
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.modules import SPPF, SPPF_PKC
from ultralytics.nn.tasks import DetectionModel, load_checkpoint
from ultralytics.utils.torch_utils import ModelEMA, initialize_weights


class PKCContracts(unittest.TestCase):
    """Exercise formula, initialization, native task gradients, serialization and experiment ownership."""

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_formula_nonzero_and_conditional_residual(self):
        for c1, c2, shortcut in ((256, 256, True), (32, 48, True), (32, 32, False)):
            block = SPPF_PKC(c1, c2, shortcut=shortcut).eval()
            initialize_weights(block)
            torch.nn.init.normal_(block.pkc["project"].weight, std=0.02)
            x = torch.randn(2, c1, 17, 23)
            native = SPPF(c1, c2, shortcut=shortcut).eval()
            initialize_weights(native)
            native.load_state_dict({k: v for k, v in block.state_dict().items() if not k.startswith("pkc.")})
            with torch.no_grad():
                z = block.cv1(x)
                u = block.pkc["reduce"](z)
                l5 = block.pkc["l5"](u)
                l9 = block.pkc["l9"](l5)
                l13 = block.pkc["l13"](l9)
                delta = torch.nn.functional.conv2d(torch.cat((l5, l9, l13), 1), block.pkc["project"].weight)
                assert torch.count_nonzero(delta) > 0
                torch.testing.assert_close(block(x), native(x) + delta, atol=0, rtol=0)
            assert isinstance(block.cv1.act, nn.Identity)
            assert block.pkc["project"].bias is None
            assert sum(p.numel() for p in SPPF_PKC(256, 256).pkc.parameters()) == 30304
            for key, kernel, dilation in (("l5", 5, 1), ("l9", 3, 2), ("l13", 3, 2)):
                conv = block.pkc[key].conv
                assert (conv.kernel_size, conv.dilation, conv.padding, conv.groups, conv.stride) == (
                    (kernel, kernel),
                    (dilation, dilation),
                    (2, 2),
                    32,
                    (1, 1),
                )
                assert conv.bias is None
            for module in block.pkc.modules():
                if isinstance(module, nn.BatchNorm2d):
                    assert torch.all(module.weight == 1) and torch.all(module.bias == 0)

    def test_receptive_support_is_dense_5_9_13(self):
        block = SPPF_PKC(32, 32).eval()
        with torch.no_grad():
            for key in ("l5", "l9", "l13"):
                block.pkc[key].conv.weight.fill_(0.1)
        u = torch.ones(1, 32, 25, 25, requires_grad=True)
        feature = u
        for key, size in (("l5", 5), ("l9", 9), ("l13", 13)):
            feature = block.pkc[key](feature)
            grad = torch.autograd.grad(feature[0, 0, 12, 12], u, retain_graph=True)[0][0, 0]
            expected = torch.zeros_like(grad, dtype=torch.bool)
            radius = size // 2
            expected[12 - radius : 13 + radius, 12 - radius : 13 + radius] = True
            assert torch.equal(grad != 0, expected)

    def test_rng_and_all_common_initialization(self):
        torch.manual_seed(42)
        native = DetectionModel(run.baseline_architecture(), verbose=False).eval()
        native_rng = torch.get_rng_state()
        torch.manual_seed(42)
        model = DetectionModel(str(run.MODEL), nc=1, verbose=False).eval()
        assert torch.equal(native_rng, torch.get_rng_state())
        audit = run.audit_weights(native, model, None)
        assert audit["common_keys"] == 708
        assert audit["added_parameters"] == 30304
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
        weight = Path(os.environ.get("PKC_TEST_PRETRAINED", str(run.ROOT / "yolo26n.pt")))
        if not weight.is_file():
            self.skipTest("Set PKC_TEST_PRETRAINED to original yolo26n.pt for offline initialization audit")
        assert run.sha256(weight) == run.PRETRAINED_SHA256
        source, _ = load_checkpoint(weight)
        trainer = object.__new__(run.AuditedTrainer)
        trainer.args = get_cfg(overrides={k: v for k, v in run.REFERENCE["args"].items() if k != "save_dir"})
        trainer.data = dict(nc=1, channels=3, names={0: "crack"})
        torch.manual_seed(42)
        trainer.get_model(str(run.MODEL), source, verbose=False)
        assert len(trainer.weight_audit["loaded_keys"]) == 606
        run.write_json(run.ROOT / "artifacts/pkc_local/pretrained.json", trainer.weight_audit)

    def test_native_musgd_task_gradients_ema_reload(self):
        torch.manual_seed(42)
        model = DetectionModel(str(run.MODEL), nc=1, verbose=False).train()
        model.args = get_cfg(overrides={k: v for k, v in run.REFERENCE["args"].items() if k != "save_dir"})
        model.names = {0: "crack"}
        trainer = object.__new__(DetectionTrainer)
        optimizer = trainer.build_optimizer(model, "MuSGD", 0.01, 0.937, 0.0005, 26400)
        assert type(optimizer).__name__ == "MuSGD"
        params = dict(model.model[9].pkc.named_parameters())
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
                assert norms["project.weight"] > 0
                assert all(v == 0 for k, v in norms.items() if k != "project.weight")
            else:
                last = norms
                assert all(v > 0 for v in norms.values())
            before = {k: p.detach().clone() for k, p in params.items()}
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            ema.update(model)
            changed = {k: not torch.equal(before[k], p) for k, p in params.items()}
        assert all(changed.values())
        assert torch.count_nonzero(ema.ema.model[9].pkc["project"].weight) > 0
        with tempfile.TemporaryDirectory(dir=run.ROOT / "artifacts/pkc_local") as tmp:
            reload = run.save_reload_check(ema.ema, Path(tmp))
        run.write_json(
            run.ROOT / "artifacts/pkc_local/gradients_reload.json",
            dict(
                first=first,
                second=last,
                second_updated=changed,
                ema_updates=ema.updates,
                reload=reload,
                scope="CPU synthetic labeled batch32 64x96; not server preflight",
            ),
        )

    def test_diagnose_native_dataset_and_reloaded_loss(self):
        import cv2
        import json
        from tools.experiments.finish_b19_pkc_sppf import diagnose
        from ultralytics.utils import YAML

        with tempfile.TemporaryDirectory(dir=run.ROOT / "artifacts/pkc_local") as tmp:
            folder = Path(tmp)
            data = folder / "dataset"
            (data / "images/val").mkdir(parents=True)
            (data / "labels/val").mkdir(parents=True)
            for i in range(16):
                image = np.random.default_rng(i).integers(0, 256, (48, 80, 3), dtype=np.uint8)
                assert cv2.imwrite(str(data / f"images/val/{i:02}.jpg"), image)
                (data / f"labels/val/{i:02}.txt").write_text("0 0.5 0.5 0.3 0.2\n")
            YAML.save(
                data / "data.yaml",
                dict(path=str(data), train="images/val", val="images/val", test="images/val", names={0: "crack"}),
            )
            model = DetectionModel(str(run.MODEL), nc=1, verbose=False).eval()
            model.names = {0: "crack"}
            with torch.no_grad():
                model.model[9].pkc["project"].weight.normal_(std=0.01)
            (folder / "weights").mkdir()
            torch.save(
                dict(model=model, train_args={k: v for k, v in run.REFERENCE["args"].items() if k != "save_dir"}),
                folder / "weights/best.pt",
            )
            diagnose(folder, data / "data.yaml", device="cpu")
            index = json.loads((folder / "diagnostics.json").read_text())
            result = json.loads((folder / index["path"]).read_text())
            assert len(result["images"]) == 16
            assert result["thresholds"] == dict(confidence=0.25, matching_iou=0.5, max_det=300)
            for row in result["images"]:
                assert row["delta_y0_norm_ratio"] > 0
                assert set(row["scales"]) == {"l5", "l9", "l13"}
                assert all(v["task_gradient"]["norm"] > 0 for v in row["scales"].values())
                assert row["detections"]["branch_on"]["TP"] + row["detections"]["branch_on"]["FN"] == 1

    def test_recipe_launcher_and_fixed_batch_oom(self):
        raw = copy.deepcopy(run.REFERENCE["args"])
        assert run.launcher_evidence(SimpleNamespace(baseline_launcher=None), raw)["verified"]
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
