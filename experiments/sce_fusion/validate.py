"""Evaluate a learned SCE checkpoint with the fixed, independent FP32 b19 comparison protocol."""

import argparse
import sys
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.sce_fusion.train import ROOT, audit_data, environment, sha256, write_report

import numpy as np
import torch

from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.nn.modules import Detect, SCEFusion
from ultralytics.nn.modules.sce import _resize
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.files import WorkingDirectory


def load_for_evaluation(weights):
    """Restore the complete checkpoint using the native user API, with no pretrained remapping."""
    weights = Path(weights).resolve(strict=True)
    # The locked asset resolver strips apostrophes from absolute checkpoint paths. Pass the local filename.
    with WorkingDirectory(weights.parent):
        model = YOLO(weights.name)
    if len(model.model.model) != 28 or not isinstance(model.model.model[23], SCEFusion):
        raise ValueError("Evaluation requires a complete SCE checkpoint with the learned fusion module")
    if type(model.model.model[-1]) is not Detect or model.model.model[-1].nc != 1:
        raise ValueError("Expected the native single-class Detect at graph node 27")
    model.model.float().eval()
    return model


def parameter_audit(model):
    """Measure identical native unfused/fused counts for b19 and SCE, without modifying the supplied model."""
    with torch.random.fork_rng(devices=[]):
        baseline = DetectionModel(str(ROOT / "ultralytics/cfg/models/26/yolo26n.yaml"), nc=1, verbose=False).eval()
    candidate = deepcopy(model).cpu().float().eval()

    def count(m):
        return sum(p.numel() for p in m.parameters())

    result = {"unfused": {"b19": count(baseline), "sce": count(candidate), "new_module": count(candidate.model[23])}}
    baseline.fuse(verbose=False)
    candidate.fuse(verbose=False)
    result["fused"] = {"b19": count(baseline), "sce": count(candidate), "new_module": count(candidate.model[23])}
    for row in result.values():
        row["net_increase"] = row["sce"] - row["b19"]
        row["replaced_parameters"] = 0
    result["static_sce_unfused"] = {
        "projections": 29056,
        "six_contexts": 209280,
        "three_routers": 10872,
        "refine": 2112,
        "out": 29568,
        "lambda": 3,
        "total": 280891,
    }
    result["fuse_note"] = (
        "Native end-to-end Detect removes O2M in addition to folding Conv/BN; both models use that policy."
    )
    result["flops_scope"] = (
        "Not measured. Conv-based tools may omit pooling, interpolation, statistics and elementwise routing."
    )
    return result


class FP32Validator(DetectionValidator):
    """Keep native matching/metrics while checking the actual model and input precision."""

    def init_metrics(self, model):
        """Record the device and check the loaded backend weights."""
        super().init_metrics(model)
        self.actual_model_dtype = str(next(model.model.parameters()).dtype)
        if self.actual_model_dtype != "torch.float32":
            raise RuntimeError(f"Independent evaluation requires FP32 weights, got {self.actual_model_dtype}")

    def preprocess(self, batch):
        """Assert the actual native-preprocessed input is FP32."""
        batch = super().preprocess(batch)
        self.actual_input_dtype = str(batch["img"].dtype)
        if batch["img"].dtype != torch.float32:
            raise RuntimeError("Independent evaluation requires FP32 inputs")
        return batch


def summarize_metrics(validator):
    """Locate AP75 by IoU value, preserving native and fixed-confidence working points separately."""
    box = validator.metrics.box
    index = np.flatnonzero(np.isclose(validator.iouv.cpu().numpy(), 0.75))
    if len(index) != 1:
        raise RuntimeError("Cannot locate the unique IoU=0.75 metric column")
    fixed = {}
    for conf in (0.25, 0.50):
        p = float(np.mean([np.interp(conf, box.px, curve) for curve in box.p_curve])) if len(box.p) else 0.0
        r = float(np.mean([np.interp(conf, box.px, curve) for curve in box.r_curve])) if len(box.r) else 0.0
        fixed[str(conf)] = {
            "precision": p,
            "recall": r,
            "f1": 2 * p * r / (p + r) if p + r else 0.0,
            "method": "curve estimate at IoU=0.5; not exact TP/FP/FN counts",
        }
    return {
        "precision": float(box.mp),
        "recall": float(box.mr),
        "f1": float(np.mean(box.f1)) if len(box.f1) else 0.0,
        "AP50": float(box.map50),
        "AP75": float(box.all_ap[:, index[0]].mean()) if len(box.all_ap) else 0.0,
        "mAP50_95": float(box.map),
        "native_working_point": "native maximum smoothed mean F1",
        "fixed_confidence": fixed,
        "iou_thresholds": validator.iouv.tolist(),
        "curves": [
            {"x": np.asarray(x).tolist(), "y": np.asarray(y).tolist(), "x_label": xl, "y_label": yl}
            for x, y, xl, yl in validator.metrics.curves_results
        ],
    }


class Diagnostics:
    """Observe at most four samples using temporary hooks; retain only aggregated numbers."""

    def __init__(self, module, limit):
        """Attach hooks only for an explicitly requested small diagnostic run."""
        if not 0 <= limit <= 4:
            raise ValueError("diagnostic-samples must be 0..4")
        self.module, self.limit, self.records, self.handles = module, limit, [], []
        self.current = None
        if limit:
            self.handles = [module.register_forward_pre_hook(self.start), module.register_forward_hook(self.finish)]
            for i, router in enumerate(module.routers):
                self.handles.append(router.logits[-1].register_forward_hook(self.logits_hook(i)))
                self.handles.append(router.register_forward_hook(self.message_hook(i)))

    def start(self, module, args):
        """Capture only input norms and sizes for the remaining sample allowance."""
        n = min(self.limit - len(self.records), args[0][0].shape[0])
        self.current = None
        if n > 0:
            self.current = [
                {"lambda": module.lambdas.detach().float().cpu().tolist(), "targets": [{}, {}, {}]} for _ in range(n)
            ]
            self.norms = [x[:n].float().flatten(1).norm(dim=1).clamp_min(1e-12) for x in args[0]]
            self.sizes = [x.shape[-2:] for x in args[0]]

    def logits_hook(self, target):
        """Summarize each group and each candidate, retaining no probability maps."""

        def observe(module, args, output):
            if self.current is None:
                return
            p = output[: len(self.current)].float().reshape(-1, 8, 3, *output.shape[-2:]).softmax(2)
            for row, values in zip(self.current, p):
                flat = values.permute(1, 0, 2, 3).reshape(3, -1)
                row["targets"][target].update(
                    candidate_mean=flat.mean(1).cpu().tolist(),
                    candidate_quantiles=torch.quantile(flat, flat.new_tensor([0, 0.25, 0.5, 0.75, 1]), dim=1)
                    .cpu()
                    .tolist(),
                    group_candidate_mean=values.mean((-2, -1)).cpu().tolist(),
                    candidates=[f"P{j + 3}" for j in SCEFusion.sources[target]] + ["zero"],
                )

        return observe

    def message_hook(self, target):
        """Measure restored external message norms relative to the original-resolution input."""

        def observe(module, args, output):
            if self.current is not None:
                ratios = (
                    _resize(output[: len(self.current)], self.sizes[target]).float().flatten(1).norm(dim=1)
                    / self.norms[target]
                )
                for row, ratio in zip(self.current, ratios):
                    row["targets"][target]["message_over_input_norm"] = float(ratio)

        return observe

    def finish(self, module, args, outputs):
        """Measure the final residual and discard all temporary input references."""
        if self.current is not None:
            for i, (x, y) in enumerate(zip(args[0], outputs)):
                n = len(self.current)
                ratios = (y[:n].float() - x[:n].float()).flatten(1).norm(dim=1) / self.norms[i]
                for row, ratio in zip(self.current, ratios):
                    row["targets"][i]["residual_over_input_norm"] = float(ratio)
            self.records.extend(self.current)
            self.current = self.norms = self.sizes = None

    def close(self):
        """Remove all observers; ordinary model forward always returns only its three features."""
        for handle in self.handles:
            handle.remove()


def main():
    """Run full evaluation only when this separate entry is explicitly invoked."""
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("weights", "data", "project", "name"):
        parser.add_argument(f"--{flag}", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--diagnostic-samples", type=int, choices=range(5), default=0)
    options = parser.parse_args()
    target = Path(options.project).resolve() / options.name
    if target.exists():
        raise FileExistsError(f"Evaluation directory already exists: {target}")
    dataset = audit_data(options.data)
    model = load_for_evaluation(options.weights)
    counts = parameter_audit(model.model)
    model.model.fuse(verbose=False)
    # half=False is normalized to quantize=None by the locked native configuration loader.
    args = dict(
        data=str(Path(options.data).resolve()),
        project=str(target.parent),
        name=target.name,
        device=options.device,
        split=options.split,
        imgsz=640,
        batch=32,
        conf=0.001,
        iou=0.7,
        max_det=300,
        rect=True,
        augment=False,
        workers=8,
        half=False,
        quantize=None,
        end2end=True,
        plots=True,
        save_json=False,
        exist_ok=False,
        task="detect",
        mode="val",
    )
    validator = FP32Validator(args=args)
    validator(model=model.model)
    records = []
    if options.diagnostic_samples:
        # Use selected split samples only after evaluation, with no threshold/checkpoint tuning.
        selected = validator.dataloader.dataset.im_files[: options.diagnostic_samples]
        model.predict(
            source=selected[:1],
            device=options.device,
            imgsz=640,
            batch=1,
            half=False,
            rect=True,
            augment=False,
            save=False,
            verbose=False,
        )
        observer = Diagnostics(model.model.model[23], options.diagnostic_samples)
        try:
            model.predict(
                source=selected,
                device=options.device,
                imgsz=640,
                batch=1,
                half=False,
                rect=True,
                augment=False,
                save=False,
                verbose=False,
            )
            records = observer.records
        finally:
            observer.close()
    write_report(
        target / "evaluation.json",
        {
            "status": "PASS",
            "environment": environment(),
            "checkpoint": str(Path(options.weights).resolve()),
            "checkpoint_sha256": sha256(options.weights),
            "dataset": dataset,
            "configuration": vars(validator.args),
            "model_dtype": validator.actual_model_dtype,
            "input_dtype": validator.actual_input_dtype,
            "native_fuse": True,
            "parameters": counts,
            "metrics": summarize_metrics(validator),
            "diagnostics": records,
            "diagnostic_interpretation": "Descriptive routing statistics, not evidence of complementarity or accuracy gains",
        },
    )


if __name__ == "__main__":
    main()
