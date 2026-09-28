"""Future native FP32 val/test evaluation of a learned CSA checkpoint (never a new-run reconstruction)."""

import argparse
import os
import sys
from pathlib import Path

os.environ["YOLO_AUTOINSTALL"] = "false"
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from experiments.csa_c3k2.common import ROOT, Diagnostics, environment, fusion_audit, sha256, write_report
from ultralytics import YOLO
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.nn.modules import CSAC3k2


def load_for_evaluation(weights):
    """Restore the checkpoint's learned graph through YOLO without invoking experimental initialization."""
    path = Path(weights).expanduser().resolve(strict=True)
    model = YOLO(str(path))
    if not isinstance(model.model.model[4], CSAC3k2) or model.model.yaml["nc"] != 1 or not model.model.end2end:
        raise ValueError("Expected a single-class, native end-to-end CSA-C3k2 checkpoint")
    return model


def metric_report(validator):
    """Use the actual IoU grid for AP75 and explicitly label fixed-confidence interpolation as estimates."""
    box = validator.metrics.box
    grid = validator.iouv.detach().cpu().numpy()
    match = np.flatnonzero(np.isclose(grid, 0.75))
    if len(match) != 1:
        raise ValueError(f"Expected exactly one IoU=0.75 metric column: {grid}")
    fixed = {}
    for threshold in (0.25, 0.50):
        if len(box.p):
            precision = float(np.interp(threshold, box.px, box.p_curve.mean(0)))
            recall = float(np.interp(threshold, box.px, box.r_curve.mean(0)))
            fixed[str(threshold)] = {
                "Precision": precision,
                "Recall": recall,
                "F1": 2 * precision * recall / (precision + recall + 1e-16),
                "method": "curve estimate at IoU=0.5; not exact TP/FP/FN",
            }
        else:
            fixed[str(threshold)] = {"status": "UNVERIFIED", "reason": "Native evaluator returned no per-class curves"}
    return {
        "Precision": float(box.mp),
        "Recall": float(box.mr),
        "F1": float(np.mean(box.f1)) if len(box.f1) else 0.0,
        "AP50": float(box.map50),
        "AP75": float(box.all_ap[:, match[0]].mean()) if len(box.all_ap) else 0.0,
        "mAP50-95": float(box.map),
        "iou_grid": grid.tolist(),
        "fixed_confidence": fixed,
        "native_working_point": "P/R/F1 at the native best smoothed mean-F1 confidence; not a shared confidence across experiments",
        "curves": [
            {"x": np.asarray(x).tolist(), "y": np.asarray(y).tolist(), "x_label": x_label, "y_label": y_label}
            for x, y, x_label, y_label in box.curves_results
        ]
        if len(box.p)
        else [],
    }


def main():
    """Run only when explicitly requested later; this development task does not call full validation."""
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("weights", "data", "project", "name"):
        parser.add_argument("--" + field, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--diagnostic-samples", type=int, choices=range(5), default=0)
    args = parser.parse_args()
    os.chdir(ROOT)
    weights = Path(args.weights).expanduser().resolve(strict=True)
    data = Path(args.data).expanduser().resolve(strict=True)
    if Path(args.name).name != args.name or args.name in {".", ".."}:
        raise ValueError("--name must be one directory name")
    output = Path(args.project).expanduser().resolve() / args.name
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "FAIL",
        "environment": environment(),
        "checkpoint": str(weights),
        "checkpoint_sha256": sha256(weights),
        "data": str(data),
        "data_sha256": sha256(data),
    }
    diagnostics = None
    try:
        wrapper = load_for_evaluation(weights)
        model = wrapper.model.float().eval()
        report["parameters"], fused = fusion_audit(model)
        report["parameters"]["note"] = (
            "Native end-to-end fusion also removes O2M; functional deform_conv2d and aggregation are not fully covered by THOP"
        )
        # Explicit legacy half=False is normalized by this locked version to quantize=None.
        config = dict(
            task="detect",
            mode="val",
            data=str(data),
            project=str(output.parent),
            name=output.name,
            device=args.device,
            split=args.split,
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
        )
        validator = DetectionValidator(args=config, save_dir=output)
        report["requested_configuration"] = config
        observed = {}

        def capture_dtype(module, inputs):
            observed["input"] = str(inputs[0].dtype)
            observed["parameters"] = sorted({str(p.dtype) for p in module.parameters()})
            assert inputs[0].dtype == torch.float32 and all(p.dtype == torch.float32 for p in module.parameters())

        dtype_hook = fused.register_forward_pre_hook(capture_dtype)

        # Install diagnostic hooks only after native warmup so they observe real split images.
        def start_diagnostics(current):
            nonlocal diagnostics
            diagnostics = Diagnostics(fused, args.diagnostic_samples)

        validator.add_callback("on_val_start", start_diagnostics)
        try:
            validator(model=fused)
        finally:
            dtype_hook.remove()
            if diagnostics:
                diagnostics.close()
        report.update(
            status="PASS",
            metrics=metric_report(validator),
            actual_configuration=vars(validator.args),
            actual_dtype=observed,
            diagnostics=diagnostics.records if diagnostics else {},
            fuse=True,
        )
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        write_report(output / "evaluation.json", report)


if __name__ == "__main__":
    main()
