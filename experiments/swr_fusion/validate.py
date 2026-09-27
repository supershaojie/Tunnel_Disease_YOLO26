"""Future b19/SWR evaluation at the same fixed FP32 native end-to-end operating settings."""

from __future__ import annotations

# ruff: noqa: E402 - bootstrap this worktree and disable auto-install before Ultralytics imports.

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ["YOLO_AUTOINSTALL"] = "false"

import numpy as np
import torch

from experiments.swr_fusion.train import DATA_SHA, environment_report, sha256
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.nn.modules.swr import SWRFusion
from ultralytics.nn.tasks import load_checkpoint


def metric_summary(metric, iouv: torch.Tensor) -> dict:
    """Locate AP75 by IoU value and label interpolated same-confidence metrics as curve estimates."""
    index = torch.where(torch.isclose(iouv.cpu(), torch.tensor(0.75)))[0]
    if index.numel() != 1:
        raise ValueError("Expected exactly one IoU=0.75 entry")
    result = {
        "precision": float(metric.mp),
        "recall": float(metric.mr),
        "f1": float(np.mean(metric.f1)) if len(metric.f1) else 0.0,
        "ap50": float(metric.map50),
        "ap75": float(metric.all_ap[:, int(index[0])].mean()) if len(metric.all_ap) else 0.0,
        "map50_95": float(metric.map),
        "native_pr_working_point": "each model's own best smoothed F1; not shared confidence",
        "fixed_confidence": {},
    }
    for threshold in (0.25, 0.50):
        p = (
            float(np.mean([np.interp(threshold, metric.px, curve) for curve in metric.p_curve]))
            if len(metric.all_ap)
            else 0.0
        )
        r = (
            float(np.mean([np.interp(threshold, metric.px, curve) for curve in metric.r_curve]))
            if len(metric.all_ap)
            else 0.0
        )
        result["fixed_confidence"][str(threshold)] = {
            "precision": p,
            "recall": r,
            "f1": 2 * p * r / (p + r) if p + r else 0.0,
            "method": "PR-curve interpolation estimate at matching IoU=0.5, not exact TP/FP/FN",
        }
    return result


def main():
    """Evaluate only when explicitly invoked; default finite development checks live in verify.py."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", default=str(ROOT / "runs/swr_eval"))
    parser.add_argument("--name", required=True)
    parser.add_argument("--diagnostic-samples", type=int, choices=range(5), default=0)
    args = parser.parse_args()
    weights, data = Path(args.weights).resolve(), Path(args.data).resolve()
    if sha256(data) != DATA_SHA:
        raise ValueError("Dataset YAML is not the locked b19 split")
    output = Path(args.project).resolve() / args.name
    output.mkdir(parents=True, exist_ok=False)
    # Load from its own directory: the pinned downloader strips apostrophes from absolute Windows paths.
    os.chdir(weights.parent)
    model, _ = load_checkpoint(weights.name)
    os.chdir(ROOT)
    model.float().eval()
    if model.model[-1].nc != 1 or model.names != {0: "crack"} or not model.end2end:
        raise ValueError("Evaluation requires a native end-to-end nc=1 crack checkpoint")
    unfused = sum(p.numel() for p in model.parameters())
    model.fuse(verbose=False)
    fused = sum(p.numel() for p in model.parameters())
    config = dict(
        task="detect",
        mode="val",
        data=str(data),
        split=args.split,
        device=args.device,
        imgsz=640,
        batch=32,
        conf=0.001,
        iou=0.7,
        max_det=300,
        rect=True,
        augment=False,
        quantize=None,
        end2end=True,
        project=str(output.parent),
        name=output.name,
        save_dir=str(output),
        exist_ok=False,
        plots=True,
        workers=8,
    )
    validator = DetectionValidator(args=config)
    observed, diagnostic_stats, handles = {}, [], []

    def attach(_validator):
        def observe(_model, inputs):
            observed.update(model_dtype=str(next(_model.parameters()).dtype), input_dtype=str(inputs[0].dtype))
            if observed != {"model_dtype": "torch.float32", "input_dtype": "torch.float32"}:
                raise RuntimeError(f"Independent evaluation must remain FP32: {observed}")

        handles.append(model.register_forward_pre_hook(observe))
        if args.diagnostic_samples and isinstance(model.model[16], SWRFusion):

            def diagnose(module, inputs):
                if len(diagnostic_stats) < args.diagnostic_samples:
                    diagnostic_stats.append(module.diagnostics([t[:1] for t in inputs[0]]))

            handles.append(model.model[16].register_forward_pre_hook(diagnose))

    validator.callbacks["on_val_start"].append(attach)
    try:
        validator(model=model)
    finally:
        for handle in handles:
            handle.remove()
    report = {
        "weights": str(weights),
        "weights_sha256": sha256(weights),
        "data_yaml_sha256": sha256(data),
        "source_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "config": config,
        "actual_precision": observed,
        "native_fuse": True,
        "parameters": {"unfused": unfused, "fused": fused},
        "metrics": metric_summary(validator.metrics.box, validator.iouv),
        "diagnostics": diagnostic_stats,
        "environment": environment_report(),
    }
    (output / "evaluation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["metrics"], indent=2))


if __name__ == "__main__":
    main()
