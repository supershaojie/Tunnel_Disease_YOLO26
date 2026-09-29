"""Evaluate b19, DTR, SCE or DTR+SCE checkpoints using one fixed independent FP32 protocol."""

import argparse
import inspect
import json
import sys
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from experiments.dtr_sce.train import audit_data, environment, identity, sha256, write_report
from experiments.sce_fusion.validate import Diagnostics, FP32Validator
from experiments.sce_fusion.validate import summarize_metrics as sce_metrics
from ultralytics import YOLO
from ultralytics.nn.modules import C2PSA_DTR, SCEFusion
from ultralytics.utils.files import WorkingDirectory
from ultralytics.utils.metrics import ConfusionMatrix, smooth
from ultralytics.utils.torch_utils import init_seeds

EXPECTED_COUNTS = {
    "b19": (2504190, 2375031),
    "dtr": (2818924, 2689637),
    "sce": (2785081, 2653554),
    "dtr_sce": (3099815, 2968160),
}


def numerical_settings():
    """Apply the single-module evaluation settings and report the actual backend flags."""
    init_seeds(42, deterministic=True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    return {
        "seed": 42,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
    }


def load_for_evaluation(weights, variant="dtr_sce"):
    """Restore the learned graph through YOLO; never consult pretrained fields or remap native keys."""
    weights = Path(weights).resolve(strict=True)
    with WorkingDirectory(weights.parent):
        model = YOLO(weights.name)
    identity(model.model, variant)
    model.model.float().eval()
    return model


def parameter_audit(model, variant="dtr_sce"):
    """Measure native unfused/fused counts without mutating the caller's model."""
    identity(model, variant)
    candidate = deepcopy(model).cpu().float().eval()
    if not any(isinstance(m, torch.nn.BatchNorm2d) for m in candidate.modules()):
        raise ValueError("Parameter audit requires an unfused training checkpoint")
    unfused = sum(p.numel() for p in candidate.parameters())
    candidate.fuse(verbose=False)
    fused = sum(p.numel() for p in candidate.parameters())
    if (unfused, fused) != EXPECTED_COUNTS[variant]:
        raise ValueError(f"Measured {variant} counts {(unfused, fused)} differ from {EXPECTED_COUNTS[variant]}")
    return {
        "measured": {"unfused": unfused, "native_fused": fused},
        "static_expected": dict(zip(("unfused", "native_fused"), EXPECTED_COUNTS[variant])),
        "fuse_note": "Native BN folding and end-to-end Detect O2M removal; no speed inference from counts",
    }


def summarize_metrics(validator):
    """Reuse native AP/curve summaries and recover the exact smoothed-F1 confidence index."""
    result = sce_metrics(validator)
    result.pop("curves")  # Full arrays are saved once, losslessly, in curves.npz.
    box = validator.metrics.box
    if len(box.f1_curve):
        index = int(smooth(box.f1_curve.mean(0), 0.1).argmax())
        best = {
            "index": index,
            "confidence": float(box.px[index]),
            "precision": float(box.p_curve[:, index].mean()),
            "recall": float(box.r_curve[:, index].mean()),
            "f1": float(box.f1_curve[:, index].mean()),
            "method": "smooth(mean F1, 0.1).argmax()",
        }
        if not np.allclose(
            [best["precision"], best["recall"], best["f1"]],
            [result["precision"], result["recall"], result["f1"]],
            rtol=0,
            atol=1e-12,
        ):
            raise RuntimeError("Metric curves and native working point disagree")
    else:
        best = {"confidence": None, "reason": "No native F1 curve"}
    result["native_best_F1_confidence"] = best
    result["ap_by_iou"] = box.all_ap.mean(0).tolist() if len(box.all_ap) else [0.0] * len(validator.iouv)
    result["images"] = int(validator.seen)
    result["targets"] = int(validator.metrics.nt_per_class.sum())
    for row in result["fixed_confidence"].values():
        row["method"] = "curve_estimate"
        row["note"] = "Interpolated P/R at IoU0.5, derived F1; no exact TP/FP/FN"
    return result


def save_evaluation(validator, target, provenance):
    """Persist the exact metric schema, raw curve arrays, actual precision and native timings."""
    box = validator.metrics.box
    np.savez_compressed(
        target / "curves.npz",
        **{
            name: np.asarray(getattr(box, name))
            for name in ("all_ap", "ap_class_index", "p_curve", "r_curve", "f1_curve", "px")
        },
        iou_thresholds=validator.iouv.cpu().numpy(),
    )
    np.save(target / "confusion_matrix.npy", validator.confusion_matrix.matrix)
    report = {
        **provenance,
        "status": "PASS",
        "metrics": summarize_metrics(validator),
        "actual_configuration": vars(validator.args),
        "model_dtype": validator.actual_model_dtype,
        "input_dtype": validator.actual_input_dtype,
        "native_fuse": True,
        "speed_ms_per_image": validator.speed,
        "timing_scope": "Native logged timings under the recorded conditions; concurrent GPU use is not a fair benchmark",
        "confusion_matrix": {
            "confidence": 0.25 if validator.args.conf in {None, 0.001} else validator.args.conf,
            "iou": inspect.signature(ConfusionMatrix.process_batch).parameters["iou_thres"].default,
            "note": "Different operating point; do not infer native summary P/R from matrix counts",
        },
    }
    write_report(target / "metrics_exact.json", report)
    return report


def diagnostic_samples(model, selected, device):
    """Reuse bounded SCE hooks and existing DTR aggregates, restoring every observer at exit."""
    if not selected:
        return []
    if len(selected) > 4:
        raise ValueError("Diagnostics are bounded to four images")
    args = {
        "device": device,
        "imgsz": 640,
        "batch": 1,
        "quantize": None,
        "rect": True,
        "augment": False,
        "save": False,
        "verbose": False,
    }
    model.predict(selected[:1], **args)  # Native predictor warmup is outside the diagnostic sample allowance.
    graph = model.model.model
    sce = Diagnostics(graph[23], len(selected)) if isinstance(graph[23], SCEFusion) else None
    blocks = list(graph[10].dtr_blocks) if isinstance(graph[10], C2PSA_DTR) else []
    previous = [(b.diagnostics_enabled, b.diagnostics) for b in blocks]
    records = []
    try:
        for block in blocks:
            block.diagnostics_enabled = True
        for sample in selected:
            model.predict([sample], **args)
            records.append(
                {
                    "image": str(sample),
                    "dtr": [deepcopy(b.diagnostics) for b in blocks],
                    "sce": sce.records[-1] if sce else None,
                }
            )
    finally:
        if sce:
            sce.close()
        for block, (enabled, stats) in zip(blocks, previous):
            block.diagnostics_enabled, block.diagnostics = enabled, stats
    return records


def parser():
    """Expose the common four-variant evaluation CLI."""
    result = argparse.ArgumentParser(description=__doc__)
    for flag in ("weights", "data", "project", "name"):
        result.add_argument(f"--{flag}", required=True)
    result.add_argument("--device", default="0")
    result.add_argument("--split", choices=("val", "test"), default="val")
    result.add_argument("--variant", choices=tuple(EXPECTED_COUNTS), default="dtr_sce")
    result.add_argument("--diagnostic-samples", type=int, choices=range(5), default=0)
    return result


def main():
    """Run full val/test only when this separate entry is explicitly invoked."""
    options = parser().parse_args()
    if not options.name or Path(options.name).name != options.name or options.name in {".", ".."}:
        raise ValueError("--name must be one output directory name")
    target = Path(options.project).expanduser().resolve() / options.name
    if target.exists():
        raise FileExistsError(f"Evaluation output already exists: {target}")
    settings = numerical_settings()
    dataset = audit_data(options.data)
    model = load_for_evaluation(options.weights, options.variant)
    model_identity = identity(model.model, options.variant)
    counts = parameter_audit(model.model, options.variant)
    model.model.fuse(verbose=False)
    target.mkdir(parents=True, exist_ok=False)
    args = {
        "data": str(Path(options.data).resolve()),
        "project": str(target.parent),
        "name": target.name,
        "save_dir": str(target),
        "device": options.device,
        "split": options.split,
        "imgsz": 640,
        "batch": 32,
        "workers": 8,
        "conf": 0.001,
        "iou": 0.7,
        "max_det": 300,
        "rect": True,
        "augment": False,
        "quantize": None,
        "end2end": True,
        "plots": True,
        "save_json": False,
        "exist_ok": False,
        "task": "detect",
        "mode": "val",
        "seed": 42,
        "deterministic": True,
    }
    validator = FP32Validator(args=args)
    validator(model=model.model)
    records = diagnostic_samples(
        model, validator.dataloader.dataset.im_files[: options.diagnostic_samples], options.device
    )
    report = save_evaluation(
        validator,
        target,
        {
            "checkpoint": str(Path(options.weights).resolve()),
            "checkpoint_sha256": sha256(options.weights),
            "identity": model_identity,
            "parameters": counts,
            "dataset": dataset,
            "environment": environment(options.device),
            "numerical_settings": settings,
            "requested_configuration": args,
            "diagnostics": records,
        },
    )
    metrics = report["metrics"]
    print(" ".join(f"{k}={metrics[k]:.8f}" for k in ("precision", "recall", "f1", "AP50", "AP75", "mAP50_95")))
    print(
        json.dumps({"result_directory": str(target), "summary": str(target / "metrics_exact.json")}, ensure_ascii=False)
    )


if __name__ == "__main__":
    main()
