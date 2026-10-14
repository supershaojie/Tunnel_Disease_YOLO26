"""Shared B19 matching/AP, exact confidence counts and lossless prediction exchange."""

import csv
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from .b19.matching import match_predictions
from .b19.metrics import ap_per_class, box_iou, smooth

IOUV = torch.linspace(0.5, 0.95, 10)


def correct(pred, gt, thresholds=IOUV):
    """Use the frozen B19 greedy IoU matching, including duplicate predictions."""
    pred = torch.as_tensor(pred, dtype=torch.float32).reshape(-1, 6)
    gt = torch.as_tensor(gt, dtype=torch.float32).reshape(-1, 5)
    return match_predictions(
        SimpleNamespace(iouv=thresholds), pred[:, 5], gt[:, 4], box_iou(gt[:, :4], pred[:, :4])
    ).numpy()


def counts(records, threshold):
    """Re-match retained detections at IoU 0.5 to obtain integer TP/FP/FN."""
    tp, fp, fn = 0, 0, 0
    for record in records:
        pred = record["predictions"]
        kept = pred[pred[:, 4] >= threshold]
        matched = int(correct(kept, record["gt"], torch.tensor([0.5])).sum())
        tp += matched
        fp += len(kept) - matched
        fn += len(record["gt"]) - matched
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {
        "confidence": float(threshold),
        "iou": 0.5,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "P": p,
        "R": r,
        "F1": 2 * p * r / (p + r) if p + r else 0.0,
        "confusion_matrix": [[tp, fp], [fn, 0]],
        "matrix_axes": "rows=prediction(crack/background), columns=GT(crack/background); TN undefined",
    }


def compute(records, split, frozen_threshold=None, collection_conf=0.001):
    """Compute AP from all low-confidence predictions; counts use separate filters."""
    if not records:
        raise ValueError("No evaluated images")
    predictions = np.concatenate([r["predictions"] for r in records], axis=0)
    target_cls = np.concatenate([r["gt"][:, 4] for r in records])
    tp = np.concatenate([correct(r["predictions"], r["gt"]) for r in records], axis=0)
    ap = ap_per_class(tp, predictions[:, 4], predictions[:, 5], target_cls)
    _, _, p, r, f1, all_ap, classes, pc, rc, fc, x, pr = ap
    have_gt = len(classes) > 0
    index = int(smooth(fc.mean(0), 0.1).argmax()) if have_gt else 0
    threshold = max(float(x[index]), collection_conf)
    peak = {
        "P": float(p.mean()) if have_gt else None,
        "R": float(r.mean()) if have_gt else None,
        "F1": float(f1.mean()) if have_gt else None,
        "confidence": float(x[index]),
        "method": "B19 interpolated curves, smoothed mean F1 maximum; diagnostic only on test",
    }
    i75 = int(torch.where(torch.isclose(IOUV, torch.tensor(0.75)))[0].item())
    metrics = {
        "units": "fraction",
        "split": split,
        "images": len(records),
        "boxes": int(len(target_cls)),
        "predictions": int(len(predictions)),
        "IoU_thresholds": IOUV.tolist(),
        "AP50": float(all_ap[:, 0].mean()) if have_gt else None,
        "AP75": float(all_ap[:, i75].mean()) if have_gt else None,
        "mAP50-95": float(all_ap.mean()) if have_gt else None,
        "legacy_curve_peak": peak,
        "fixed_confidence": {
            str(c): counts(records, c)
            if c >= collection_conf
            else {"status": "UNVERIFIED", "reason": "threshold below collected prediction confidence", "confidence": c}
            for c in [0.25, 0.50]
        },
        "threshold_selection": None,
        "val_frozen": None,
        "P_R_display_rule": "val_frozen when supplied; fixed thresholds and legacy diagnostics separately named",
    }
    if split == "val":
        metrics["threshold_selection"] = {
            "confidence": threshold,
            "source_split": "val",
            "method": "B19 smoothed interpolated F1 maximum",
            "exact_counts": counts(records, threshold),
        }
    if frozen_threshold is not None:
        if frozen_threshold < collection_conf:
            raise ValueError("Frozen val threshold is below the prediction collection confidence")
        metrics["val_frozen"] = counts(records, frozen_threshold)
        metrics["val_frozen"]["threshold_source"] = "independent val evaluation of same checkpoint/protocol"
    curves = {
        "confidence": x,
        "precision": pc,
        "recall": rc,
        "F1": fc,
        "PR_recall": x,
        "PR_precision": pr,
        "AP": all_ap,
        "IoU_thresholds": IOUV.numpy(),
    }
    return metrics, curves


def save_predictions(directory, records):
    """Exchange all predictions and GT in float32 without truncating low scores."""
    directory = Path(directory)
    pred = np.concatenate([r["predictions"] for r in records], axis=0)
    gt = np.concatenate([r["gt"] for r in records], axis=0)
    ids = np.array([r["image_id"] for r in records])
    offsets = np.cumsum([0] + [len(r["predictions"]) for r in records], dtype=np.int64)
    gt_offsets = np.cumsum([0] + [len(r["gt"]) for r in records], dtype=np.int64)
    np.savez_compressed(
        directory / "predictions.npz",
        boxes=pred[:, :4].astype(np.float32),
        scores=pred[:, 4].astype(np.float32),
        classes=pred[:, 5].astype(np.int32),
        image_offsets=offsets,
        image_ids=ids,
    )
    np.savez_compressed(
        directory / "ground_truth.npz",
        boxes=gt[:, :4].astype(np.float32),
        classes=gt[:, 4].astype(np.int32),
        image_offsets=gt_offsets,
        image_ids=ids,
    )


def summary(path, metrics):
    """Write a stable cross-model numeric summary with unambiguous units."""
    counts_at_val = metrics.get("val_frozen") or {}
    fields = [
        "model",
        "split",
        "units",
        "images",
        "boxes",
        "P",
        "R",
        "F1",
        "AP50",
        "AP75",
        "mAP50-95",
        "protocol_sha256",
        "weight_sha256",
    ]
    values = {**metrics, **{k: counts_at_val.get(k) for k in ["P", "R", "F1"]}}
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerow({k: values.get(k) for k in fields})
