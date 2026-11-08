"""Rank real, cached test detections without changing the formal evaluation working point."""

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from ultralytics.engine.validator import BaseValidator
from ultralytics.utils.metrics import box_iou

MODELS = ("yolov5n", "yolov8n", "yolo11n", "yolov13n", "yolo26n", "rtdetr_r18", "drsc")
THRESHOLD = 0.25


def measurements(predictions, ground_truth):
    """Reuse native B19 matching; retain actual greedy pairs for localization inspection."""
    pred = torch.tensor(predictions, dtype=torch.float32).reshape(-1, 6)
    pred = pred[pred[:, 4] >= THRESHOLD]
    gt = torch.tensor(ground_truth, dtype=torch.float32).reshape(-1, 5)
    iou = box_iou(gt[:, :4], pred[:, :4])
    same_class_iou = (iou * (gt[:, 4, None] == pred[:, 5])).numpy()
    correct = BaseValidator.match_predictions(
        SimpleNamespace(iouv=torch.tensor([0.50, 0.75])), pred[:, 5], gt[:, 4], iou
    ).numpy()
    result = {}
    for column, threshold in enumerate((0.50, 0.75)):
        # The native boolean interface omits pair IDs; replay only its pair selection to expose exact matched IoUs.
        pairs = np.stack(np.nonzero(same_class_iou >= threshold), axis=1)
        if len(pairs) > 1:
            pairs = pairs[same_class_iou[pairs[:, 0], pairs[:, 1]].argsort()[::-1]]
            pairs = pairs[np.unique(pairs[:, 1], return_index=True)[1]]
            pairs = pairs[np.unique(pairs[:, 0], return_index=True)[1]]
        assert set(pairs[:, 1]) == set(np.flatnonzero(correct[:, column]))
        tp = int(correct[:, column].sum())
        unmatched = ~correct[:, column]
        has_overlap = (same_class_iou >= threshold).any(axis=0)
        matched_ious = same_class_iou[pairs[:, 0], pairs[:, 1]].tolist()
        result[str(threshold)] = {
            "tp": tp,
            "fp": len(pred) - tp,
            "fn": len(gt) - tp,
            "f1": 2 * tp / max(len(gt) + len(pred), 1),
            "matched_ious": matched_ious,
            "mean_matched_iou": float(np.mean(matched_ious)) if matched_ious else 0.0,
            "redundant": int((unmatched & has_overlap).sum()),
            "unmatched": int((unmatched & ~has_overlap).sum()),
            "pairs": pairs.tolist(),
        }
    return result


def rank_candidates(bundle, output, limit=80):
    """Write all-test counts and propose distinct, numerically supported cases for visual review."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows, ranked = [], []
    dataset_root = Path(bundle.get("dataset_root", Path(bundle["images"][0]["path"]).parents[2]))
    manifest = dataset_root / "metadata/split_manifest.csv"
    parents = {}
    if manifest.exists():
        with manifest.open(encoding="utf-8-sig", newline="") as handle:
            parents = {row["output_image"]: row["parent_id"] for row in csv.DictReader(handle)}
    for image in tqdm(bundle["images"], desc="Statistics and visual cues", dynamic_ncols=True):
        stats = {key: measurements(image["predictions"][key], image["gt"]) for key in MODELS}
        for key, values in stats.items():
            for iou, counts in values.items():
                rows.append({"image_id": image["id"], "model": key, "iou": iou, **counts})
        ours, base = stats["drsc"]["0.5"], stats["yolo26n"]["0.5"]
        error = lambda x: x["fp"] + x["fn"]
        delta = error(base) - error(ours)
        loc_delta = ours["mean_matched_iou"] - base["mean_matched_iou"]
        strong_delta = max(error(stats[key]["0.5"]) - error(ours) for key in ("yolov13n", "rtdetr_r18"))
        if ours["tp"] == 0 or ours["f1"] < 2 / 3:
            continue
        if delta <= 0 and loc_delta < 0.03 and strong_delta <= 0:
            continue
        with Image.open(image["path"]) as photo:
            gray = np.asarray(photo.convert("L").resize((64, 64)), dtype=float)
        small = np.asarray(Image.fromarray(gray.astype("uint8")).resize((9, 8)), dtype=int)
        fingerprint = (small[:, 1:] > small[:, :-1]).ravel()
        gt = np.asarray(image["gt"], dtype=float).reshape(-1, 5)
        sizes = gt[:, 2:4] - gt[:, :2]
        elongation = float(np.max(sizes.max(axis=1) / np.maximum(sizes.min(axis=1), 1)))
        cues = {
            "mean_brightness": round(float(gray.mean()), 2),
            "brightness_spread": round(float(np.std(gray.reshape(4, 16, 4, 16).mean(axis=(1, 3)))), 2),
            "box_elongation": round(elongation, 2),
            "gt_count": len(gt),
            "baseline_unmatched": base["unmatched"],
        }
        cue = max(
            (
                ("low_brightness", (110 - gray.mean()) / 60),
                ("uneven_brightness", cues["brightness_spread"] / 35),
                ("elongated_boxes", elongation / 5),
                ("multiple_targets", (len(gt) - 1) / 2),
                ("unmatched_reduction", (base["unmatched"] - ours["unmatched"]) / 2),
            ),
            key=lambda pair: pair[1],
        )[0]
        ranked.append(
            {
                "image_id": image["id"],
                "status": "proposed",
                "cue": cue,
                "cues": cues,
                "score": 3 * ours["f1"] + 2 * delta + strong_delta + 2 * loc_delta,
                "parent": parents.get(image["id"], Path(image["id"]).stem.split("__")[0]),
                "reason": f"DRSC TP/FP/FN={ours['tp']}/{ours['fp']}/{ours['fn']}; "
                f"YOLOv26n={base['tp']}/{base['fp']}/{base['fn']}; "
                f"mean matched IoU delta={loc_delta:+.3f}; "
                f"YOLOv13n FP/FN={stats['yolov13n']['0.5']['fp']}/{stats['yolov13n']['0.5']['fn']}; "
                f"RT-DETR-R18 FP/FN={stats['rtdetr_r18']['0.5']['fp']}/{stats['rtdetr_r18']['0.5']['fn']}",
                "statistics": stats,
                "fingerprint": fingerprint,
            }
        )
    with (output / "ranking.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_id", "score", "cue", "parent", "reason"])
        writer.writeheader()
        writer.writerows(
            {key: candidate[key] for key in writer.fieldnames}
            for candidate in sorted(ranked, key=lambda row: (-row["score"], row["image_id"]))
        )
    with (output / "stats.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({k: json.dumps(v) if isinstance(v, list) else v for k, v in row.items()} for row in rows)
    # Round-robin numeric cue groups; labels remain hypotheses, never confirmed scene descriptions.
    queues = {}
    for candidate in sorted(ranked, key=lambda row: (-row["score"], row["image_id"])):
        queues.setdefault(candidate["cue"], []).append(candidate)
    selected, parent_ids, fingerprints = [], set(), []
    while any(queues.values()) and len(selected) < limit:
        for queue in queues.values():
            while queue:
                candidate = queue.pop(0)
                fp = candidate.pop("fingerprint")
                if candidate["parent"] in parent_ids or any(np.mean(fp != old) < 0.10 for old in fingerprints):
                    continue
                candidate["case_id"] = f"C{len(selected) + 1:03d}"
                selected.append(candidate)
                parent_ids.add(candidate["parent"])
                fingerprints.append(fp)
                break
            if len(selected) == limit:
                break
    (output / "candidates.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")
    proposed = {
        "status": "proposed",
        "note": "Numeric shortlist only; inspect real photos before final selection.",
        "confidence_threshold": THRESHOLD,
        "cases": [{"case_id": row["case_id"], "image_id": row["image_id"]} for row in selected[:5]],
    }
    (output / "proposed_cases.json").write_text(json.dumps(proposed, indent=2), encoding="utf-8")
    print(f"Proposed {len(selected)} distinct candidates from {len(ranked)} eligible images; no filler cases.")
    return selected
