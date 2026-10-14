import json
from pathlib import Path
from comparison_io import write_json


def comparisons(directory, records, cfg, data_report):
    """Select images by GT and a fixed hash before inspecting model success/failure."""
    import hashlib
    import cv2
    import numpy as np
    from shared.evaluation import correct

    split = cfg["eval"]["split"]
    entries = data_report["splits"][split]["entries"]
    fixed = cfg["eval"]["fixed_images"]
    if fixed:
        manifest = json.loads(Path(fixed).read_text(encoding="utf-8"))
        if manifest["data_sha256"] != data_report["data_sha256"] or manifest["split"] != split:
            raise ValueError("Fixed image manifest has a different data/split identity")
        ids = manifest["image_ids"]
    else:
        ranked = sorted(entries, key=lambda x: hashlib.sha256(("42:" + x["image_id"]).encode()).hexdigest())
        positive = [e for e in ranked if e["boxes"] > 0]
        empty = [e for e in ranked if not e["boxes"]]
        ids = [e["image_id"] for e in positive[:4] + empty[:2]]
        manifest = {
            "data_sha256": data_report["data_sha256"],
            "split": split,
            "image_ids": ids,
            "selection": "fixed seed 42 SHA256 ranking, first 4 nonempty GT + first 2 empty GT; independent of predictions",
        }
    lookup = {r["image_id"]: r for r in records}
    if len(set(ids)) != len(ids) or any(i not in lookup for i in ids):
        raise ValueError("Fixed image selection is duplicate or not covered by evaluation")
    write_json(Path(directory) / "fixed_images.json", manifest)
    out = Path(directory) / "examples"
    out.mkdir()
    diagnoses = []
    for index, image_id in enumerate(ids):
        r = lookup[image_id]
        raw = cv2.imdecode(np.fromfile(Path(data_report["root"]) / image_id, dtype=np.uint8), cv2.IMREAD_COLOR)
        truth, prediction = raw.copy(), raw.copy()
        for box in r["gt"]:
            a, b, c, d = np.rint(box[:4]).astype(int)
            cv2.rectangle(truth, (a, b), (c, d), (0, 190, 0), 2)
        kept = r["predictions"][r["predictions"][:, 4] >= cfg["eval"]["display_conf"]]
        tp = int(correct(kept, r["gt"]).sum(axis=0)[0])
        for box in kept:
            a, b, c, d = np.rint(box[:4]).astype(int)
            cv2.rectangle(prediction, (a, b), (c, d), (0, 0, 230), 2)
            cv2.putText(prediction, f"{box[4]:.2f}", (a, max(b, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 230), 1)
        pair = np.concatenate([truth, prediction], axis=1)
        max_width = 1800
        if pair.shape[1] > max_width:
            pair = cv2.resize(pair, (max_width, round(pair.shape[0] * max_width / pair.shape[1])))
        cv2.imencode(".jpg", pair, [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tofile(out / f"{index:02d}_gt_prediction.jpg")
        if len(r["gt"]):
            # Deterministic largest GT crop, independent of whether this model detected it.
            box = max(r["gt"], key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
            a, b, c, d = np.rint(box[:4]).astype(int)
            margin = 24
            h, w = raw.shape[:2]
            crop = np.concatenate(
                [
                    truth[max(0, b - margin) : min(h, d + margin), max(0, a - margin) : min(w, c + margin)],
                    prediction[max(0, b - margin) : min(h, d + margin), max(0, a - margin) : min(w, c + margin)],
                ],
                axis=1,
            )
            cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tofile(out / f"{index:02d}_crop.jpg")
        diagnoses.append(
            {
                "image_id": image_id,
                "confidence": cfg["eval"]["display_conf"],
                "TP": tp,
                "FP": len(kept) - tp,
                "FN": len(r["gt"]) - tp,
            }
        )
    write_json(
        out / "diagnostics.json",
        {
            "selection_is_prediction_independent": True,
            "images": diagnoses,
            "coverage_note": "Success/miss/false-positive counts retained; do not cherry-pick examples to guarantee categories",
        },
    )
