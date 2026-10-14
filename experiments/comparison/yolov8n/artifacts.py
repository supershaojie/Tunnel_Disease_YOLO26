"""Paper figures, model-independent example selection and strictly bounded lossless packages."""

import csv
import gzip
import io
import json
import shutil
import zipfile
from pathlib import Path

from common import HERE, file_hash, status, write_json


def figures(directory, metrics, curves, run):
    """Plot frozen numeric curves and training CSV; figures remain reproducible."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    out = Path(directory) / "figures"
    out.mkdir()
    for title, x_key, y_key, xlabel in [
        ("PR", "PR_recall", "PR_precision", "Recall"),
        ("P", "confidence", "precision", "Confidence"),
        ("R", "confidence", "recall", "Confidence"),
        ("F1", "confidence", "F1", "Confidence"),
    ]:
        fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
        y = curves[y_key]
        ax.plot(curves[x_key], y.mean(0) if len(y) else np.zeros_like(curves[x_key]), label="crack")
        ax.set(
            xlabel=xlabel,
            ylabel="Precision" if title in ["P", "PR"] else "Recall" if title == "R" else "F1",
            xlim=(0, 1),
            ylim=(0, 1),
            title=title + " (" + metrics["split"] + ")",
        )
        ax.grid(alpha=0.2)
        fig.savefig(out / (title + ".svg"))
        fig.savefig(out / (title + ".png"), dpi=220)
        plt.close(fig)
    fixed = metrics["fixed_confidence"]["0.25"]
    if "confusion_matrix" not in fixed:
        # High collection-conf variants cannot reconstruct the omitted low-score predictions.
        matrix = np.array([[np.nan, np.nan], [np.nan, np.nan]])
    else:
        matrix = np.array(fixed["confusion_matrix"])
    np.savetxt(Path(directory) / "confusion_matrix.csv", matrix, delimiter=",", fmt="%g")
    fig, ax = plt.subplots(figsize=(4, 4), constrained_layout=True)
    ax.imshow(matrix, cmap="Blues")
    for (i, j), value in np.ndenumerate(matrix):
        ax.text(j, i, str(value) if (i, j) != (1, 1) else "N/A", ha="center", va="center")
    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["crack", "background"],
        yticklabels=["crack", "background"],
        xlabel="Ground truth",
        ylabel="Prediction",
        title="IoU 0.5, confidence 0.25",
    )
    fig.savefig(out / "confusion_matrix.svg")
    fig.savefig(out / "confusion_matrix.png", dpi=220)
    plt.close(fig)
    results = Path(run) / "results.csv"
    if results.is_file():
        shutil.copyfile(results, Path(directory) / "results.csv")
        with results.open(encoding="utf-8") as f:
            rows = [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(f)]
        names = [k for k in rows[0] if "loss" in k or "mAP" in k]
        fig, axes = plt.subplots(2, 1, figsize=(7, 7), constrained_layout=True)
        for key in names:
            ax = axes[0 if "loss" in key else 1]
            ax.plot([int(r["epoch"]) for r in rows], [float(r[key]) for r in rows], label=key)
        for ax in axes:
            ax.set_xlabel("Epoch")
            ax.legend(fontsize=7)
            ax.grid(alpha=0.2)
        fig.savefig(out / "training.svg")
        fig.savefig(out / "training.png", dpi=220)
        plt.close(fig)


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


def _members(directory):
    """Select required provenance/numerics, excluding weights, data, caches and duplicate images."""
    directory = Path(directory).resolve()
    required = [
        "metrics.json",
        "summary.csv",
        "protocol.json",
        "run_config.json",
        "data_manifest.json",
        "environment.json",
        "predictions.npz",
        "ground_truth.npz",
        "curves.npz",
        "confusion_matrix.csv",
        "evaluate_status.json",
        "fixed_images.json",
        "results.csv",
    ]
    missing = [name for name in required if not (directory / name).is_file()]
    if missing:
        raise ValueError(f"Incomplete evaluation package: {missing}")
    evaluation_status = json.loads((directory / "evaluate_status.json").read_text(encoding="utf-8"))
    if evaluation_status["status"] != "COMPLETED":
        raise ValueError("Only completed evaluations may be packaged")
    metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    run = Path(metrics["weight"]).parent.parent
    metadata = [
        "user_config.json",
        "official_defaults.json",
        "requested_config.json",
        "effective_config.json",
        "environment.json",
        "requested_effective_diff.json",
        "parameter_sources.json",
        "initialization.json",
        "best_selection.json",
        "checkpoint_manifest.json",
        "training_summary.json",
        "train_status.json",
        "protocol.json",
        "data_manifest.json",
        "run_config.json",
        "backend.lock.json",
    ]
    missing = [name for name in metadata if not (run / name).is_file()]
    if missing:
        raise ValueError(f"Incomplete training provenance: {missing}")
    members = {f"evaluation/{name}": directory / name for name in required}
    members.update({f"training/{name}": run / name for name in metadata})
    for name in ["augmentation_close.json", "optimizer_epochs.jsonl", "train_status.jsonl", "benchmark_location.json"]:
        if (run / name).is_file():
            members["training/" + name] = run / name
    members.update({"training/" + p.name: p for p in run.glob("resume_*.json")})
    members.update({f"evaluation/figures/{p.name}": p for p in (directory / "figures").glob("*.svg")})
    if len([name for name in members if "/figures/" in name]) < 6:
        raise ValueError("Required paper figures are missing")
    members.update({f"evaluation/examples/{p.name}": p for p in (directory / "examples").glob("*") if p.is_file()})
    location = run / "benchmark_location.json"
    benchmark_status = {"status": "NOT_RUN", "FLOPs": None, "FPS": None}
    if location.is_file():
        benchmark = Path(json.loads(location.read_text(encoding="utf-8"))["directory"]) / "benchmark.json"
        measured = json.loads(benchmark.read_text(encoding="utf-8"))
        if (
            measured["weight_sha256"] == metrics["weight_sha256"]
            and measured["protocol_sha256"] == metrics["protocol_sha256"]
        ):
            members["benchmark/benchmark.json"] = benchmark
            members["benchmark/timing_samples.npz"] = benchmark.parent / "timing_samples.npz"
        else:
            members["benchmark/unmatched_benchmark.json"] = benchmark
            benchmark_status = {
                "status": "UNVERIFIED",
                "reason": "Completed benchmark belongs to another checkpoint or evaluation protocol",
            }
    # Include local/source provenance only, without cloning the large source tree into result packages.
    members["source/b19_source.lock.json"] = HERE.parent / "shared/b19/source.lock.json"
    return members, metrics, run, benchmark_status


def package(cfg, args):
    """Build a strict-size single-model package or protocol-checked multi-model aggregate."""
    directories = args.results or ([args.result_dir] if args.result_dir else [])
    if not directories:
        raise ValueError("package requires --result-dir or --results DIR...")
    if args.dry_run:
        print(json.dumps({"status": "DRY_RUN", "directories": directories, "max_bytes": cfg["package"]["max_bytes"]}))
        return
    all_members, summaries, payloads = {}, [], {}
    expected = None
    for index, directory in enumerate(directories):
        members, metrics, run, benchmark_status = _members(directory)
        if expected is not None and metrics["protocol_sha256"] != expected:
            raise ValueError("Aggregate protocol hashes differ; no misleading comparison package is created")
        expected = metrics["protocol_sha256"]
        prefix = "" if len(directories) == 1 else f"{index:02d}_{metrics['model']}/"
        all_members.update({prefix + name: p for name, p in members.items()})
        summaries.append(metrics)
        logs = sorted(run.glob("*.log"))
        for p in logs[:2]:
            # Lossless gzip preserves full logs while making the result package compact.
            payloads[prefix + "logs/" + p.name + ".gz"] = gzip.compress(p.read_bytes(), mtime=0)
        if "benchmark/benchmark.json" not in members:
            payloads[prefix + "benchmark/benchmark.json"] = json.dumps(benchmark_status).encode()
    output = Path(args.output or Path(directories[0]).parent / "yolov8n_core.zip")
    if output.exists():
        raise FileExistsError(f"Package output exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    maximum = cfg["package"]["max_bytes"] if len(directories) == 1 else 100000000
    listing = [
        {
            "name": name,
            "bytes": p.stat().st_size,
            "sha256": file_hash(p),
            "selection": "required numeric/provenance, SVG paper figure or fixed example",
        }
        for name, p in sorted(all_members.items())
    ]
    listing.extend(
        {
            "name": name,
            "bytes": len(content),
            "sha256": __import__("hashlib").sha256(content).hexdigest(),
            "selection": "compressed log or explicit NOT_RUN benchmark",
        }
        for name, content in sorted(payloads.items())
    )
    payloads["package_manifest.json"] = json.dumps(
        {
            "members": listing,
            "max_bytes_strict": maximum,
            "protocol_sha256": expected,
            "excluded": ["weights", "dataset originals", "caches", "events", "duplicate PNG figures"],
            "prediction_precision": "full float32 coordinates/scores and all evaluated detections",
            "benchmark_status": "see per-model benchmark metadata",
        },
        indent=2,
    ).encode()
    if len(directories) > 1:
        buffer = io.StringIO()
        fields = ["model", "split", "units", "AP50", "AP75", "mAP50-95", "protocol_sha256"]
        writer = csv.DictWriter(buffer, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: m[k] for k in fields} for m in summaries)
        payloads["aggregate_summary.csv"] = buffer.getvalue().encode()
    temp = output.with_name(output.name + ".partial")
    try:
        with zipfile.ZipFile(temp, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
            for name, path in sorted(all_members.items()):
                z.write(path, name)
            for name, content in sorted(payloads.items()):
                z.writestr(name, content)
        size = temp.stat().st_size
        if size >= maximum:
            raise ValueError(
                f"Core package is {size} bytes, limit is strictly <{maximum}; required data retained. Largest members: {sorted(listing, key=lambda x: x['bytes'], reverse=True)[:5]}"
            )
        with zipfile.ZipFile(temp) as z:
            if z.testzip() is not None:
                raise ValueError("Package ZIP CRC failure")
        temp.rename(output)
        write_json(
            output.with_suffix(".manifest.json"),
            {
                "bytes": size,
                "sha256": file_hash(output),
                "max_bytes_strict": maximum,
                "members": listing,
                "status": "COMPLETED",
            },
        )
        status(output.parent, "package", "COMPLETED", output=str(output), bytes=size)
        print(
            json.dumps({"output": str(output), "bytes": size, "max_bytes_strict": maximum, "sha256": file_hash(output)})
        )
    except BaseException as e:
        temp.unlink(missing_ok=True)
        status(output.parent, "package", "FAILED", reason=f"{type(e).__name__}: {e}")
        raise
