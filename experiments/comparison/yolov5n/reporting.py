"""Unified B19 metrics, compact lossless exports, and reproducible paper figures."""

import csv
from pathlib import Path

import numpy as np

from comparison_io import write_json


def records_for(predictions, targets, records=None):
    return [
        {"image_id": records[i]["image_id"] if records else str(i), "predictions": p, "gt": t}
        for i, (p, t) in enumerate(zip(predictions, targets))
    ]


def matches(predictions, targets):
    from shared.evaluation import correct

    return correct(predictions, targets)


def counts(predictions, targets, threshold):
    from shared.evaluation import counts as exact_counts

    return {
        **exact_counts(records_for(predictions, targets), threshold),
        "units": "fraction",
        "method": "exact rematching after confidence filtering",
    }


def metric_values(predictions, targets, split="test", collection_conf=0.001):
    from shared.evaluation import compute

    if not any(len(t) for t in targets):
        return {
            "AP50": None,
            "AP75": None,
            "mAP50-95": None,
            "fixed_confidence": {str(c): counts(predictions, targets, c) for c in (0.25, 0.5)},
            "legacy_curve_peak": {"state": "NO_GT"},
            "units": "fraction",
        }, None
    metrics, arrays = compute(records_for(predictions, targets), split, collection_conf=collection_conf)
    curves = {
        "confidence": arrays["confidence"],
        "P": arrays["precision"].mean(0),
        "R": arrays["recall"].mean(0),
        "F1": arrays["F1"].mean(0),
        "recall": arrays["PR_recall"],
        "PR": arrays["PR_precision"].mean(0),
        "ap_by_iou": arrays["AP"].mean(0),
        "iou_thresholds": arrays["IoU_thresholds"],
    }
    return metrics, curves


def save_exports(directory, predictions, targets, records):
    from shared.evaluation import save_predictions

    save_predictions(directory, records_for(predictions, targets, records))


def summary(path, metrics):
    from shared.evaluation import summary as shared_summary

    shared_summary(path, metrics)


def figures(directory, curves, metrics):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory = Path(directory)
    if curves is not None:
        np.savez_compressed(directory / "curves.npz", **curves)
        with (directory / "curves.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["confidence_or_recall", "P", "R", "F1", "PR"])
            writer.writerows(zip(curves["confidence"], curves["P"], curves["R"], curves["F1"], curves["PR"]))
        for key in ("PR", "P", "R", "F1"):
            fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")
            ax.plot(curves["recall"] if key == "PR" else curves["confidence"], curves[key])
            ax.set(
                xlabel="Recall" if key == "PR" else "Confidence",
                ylabel="Precision" if key == "PR" else key,
                xlim=(0, 1),
                ylim=(0, 1),
            )
            ax.grid(alpha=0.25)
            fig.savefig(directory / f"{key}_curve.svg")
            plt.close(fig)
    fixed = metrics["fixed_confidence"]["0.25"]
    if "TP" not in fixed:
        write_json(directory / "confusion_matrix.json", fixed)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.axis("off")
        ax.text(0.5, 0.5, "UNVERIFIED: confidence 0.25 below collection threshold", ha="center", wrap=True)
        fig.savefig(directory / "confusion_matrix.svg")
        plt.close(fig)
        return
    matrix = np.array([[fixed["TP"], fixed["FP"]], [fixed["FN"], 0]])
    write_json(
        directory / "confusion_matrix.json",
        {
            "matrix": matrix.tolist(),
            "axes": "rows prediction [crack,background]; columns GT [crack,background]",
            "background_background": "undefined (displayed as 0)",
            "conf": 0.25,
            "iou": 0.5,
        },
    )
    fig, ax = plt.subplots(figsize=(5, 4), layout="constrained")
    ax.imshow(matrix, cmap="Blues")
    for (y, x), value in np.ndenumerate(matrix):
        ax.text(x, y, str(value) if (y, x) != (1, 1) else "NA", ha="center", va="center")
    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["crack", "background"],
        yticklabels=["crack", "background"],
        xlabel="Ground truth",
        ylabel="Prediction",
    )
    fig.savefig(directory / "confusion_matrix.svg")
    plt.close(fig)


def training_figure(directory):
    import matplotlib.pyplot as plt

    path = Path(directory) / "results.csv"
    if not path.is_file():
        return
    with path.open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    x = [int(r["epoch"]) for r in rows]
    for name in ("train_box", "train_obj", "train_cls", "val_box", "val_obj", "val_cls"):
        axes[0].plot(x, [float(r[name]) for r in rows], label=name)
    for name in ("AP50", "mAP50_95"):
        axes[1].plot(x, [float(r[name]) for r in rows], label=name)
    for axis in axes:
        axis.set_xlabel("Epoch (zero based)")
        axis.legend(fontsize=8)
        axis.grid(alpha=0.25)
    fig.savefig(Path(directory) / "training_curves.svg")
    plt.close(fig)
