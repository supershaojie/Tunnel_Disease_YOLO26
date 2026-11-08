"""Draw cached detections and make portable, candidate-only review archives."""

import json
import math
import shutil
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from PIL import Image

MODEL_ROWS = (
    (None, "Ground truth"),
    ("yolov5n", "YOLOv5n"),
    ("yolov8n", "YOLOv8n"),
    ("yolo11n", "YOLOv11n"),
    ("yolov13n", "YOLOv13n"),
    ("yolo26n", "YOLOv26n"),
    ("rtdetr_r18", "RT-DETR-R18"),
    ("drsc", "DRSC-YOLO"),
)
THRESHOLD = 0.25
GT_COLOR = "#00bfc9"
PRED_COLOR = "#ed1c24"


def _write_json(path, data):
    """Write a small UTF-8 JSON artifact."""
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _source_path(record, bundle):
    """Resolve an original image, including one in an extracted candidate archive."""
    path = Path(record["path"])
    return path if path.is_absolute() else Path(bundle.get("root", ".")) / path


def _records(bundle, cases):
    """Resolve explicit image IDs without relying on a model's integer image index."""
    by_id = {record["id"]: record for record in bundle["images"]}
    records = []
    for case in cases:
        record = by_id[case["image_id"]]
        for key, _ in MODEL_ROWS[1:]:
            if key not in record["predictions"]:
                raise ValueError(f"{record['id']}: missing predictions for {key}; empty predictions must be []")
        records.append(record)
    return records


def _crop(record, case):
    """Validate one shared original-coordinate view for every row in a column."""
    crop = case.get("crop", [0, 0, record["width"], record["height"]])
    x1, y1, x2, y2 = map(float, crop)
    if not (0 <= x1 < x2 <= record["width"] and 0 <= y1 < y2 <= record["height"]):
        raise ValueError(f"{record['id']}: crop must lie inside the original image: {crop}")
    return [x1, y1, x2, y2]


def _draw(bundle, cases, *, preview=False):
    """Lay out original photographs with vector annotations at a fixed display threshold."""
    records = _records(bundle, cases)
    crops = [_crop(record, case) for record, case in zip(records, cases)]
    column_width = 6.6 if preview else 2.15
    label_width = 1.6 if preview else 1.25
    gap = 0.08
    row_height = min(3.8 if preview else 2.8, max((c[3] - c[1]) / (c[2] - c[0]) for c in crops) * column_width)
    top, bottom, outer = (0.54, 0.32, 0.08)
    width = label_width + len(cases) * column_width + (len(cases) - 1) * gap + outer
    height = top + 8 * row_height + 7 * gap + bottom
    fig = plt.figure(figsize=(width, height), facecolor="white")
    font_size, line_width = (11, 1.0) if preview else (7.5, 0.8)
    for column, (case, record, crop) in enumerate(zip(cases, records, crops)):
        x1, y1, x2, y2 = crop
        pixels = (math.floor(x1), math.floor(y1), math.ceil(x2), math.ceil(y2))
        with Image.open(_source_path(record, bundle)) as source:
            if source.size != (record["width"], record["height"]):
                raise ValueError(f"{record['id']}: original image dimensions changed: {source.size}")
            image = source.crop(pixels).convert("RGB")
        # Embed enough photographic pixels for the actual output size, keeping annotations vector.
        dpi = 120 if preview else 600
        image.thumbnail((math.ceil(column_width * dpi), math.ceil(row_height * dpi)), Image.Resampling.LANCZOS)
        for row, (model, label) in enumerate(MODEL_ROWS):
            left = label_width + column * (column_width + gap)
            lower = height - top - (row + 1) * row_height - row * gap
            ax = fig.add_axes([left / width, lower / height, column_width / width, row_height / height])
            ax.imshow(image, interpolation="none", extent=(pixels[0], pixels[2], pixels[3], pixels[1]))
            ax.set_xlim(x1, x2)
            ax.set_ylim(y2, y1)
            ax.set_aspect("equal", adjustable="box")
            ax.set_axis_off()
            boxes = record["gt"] if model is None else record["predictions"][model]
            color = GT_COLOR if model is None else PRED_COLOR
            for box in boxes:
                if model is not None and float(box[4]) < THRESHOLD:
                    continue
                bx1, by1, bx2, by2 = map(float, box[:4])
                ax.add_patch(Rectangle((bx1, by1), bx2 - bx1, by2 - by1, fill=False, ec=color, lw=line_width))
                if model is not None:
                    ax.annotate(
                        f"{float(box[4]):.2f}",
                        (bx1, by1),
                        xytext=(2, -1),
                        textcoords="offset points",
                        fontsize=font_size,
                        ha="left",
                        va="top",
                        color=color,
                        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 0.1},
                        clip_on=True,
                    )
            if column == 0:
                fig.text(
                    0.03 / width,
                    (lower + row_height / 2) / height,
                    label,
                    ha="left",
                    va="center",
                    fontsize=font_size + 1,
                    weight="bold" if model == "drsc" else "normal",
                )
        fig.text(
            (label_width + column * (column_width + gap) + column_width / 2) / width,
            0.09 / height,
            case["case_id"] if preview else f"({chr(97 + column)})",
            ha="center",
            va="bottom",
            fontsize=font_size + 1,
        )
    separator_y = (height - top - row_height - gap / 2) / height
    fig.add_artist(Line2D([0, 1], [separator_y, separator_y], transform=fig.transFigure, lw=0.35, color="#cccccc"))
    fig.legend(
        handles=[
            Patch(ec=GT_COLOR, fill=False, label="Ground truth"),
            Patch(ec=PRED_COLOR, fill=False, label="Prediction"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.43, 1),
        ncol=2,
        frameon=False,
        fontsize=font_size,
    )
    fig.text(0.98, 1 - 0.22 / height, "Confidence threshold: 0.25", ha="right", va="center", fontsize=font_size)
    return fig


def previews(bundle, candidates, output):
    """Write readable eight-row case previews and stable, numbered overview pages."""
    output = Path(output) / "review"
    preview_dir, overview_dir = output / "previews", output / "overview"
    preview_dir.mkdir(parents=True, exist_ok=True)
    overview_dir.mkdir(parents=True, exist_ok=True)
    for i, case in enumerate(candidates, 1):
        fig = _draw(bundle, [case], preview=True)
        fig.savefig(preview_dir / f"{case['case_id']}.png", dpi=120, facecolor="white")
        plt.close(fig)
        print(f"\rCandidate previews: {i}/{len(candidates)}", end="", flush=True)
    print()
    records = _records(bundle, candidates)
    per_page = 12
    for page, start in enumerate(range(0, len(candidates), per_page), 1):
        fig, axes = plt.subplots(4, 3, figsize=(13, 10), squeeze=False)
        for ax in axes.flat:
            ax.axis("off")
        for ax, case, record in zip(axes.flat, candidates[start : start + per_page], records[start : start + per_page]):
            with Image.open(_source_path(record, bundle)) as source:
                image = source.convert("RGB")
            image.thumbnail((600, 400), Image.Resampling.LANCZOS)
            ax.imshow(image)
            ax.set_title(case["case_id"], fontsize=11, weight="bold")
        fig.suptitle(f"Proposed candidates | Page {page:02d} | Review individual previews", fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(overview_dir / f"page_{page:02d}.jpg", dpi=140, facecolor="white")
        plt.close(fig)
    _write_json(output / "candidates.json", candidates)
    return output


def render_final(bundle, cases, output):
    """Render five explicitly selected cases as 600 dpi PNG and vector SVG/PDF."""
    selected = cases["cases"]
    if len(selected) != 5:
        raise ValueError("The final figure requires exactly five cases")
    if len({case["image_id"] for case in selected}) != 5:
        raise ValueError("The final figure requires five distinct image IDs")
    output = Path(output) / "final"
    output.mkdir(parents=True, exist_ok=True)
    with matplotlib.rc_context({"svg.fonttype": "none", "pdf.fonttype": 42, "image.composite_image": False}):
        fig = _draw(bundle, selected)
        for extension in ("png", "svg", "pdf"):
            fig.savefig(output / f"comparison.{extension}", dpi=600, facecolor="white")
        plt.close(fig)
    records = _records(bundle, selected)
    _write_json(
        output / "cases.json",
        {
            "status": cases.get("status", "proposed"),
            "confidence_threshold": THRESHOLD,
            "model_order": [label for _, label in MODEL_ROWS],
            "cases": [dict(case, crop=_crop(record, case)) for case, record in zip(selected, records)],
        },
    )
    return output


def _archive(paths, root, destination):
    """Zip only an explicit artifact list and report the actual compressed size."""
    with ZipFile(destination, "w", compression=ZIP_DEFLATED) as archive:
        for path in sorted(paths):
            archive.write(path, path.relative_to(root).as_posix())
    size = destination.stat().st_size
    print(f"Archive: {destination.resolve()} ({size:,} bytes)")
    return destination


def package_candidates(bundle, candidates, output):
    """Package originals once, cached candidate detections, previews, statistics and provenance."""
    output = Path(output)
    review = output / "review"
    review.mkdir(parents=True, exist_ok=True)
    original_dir = review / "originals"
    original_dir.mkdir(exist_ok=True)
    records = _records(bundle, candidates)
    if len({record["id"] for record in records}) != len(records):
        raise ValueError("Candidate IDs must refer to distinct original images")
    images = []
    for case, record in zip(candidates, records):
        source = _source_path(record, bundle)
        destination = original_dir / f"{case['case_id']}{source.suffix.lower()}"
        shutil.copy2(source, destination)
        images.append(dict(record, path=destination.relative_to(review).as_posix()))
    _write_json(review / "bundle.json", {"images": images, "sources": bundle["sources"]})
    _write_json(review / "candidates.json", candidates)
    _write_json(review / "sources.json", bundle["sources"])
    _write_json(review / "display.json", {"confidence_threshold": THRESHOLD, "model_rows": MODEL_ROWS})
    for name in ("stats.csv", "proposed_cases.json", "ranking.csv"):
        source = output / name
        if source.is_file():
            shutil.copy2(source, review / name)
    for source in output.glob("*.log"):
        shutil.copy2(source, review / source.name)
    # Explicit membership excludes stale originals/previews after a candidate list is changed.
    paths = [review / name for name in ("bundle.json", "candidates.json", "sources.json", "display.json")]
    paths += [review / record["path"] for record in images]
    paths += [review / "previews" / f"{case['case_id']}.png" for case in candidates]
    paths += [review / "overview" / f"page_{page:02d}.jpg" for page in range(1, (len(candidates) + 11) // 12 + 1)]
    paths += [
        review / name for name in ("stats.csv", "proposed_cases.json", "ranking.csv") if (review / name).is_file()
    ]
    paths += list(review.glob("*.log"))
    return _archive(paths, review, output / "candidates.zip")


def package_final(output):
    """Package only the three final figure formats, leaving reproduction data on disk."""
    output = Path(output)
    final = output / "final"
    return _archive([final / f"comparison.{ext}" for ext in ("png", "svg", "pdf")], final, output / "final_figures.zip")
