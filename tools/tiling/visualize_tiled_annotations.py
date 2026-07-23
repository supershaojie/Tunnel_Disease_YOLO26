# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Reproducibly visualize safe and ambiguous tiles without changing either dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

CATEGORIES = ("safe_positive", "safe_negative", "ambiguous")


class VisualizationError(RuntimeError):
    """Raised when visualization inputs are invalid or incomplete."""


def _read_manifest(path: Path) -> list[dict[str, str]]:
    """Read the tile manifest required to reconstruct every category."""
    required = {
        "split",
        "tile_file",
        "source_image",
        "tile_x",
        "tile_y",
        "tile_w",
        "tile_h",
        "category",
        "source_boxes",
        "failed_boxes",
        "rejected_reason",
    }
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise VisualizationError(f"manifest missing fields: {sorted(missing)}")
            return list(reader)
    except (OSError, UnicodeError, csv.Error) as error:
        raise VisualizationError(f"cannot read manifest {path}: {error}") from error


def _derived_seed(seed: int, category: str) -> int:
    """Derive a stable per-category random seed."""
    return int.from_bytes(hashlib.sha256(f"{seed}:{category}".encode()).digest()[:8], "big")


def select_samples(
    rows: list[dict[str, str]], categories: list[str], samples_per_category: int, seed: int
) -> dict[str, list[dict[str, str]]]:
    """Select reproducible manifest samples independently for each category."""
    if samples_per_category <= 0:
        raise VisualizationError("samples_per_category must be positive")
    invalid = sorted(set(categories) - set(CATEGORIES))
    if invalid:
        raise VisualizationError(f"unsupported categories: {invalid}")
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["category"]].append(row)
    selected: dict[str, list[dict[str, str]]] = {}
    for category in categories:
        candidates = sorted(grouped[category], key=lambda row: (row["split"], row["tile_file"]))
        count = min(samples_per_category, len(candidates))
        selected[category] = sorted(
            random.Random(_derived_seed(seed, category)).sample(candidates, count),
            key=lambda row: (row["split"], row["tile_file"]),
        )
    return selected


def _parse_json_list(row: dict[str, str], field: str) -> list[dict[str, Any]]:
    """Read one JSON-list manifest field."""
    try:
        value = json.loads(row[field])
    except (KeyError, json.JSONDecodeError) as error:
        raise VisualizationError(f"{row.get('tile_file')}: invalid {field}: {error}") from error
    if not isinstance(value, list):
        raise VisualizationError(f"{row.get('tile_file')}: {field} must contain a JSON list")
    return value


def failure_annotation_lines(failed_boxes: list[dict[str, Any]]) -> list[str]:
    """Format one recoverable annotation line per failed source box."""
    lines = []
    for box in failed_boxes:
        try:
            source_bbox_id = str(box["source_bbox_id"])
            visibility = float(box["visibility"])
            reasons = ",".join(str(reason) for reason in box["reasons"])
            source_xyxy = ",".join(f"{float(value):.2f}" for value in box["source_xyxy"])
            intersection_xyxy = ",".join(f"{float(value):.2f}" for value in box["intersection_xyxy"])
        except (KeyError, TypeError, ValueError) as error:
            raise VisualizationError(f"invalid failed-box record: {box!r}: {error}") from error
        lines.append(
            f"{source_bbox_id} | visibility={visibility:.10f} | reasons={reasons} | "
            f"source=[{source_xyxy}] | intersection=[{intersection_xyxy}]"
        )
    return lines


def _open_rgb(path: Path) -> Image.Image:
    """Fully decode one image and return an independent RGB copy."""
    try:
        with Image.open(path) as image:
            image.load()
            return image.convert("RGB")
    except (OSError, ValueError) as error:
        raise VisualizationError(f"cannot decode image {path}: {error}") from error


def _draw_banner(image: Image.Image, text: str, color: tuple[int, int, int]) -> None:
    """Draw a readable category banner."""
    draw = ImageDraw.Draw(image)
    lines = [text[index : index + 100] for index in range(0, len(text), 100)] or [text]
    height = 18 * len(lines) + 8
    draw.rectangle((0, 0, image.width, height), fill=(20, 20, 20))
    for index, line in enumerate(lines):
        draw.text((5, 4 + 18 * index), line, fill=color)


def _render_safe(row: dict[str, str], dataset: Path) -> Image.Image:
    """Render a formal safe tile and its output labels."""
    image_path = dataset / "images" / row["split"] / row["tile_file"]
    label_path = dataset / "labels" / row["split"] / f"{Path(row['tile_file']).stem}.txt"
    image = _open_rgb(image_path)
    try:
        lines = [line.split() for line in label_path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    except (OSError, UnicodeError) as error:
        raise VisualizationError(f"cannot read label {label_path}: {error}") from error
    draw = ImageDraw.Draw(image)
    for fields in lines:
        if len(fields) != 5 or fields[0] != "0":
            raise VisualizationError(f"invalid output label in {label_path}")
        try:
            x_center, y_center, width, height = (float(value) for value in fields[1:])
        except ValueError as error:
            raise VisualizationError(f"nonnumeric output label in {label_path}") from error
        x1, y1 = (x_center - width / 2) * image.width, (y_center - height / 2) * image.height
        x2, y2 = (x_center + width / 2) * image.width, (y_center + height / 2) * image.height
        draw.rectangle((x1, y1, x2, y2), outline=(0, 255, 0), width=4)
    if row["category"] == "safe_negative":
        _draw_banner(image, "SAFE NEGATIVE — NO LABELS", (255, 220, 0))
    else:
        _draw_banner(image, f"SAFE POSITIVE — {len(lines)} retained boxes", (0, 255, 0))
    return image


def _crop_with_padding(source_image: Image.Image, x: int, y: int, width: int, height: int) -> Image.Image:
    """Reconstruct a tile from the source with value-114 right/bottom padding."""
    valid_width, valid_height = min(width, source_image.width - x), min(height, source_image.height - y)
    if valid_width <= 0 or valid_height <= 0:
        raise VisualizationError(f"tile anchor {(x, y)} lies outside source image {source_image.size}")
    crop = source_image.crop((x, y, x + valid_width, y + valid_height))
    tile = Image.new("RGB", (width, height), (114, 114, 114))
    tile.paste(crop, (0, 0))
    return tile


def _render_ambiguous(row: dict[str, str], source: Path) -> Image.Image:
    """Reconstruct an ambiguous tile with source context, intersections, and rejection reasons."""
    source_image = _open_rgb(source / row["source_image"])
    tile_x, tile_y = int(row["tile_x"]), int(row["tile_y"])
    tile_w, tile_h = int(row["tile_w"]), int(row["tile_h"])
    source_boxes = _parse_json_list(row, "source_boxes")
    failed_boxes = _parse_json_list(row, "failed_boxes")
    tile = _crop_with_padding(source_image, tile_x, tile_y, tile_w, tile_h)
    tile_draw = ImageDraw.Draw(tile)
    for box in source_boxes:
        x1, y1, x2, y2 = box["source_xyxy"]
        tile_draw.rectangle(
            (x1 - tile_x, y1 - tile_y, x2 - tile_x, y2 - tile_y),
            outline=(255, 165, 0),
            width=3,
        )
    for box in failed_boxes:
        ix1, iy1, ix2, iy2 = box["intersection_xyxy"]
        tile_draw.rectangle(
            (ix1 - tile_x, iy1 - tile_y, ix2 - tile_x, iy2 - tile_y),
            outline=(255, 0, 0),
            width=5,
        )
        tile_draw.text(
            (max(0, ix1 - tile_x + 3), max(24, iy1 - tile_y + 3)),
            f"{box['source_bbox_id']} vis={float(box['visibility']):.4f}",
            fill=(255, 255, 255),
            stroke_width=2,
            stroke_fill=(0, 0, 0),
        )

    overview = source_image.copy()
    resampling = getattr(Image, "Resampling", Image)
    overview.thumbnail((768, 768), resampling.LANCZOS)
    scale_x, scale_y = overview.width / source_image.width, overview.height / source_image.height
    overview_draw = ImageDraw.Draw(overview)
    overview_draw.rectangle(
        (
            tile_x * scale_x,
            tile_y * scale_y,
            (tile_x + tile_w) * scale_x,
            (tile_y + tile_h) * scale_y,
        ),
        outline=(255, 0, 0),
        width=4,
    )
    for box in source_boxes:
        x1, y1, x2, y2 = box["source_xyxy"]
        overview_draw.rectangle(
            (x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y),
            outline=(255, 165, 0),
            width=3,
        )

    annotation_lines = failure_annotation_lines(failed_boxes)
    image_height = max(overview.height, tile.height)
    annotation_height = 22 * len(annotation_lines) + 8
    canvas = Image.new(
        "RGB",
        (overview.width + tile.width, image_height + annotation_height),
        (30, 30, 30),
    )
    canvas.paste(overview, (0, 0))
    canvas.paste(tile, (overview.width, 0))
    reasons = ", ".join(json.loads(row["rejected_reason"])) or "UNKNOWN"
    _draw_banner(
        canvas,
        f"AMBIGUOUS | red=tile/intersection orange=source bbox | rejected={reasons}",
        (255, 80, 80),
    )
    canvas_draw = ImageDraw.Draw(canvas)
    for index, line in enumerate(annotation_lines):
        canvas_draw.text((5, image_height + 4 + index * 22), line, fill=(255, 220, 220))
    return canvas


def visualize(
    source: Path,
    dataset: Path,
    output: Path,
    categories: list[str],
    samples_per_category: int,
    seed: int,
) -> dict[str, int]:
    """Create deterministic review images without modifying source or dataset."""
    source, dataset, output = (
        source.expanduser().resolve(),
        dataset.expanduser().resolve(),
        output.expanduser().resolve(),
    )
    if not source.is_dir() or not dataset.is_dir():
        raise VisualizationError(f"source and dataset must exist: source={source}, dataset={dataset}")
    if output.exists():
        raise VisualizationError(f"output already exists; refusing implicit overwrite: {output}")
    rows = _read_manifest(dataset / "metadata" / "tile_manifest.csv")
    selected = select_samples(rows, categories, samples_per_category, seed)
    staging = output.parent / f".{output.name}.building-{os.getpid()}"
    if staging.exists():
        raise VisualizationError(f"staging path already exists: {staging}")
    counts: dict[str, int] = {}
    try:
        staging.mkdir(parents=True)
        for category, category_rows in selected.items():
            category_dir = staging / category
            category_dir.mkdir()
            for index, row in enumerate(category_rows, start=1):
                image = (
                    _render_ambiguous(row, source)
                    if category == "ambiguous"
                    else _render_safe(row, dataset)
                )
                image.save(
                    category_dir / f"{index:03d}__{Path(row['tile_file']).stem}.png",
                    format="PNG",
                    compress_level=6,
                    optimize=False,
                )
            counts[category] = len(category_rows)
        staging.replace(output)
    except Exception:
        if staging.is_dir():
            shutil.rmtree(staging)
        raise
    return counts


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--categories", nargs="+", choices=CATEGORIES, default=list(CATEGORIES))
    parser.add_argument("--samples-per-category", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    """Run the visualization CLI."""
    args = parse_args()
    try:
        counts = visualize(
            args.source,
            args.dataset,
            args.output,
            args.categories,
            args.samples_per_category,
            args.seed,
        )
    except VisualizationError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"status": "PASS", "samples": counts}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
