"""Build and validate the augmented-first single-class tunnel crack dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import shutil
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
VARIANTS = ("orig", "hflip", "vflip", "bright120", "gauss_s10")
SPLIT_RATIOS = (("train", 0.70), ("val", 0.20), ("test", 0.10))
CLASS_ID_PATTERN = re.compile(r"^[+-]?\d+$")
BRIGHTNESS_FACTOR = 1.20
NOISE_MEAN = 0.0
NOISE_SIGMA = 10.0
JPEG_QUALITY = 95
MIN_FREE_BYTES = 20 * 1024**3
HASH_CHUNK_BYTES = 4 * 1024**2


class BuildStop(RuntimeError):
    """Signal a safety or preflight condition that forbids starting the build."""


@dataclass
class SourceRecord:
    """One readable source image with at least one valid crack box."""

    stem: str
    image_path: Path
    label_path: Path
    raw_crack_box_count: int
    valid_boxes: list[tuple[float, float, float, float]]
    skipped_invalid_box_count: int
    width: int
    height: int


def parse_nonnegative_int(value: str) -> int:
    """Parse a nonnegative integer argument."""
    if not CLASS_ID_PATTERN.fullmatch(value) or int(value) < 0:
        raise argparse.ArgumentTypeError(f"expected a nonnegative integer, got {value!r}")
    return int(value)


def parse_positive_int(value: str) -> int:
    """Parse a positive integer argument."""
    parsed = parse_nonnegative_int(value)
    if parsed == 0:
        raise argparse.ArgumentTypeError("expected an integer greater than zero")
    return parsed


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--source-class-id", required=True, type=parse_nonnegative_int)
    parser.add_argument("--target-class-id", required=True, type=parse_nonnegative_int)
    parser.add_argument("--seed", required=True, type=parse_nonnegative_int)
    parser.add_argument("--workers", default=2, type=parse_positive_int, help="Concurrent image encode/decode workers")
    return parser.parse_args()


def human_bytes(value: int) -> str:
    """Format bytes using binary units."""
    amount = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if amount < 1024 or unit == "TiB":
            return f"{amount:.2f} {unit}"
        amount /= 1024
    return f"{amount:.2f} TiB"


def find_files(root: Path, suffixes: set[str]) -> list[Path]:
    """Recursively find files with supported case-insensitive suffixes."""
    return sorted(
        (path for path in root.rglob("*") if path.is_file() and path.suffix.casefold() in suffixes),
        key=lambda path: path.as_posix().casefold(),
    )


def ensure_empty_or_absent(path: Path, name: str) -> None:
    """Stop if a destination exists and is nonempty or is not a directory."""
    if path.exists() and not path.is_dir():
        raise BuildStop(f"{name} exists and is not a directory: {path}")
    if path.is_dir() and next(path.iterdir(), None) is not None:
        raise BuildStop(f"{name} already exists and is nonempty; refusing to overwrite: {path}")


def sha256_file(path: Path) -> str:
    """Hash a file without modifying it."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_raw_files(files: list[Path], dataset_root: Path, phase: str) -> dict[str, dict[str, Any]]:
    """Capture path, timestamp, size, and SHA-256 for every raw image and label."""
    snapshot: dict[str, dict[str, Any]] = {}
    total = len(files)
    print(f"[{phase}] Hashing {total} raw image/label files...", flush=True)
    for index, path in enumerate(files, 1):
        before = path.stat()
        digest = sha256_file(path)
        after = path.stat()
        if before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns:
            raise BuildStop(f"raw file changed while hashing: {path}")
        relative = path.relative_to(dataset_root).as_posix()
        snapshot[relative] = {
            "size": before.st_size,
            "mtime_ns": before.st_mtime_ns,
            "sha256": digest,
        }
        if index % 1000 == 0 or index == total:
            print(f"[{phase}] Hashed {index}/{total}", flush=True)
    return snapshot


def snapshot_digest(snapshot: dict[str, dict[str, Any]]) -> str:
    """Create a stable digest of a raw-file snapshot."""
    digest = hashlib.sha256()
    for relative, values in sorted(snapshot.items()):
        digest.update(relative.encode("utf-8"))
        digest.update(str(values["size"]).encode())
        digest.update(str(values["mtime_ns"]).encode())
        digest.update(values["sha256"].encode())
    return digest.hexdigest()


def compare_snapshots(
    before: dict[str, dict[str, Any]], after: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    """Return every added, removed, or changed raw file."""
    differences: list[dict[str, Any]] = []
    for relative in sorted(set(before) | set(after)):
        if relative not in before:
            differences.append({"path": relative, "change": "added"})
        elif relative not in after:
            differences.append({"path": relative, "change": "removed"})
        elif before[relative] != after[relative]:
            changed_fields = [key for key in before[relative] if before[relative][key] != after[relative][key]]
            differences.append({"path": relative, "change": "modified", "fields": changed_fields})
    return differences


def parse_label_tokens(tokens: list[str]) -> tuple[int | None, tuple[float, float, float, float] | None, list[str]]:
    """Validate one raw YOLO detection row."""
    errors: list[str] = []
    class_id: int | None = None
    coordinates: tuple[float, float, float, float] | None = None
    if len(tokens) != 5:
        errors.append("COLUMN_COUNT_NOT_5")
    if tokens and CLASS_ID_PATTERN.fullmatch(tokens[0]):
        candidate = int(tokens[0])
        if candidate >= 0:
            class_id = candidate
        else:
            errors.append("NEGATIVE_CLASS_ID")
    elif tokens:
        errors.append("CLASS_ID_NOT_INTEGER")
    else:
        errors.append("MISSING_CLASS_ID")

    if len(tokens) >= 5:
        values: list[float] = []
        for token in tokens[1:5]:
            try:
                value = float(token)
            except ValueError:
                errors.append("COORDINATE_NOT_NUMBER")
                continue
            if not math.isfinite(value):
                errors.append("COORDINATE_NOT_FINITE")
            values.append(value)
        if len(values) == 4 and all(math.isfinite(value) for value in values):
            coordinates = (values[0], values[1], values[2], values[3])

    if coordinates is not None:
        x_center, y_center, width, height = coordinates
        if not 0 <= x_center <= 1 or not 0 <= y_center <= 1:
            errors.append("CENTER_OUT_OF_RANGE")
        if not 0 < width <= 1 or not 0 < height <= 1:
            errors.append("SIZE_OUT_OF_RANGE")
        if 0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 < width <= 1 and 0 < height <= 1:
            left, right = x_center - width / 2, x_center + width / 2
            top, bottom = y_center - height / 2, y_center + height / 2
            if left < 0 or top < 0 or right > 1 or bottom > 1:
                errors.append("BBOX_OUT_OF_BOUNDS")
    return class_id, coordinates, list(dict.fromkeys(errors))


def discover_pairs(images_dir: Path, labels_dir: Path) -> tuple[list[Path], list[Path], dict[str, tuple[Path, Path]]]:
    """Discover and strictly pair raw images and labels by globally unique stem."""
    image_files = find_files(images_dir, IMAGE_SUFFIXES)
    label_files = find_files(labels_dir, {".txt"})
    image_map: dict[str, list[Path]] = defaultdict(list)
    label_map: dict[str, list[Path]] = defaultdict(list)
    for path in image_files:
        image_map[path.stem.casefold()].append(path)
    for path in label_files:
        label_map[path.stem.casefold()].append(path)
    conflicts = {
        stem: paths
        for stem, paths in {**image_map, **label_map}.items()
        if len(image_map.get(stem, [])) != 1 or len(label_map.get(stem, [])) != 1
    }
    missing_images = sorted(set(label_map) - set(image_map))
    missing_labels = sorted(set(image_map) - set(label_map))
    if conflicts or missing_images or missing_labels:
        raise BuildStop(
            "raw image/label pairing is not globally one-to-one: "
            f"conflicts={len(conflicts)}, labels_without_images={len(missing_images)}, "
            f"images_without_labels={len(missing_labels)}"
        )
    pairs = {stem: (image_map[stem][0], label_map[stem][0]) for stem in sorted(image_map)}
    return image_files, label_files, pairs


def inspect_sources(
    pairs: dict[str, tuple[Path, Path]], source_class_id: int
) -> tuple[list[SourceRecord], dict[str, Any]]:
    """Filter valid source crack boxes and record every invalid raw row."""
    sources: list[SourceRecord] = []
    invalid_rows: list[dict[str, Any]] = []
    excluded_sources: list[dict[str, Any]] = []
    raw_crack_image_count = 0
    raw_crack_box_count = 0
    valid_crack_box_count = 0
    dropped_invalid_crack_box_count = 0
    filtered_non_crack_row_count = 0
    images_without_crack = 0
    unreadable_target_images = 0

    for _, (image_path, label_path) in pairs.items():
        try:
            text = label_path.read_text(encoding="utf-8-sig")
        except (OSError, UnicodeError) as error:
            raise BuildStop(f"unable to read raw label {label_path}: {error}") from error
        raw_target_count = 0
        valid_boxes: list[tuple[float, float, float, float]] = []
        invalid_target_count = 0
        seen_rows: dict[str, int] = {}
        for line_number, raw_line in enumerate(text.splitlines(), 1):
            line = raw_line.strip()
            if not line:
                continue
            tokens = line.split()
            class_id, coordinates, errors = parse_label_tokens(tokens)
            canonical = " ".join(tokens)
            if canonical in seen_rows:
                errors.append("DUPLICATE_LABEL_LINE")
            else:
                seen_rows[canonical] = line_number
            if class_id == source_class_id:
                raw_target_count += 1
                raw_crack_box_count += 1
                if errors or coordinates is None:
                    invalid_target_count += 1
                    dropped_invalid_crack_box_count += 1
                    action = "DROP_INVALID_CRACK_BOX"
                else:
                    valid_boxes.append(coordinates)
                    valid_crack_box_count += 1
                    action = "KEEP_AND_MAP"
            elif class_id is not None:
                filtered_non_crack_row_count += 1
                action = "FILTER_NON_CRACK_CLASS"
            else:
                action = "DROP_UNPARSABLE_ROW"
            if errors:
                invalid_rows.append(
                    {
                        "label": label_path.name,
                        "source_stem": image_path.stem,
                        "line_number": line_number,
                        "class_id": class_id,
                        "line": line,
                        "errors": errors,
                        "action": action,
                    }
                )

        if raw_target_count == 0:
            images_without_crack += 1
            continue
        raw_crack_image_count += 1
        if not valid_boxes:
            excluded_sources.append(
                {
                    "source_stem": image_path.stem,
                    "reason": "NO_VALID_CRACK_BOX_AFTER_INVALID_DROP",
                    "raw_crack_box_count": raw_target_count,
                    "dropped_invalid_crack_box_count": invalid_target_count,
                }
            )
            continue
        try:
            with Image.open(image_path) as image:
                image.load()
                width, height = image.size
            if width <= 0 or height <= 0:
                raise ValueError(f"invalid dimensions {width}x{height}")
        except Exception as error:
            unreadable_target_images += 1
            valid_crack_box_count -= len(valid_boxes)
            excluded_sources.append(
                {
                    "source_stem": image_path.stem,
                    "reason": "UNREADABLE_SOURCE_IMAGE",
                    "error": f"{type(error).__name__}: {error}",
                    "raw_crack_box_count": raw_target_count,
                    "dropped_invalid_crack_box_count": invalid_target_count,
                }
            )
            continue
        sources.append(
            SourceRecord(
                stem=image_path.stem,
                image_path=image_path,
                label_path=label_path,
                raw_crack_box_count=raw_target_count,
                valid_boxes=valid_boxes,
                skipped_invalid_box_count=invalid_target_count,
                width=width,
                height=height,
            )
        )
    return sources, {
        "raw_crack_image_count": raw_crack_image_count,
        "valid_crack_image_count": len(sources),
        "excluded_crack_image_count": len(excluded_sources),
        "excluded_sources": excluded_sources,
        "images_without_crack_count": images_without_crack,
        "raw_crack_box_count": raw_crack_box_count,
        "valid_crack_box_count": valid_crack_box_count,
        "dropped_invalid_crack_box_count": dropped_invalid_crack_box_count,
        "filtered_non_crack_row_count": filtered_non_crack_row_count,
        "unreadable_target_image_count": unreadable_target_images,
        "invalid_raw_rows": invalid_rows,
        "bbox_boundary_tolerance": 0.0,
    }


def transform_boxes(
    boxes: list[tuple[float, float, float, float]], variant: str
) -> list[tuple[float, float, float, float]]:
    """Transform crack boxes independently from the original coordinates."""
    if variant == "hflip":
        return [(1 - x, y, width, height) for x, y, width, height in boxes]
    if variant == "vflip":
        return [(x, 1 - y, width, height) for x, y, width, height in boxes]
    return list(boxes)


def format_output_boxes(
    boxes: list[tuple[float, float, float, float]], target_class_id: int
) -> tuple[list[tuple[float, float, float, float]], str]:
    """Round output boxes to six decimals and revalidate the rounded values."""
    rounded_boxes: list[tuple[float, float, float, float]] = []
    lines: list[str] = []
    for box in boxes:
        formatted = [f"{value:.6f}" for value in box]
        rounded = tuple(float(value) for value in formatted)
        _, _, errors = parse_label_tokens([str(target_class_id), *formatted])
        if errors:
            raise BuildStop(f"six-decimal output rounding created an invalid box {formatted}: {errors}")
        rounded_boxes.append((rounded[0], rounded[1], rounded[2], rounded[3]))
        lines.append(f"{target_class_id} {' '.join(formatted)}")
    return rounded_boxes, "\n".join(lines) + "\n"


def derived_noise_seed(seed: int, source_stem: str) -> int:
    """Derive a stable per-source NumPy seed from the global seed and source stem."""
    payload = f"{seed}:{source_stem}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def largest_remainder_split_counts(total: int) -> dict[str, int]:
    """Allocate 70/20/10 counts with largest-remainder rounding and an exact total."""
    exact = [(name, total * ratio) for name, ratio in SPLIT_RATIOS]
    counts = {name: math.floor(value) for name, value in exact}
    remainder = total - sum(counts.values())
    ranked = sorted(enumerate(exact), key=lambda item: (-(item[1][1] - math.floor(item[1][1])), item[0]))
    for index in range(remainder):
        counts[ranked[index][1][0]] += 1
    return counts


def create_variant_records(
    sources: list[SourceRecord], output_root: Path, target_class_id: int, seed: int
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Create all five variant records, then deterministically shuffle and assign splits."""
    records: list[dict[str, Any]] = []
    output_stems: set[str] = set()
    for source in sorted(sources, key=lambda item: item.stem.casefold()):
        for variant in VARIANTS:
            output_stem = f"{source.stem}__{variant}"
            key = output_stem.casefold()
            if key in output_stems:
                raise BuildStop(f"duplicate output stem: {output_stem}")
            output_stems.add(key)
            output_boxes, label_text = format_output_boxes(
                transform_boxes(source.valid_boxes, variant), target_class_id
            )
            records.append(
                {
                    "source_stem": source.stem,
                    "output_stem": output_stem,
                    "original_image": str(source.image_path),
                    "variant": variant,
                    "split": "",
                    "brightness_factor": f"{BRIGHTNESS_FACTOR:.2f}" if variant == "bright120" else "",
                    "noise_mean": f"{NOISE_MEAN:.1f}" if variant == "gauss_s10" else "",
                    "noise_sigma": f"{NOISE_SIGMA:.1f}" if variant == "gauss_s10" else "",
                    "source_crack_box_count": source.raw_crack_box_count,
                    "output_crack_box_count": len(output_boxes),
                    "skipped_invalid_box_count": source.skipped_invalid_box_count,
                    "seed": seed,
                    "noise_derived_seed": derived_noise_seed(seed, source.stem) if variant == "gauss_s10" else "",
                    "source_width": source.width,
                    "source_height": source.height,
                    "output_boxes": output_boxes,
                    "label_text": label_text,
                    "output_image": "",
                    "output_label": "",
                    "image_path": None,
                    "label_path": None,
                }
            )
    random.Random(seed).shuffle(records)
    split_counts = largest_remainder_split_counts(len(records))
    split_sequence = [split for split, _ in SPLIT_RATIOS for _ in range(split_counts[split])]
    for shuffle_index, (record, split) in enumerate(zip(records, split_sequence)):
        record["shuffle_index"] = shuffle_index
        record["split"] = split
        image_suffix = Path(record["original_image"]).suffix if record["variant"] == "orig" else ".jpg"
        image_relative = Path("images") / split / f"{record['output_stem']}{image_suffix}"
        label_relative = Path("labels") / split / f"{record['output_stem']}.txt"
        record["output_image"] = image_relative.as_posix()
        record["output_label"] = label_relative.as_posix()
        record["image_path"] = output_root / image_relative
        record["label_path"] = output_root / label_relative
    return records, split_counts


def write_data_yaml(output_root: Path) -> None:
    """Write the dataset YAML using a forward-slash absolute path."""
    content = (
        f"path: {output_root.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n\n"
        "names:\n"
        "  0: crack\n"
    )
    (output_root / "data.yaml").write_text(content, encoding="utf-8")


def generate_source_variants(source: SourceRecord, records: list[dict[str, Any]]) -> None:
    """Generate all variants independently from one decoded original image."""
    by_variant = {record["variant"]: record for record in records}
    original_record = by_variant["orig"]
    shutil.copy2(source.image_path, original_record["image_path"])
    original_record["label_path"].write_text(original_record["label_text"], encoding="utf-8")

    with Image.open(source.image_path) as image:
        image.load()
        rgb = image.convert("RGB")
    if rgb.size != (source.width, source.height):
        raise RuntimeError(f"source dimensions changed during build: {source.image_path}")
    base = np.asarray(rgb, dtype=np.uint8)

    hflip = rgb.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    hflip.save(by_variant["hflip"]["image_path"], format="JPEG", quality=JPEG_QUALITY)
    by_variant["hflip"]["label_path"].write_text(by_variant["hflip"]["label_text"], encoding="utf-8")
    hflip.close()

    vflip = rgb.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    vflip.save(by_variant["vflip"]["image_path"], format="JPEG", quality=JPEG_QUALITY)
    by_variant["vflip"]["label_path"].write_text(by_variant["vflip"]["label_text"], encoding="utf-8")
    vflip.close()

    bright_array = np.multiply(base, BRIGHTNESS_FACTOR, dtype=np.float32)
    np.clip(bright_array, 0, 255, out=bright_array)
    bright = Image.fromarray(bright_array.astype(np.uint8), "RGB")
    bright.save(by_variant["bright120"]["image_path"], format="JPEG", quality=JPEG_QUALITY)
    by_variant["bright120"]["label_path"].write_text(
        by_variant["bright120"]["label_text"], encoding="utf-8"
    )
    bright.close()
    del bright_array

    noise_record = by_variant["gauss_s10"]
    rng = np.random.default_rng(noise_record["noise_derived_seed"])
    noisy_array = rng.standard_normal(base.shape, dtype=np.float32)
    noisy_array *= NOISE_SIGMA
    noisy_array += NOISE_MEAN
    noisy_array += base
    np.clip(noisy_array, 0, 255, out=noisy_array)
    noisy = Image.fromarray(noisy_array.astype(np.uint8), "RGB")
    noisy.save(noise_record["image_path"], format="JPEG", quality=JPEG_QUALITY)
    noise_record["label_path"].write_text(noise_record["label_text"], encoding="utf-8")
    noisy.close()
    rgb.close()


def write_manifest(path: Path, records: list[dict[str, Any]]) -> None:
    """Write the auditable split manifest."""
    fields = [
        "source_stem",
        "output_stem",
        "original_image",
        "output_image",
        "output_label",
        "variant",
        "split",
        "brightness_factor",
        "noise_mean",
        "noise_sigma",
        "source_crack_box_count",
        "output_crack_box_count",
        "skipped_invalid_box_count",
        "seed",
        "noise_derived_seed",
        "source_width",
        "source_height",
        "shuffle_index",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for record in records:
            writer.writerow({field: record[field] for field in fields})


def assertion(name: str, passed: bool, details: str) -> dict[str, Any]:
    """Create one serializable quality assertion."""
    return {"name": name, "passed": bool(passed), "details": details}


def validate_built_dataset(
    output_root: Path,
    records: list[dict[str, Any]],
    split_counts: dict[str, int],
    target_class_id: int,
    raw_before: dict[str, dict[str, Any]],
    dataset_root: Path,
    workers: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Fully decode every output and validate all labels, manifest rows, and split counts."""
    manifest_images = {record["output_image"] for record in records}
    manifest_labels = {record["output_label"] for record in records}
    disk_images = {
        path.relative_to(output_root).as_posix()
        for path in find_files(output_root / "images", IMAGE_SUFFIXES)
    }
    disk_labels = {
        path.relative_to(output_root).as_posix() for path in find_files(output_root / "labels", {".txt"})
    }
    disk_image_split_counts = Counter(Path(path).parts[1] for path in disk_images)
    disk_label_split_counts = Counter(Path(path).parts[1] for path in disk_labels)
    image_errors: list[dict[str, str]] = []
    orig_hash_mismatches: list[str] = []

    def validate_image(record: dict[str, Any]) -> tuple[list[str], bool]:
        errors: list[str] = []
        path = record["image_path"]
        image_format = ""
        try:
            with Image.open(path) as image:
                image_format = image.format or ""
                image.load()
                size = image.size
            if size != (record["source_width"], record["source_height"]):
                errors.append(f"DIMENSION_MISMATCH: {size}")
            if record["variant"] != "orig" and image_format.upper() != "JPEG":
                errors.append(f"FORMAT_NOT_JPEG: {image_format}")
        except Exception as error:
            errors.append(f"UNREADABLE: {type(error).__name__}: {error}")
        orig_identical = True
        if record["variant"] == "orig" and not errors:
            relative_source = Path(record["original_image"]).relative_to(dataset_root).as_posix()
            orig_identical = sha256_file(path) == raw_before[relative_source]["sha256"]
        return errors, orig_identical

    print(f"[validate] Fully decoding {len(records)} output images...", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for index, (record, result) in enumerate(zip(records, executor.map(validate_image, records)), 1):
            errors, orig_identical = result
            if errors:
                image_errors.append({"path": record["output_image"], "errors": "; ".join(errors)})
            if not orig_identical:
                orig_hash_mismatches.append(record["output_image"])
            if index % 500 == 0 or index == len(records):
                print(f"[validate] Decoded {index}/{len(records)}", flush=True)

    empty_labels: list[str] = []
    label_errors: list[dict[str, Any]] = []
    split_box_counts: Counter[str] = Counter()
    for record in records:
        path = record["label_path"]
        try:
            lines = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        except (OSError, UnicodeError) as error:
            label_errors.append({"path": record["output_label"], "line": None, "errors": [str(error)]})
            continue
        if not lines:
            empty_labels.append(record["output_label"])
            continue
        for line_number, line in enumerate(lines, 1):
            class_id, _, errors = parse_label_tokens(line.split())
            if class_id != target_class_id:
                errors.append("CLASS_ID_NOT_TARGET")
            if errors:
                label_errors.append(
                    {"path": record["output_label"], "line": line_number, "errors": list(dict.fromkeys(errors))}
                )
            else:
                split_box_counts[record["split"]] += 1

    source_variants: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        source_variants[record["source_stem"]].append(record)
    bad_source_variants = {
        source: [record["variant"] for record in source_records]
        for source, source_records in source_variants.items()
        if len(source_records) != len(VARIANTS)
        or set(record["variant"] for record in source_records) != set(VARIANTS)
    }
    manifest_split_counts = Counter(record["split"] for record in records)
    all_output_stems = [record["output_stem"].casefold() for record in records]
    unreadable_image_count = sum(
        any(error.startswith("UNREADABLE:") for error in item["errors"].split("; ")) for item in image_errors
    )
    dimension_mismatch_count = sum("DIMENSION_MISMATCH:" in item["errors"] for item in image_errors)
    augmented_format_error_count = sum("FORMAT_NOT_JPEG:" in item["errors"] for item in image_errors)
    checks = [
        assertion(
            "split_image_label_counts_match",
            all(disk_image_split_counts[split] == disk_label_split_counts[split] for split, _ in SPLIT_RATIOS),
            str(
                {
                    split: {
                        "images": disk_image_split_counts[split],
                        "labels": disk_label_split_counts[split],
                    }
                    for split, _ in SPLIT_RATIOS
                }
            ),
        ),
        assertion("all_output_images_readable", not unreadable_image_count, f"errors={unreadable_image_count}"),
        assertion(
            "all_output_resolutions_match_sources",
            not dimension_mismatch_count,
            f"mismatches={dimension_mismatch_count}",
        ),
        assertion(
            "all_augmented_outputs_are_jpeg",
            not augmented_format_error_count,
            f"errors={augmented_format_error_count}",
        ),
        assertion("all_labels_have_five_columns", not label_errors, f"errors={len(label_errors)}"),
        assertion("all_output_class_ids_are_target", not label_errors, f"target_class_id={target_class_id}"),
        assertion("all_output_coordinates_valid", not label_errors, f"errors={len(label_errors)}"),
        assertion("no_empty_label_files", not empty_labels, f"empty={len(empty_labels)}"),
        assertion(
            "no_unmatched_output_files",
            disk_images == manifest_images and disk_labels == manifest_labels,
            f"disk_images={len(disk_images)}, disk_labels={len(disk_labels)}, manifest={len(records)}",
        ),
        assertion(
            "five_variants_per_valid_source",
            not bad_source_variants,
            f"sources={len(source_variants)}, bad_sources={len(bad_source_variants)}",
        ),
        assertion(
            "manifest_matches_disk",
            disk_images == manifest_images and disk_labels == manifest_labels,
            f"image_delta={len(disk_images ^ manifest_images)}, label_delta={len(disk_labels ^ manifest_labels)}",
        ),
        assertion(
            "split_counts_match_largest_remainder_70_20_10",
            all(
                disk_image_split_counts[split] == split_counts[split]
                and manifest_split_counts[split] == split_counts[split]
                for split, _ in SPLIT_RATIOS
            ),
            f"expected={split_counts}, disk={dict(disk_image_split_counts)}, manifest={dict(manifest_split_counts)}",
        ),
        assertion(
            "total_output_is_exactly_five_times_valid_sources",
            len(records) == len(source_variants) * len(VARIANTS),
            f"outputs={len(records)}, sources={len(source_variants)}",
        ),
        assertion(
            "output_stems_are_unique",
            len(all_output_stems) == len(set(all_output_stems)),
            f"outputs={len(all_output_stems)}, unique={len(set(all_output_stems))}",
        ),
        assertion(
            "orig_variants_are_byte_identical_copies",
            not orig_hash_mismatches,
            f"mismatches={len(orig_hash_mismatches)}",
        ),
    ]
    return checks, {
        "image_errors": image_errors,
        "label_errors": label_errors,
        "empty_labels": empty_labels,
        "bad_source_variants": bad_source_variants,
        "orig_hash_mismatches": orig_hash_mismatches,
        "split_image_counts": {split: disk_image_split_counts[split] for split, _ in SPLIT_RATIOS},
        "split_label_counts": {split: disk_label_split_counts[split] for split, _ in SPLIT_RATIOS},
        "split_crack_box_counts": {split: split_box_counts[split] for split, _ in SPLIT_RATIOS},
    }


def annotated_thumbnail(record: dict[str, Any], width: int) -> Image.Image:
    """Load one output image, resize it, and draw its crack boxes."""
    with Image.open(record["image_path"]) as image:
        image.load()
        rgb = image.convert("RGB")
    height = max(1, round(rgb.height * width / rgb.width))
    resized = rgb.resize((width, height), Image.Resampling.LANCZOS)
    rgb.close()
    draw = ImageDraw.Draw(resized)
    line_width = max(2, width // 320)
    for x_center, y_center, box_width, box_height in record["output_boxes"]:
        left = (x_center - box_width / 2) * width
        right = (x_center + box_width / 2) * width
        top = (y_center - box_height / 2) * height
        bottom = (y_center + box_height / 2) * height
        draw.rectangle((left, top, right, bottom), outline=(255, 32, 32), width=line_width)
    return resized


def create_visual_audits(records: list[dict[str, Any]], visual_root: Path, seed: int) -> dict[str, Any]:
    """Create deterministic per-split boxed samples and ten five-variant comparison grids."""
    visual_root.mkdir(parents=True, exist_ok=True)
    split_sample_counts: dict[str, int] = {}
    for split, _ in SPLIT_RATIOS:
        candidates = sorted((record for record in records if record["split"] == split), key=lambda x: x["output_stem"])
        sample_count = min(10, len(candidates))
        rng = random.Random(derived_noise_seed(seed, f"visual:{split}"))
        selected = rng.sample(candidates, sample_count)
        destination = visual_root / "split_samples" / split
        destination.mkdir(parents=True, exist_ok=True)
        for record in selected:
            thumbnail = annotated_thumbnail(record, 1280)
            thumbnail.save(destination / f"{record['output_stem']}__boxed.jpg", format="JPEG", quality=92)
            thumbnail.close()
        split_sample_counts[split] = sample_count

    by_source: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        by_source[record["source_stem"]][record["variant"]] = record
    source_candidates = sorted(source for source, variants in by_source.items() if set(variants) == set(VARIANTS))
    comparison_count = min(10, len(source_candidates))
    rng = random.Random(derived_noise_seed(seed, "visual:comparisons"))
    selected_sources = rng.sample(source_candidates, comparison_count)
    comparison_dir = visual_root / "five_variant_comparisons"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()
    cell_width = 640
    header_height = 28
    for source_stem in selected_sources:
        thumbnails = [annotated_thumbnail(by_source[source_stem][variant], cell_width) for variant in VARIANTS]
        cell_height = max(image.height for image in thumbnails)
        canvas = Image.new("RGB", (cell_width * len(VARIANTS), cell_height + header_height), "white")
        draw = ImageDraw.Draw(canvas)
        for index, (variant, thumbnail) in enumerate(zip(VARIANTS, thumbnails)):
            record = by_source[source_stem][variant]
            x_offset = index * cell_width
            canvas.paste(thumbnail, (x_offset, header_height))
            draw.text((x_offset + 8, 7), f"{variant} | {record['split']}", fill="black", font=font)
            thumbnail.close()
        canvas.save(comparison_dir / f"{source_stem}__5variants.jpg", format="JPEG", quality=92)
        canvas.close()
    return {
        "root": str(visual_root),
        "split_sample_counts": split_sample_counts,
        "five_variant_comparison_count": comparison_count,
        "selected_comparison_sources": selected_sources,
    }


def directory_size(root: Path) -> int:
    """Sum all regular files beneath a directory."""
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def split_statistics(records: list[dict[str, Any]], validation: dict[str, Any]) -> dict[str, Any]:
    """Summarize images, boxes, and variants for each split."""
    result: dict[str, Any] = {}
    total = len(records)
    for split, _ in SPLIT_RATIOS:
        split_records = [record for record in records if record["split"] == split]
        variants = Counter(record["variant"] for record in split_records)
        result[split] = {
            "image_count": len(split_records),
            "label_count": validation["split_label_counts"][split],
            "crack_box_count": validation["split_crack_box_counts"][split],
            "ratio": len(split_records) / total if total else 0,
            "variant_counts": {variant: variants[variant] for variant in VARIANTS},
        }
    return result


def write_split_summary(path: Path, split_stats: dict[str, Any]) -> None:
    """Write the report-level split summary CSV."""
    fields = ["split", "image_count", "label_count", "crack_box_count", "ratio", *VARIANTS]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for split, _ in SPLIT_RATIOS:
            values = split_stats[split]
            writer.writerow(
                {
                    "split": split,
                    "image_count": values["image_count"],
                    "label_count": values["label_count"],
                    "crack_box_count": values["crack_box_count"],
                    "ratio": f"{values['ratio']:.8f}",
                    **values["variant_counts"],
                }
            )


def render_markdown(report: dict[str, Any]) -> str:
    """Render the build report as Markdown."""
    filtering = report["filtering"]
    augmentation = report["augmentation"]
    lines = [
        "# Tunnel_Crack_AugFirst_5x 构建报告",
        "",
        f"- 开始时间：`{report['execution']['started_at']}`",
        f"- 结束时间：`{report['execution']['finished_at']}`",
        f"- 执行时间：{report['execution']['seconds']:.2f} 秒",
        f"- 原始数据集：`{report['input']['dataset_root']}`",
        f"- 输出数据集：`{report['output']['dataset_root']}`",
        "- 原始数据操作：**只读；没有修改、删除、重命名或覆盖原始图片和标签**。",
        f"- 类别映射：原始 `{report['input']['source_class_id']}` → 新 `{report['output']['target_class_id']}`（`crack`）。",
        "",
        "## 筛选结果",
        "",
        "| 指标 | 数量 |",
        "| --- | ---: |",
        f"| 原始裂缝图片 | {filtering['raw_crack_image_count']} |",
        f"| 有效裂缝图片 | {filtering['valid_crack_image_count']} |",
        f"| 被排除裂缝图片 | {filtering['excluded_crack_image_count']} |",
        f"| 不含裂缝图片 | {filtering['images_without_crack_count']} |",
        f"| 原始裂缝框 | {filtering['raw_crack_box_count']} |",
        f"| 有效裂缝框 | {filtering['valid_crack_box_count']} |",
        f"| 删除的越界/无效裂缝框 | {filtering['dropped_invalid_crack_box_count']} |",
        f"| 筛除的非裂缝标签行 | {filtering['filtered_non_crack_row_count']} |",
        "",
        "被排除裂缝图片：",
        "",
    ]
    if filtering["excluded_sources"]:
        lines.extend(
            f"- `{item['source_stem']}`：{item['reason']}（原始裂缝框 {item['raw_crack_box_count']}）"
            for item in filtering["excluded_sources"]
        )
    else:
        lines.append("- 无")
    lines.extend(["", "严格零容差复检发现的无效/越界记录（包含两条已知记录）：", ""])
    for item in filtering["invalid_raw_rows"]:
        lines.append(
            f"- `{item['label']}` 第 {item['line_number']} 行，类别 `{item['class_id']}`，"
            f"处理：`{item['action']}`，错误：`{', '.join(item['errors'])}`。"
        )
    lines.extend(
        [
            "",
            "## 增强与输出",
            "",
            f"- 每个有效 source 生成 {len(VARIANTS)} 个独立版本；总输出图片：{augmentation['total_output_images']}。",
            f"- 全局 seed：`{report['input']['seed']}`。",
            f"- 数据集容量：{report['output']['total_size_bytes']} B（{report['output']['total_size_human']}）。",
            "",
            "| variant | 数量 |",
            "| --- | ---: |",
        ]
    )
    lines.extend(f"| {variant} | {augmentation['variant_counts'][variant]} |" for variant in VARIANTS)
    lines.extend(
        [
            "",
            "## 7:2:1 variant 级随机划分",
            "",
            "同一 source 不分组；不同 variant 允许进入不同集合。",
            "",
            "| split | 图片 | 标签 | 裂缝框 | 实际比例 | orig | hflip | vflip | bright120 | gauss_s10 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for split, _ in SPLIT_RATIOS:
        values = report["splits"][split]
        variants = values["variant_counts"]
        lines.append(
            f"| {split} | {values['image_count']} | {values['label_count']} | {values['crack_box_count']} | "
            f"{values['ratio']:.6%} | {variants['orig']} | {variants['hflip']} | {variants['vflip']} | "
            f"{variants['bright120']} | {variants['gauss_s10']} |"
        )
    multi = report["source_split_audit"]
    lines.extend(
        [
            "",
            f"- 出现在多个 split 的 source：{multi['multi_split_source_count']} / {multi['source_count']} "
            f"（{multi['multi_split_source_ratio']:.6%}）。",
            "",
            "## 质量断言",
            "",
            "| 断言 | 结果 | 详情 |",
            "| --- | --- | --- |",
        ]
    )
    for check in report["quality_assertions"]:
        lines.append(f"| {check['name']} | {'PASS' if check['passed'] else 'FAIL'} | {check['details']} |")
    lines.extend(
        [
            "",
            f"- 原始文件前后快照一致：{'是' if report['raw_integrity']['unchanged'] else '否'}。",
            f"- 原始文件数量：{report['raw_integrity']['file_count']}。",
            f"- 可视化目录：`{report['visual_audit']['root']}`。",
            "",
            "## 冒烟训练就绪性",
            "",
            f"**{'满足下一步 YOLO26n 冒烟训练条件' if report['readiness']['can_smoke_train'] else '暂不满足冒烟训练条件'}。**",
            "",
        ]
    )
    if report["readiness"]["blockers"]:
        lines.append("阻断项：")
        lines.append("")
        lines.extend(f"- {item}" for item in report["readiness"]["blockers"])
        lines.append("")
    return "\n".join(lines)


def write_reports(report_dir: Path, report: dict[str, Any]) -> None:
    """Write JSON, Markdown, and CSV build reports in UTF-8."""
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "crack_augfirst_build_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (report_dir / "crack_augfirst_build_report.md").write_text(render_markdown(report), encoding="utf-8")
    write_split_summary(report_dir / "crack_augfirst_split_summary.csv", report["splits"])


def main() -> int:
    """Run the guarded build, validation, visualization, and reporting pipeline."""
    args = parse_args()
    started_clock = time.perf_counter()
    started_at = datetime.now().astimezone()
    dataset_root = args.dataset_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    visual_root = output_root.parent / "_audit_crack_augfirst_visuals"
    report_dir = Path(__file__).resolve().parents[1] / "reports"

    try:
        images_dir = dataset_root / "images"
        labels_dir = dataset_root / "labels"
        for path in (dataset_root, images_dir, labels_dir):
            if not path.is_dir():
                raise BuildStop(f"required raw directory is missing: {path}")
        ensure_empty_or_absent(output_root, "dataset output directory")
        ensure_empty_or_absent(visual_root, "visual audit directory")
        free_before = shutil.disk_usage(output_root.drive + "\\").free
        if free_before < MIN_FREE_BYTES:
            raise BuildStop(
                f"output drive has only {human_bytes(free_before)} free; "
                f"at least {human_bytes(MIN_FREE_BYTES)} required"
            )
        image_files, label_files, pairs = discover_pairs(images_dir, labels_dir)
        raw_files = sorted([*image_files, *label_files], key=lambda path: path.as_posix().casefold())
        raw_before = snapshot_raw_files(raw_files, dataset_root, "before")
        sources, filtering = inspect_sources(pairs, args.source_class_id)
        if not sources:
            raise BuildStop("no readable source image has a valid crack box")
        records, expected_split_counts = create_variant_records(
            sources, output_root, args.target_class_id, args.seed
        )

        for split, _ in SPLIT_RATIOS:
            (output_root / "images" / split).mkdir(parents=True, exist_ok=True)
            (output_root / "labels" / split).mkdir(parents=True, exist_ok=True)
        write_data_yaml(output_root)
        records_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            records_by_source[record["source_stem"]].append(record)
        print(
            f"[build] Generating {len(records)} variants from {len(sources)} valid source images "
            f"with {args.workers} workers...",
            flush=True,
        )
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            results = executor.map(
                lambda source: generate_source_variants(
                    source,
                    records_by_source[source.stem],
                ),
                sources,
            )
            for index, _ in enumerate(results, 1):
                if index % 25 == 0 or index == len(sources):
                    print(f"[build] Generated {index}/{len(sources)} sources", flush=True)
        write_manifest(output_root / "split_manifest.csv", records)

        checks, validation = validate_built_dataset(
            output_root,
            records,
            expected_split_counts,
            args.target_class_id,
            raw_before,
            dataset_root,
            args.workers,
        )
        visual_audit = create_visual_audits(records, visual_root, args.seed)
        raw_after = snapshot_raw_files(raw_files, dataset_root, "after")
        raw_differences = compare_snapshots(raw_before, raw_after)
        checks.append(
            assertion(
                "raw_images_and_labels_unchanged",
                not raw_differences,
                f"files={len(raw_before)}, differences={len(raw_differences)}",
            )
        )
        split_stats = split_statistics(records, validation)
        variant_counts = Counter(record["variant"] for record in records)
        source_splits: dict[str, set[str]] = defaultdict(set)
        for record in records:
            source_splits[record["source_stem"]].add(record["split"])
        multi_split_count = sum(len(splits) > 1 for splits in source_splits.values())
        output_size = directory_size(output_root)
        visual_size = directory_size(visual_root)
        finished_at = datetime.now().astimezone()
        elapsed = time.perf_counter() - started_clock
        blockers = [f"Quality assertion failed: {check['name']}" for check in checks if not check["passed"]]
        report = {
            "schema_version": 1,
            "execution": {
                "started_at": started_at.isoformat(timespec="seconds"),
                "finished_at": finished_at.isoformat(timespec="seconds"),
                "seconds": elapsed,
                "python_executable": sys.executable,
                "workers": args.workers,
            },
            "input": {
                "dataset_root": str(dataset_root),
                "image_count": len(image_files),
                "label_count": len(label_files),
                "source_class_id": args.source_class_id,
                "seed": args.seed,
                "free_bytes_before": free_before,
                "free_human_before": human_bytes(free_before),
            },
            "output": {
                "dataset_root": str(output_root),
                "target_class_id": args.target_class_id,
                "class_name": "crack",
                "total_size_bytes": output_size,
                "total_size_human": human_bytes(output_size),
                "visual_size_bytes": visual_size,
                "visual_size_human": human_bytes(visual_size),
                "free_bytes_after": shutil.disk_usage(output_root.drive + "\\").free,
            },
            "filtering": filtering,
            "augmentation": {
                "variants": list(VARIANTS),
                "variant_counts": {variant: variant_counts[variant] for variant in VARIANTS},
                "total_output_images": len(records),
                "brightness_factor": BRIGHTNESS_FACTOR,
                "noise_mean": NOISE_MEAN,
                "noise_sigma": NOISE_SIGMA,
                "jpeg_quality": JPEG_QUALITY,
                "noise_seed_derivation": "uint64_be(sha256(f'{global_seed}:{source_stem}')[:8])",
            },
            "splits": split_stats,
            "source_split_audit": {
                "source_count": len(source_splits),
                "multi_split_source_count": multi_split_count,
                "multi_split_source_ratio": multi_split_count / len(source_splits),
            },
            "validation_details": validation,
            "quality_assertions": checks,
            "raw_integrity": {
                "unchanged": not raw_differences,
                "file_count": len(raw_before),
                "before_snapshot_digest": snapshot_digest(raw_before),
                "after_snapshot_digest": snapshot_digest(raw_after),
                "differences": raw_differences,
            },
            "visual_audit": visual_audit,
            "readiness": {"can_smoke_train": not blockers, "blockers": blockers},
        }
        write_reports(report_dir, report)
        print(f"Build complete in {elapsed:.2f} seconds", flush=True)
        print(f"Dataset: {output_root} ({human_bytes(output_size)})", flush=True)
        print(f"Ready for YOLO26n smoke training: {not blockers}", flush=True)
        return 0 if not blockers else 1
    except BuildStop as error:
        print(f"SAFETY STOP: {error}", file=sys.stderr, flush=True)
        return 2
    except Exception as error:
        print(f"BUILD FAILED: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
