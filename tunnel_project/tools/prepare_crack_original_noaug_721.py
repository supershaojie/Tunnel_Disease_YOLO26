# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Build and independently validate an original-image, single-class crack dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

import yaml
from PIL import Image, ImageDraw


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
SPLIT_RATIOS = (("train", 0.70), ("val", 0.20), ("test", 0.10))
CLASS_ID_PATTERN = re.compile(r"^[+-]?\d+$")
AUGMENTATION_MARKER = re.compile(
    r"(aug|flip|rotate|rotated|mosaic|mixup|hsv|brightness|contrast|copy[_-]?paste)", re.IGNORECASE
)
HASH_CHUNK_BYTES = 4 * 1024**2
FLOAT_PIXEL_TOLERANCE = 1e-9
OUTPUT_NORMALIZED_TOLERANCE = 1e-9
REPAIR_THRESHOLD_PX = 10.0
REPAIR_THRESHOLD_RATIO = 0.005
NEAR_DUPLICATE_HAMMING_THRESHOLD = 5
DEFAULT_SOURCE = Path(r"E:\ditie_dataset\隧道数据集")
DEFAULT_OUTPUT = Path(
    r"E:\PycharmProjects\Tunnel_Disease_YOLO26\datasets\Tunnel_Crack_Original_NoAug_7_2_1_seed42"
)
SERVER_DATASET_PATH = "/root/autodl-tmp/projects/Tunnel_Disease_YOLO26/datasets/Tunnel_Crack_Original_NoAug_7_2_1_seed42"
DEFAULT_ZIP_OUTPUT = Path(
    r"E:\PycharmProjects\Tunnel_Disease_YOLO26\artifacts\datasets\Tunnel_Crack_Original_NoAug_7_2_1_seed42.zip"
)


class BuildStop(RuntimeError):
    """Signal a condition that forbids publishing the final dataset."""


@dataclass
class SourceSample:
    """One validated source image that contains at least one crack box."""

    source_image: Path
    source_label: Path
    relative_image: str
    relative_label: str
    width: int
    height: int
    crack_tokens: list[tuple[str, str, str, str]]
    crack_values: list[tuple[float, float, float, float]]
    image_bytes: int
    image_sha256: str = ""
    source_label_sha256: str = ""
    dhash: str = ""
    canonical_id: str = ""
    split: str = ""
    output_image: str = ""
    output_label: str = ""
    repairs: list[dict[str, Any]] = field(default_factory=list)

    @property
    def crack_box_count(self) -> int:
        """Return the number of retained crack boxes."""
        return len(self.crack_tokens)

    @property
    def equivalent_crack_label(self) -> tuple[tuple[float, float, float, float], ...]:
        """Return an order-independent numeric representation for duplicate comparison."""
        return tuple(sorted(self.crack_values))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--crack-class-name", default="crack")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--zip-output", type=Path, default=DEFAULT_ZIP_OUTPUT)
    parser.add_argument(
        "--verify-only", action="store_true", help="Independently rescan an already-built dataset using its manifest"
    )
    return parser.parse_args()


def normalized_relative(path: Path, root: Path) -> str:
    """Return a stable POSIX-style relative path."""
    return path.relative_to(root).as_posix()


def find_files(root: Path, suffixes: set[str]) -> list[Path]:
    """Recursively find regular files with case-insensitive suffix matching."""
    return sorted(
        (path for path in root.rglob("*") if path.is_file() and path.suffix.casefold() in suffixes),
        key=lambda path: normalized_relative(path, root).casefold(),
    )


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest without changing the file."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def directory_size(root: Path) -> int:
    """Return the total byte size of regular files below a directory."""
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def human_bytes(value: int) -> str:
    """Format a byte count with binary units."""
    amount = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if amount < 1024 or unit == "TiB":
            return f"{amount:.2f} {unit}"
        amount /= 1024
    return f"{amount:.2f} TiB"


def load_class_mapping(yaml_path: Path, requested_name: str) -> tuple[dict[int, str], int, str]:
    """Load names from source YAML and uniquely resolve the crack class."""
    try:
        payload = yaml.safe_load(yaml_path.read_text(encoding="utf-8-sig"))
    except Exception as error:
        raise BuildStop(f"cannot read source data.yaml: {type(error).__name__}: {error}") from error
    if not isinstance(payload, dict) or "names" not in payload:
        raise BuildStop("source data.yaml does not contain a names mapping")
    raw_names = payload["names"]
    if isinstance(raw_names, list):
        names = {index: str(name) for index, name in enumerate(raw_names)}
    elif isinstance(raw_names, dict):
        try:
            names = {int(index): str(name) for index, name in raw_names.items()}
        except (TypeError, ValueError) as error:
            raise BuildStop(f"source data.yaml contains a non-integer class key: {error}") from error
    else:
        raise BuildStop(f"source data.yaml names must be a list or mapping, got {type(raw_names).__name__}")
    if len(names) != len(set(names)) or any(index < 0 for index in names):
        raise BuildStop(f"source data.yaml contains invalid class IDs: {names}")
    if "nc" in payload and int(payload["nc"]) != len(names):
        raise BuildStop(f"source data.yaml nc={payload['nc']} but names contains {len(names)} classes")

    requested = requested_name.strip().casefold()
    accepted = {requested}
    if requested in {"crack", "裂缝"}:
        accepted.update({"crack", "裂缝"})
    matches = [(index, name) for index, name in names.items() if name.strip().casefold() in accepted]
    if len(matches) != 1:
        raise BuildStop(
            f"crack class cannot be uniquely resolved for {requested_name!r}; matches={matches}; complete names={names}"
        )
    return dict(sorted(names.items())), matches[0][0], matches[0][1]


def pairing_key(path: Path, root: Path) -> str:
    """Pair images and labels by relative directory and case-insensitive stem."""
    return Path(normalized_relative(path, root)).with_suffix("").as_posix().casefold()


def discover_pairs(images_dir: Path, labels_dir: Path) -> tuple[list[Path], list[Path], list[tuple[Path, Path]]]:
    """Discover strict one-to-one image/label pairs and reject conflicts."""
    images = find_files(images_dir, IMAGE_SUFFIXES)
    labels = find_files(labels_dir, {".txt"})
    image_map: dict[str, list[Path]] = defaultdict(list)
    label_map: dict[str, list[Path]] = defaultdict(list)
    for path in images:
        image_map[pairing_key(path, images_dir)].append(path)
    for path in labels:
        label_map[pairing_key(path, labels_dir)].append(path)

    extension_conflicts = {
        key: [str(path) for path in paths]
        for key, paths in image_map.items()
        if len(paths) != 1
    }
    label_conflicts = {
        key: [str(path) for path in paths]
        for key, paths in label_map.items()
        if len(paths) != 1
    }
    missing_labels = sorted(set(image_map) - set(label_map))
    labels_without_images = sorted(set(label_map) - set(image_map))
    if extension_conflicts or label_conflicts or missing_labels or labels_without_images:
        details = {
            "same_stem_image_extension_conflicts": extension_conflicts,
            "same_relative_label_conflicts": label_conflicts,
            "images_without_labels": missing_labels,
            "labels_without_images": labels_without_images,
        }
        raise BuildStop("source image/label pairing is not one-to-one:\n" + json.dumps(details, ensure_ascii=False, indent=2))
    pairs = [(image_map[key][0], label_map[key][0]) for key in sorted(image_map)]
    return images, labels, pairs


def parse_label_row(tokens: list[str], class_ids: set[int]) -> tuple[int | None, tuple[float, ...] | None, list[str]]:
    """Validate only the syntax needed to reliably identify and parse one YOLO row."""
    errors: list[str] = []
    class_id: int | None = None
    coordinates: tuple[float, ...] | None = None
    if len(tokens) != 5:
        errors.append("COLUMN_COUNT_NOT_5")
    if tokens and CLASS_ID_PATTERN.fullmatch(tokens[0]):
        class_id = int(tokens[0])
        if class_id not in class_ids:
            errors.append("CLASS_ID_NOT_IN_DATA_YAML")
    else:
        errors.append("CLASS_ID_NOT_INTEGER")
    if len(tokens) == 5:
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
            coordinates = tuple(values)
    return class_id, coordinates, list(dict.fromkeys(errors))


def pixel_geometry(
    coordinates: tuple[float, ...], width: int, height: int
) -> tuple[tuple[float, float, float, float], tuple[float, float, float, float]]:
    """Convert YOLO coordinates to pixel xyxy and return per-side positive overflow."""
    x_center, y_center, box_width, box_height = coordinates
    x1 = (x_center - box_width / 2) * width
    y1 = (y_center - box_height / 2) * height
    x2 = (x_center + box_width / 2) * width
    y2 = (y_center + box_height / 2) * height
    overflow = (max(0.0, -x1), max(0.0, -y1), max(0.0, x2 - width), max(0.0, y2 - height))
    overflow = tuple(0.0 if value <= FLOAT_PIXEL_TOLERANCE else value for value in overflow)
    return (x1, y1, x2, y2), overflow


def non_target_coordinate_issues(
    coordinates: tuple[float, ...], width: int, height: int
) -> tuple[list[str], tuple[float, ...], tuple[float, ...]]:
    """Describe non-target geometry problems for audit without affecting the build."""
    x_center, y_center, box_width, box_height = coordinates
    xyxy, overflow = pixel_geometry(coordinates, width, height)
    issues: list[str] = []
    if not 0 <= x_center <= 1 or not 0 <= y_center <= 1:
        issues.append("CENTER_OUT_OF_RANGE")
    if box_width <= 0 or box_height <= 0:
        issues.append("NON_POSITIVE_SIZE")
    if box_width > 1 or box_height > 1:
        issues.append("SIZE_ABOVE_ONE")
    if any(overflow):
        issues.append("BBOX_OUT_OF_BOUNDS")
    return issues, xyxy, overflow


def process_crack_box(
    coordinates: tuple[float, ...],
    coordinate_tokens: tuple[str, str, str, str],
    width: int,
    height: int,
    image_path: Path,
    label_path: Path,
    line_number: int,
    crack_class_id: int,
) -> tuple[
    tuple[str, str, str, str] | None,
    tuple[float, float, float, float] | None,
    dict[str, Any] | None,
    dict[str, Any] | None,
    bool,
]:
    """Keep an in-bounds crack box, safely repair a minor overflow, or return a severe error."""
    x_center, y_center, box_width, box_height = coordinates
    original_xyxy, overflow = pixel_geometry(coordinates, width, height)
    overflow_left, overflow_top, overflow_right, overflow_bottom = overflow
    overflow_found = any(overflow)
    severe_reasons: list[str] = []
    if box_width <= 0 or box_height <= 0:
        severe_reasons.append("NON_POSITIVE_SIZE")

    if not overflow_found and not severe_reasons:
        return coordinate_tokens, (x_center, y_center, box_width, box_height), None, None, False

    horizontal_limit = width * REPAIR_THRESHOLD_RATIO
    vertical_limit = height * REPAIR_THRESHOLD_RATIO
    if max(overflow, default=0.0) > REPAIR_THRESHOLD_PX:
        severe_reasons.append("OVERFLOW_EXCEEDS_10_PIXELS")
    if overflow_left > horizontal_limit or overflow_right > horizontal_limit:
        severe_reasons.append("HORIZONTAL_OVERFLOW_EXCEEDS_0.5_PERCENT")
    if overflow_top > vertical_limit or overflow_bottom > vertical_limit:
        severe_reasons.append("VERTICAL_OVERFLOW_EXCEEDS_0.5_PERCENT")

    x1, y1, x2, y2 = original_xyxy
    repaired_xyxy = (
        max(0.0, min(x1, float(width))),
        max(0.0, min(y1, float(height))),
        max(0.0, min(x2, float(width))),
        max(0.0, min(y2, float(height))),
    )
    repaired_x1, repaired_y1, repaired_x2, repaired_y2 = repaired_xyxy
    if repaired_x2 - repaired_x1 <= 0 or repaired_y2 - repaired_y1 <= 0:
        severe_reasons.append("NON_POSITIVE_SIZE_AFTER_CLIP")

    common = {
        "source_image": str(image_path),
        "source_label": str(label_path),
        "line_number": line_number,
        "original_class_id": crack_class_id,
        "image_width": width,
        "image_height": height,
        "original_yolo_bbox": " ".join(coordinate_tokens),
        "original_pixel_xyxy": " ".join(f"{value:.12f}" for value in original_xyxy),
        "overflow_left_px": f"{overflow_left:.12f}",
        "overflow_top_px": f"{overflow_top:.12f}",
        "overflow_right_px": f"{overflow_right:.12f}",
        "overflow_bottom_px": f"{overflow_bottom:.12f}",
    }
    if severe_reasons:
        severe = {
            **common,
            "issue_type": ";".join(dict.fromkeys(severe_reasons)),
            "repair_threshold_px": REPAIR_THRESHOLD_PX,
            "repair_threshold_ratio": REPAIR_THRESHOLD_RATIO,
            "action": "stop_build_for_manual_review",
        }
        return None, None, None, severe, overflow_found

    repaired = (
        ((repaired_x1 + repaired_x2) / 2) / width,
        ((repaired_y1 + repaired_y2) / 2) / height,
        (repaired_x2 - repaired_x1) / width,
        (repaired_y2 - repaired_y1) / height,
    )
    repaired_tokens = tuple(f"{value:.10f}" for value in repaired)
    repaired_values = tuple(float(value) for value in repaired_tokens)
    repair = {
        **common,
        "output_class_id": 0,
        "repaired_pixel_xyxy": " ".join(f"{value:.12f}" for value in repaired_xyxy),
        "repaired_yolo_bbox": " ".join(repaired_tokens),
        "repair_reason": "minor_target_bbox_overflow_clipped_in_pixel_xyxy",
        "repair_threshold_px": REPAIR_THRESHOLD_PX,
        "repair_threshold_ratio": REPAIR_THRESHOLD_RATIO,
        "source_file_modified": "false",
    }
    return repaired_tokens, repaired_values, repair, None, True


def difference_hash(image: Image.Image) -> str:
    """Compute a 64-bit dHash for audit-only near-duplicate screening."""
    gray = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    pixels = list(gray.get_flattened_data())
    gray.close()
    bits = 0
    for row in range(8):
        for column in range(8):
            bits = (bits << 1) | int(pixels[row * 9 + column] > pixels[row * 9 + column + 1])
    return f"{bits:016x}"


def inspect_pair(
    pair: tuple[Path, Path], images_dir: Path, labels_dir: Path, class_ids: set[int], crack_class_id: int
) -> tuple[
    SourceSample | None,
    list[dict[str, Any]],
    str,
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, int],
]:
    """Decode one image, validate its full label, and retain only crack rows."""
    image_path, label_path = pair
    errors: list[dict[str, Any]] = []
    width = height = 0
    dhash = ""
    try:
        with Image.open(image_path) as image:
            image.load()
            width, height = image.size
            if width <= 0 or height <= 0:
                raise ValueError(f"invalid image dimensions {width}x{height}")
            dhash = difference_hash(image)
    except Exception as error:
        errors.append(
            {
                "kind": "damaged_image",
                "image": str(image_path),
                "label": str(label_path),
                "detail": f"{type(error).__name__}: {error}",
            }
        )

    crack_tokens: list[tuple[str, str, str, str]] = []
    crack_values: list[tuple[float, float, float, float]] = []
    repairs: list[dict[str, Any]] = []
    non_target_issues: list[dict[str, Any]] = []
    severe_errors: list[dict[str, Any]] = []
    stats = {"filtered_non_target_boxes": 0, "crack_overflow_boxes": 0, "repaired_crack_boxes": 0}
    try:
        text = label_path.read_text(encoding="utf-8-sig")
    except Exception as error:
        errors.append(
            {
                "kind": "invalid_label",
                "image": str(image_path),
                "label": str(label_path),
                "detail": f"cannot read label: {type(error).__name__}: {error}",
            }
        )
        text = ""
    for line_number, raw_line in enumerate(text.splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        tokens = line.split()
        class_id, coordinates, row_errors = parse_label_row(tokens, class_ids)
        if row_errors:
            errors.append(
                {
                    "kind": "invalid_label",
                    "image": str(image_path),
                    "label": str(label_path),
                    "line_number": line_number,
                    "line": line,
                    "detail": ";".join(row_errors),
                }
            )
            continue
        if class_id == crack_class_id and coordinates is not None:
            output_tokens, output_values, repair, severe, overflow_found = process_crack_box(
                coordinates,
                (tokens[1], tokens[2], tokens[3], tokens[4]),
                width,
                height,
                image_path,
                label_path,
                line_number,
                crack_class_id,
            )
            stats["crack_overflow_boxes"] += int(overflow_found)
            if severe is not None:
                severe_errors.append(severe)
            elif output_tokens is not None and output_values is not None:
                crack_tokens.append(output_tokens)
                crack_values.append(output_values)
                if repair is not None:
                    repairs.append(repair)
                    stats["repaired_crack_boxes"] += 1
        elif class_id is not None and coordinates is not None:
            stats["filtered_non_target_boxes"] += 1
            issues, _, overflow = non_target_coordinate_issues(coordinates, width, height)
            if issues:
                non_target_issues.append(
                    {
                        "source_image": str(image_path),
                        "source_label": str(label_path),
                        "line_number": line_number,
                        "original_class_id": class_id,
                        "original_yolo_bbox": " ".join(tokens[1:]),
                        "issue_type": ";".join(issues),
                        "overflow_left_px": f"{overflow[0]:.12f}",
                        "overflow_top_px": f"{overflow[1]:.12f}",
                        "overflow_right_px": f"{overflow[2]:.12f}",
                        "overflow_bottom_px": f"{overflow[3]:.12f}",
                        "action": "filtered_non_target_box",
                        "affects_dataset_build": "false",
                    }
                )
    if errors or severe_errors:
        return None, errors, "invalid", non_target_issues, severe_errors, stats
    if not crack_tokens:
        return None, [], "no_crack", non_target_issues, severe_errors, stats
    return (
        SourceSample(
            source_image=image_path,
            source_label=label_path,
            relative_image=normalized_relative(image_path, images_dir),
            relative_label=normalized_relative(label_path, labels_dir),
            width=width,
            height=height,
            crack_tokens=crack_tokens,
            crack_values=crack_values,
            image_bytes=image_path.stat().st_size,
            source_label_sha256=sha256_file(label_path),
            dhash=dhash,
            repairs=repairs,
        ),
        [],
        "crack",
        non_target_issues,
        severe_errors,
        stats,
    )


def inspect_all_sources(
    pairs: list[tuple[Path, Path]],
    images_dir: Path,
    labels_dir: Path,
    class_ids: set[int],
    crack_class_id: int,
    workers: int,
) -> tuple[
    list[SourceSample],
    list[dict[str, Any]],
    list[dict[str, str]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, int],
]:
    """Fully validate every paired source image and label."""
    samples: list[SourceSample] = []
    errors: list[dict[str, Any]] = []
    excluded: list[dict[str, str]] = []
    non_target_issues: list[dict[str, Any]] = []
    severe_errors: list[dict[str, Any]] = []
    totals: Counter[str] = Counter()

    def inspect(pair: tuple[Path, Path]) -> tuple[Any, ...]:
        return inspect_pair(pair, images_dir, labels_dir, class_ids, crack_class_id)

    print(f"[validate-source] Decoding and validating {len(pairs)} image/label pairs...", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for index, (pair, result) in enumerate(zip(pairs, executor.map(inspect, pairs)), 1):
            sample, pair_errors, disposition, pair_non_target_issues, pair_severe_errors, pair_stats = result
            errors.extend(pair_errors)
            non_target_issues.extend(pair_non_target_issues)
            severe_errors.extend(pair_severe_errors)
            totals.update(pair_stats)
            if sample is not None:
                samples.append(sample)
            elif disposition == "no_crack":
                image_path, label_path = pair
                excluded.append(
                    {
                        "source_image": str(image_path),
                        "source_label": str(label_path),
                        "reason": "no_crack",
                    }
                )
            if index % 500 == 0 or index == len(pairs):
                print(f"[validate-source] {index}/{len(pairs)}", flush=True)
    return samples, errors, excluded, non_target_issues, severe_errors, dict(totals)


def hash_samples(samples: list[SourceSample], workers: int) -> None:
    """Hash every selected original image in place."""
    print(f"[hash] Computing SHA-256 for {len(samples)} crack images...", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for index, (sample, digest) in enumerate(
            zip(samples, executor.map(lambda item: sha256_file(item.source_image), samples)), 1
        ):
            sample.image_sha256 = digest
            if index % 500 == 0 or index == len(samples):
                print(f"[hash] {index}/{len(samples)}", flush=True)


def snapshot_source_labels(labels: list[Path], labels_dir: Path, workers: int) -> list[dict[str, str]]:
    """Hash every source label so the independent validator can prove the source stayed unchanged."""
    print(f"[source-label-snapshot] Hashing {len(labels)} source labels...", flush=True)
    rows: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for path, digest in zip(labels, executor.map(sha256_file, labels)):
            rows.append(
                {
                    "source_label": str(path),
                    "source_label_relative": normalized_relative(path, labels_dir),
                    "sha256": digest,
                }
            )
    return rows


def source_label_snapshot_fingerprint(rows: list[dict[str, str]]) -> str:
    """Return a stable combined fingerprint for all source label files."""
    digest = hashlib.sha256()
    for row in sorted(rows, key=lambda item: item["source_label_relative"].casefold()):
        digest.update(f"{row['source_label_relative']}\0{row['sha256']}\n".encode("utf-8"))
    return digest.hexdigest()


def remove_exact_duplicates(
    samples: list[SourceSample],
) -> tuple[list[SourceSample], list[dict[str, str]], list[dict[str, str]]]:
    """Remove equivalent exact duplicates and report conflicting duplicate labels."""
    groups: dict[str, list[SourceSample]] = defaultdict(list)
    for sample in samples:
        groups[sample.image_sha256].append(sample)
    kept: list[SourceSample] = []
    duplicates: list[dict[str, str]] = []
    conflicts: list[dict[str, str]] = []
    for image_hash in sorted(groups):
        group = sorted(groups[image_hash], key=lambda item: item.relative_image.casefold())
        canonical = group[0]
        kept.append(canonical)
        for duplicate in group[1:]:
            if duplicate.equivalent_crack_label == canonical.equivalent_crack_label:
                duplicates.append(
                    {
                        "image_sha256": image_hash,
                        "canonical_source_image": str(canonical.source_image),
                        "canonical_source_label": str(canonical.source_label),
                        "excluded_source_image": str(duplicate.source_image),
                        "excluded_source_label": str(duplicate.source_label),
                        "reason": "identical_image_and_equivalent_crack_label",
                    }
                )
            else:
                conflicts.append(
                    {
                        "image_sha256": image_hash,
                        "canonical_source_image": str(canonical.source_image),
                        "canonical_source_label": str(canonical.source_label),
                        "conflicting_source_image": str(duplicate.source_image),
                        "conflicting_source_label": str(duplicate.source_label),
                        "canonical_crack_boxes": json.dumps(canonical.equivalent_crack_label),
                        "conflicting_crack_boxes": json.dumps(duplicate.equivalent_crack_label),
                    }
                )
    return sorted(kept, key=lambda item: item.relative_image.casefold()), duplicates, conflicts


def largest_remainder_counts(total: int) -> dict[str, int]:
    """Allocate an exact 70/20/10 split using deterministic largest remainders."""
    quotas = [(split, total * ratio) for split, ratio in SPLIT_RATIOS]
    counts = {split: math.floor(quota) for split, quota in quotas}
    remaining = total - sum(counts.values())
    order = sorted(
        range(len(quotas)), key=lambda index: (-(quotas[index][1] - math.floor(quotas[index][1])), index)
    )
    for index in order[:remaining]:
        counts[quotas[index][0]] += 1
    return counts


def assign_splits(samples: list[SourceSample], seed: int) -> dict[str, int]:
    """Sort, shuffle once with the fixed seed, and assign disjoint splits."""
    ordered = sorted(samples, key=lambda item: item.relative_image.casefold())
    random.Random(seed).shuffle(ordered)
    counts = largest_remainder_counts(len(ordered))
    cursor = 0
    for split, _ in SPLIT_RATIOS:
        for sample in ordered[cursor : cursor + counts[split]]:
            sample.split = split
        cursor += counts[split]
    for sample in samples:
        digest = hashlib.sha256(sample.relative_image.casefold().encode("utf-8")).hexdigest()
        sample.canonical_id = f"sample_{digest[:20]}"
        sample.output_image = f"images/{sample.split}/{sample.canonical_id}{sample.source_image.suffix.casefold()}"
        sample.output_label = f"labels/{sample.split}/{sample.canonical_id}.txt"
    return counts


def hamming_distance(left: str, right: str) -> int:
    """Return the Hamming distance between two hexadecimal hashes."""
    return (int(left, 16) ^ int(right, 16)).bit_count()


def near_duplicate_candidates(samples: list[SourceSample]) -> list[dict[str, Any]]:
    """Find audit-only cross-split dHash candidates without deleting any sample."""
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    segments = ((0, 11), (11, 22), (22, 33), (33, 44), (44, 54), (54, 64))
    for index, sample in enumerate(samples):
        value = int(sample.dhash, 16)
        for segment_index, (start, end) in enumerate(segments):
            width = end - start
            segment_value = (value >> (64 - end)) & ((1 << width) - 1)
            buckets[(segment_index, segment_value)].append(index)
    candidate_pairs: set[tuple[int, int]] = set()
    for indices in buckets.values():
        for left_position in range(len(indices)):
            for right_position in range(left_position + 1, len(indices)):
                left, right = indices[left_position], indices[right_position]
                if samples[left].split != samples[right].split:
                    candidate_pairs.add((min(left, right), max(left, right)))
    result: list[dict[str, Any]] = []
    for left_index, right_index in sorted(candidate_pairs):
        left, right = samples[left_index], samples[right_index]
        distance = hamming_distance(left.dhash, right.dhash)
        if distance <= NEAR_DUPLICATE_HAMMING_THRESHOLD and left.image_sha256 != right.image_sha256:
            result.append(
                {
                    "left_split": left.split,
                    "left_source_image": str(left.source_image),
                    "left_dhash": left.dhash,
                    "right_split": right.split,
                    "right_source_image": str(right.source_image),
                    "right_dhash": right.dhash,
                    "hamming_distance": distance,
                    "action": "audit_only_no_automatic_removal",
                }
            )
    return result


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    """Write a CSV with a header even when it has no data rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def label_text(sample: SourceSample) -> str:
    """Remap crack class IDs to zero while preserving all coordinate tokens."""
    return "".join(f"0 {' '.join(tokens)}\n" for tokens in sample.crack_tokens)


def copy_dataset_files(staging: Path, samples: list[SourceSample]) -> None:
    """Byte-copy original images and write filtered, remapped labels."""
    for split, _ in SPLIT_RATIOS:
        (staging / "images" / split).mkdir(parents=True, exist_ok=True)
        (staging / "labels" / split).mkdir(parents=True, exist_ok=True)
        (staging / "audit_samples" / split).mkdir(parents=True, exist_ok=True)
    print(f"[copy] Copying {len(samples)} original images without re-encoding...", flush=True)
    for index, sample in enumerate(samples, 1):
        destination_image = staging / sample.output_image
        destination_label = staging / sample.output_label
        shutil.copyfile(sample.source_image, destination_image)
        destination_label.write_text(label_text(sample), encoding="utf-8", newline="\n")
        if index % 500 == 0 or index == len(samples):
            print(f"[copy] {index}/{len(samples)}", flush=True)


def create_audit_samples(staging: Path, samples: list[SourceSample], seed: int) -> dict[str, int]:
    """Create six deterministic boxed visualization copies per split (or all if fewer)."""
    counts: dict[str, int] = {}
    for split, _ in SPLIT_RATIOS:
        candidates = sorted((sample for sample in samples if sample.split == split), key=lambda item: item.canonical_id)
        sample_count = min(6, len(candidates))
        derived_seed = int.from_bytes(hashlib.sha256(f"audit:{seed}:{split}".encode()).digest()[:8], "big")
        selected = random.Random(derived_seed).sample(candidates, sample_count)
        for sample in selected:
            with Image.open(staging / sample.output_image) as image:
                image.load()
                canvas = image.convert("RGB")
            scale = min(1.0, 1600 / max(canvas.size))
            if scale < 1:
                resized = canvas.resize(
                    (max(1, round(canvas.width * scale)), max(1, round(canvas.height * scale))),
                    Image.Resampling.LANCZOS,
                )
                canvas.close()
                canvas = resized
            draw = ImageDraw.Draw(canvas)
            line_width = max(2, round(max(canvas.size) / 500))
            for x_center, y_center, width, height in sample.crack_values:
                left = (x_center - width / 2) * canvas.width
                right = (x_center + width / 2) * canvas.width
                top = (y_center - height / 2) * canvas.height
                bottom = (y_center + height / 2) * canvas.height
                draw.rectangle((left, top, right, bottom), outline=(255, 32, 32), width=line_width)
            canvas.save(
                staging / "audit_samples" / split / f"{sample.canonical_id}_boxed.jpg",
                format="JPEG",
                quality=92,
            )
            canvas.close()
        counts[split] = sample_count
    return counts


def manifest_rows(samples: list[SourceSample], crack_class_id: int) -> list[dict[str, Any]]:
    """Create deterministic manifest rows."""
    return [
        {
            "canonical_id": sample.canonical_id,
            "split": sample.split,
            "source_image": str(sample.source_image),
            "source_label": str(sample.source_label),
            "source_image_relative": sample.relative_image,
            "source_label_relative": sample.relative_label,
            "output_image": sample.output_image,
            "output_label": sample.output_label,
            "original_class_id": crack_class_id,
            "output_class_id": 0,
            "image_width": sample.width,
            "image_height": sample.height,
            "crack_box_count": sample.crack_box_count,
            "repaired_crack_box_count": len(sample.repairs),
            "image_sha256": sample.image_sha256,
            "source_label_sha256": sample.source_label_sha256,
            "dhash": sample.dhash,
        }
        for sample in sorted(samples, key=lambda item: item.canonical_id)
    ]


MANIFEST_FIELDS = [
    "canonical_id",
    "split",
    "source_image",
    "source_label",
    "source_image_relative",
    "source_label_relative",
    "output_image",
    "output_label",
    "original_class_id",
    "output_class_id",
    "image_width",
    "image_height",
    "crack_box_count",
    "repaired_crack_box_count",
    "image_sha256",
    "source_label_sha256",
    "dhash",
]

BBOX_REPAIR_FIELDS = [
    "canonical_id",
    "split",
    "output_label",
    "source_image",
    "source_label",
    "line_number",
    "original_class_id",
    "output_class_id",
    "image_width",
    "image_height",
    "original_yolo_bbox",
    "original_pixel_xyxy",
    "overflow_left_px",
    "overflow_top_px",
    "overflow_right_px",
    "overflow_bottom_px",
    "repaired_pixel_xyxy",
    "repaired_yolo_bbox",
    "repair_reason",
    "repair_threshold_px",
    "repair_threshold_ratio",
    "source_file_modified",
]
NON_TARGET_ISSUE_FIELDS = [
    "source_image",
    "source_label",
    "line_number",
    "original_class_id",
    "original_yolo_bbox",
    "issue_type",
    "overflow_left_px",
    "overflow_top_px",
    "overflow_right_px",
    "overflow_bottom_px",
    "action",
    "affects_dataset_build",
]
SEVERE_CRACK_ERROR_FIELDS = [
    "source_image",
    "source_label",
    "line_number",
    "original_class_id",
    "image_width",
    "image_height",
    "original_yolo_bbox",
    "original_pixel_xyxy",
    "overflow_left_px",
    "overflow_top_px",
    "overflow_right_px",
    "overflow_bottom_px",
    "issue_type",
    "repair_threshold_px",
    "repair_threshold_ratio",
    "action",
]


def retained_repair_rows(samples: list[SourceSample]) -> list[dict[str, Any]]:
    """Attach output identity to repairs belonging to retained canonical samples."""
    rows: list[dict[str, Any]] = []
    for sample in samples:
        for repair in sample.repairs:
            rows.append(
                {
                    "canonical_id": sample.canonical_id,
                    "split": sample.split,
                    "output_label": sample.output_label,
                    **repair,
                }
            )
    return sorted(rows, key=lambda row: (row["source_label"].casefold(), int(row["line_number"])))


def dataset_fingerprint(rows: list[dict[str, Any]], root: Path) -> str:
    """Fingerprint split membership, source identity, image bytes, and output label bytes."""
    digest = hashlib.sha256()
    for row in sorted(rows, key=lambda item: item["canonical_id"]):
        fields = (
            row["canonical_id"],
            row["split"],
            row["source_image_relative"],
            row["output_image"],
            row["image_sha256"],
            sha256_file(root / row["output_label"]),
        )
        digest.update(("\0".join(str(field) for field in fields) + "\n").encode("utf-8"))
    return digest.hexdigest()


def write_yaml_files(root: Path, final_output: Path) -> None:
    """Write exact local and server dataset configurations."""
    common = "train: images/train\nval: images/val\ntest: images/test\nnames:\n  0: crack\n"
    local_path = final_output.resolve().as_posix()
    (root / "data_local.yaml").write_text(f"path: {local_path}\n{common}", encoding="utf-8", newline="\n")
    (root / "data.yaml").write_text(f"path: {SERVER_DATASET_PATH}\n{common}", encoding="utf-8", newline="\n")


def write_readme(
    root: Path,
    source: Path,
    crack_class_id: int,
    seed: int,
    fingerprint: str,
    bbox_audit: dict[str, Any],
) -> None:
    """Write dataset provenance and usage documentation."""
    text = f"""# Tunnel Crack Original NoAug 7:2:1 seed{seed}

- Data source: `{source}` (`images`, `labels`, and `data.yaml` only).
- Generated: {date.today().isoformat()}.
- Selection: retain original images with at least one crack annotation; remove all non-crack boxes.
- Split: original-image-level train/val/test = 70%/20%/10%, largest-remainder allocation after exact deduplication.
- Random seed: `{seed}`. Inputs are sorted by normalized relative image path before one deterministic shuffle.
- Class remap: source class `{crack_class_id}` (`裂缝`/`crack`) -> output class `0` (`crack`).
- Offline augmentation: **none**. Images are byte-for-byte copies; no re-encoding, synthesis, cropping, scaling, or color change.
- Bounding-box normalization is not augmentation. Crack boxes with per-side overflow <= {REPAIR_THRESHOLD_PX:g} px and
  <= {REPAIR_THRESHOLD_RATIO:.3%} of the relevant image dimension are clipped in pixel `xyxy` space and converted
  back to YOLO coordinates. Source labels are never modified.
- Non-target boxes with coordinate issues: {bbox_audit['non_target_bbox_issue_count']}; crack boxes with overflow:
  {bbox_audit['crack_overflow_bbox_count']}; safely repaired crack boxes: {bbox_audit['repaired_crack_bbox_count']};
  severe crack errors: {bbox_audit['severe_crack_bbox_error_count']}.
- Repaired source rows: {bbox_audit['repaired_source_rows']}.
- Online augmentation belongs to the later training stage and is not controlled by this dataset preparation script.
- The test split is final evaluation data and must not be used for tuning or model selection.
- Dataset fingerprint: `{fingerprint}`.

## Usage

For local checks or training, use `data_local.yaml`. After uploading the complete top-level dataset directory to the
server path recorded in `data.yaml`, use `data.yaml`.

`metadata/split_manifest.csv` is the authoritative split manifest. Reuse it rather than reshuffling for subsequent
runs. Each row records the absolute source image/label, output paths, source SHA-256, dimensions, and crack box count.
The preparation script refuses to overwrite an existing output. Use `--verify-only --output <dataset>` to rescan the
built dataset against this manifest.
"""
    (root / "README.md").write_text(text, encoding="utf-8", newline="\n")


def validate_output_label(path: Path) -> tuple[int, list[str], list[str]]:
    """Strictly validate one output label with at most 1e-9 normalized boundary tolerance."""
    errors: list[str] = []
    try:
        lines = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    except Exception as error:
        return 0, [], [f"cannot read: {type(error).__name__}: {error}"]
    if not lines:
        return 0, [], ["empty output label"]
    for line_number, line in enumerate(lines, 1):
        class_id, coordinates, row_errors = parse_label_row(line.split(), {0})
        if class_id != 0:
            row_errors.append("OUTPUT_CLASS_NOT_ZERO")
        if coordinates is not None:
            x_center, y_center, width, height = coordinates
            if not 0 <= x_center <= 1 or not 0 <= y_center <= 1:
                row_errors.append("CENTER_OUT_OF_RANGE")
            if not 0 < width <= 1 or not 0 < height <= 1:
                row_errors.append("SIZE_OUT_OF_RANGE")
            x1, y1 = x_center - width / 2, y_center - height / 2
            x2, y2 = x_center + width / 2, y_center + height / 2
            if (
                x1 < -OUTPUT_NORMALIZED_TOLERANCE
                or y1 < -OUTPUT_NORMALIZED_TOLERANCE
                or x2 > 1 + OUTPUT_NORMALIZED_TOLERANCE
                or y2 > 1 + OUTPUT_NORMALIZED_TOLERANCE
            ):
                row_errors.append("OUTPUT_BBOX_OUT_OF_BOUNDS")
        if row_errors:
            errors.append(f"line {line_number}: {','.join(dict.fromkeys(row_errors))}")
    return len(lines), lines, errors


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Read a UTF-8 CSV and fail clearly if it is missing."""
    if not path.is_file():
        raise BuildStop(f"required audit CSV is missing: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def reconstruct_expected_label(row: dict[str, str], class_ids: set[int]) -> tuple[list[str], list[dict[str, Any]]]:
    """Rebuild the expected single-class output directly from the unchanged source label."""
    source_image = Path(row["source_image"])
    source_label = Path(row["source_label"])
    width, height = int(row["image_width"]), int(row["image_height"])
    expected_lines: list[str] = []
    expected_repairs: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(source_label.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        tokens = line.split()
        class_id, coordinates, basic_errors = parse_label_row(tokens, class_ids)
        if basic_errors or class_id is None or coordinates is None:
            raise BuildStop(f"source label changed or became invalid: {source_label}:{line_number} {basic_errors}")
        if class_id != int(row["original_class_id"]):
            continue
        output_tokens, _, repair, severe, _ = process_crack_box(
            coordinates,
            (tokens[1], tokens[2], tokens[3], tokens[4]),
            width,
            height,
            source_image,
            source_label,
            line_number,
            class_id,
        )
        if severe is not None or output_tokens is None:
            raise BuildStop(f"retained source now has a severe crack box: {source_label}:{line_number} {severe}")
        expected_lines.append(f"0 {' '.join(output_tokens)}")
        if repair is not None:
            expected_repairs.append(
                {
                    "canonical_id": row["canonical_id"],
                    "split": row["split"],
                    "output_label": row["output_label"],
                    **repair,
                }
            )
    return expected_lines, expected_repairs


def independent_validate(root: Path, expected_local_root: Path | None = None) -> dict[str, Any]:
    """Rescan all final files independently using the persisted manifest."""
    expected_local_root = (expected_local_root or root).resolve()
    manifest_path = root / "metadata" / "split_manifest.csv"
    if not manifest_path.is_file():
        raise BuildStop(f"manifest is missing: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    missing_manifest_fields = set(MANIFEST_FIELDS) - set(rows[0] if rows else [])
    errors: list[str] = []
    if missing_manifest_fields:
        errors.append(f"manifest missing fields: {sorted(missing_manifest_fields)}")

    metadata = root / "metadata"
    try:
        repair_rows = read_csv_rows(metadata / "bbox_repairs.csv")
        non_target_issue_rows = read_csv_rows(metadata / "non_target_bbox_issues.csv")
        severe_rows = read_csv_rows(metadata / "severe_crack_bbox_errors.csv")
        source_label_hash_rows = read_csv_rows(metadata / "source_label_sha256.csv")
    except BuildStop as error:
        repair_rows, non_target_issue_rows, severe_rows, source_label_hash_rows = [], [], [], []
        errors.append(str(error))
    try:
        source_payload = yaml.safe_load((metadata / "source_data.yaml").read_text(encoding="utf-8-sig"))
        raw_names = source_payload["names"]
        class_ids = set(range(len(raw_names))) if isinstance(raw_names, list) else {int(key) for key in raw_names}
    except Exception as error:
        class_ids = set()
        errors.append(f"invalid metadata/source_data.yaml: {type(error).__name__}: {error}")

    split_results: dict[str, dict[str, int | float]] = {}
    filenames_by_split: dict[str, set[str]] = {}
    source_paths_by_split: dict[str, set[str]] = {}
    hashes_by_split: dict[str, set[str]] = {}
    manifest_by_image = {row["output_image"]: row for row in rows}
    all_output_images: set[str] = set()
    all_output_labels: set[str] = set()
    source_hash_mismatches = 0
    output_hash_mismatches = 0
    output_label_errors = 0
    output_image_errors = 0
    output_classes_only_zero = True
    marker_files: list[str] = []
    reconstructed_mismatches = 0
    expected_repairs: list[dict[str, Any]] = []

    print(f"[independent-validate] Rescanning {len(rows)} manifest samples...", flush=True)
    for split, _ in SPLIT_RATIOS:
        image_dir = root / "images" / split
        label_dir = root / "labels" / split
        image_files = find_files(image_dir, IMAGE_SUFFIXES) if image_dir.is_dir() else []
        label_files = find_files(label_dir, {".txt"}) if label_dir.is_dir() else []
        image_stems = [path.stem.casefold() for path in image_files]
        label_stems = [path.stem.casefold() for path in label_files]
        if len(image_stems) != len(set(image_stems)) or len(label_stems) != len(set(label_stems)):
            errors.append(f"{split}: duplicate output stems")
        if set(image_stems) != set(label_stems):
            errors.append(f"{split}: image/label stem mismatch")
        if len(image_files) != len(label_files):
            errors.append(f"{split}: images={len(image_files)} labels={len(label_files)}")
        filenames_by_split[split] = {path.name.casefold() for path in image_files}
        split_rows = [row for row in rows if row.get("split") == split]
        source_paths_by_split[split] = {row["source_image"].casefold() for row in split_rows}
        hashes_by_split[split] = set()
        split_box_count = 0
        for image_path in image_files:
            relative_image = normalized_relative(image_path, root)
            all_output_images.add(relative_image)
            if AUGMENTATION_MARKER.search(image_path.stem):
                marker_files.append(relative_image)
            row = manifest_by_image.get(relative_image)
            if row is None:
                errors.append(f"unmanifested image: {relative_image}")
                continue
            try:
                with Image.open(image_path) as image:
                    image.load()
                    dimensions = image.size
                if dimensions != (int(row["image_width"]), int(row["image_height"])):
                    errors.append(f"dimension mismatch: {relative_image} {dimensions}")
            except Exception as error:
                output_image_errors += 1
                errors.append(f"unreadable output image {relative_image}: {type(error).__name__}: {error}")
                continue
            output_hash = sha256_file(image_path)
            hashes_by_split[split].add(output_hash)
            if output_hash != row["image_sha256"]:
                output_hash_mismatches += 1
                errors.append(f"output hash mismatch: {relative_image}")
            source_path = Path(row["source_image"])
            if not source_path.is_file() or sha256_file(source_path) != row["image_sha256"]:
                source_hash_mismatches += 1
                errors.append(f"source hash mismatch or missing: {source_path}")
            output_label = root / row["output_label"]
            box_count, output_lines, label_errors = validate_output_label(output_label)
            split_box_count += box_count
            all_output_labels.add(row["output_label"])
            if label_errors:
                output_label_errors += 1
                output_classes_only_zero = False
                errors.append(f"invalid output label {row['output_label']}: {'; '.join(label_errors)}")
            if box_count != int(row["crack_box_count"]):
                errors.append(
                    f"box count mismatch {row['output_label']}: manifest={row['crack_box_count']} disk={box_count}"
                )
            try:
                expected_lines, row_expected_repairs = reconstruct_expected_label(row, class_ids)
                expected_repairs.extend(row_expected_repairs)
                if output_lines != expected_lines:
                    reconstructed_mismatches += 1
                    errors.append(f"output label differs from reconstructed source policy: {row['output_label']}")
            except Exception as error:
                reconstructed_mismatches += 1
                errors.append(f"cannot reconstruct {row['output_label']}: {type(error).__name__}: {error}")
        split_results[split] = {
            "images": len(image_files),
            "labels": len(label_files),
            "crack_boxes": split_box_count,
            "ratio": len(image_files) / len(rows) if rows else 0.0,
        }

    manifest_images = {row["output_image"] for row in rows}
    manifest_labels = {row["output_label"] for row in rows}
    if all_output_images != manifest_images:
        errors.append(f"image/manifest mismatch: delta={len(all_output_images ^ manifest_images)}")
    if all_output_labels != manifest_labels:
        errors.append(f"label/manifest mismatch: delta={len(all_output_labels ^ manifest_labels)}")
    for left_index, (left, _) in enumerate(SPLIT_RATIOS):
        for right, _ in SPLIT_RATIOS[left_index + 1 :]:
            if overlap := filenames_by_split[left] & filenames_by_split[right]:
                errors.append(f"filename overlap {left}/{right}: {len(overlap)}")
            if overlap := source_paths_by_split[left] & source_paths_by_split[right]:
                errors.append(f"source path overlap {left}/{right}: {len(overlap)}")
            if overlap := hashes_by_split[left] & hashes_by_split[right]:
                errors.append(f"SHA-256 overlap {left}/{right}: {len(overlap)}")
    if marker_files:
        errors.append(f"augmentation markers in output images: {marker_files[:20]}")

    def repair_key(item: dict[str, Any]) -> tuple[str, int, str]:
        return (str(item.get("source_label", "")).casefold(), int(item.get("line_number", 0)), str(item.get("repaired_yolo_bbox", "")))

    recorded_repair_keys = Counter(repair_key(row) for row in repair_rows)
    expected_repair_keys = Counter(repair_key(row) for row in expected_repairs)
    manifest_repair_count = sum(int(row.get("repaired_crack_box_count", 0)) for row in rows)
    if recorded_repair_keys != expected_repair_keys:
        errors.append(
            f"bbox repair audit differs from reconstructed repairs: recorded={sum(recorded_repair_keys.values())} "
            f"expected={sum(expected_repair_keys.values())}"
        )
    if len(repair_rows) != manifest_repair_count:
        errors.append(f"bbox repair count mismatch: csv={len(repair_rows)} manifest={manifest_repair_count}")
    if severe_rows:
        errors.append(f"severe crack error CSV is not empty: {len(severe_rows)}")

    source_label_hash_mismatches = 0
    for snapshot_row in source_label_hash_rows:
        source_label = Path(snapshot_row["source_label"])
        if not source_label.is_file() or sha256_file(source_label) != snapshot_row["sha256"]:
            source_label_hash_mismatches += 1
            errors.append(f"source label changed or missing: {source_label}")
            if source_label_hash_mismatches >= 20:
                break

    t798_rows = [
        row
        for row in non_target_issue_rows
        if Path(row.get("source_label", "")).name.casefold() == "t798_620_015304743.txt"
        and row.get("original_class_id") == "4"
    ]
    t798_manifest_rows = [row for row in rows if Path(row["source_label"]).name.casefold() == "t798_620_015304743.txt"]
    t812_manifest_rows = [row for row in rows if Path(row["source_label"]).name.casefold() == "t812_812_011906777.txt"]
    t812_repair_rows = [
        row for row in repair_rows if Path(row.get("source_label", "")).name.casefold() == "t812_812_011906777.txt"
    ]
    if len(t798_rows) != 1 or t798_rows[0].get("action") != "filtered_non_target_box":
        errors.append(f"t798 class-4 audit mismatch: rows={len(t798_rows)}")
    if t798_manifest_rows:
        errors.append("t798 unexpectedly retained despite having no crack boxes")
    if len(t812_manifest_rows) != 1 or len(t812_repair_rows) != 1:
        errors.append(
            f"t812 repaired crack retention mismatch: manifest={len(t812_manifest_rows)} repairs={len(t812_repair_rows)}"
        )

    try:
        local_yaml = yaml.safe_load((root / "data_local.yaml").read_text(encoding="utf-8-sig"))
        if Path(local_yaml["path"]).resolve() != expected_local_root:
            errors.append(f"data_local.yaml path mismatch: {local_yaml.get('path')}")
        for split, _ in SPLIT_RATIOS:
            if not (root / local_yaml[split]).is_dir():
                errors.append(f"data_local.yaml missing {split} path: {local_yaml.get(split)}")
        if local_yaml.get("names") != {0: "crack"}:
            errors.append(f"data_local.yaml names mismatch: {local_yaml.get('names')}")
    except Exception as error:
        errors.append(f"invalid data_local.yaml: {type(error).__name__}: {error}")
    try:
        server_yaml = yaml.safe_load((root / "data.yaml").read_text(encoding="utf-8-sig"))
        if server_yaml.get("path") != SERVER_DATASET_PATH:
            errors.append(f"data.yaml server path mismatch: {server_yaml.get('path')}")
        if server_yaml.get("names") != {0: "crack"}:
            errors.append(f"data.yaml names mismatch: {server_yaml.get('names')}")
    except Exception as error:
        errors.append(f"invalid data.yaml: {type(error).__name__}: {error}")

    old_source_references = [row["source_image"] for row in rows if "tunnel_crack_augfirst_5x" in row["source_image"].casefold()]
    if old_source_references:
        errors.append(f"old AugFirst source references found: {len(old_source_references)}")

    checks = {
        "image_label_counts_equal": all(
            values["images"] == values["labels"] for values in split_results.values()
        ),
        "one_label_per_image": all_output_images == manifest_images and all_output_labels == manifest_labels,
        "every_label_has_crack": output_label_errors == 0,
        "all_output_classes_zero": output_classes_only_zero,
        "all_output_boxes_valid": output_label_errors == 0,
        "all_output_labels_match_source_filter_and_repair_policy": reconstructed_mismatches == 0,
        "bbox_repairs_fully_audited": recorded_repair_keys == expected_repair_keys and len(repair_rows) == manifest_repair_count,
        "no_severe_crack_bbox_errors": not severe_rows,
        "non_target_bbox_issues_do_not_affect_build": all(
            row.get("action") == "filtered_non_target_box" and row.get("affects_dataset_build", "").casefold() == "false"
            for row in non_target_issue_rows
        ),
        "source_labels_unchanged": source_label_hash_mismatches == 0,
        "t798_class4_filtered_and_no_crack_image_excluded": len(t798_rows) == 1 and not t798_manifest_rows,
        "t812_crack_repaired_and_retained": len(t812_manifest_rows) == 1 and len(t812_repair_rows) == 1,
        "filenames_disjoint": not any("filename overlap" in error for error in errors),
        "source_paths_disjoint": not any("source path overlap" in error for error in errors),
        "sha256_disjoint": not any("SHA-256 overlap" in error for error in errors),
        "output_images_match_source_sha256": source_hash_mismatches == 0 and output_hash_mismatches == 0,
        "no_augmentation_markers": not marker_files,
        "data_local_paths_exist": not any("data_local.yaml" in error for error in errors),
        "data_yaml_server_path_correct": not any("data.yaml" in error for error in errors),
        "no_old_augfirst_source_references": not old_source_references,
        "all_output_images_decodable": output_image_errors == 0,
    }
    passed = not errors and all(checks.values())
    return {
        "passed": passed,
        "checks": checks,
        "errors": errors,
        "split_results": split_results,
        "source_hash_mismatches": source_hash_mismatches,
        "output_hash_mismatches": output_hash_mismatches,
        "output_image_errors": output_image_errors,
        "output_label_errors": output_label_errors,
        "reconstructed_label_mismatches": reconstructed_mismatches,
        "bbox_repair_csv_rows": len(repair_rows),
        "expected_bbox_repairs": len(expected_repairs),
        "bbox_repair_details": repair_rows,
        "non_target_bbox_issue_rows": len(non_target_issue_rows),
        "non_target_bbox_issue_details": non_target_issue_rows,
        "severe_crack_bbox_error_rows": len(severe_rows),
        "source_label_hash_mismatches": source_label_hash_mismatches,
        "manifest_rows": len(rows),
    }


def write_validation_report(path: Path, validation: dict[str, Any]) -> None:
    """Write a plain-text independent validation report."""
    lines = [
        f"generated_at={datetime.now().isoformat(timespec='seconds')}",
        f"DATASET_BUILD_STATUS={'PASS' if validation['passed'] else 'FAIL'}",
        f"manifest_rows={validation['manifest_rows']}",
    ]
    lines.extend(f"{name}={'PASS' if passed else 'FAIL'}" for name, passed in validation["checks"].items())
    lines.append("split_results=" + json.dumps(validation["split_results"], ensure_ascii=False, sort_keys=True))
    lines.append(f"repair_threshold_px={REPAIR_THRESHOLD_PX}")
    lines.append(f"repair_threshold_ratio={REPAIR_THRESHOLD_RATIO}")
    lines.append(f"non_target_bbox_issue_count={validation.get('non_target_bbox_issue_rows', 0)}")
    lines.append(f"crack_bbox_repair_count={validation.get('bbox_repair_csv_rows', 0)}")
    lines.append(f"severe_crack_bbox_error_count={validation.get('severe_crack_bbox_error_rows', 0)}")
    lines.append("source_files_modified=false")
    lines.append("bbox_normalization_is_offline_augmentation=false")
    lines.append("bbox_repair_details=" + json.dumps(validation.get("bbox_repair_details", []), ensure_ascii=False))
    lines.append(
        "non_target_bbox_issue_details="
        + json.dumps(validation.get("non_target_bbox_issue_details", []), ensure_ascii=False)
    )
    lines.append(f"errors={len(validation['errors'])}")
    lines.extend(f"ERROR: {error}" for error in validation["errors"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def write_summary_with_stable_size(path: Path, summary: dict[str, Any], dataset_root: Path) -> None:
    """Write the summary until its recorded directory size matches the completed directory."""
    for _ in range(10):
        path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
        actual_size = directory_size(dataset_root)
        if summary.get("dataset_total_bytes") == actual_size:
            return
        summary["dataset_total_bytes"] = actual_size
        summary["dataset_total_human"] = human_bytes(actual_size)
    raise BuildStop("dataset size summary did not stabilize")


def package_and_validate_dataset(dataset_root: Path, zip_path: Path) -> dict[str, Any]:
    """Create an atomic standalone ZIP and independently validate its layout and counts."""
    zip_path = zip_path.resolve()
    checksum_path = zip_path.with_suffix(zip_path.suffix + ".sha256")
    if zip_path.exists() or checksum_path.exists():
        raise BuildStop(f"ZIP output or checksum already exists; refusing to overwrite: {zip_path}")
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_bytes = directory_size(dataset_root)
    free_bytes = shutil.disk_usage(zip_path.parent).free
    if free_bytes < dataset_bytes + 512 * 1024**2:
        raise BuildStop(
            f"insufficient space for ZIP: free={human_bytes(free_bytes)} required={human_bytes(dataset_bytes + 512 * 1024**2)}"
        )
    with tempfile.NamedTemporaryFile(
        prefix=f".{zip_path.name}.", suffix=".tmp", dir=zip_path.parent, delete=False
    ) as temporary:
        temporary_zip = Path(temporary.name)
    files = sorted(
        (path for path in dataset_root.rglob("*") if path.is_file()),
        key=lambda path: normalized_relative(path, dataset_root).casefold(),
    )
    try:
        print(f"[zip] Packaging {len(files)} files...", flush=True)
        with zipfile.ZipFile(temporary_zip, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
            for index, path in enumerate(files, 1):
                archive.write(path, (Path(dataset_root.name) / path.relative_to(dataset_root)).as_posix())
                if index % 500 == 0 or index == len(files):
                    print(f"[zip] {index}/{len(files)}", flush=True)
        with zipfile.ZipFile(temporary_zip, "r") as archive:
            bad_member = archive.testzip()
            names = [name for name in archive.namelist() if not name.endswith("/")]
            top_levels = {Path(name).parts[0] for name in names}
            if bad_member is not None:
                raise BuildStop(f"ZIP CRC validation failed: {bad_member}")
            if top_levels != {dataset_root.name}:
                raise BuildStop(f"ZIP top-level layout is invalid: {sorted(top_levels)}")
            forbidden = [
                name
                for name in names
                if any(
                    marker in name.casefold()
                    for marker in (".building-", "__pycache__", "tunnel_crack_augfirst_5x", "_audit_crack_augfirst")
                )
            ]
            if forbidden:
                raise BuildStop(f"ZIP contains forbidden files: {forbidden[:20]}")
            required = {
                f"{dataset_root.name}/data.yaml",
                f"{dataset_root.name}/data_local.yaml",
                f"{dataset_root.name}/README.md",
                f"{dataset_root.name}/metadata/split_manifest.csv",
                f"{dataset_root.name}/metadata/bbox_repairs.csv",
                f"{dataset_root.name}/metadata/non_target_bbox_issues.csv",
            }
            missing = required - set(names)
            if missing:
                raise BuildStop(f"ZIP is missing required files: {sorted(missing)}")
            zip_counts: dict[str, dict[str, int]] = {}
            for split, _ in SPLIT_RATIOS:
                image_prefix = f"{dataset_root.name}/images/{split}/"
                label_prefix = f"{dataset_root.name}/labels/{split}/"
                zip_images = sum(
                    name.startswith(image_prefix) and Path(name).suffix.casefold() in IMAGE_SUFFIXES for name in names
                )
                zip_labels = sum(name.startswith(label_prefix) and Path(name).suffix.casefold() == ".txt" for name in names)
                disk_images = len(find_files(dataset_root / "images" / split, IMAGE_SUFFIXES))
                disk_labels = len(find_files(dataset_root / "labels" / split, {".txt"}))
                if (zip_images, zip_labels) != (disk_images, disk_labels):
                    raise BuildStop(
                        f"ZIP count mismatch for {split}: zip={zip_images}/{zip_labels} disk={disk_images}/{disk_labels}"
                    )
                zip_counts[split] = {"images": zip_images, "labels": zip_labels}
            manifest_text = archive.read(f"{dataset_root.name}/metadata/split_manifest.csv").decode("utf-8-sig")
            manifest_count = sum(1 for _ in csv.DictReader(io.StringIO(manifest_text)))
            if manifest_count != sum(values["images"] for values in zip_counts.values()):
                raise BuildStop(
                    f"ZIP manifest count mismatch: manifest={manifest_count} images={sum(v['images'] for v in zip_counts.values())}"
                )
        temporary_zip.rename(zip_path)
        zip_sha256 = sha256_file(zip_path)
        checksum_path.write_text(f"{zip_sha256}  {zip_path.name}\n", encoding="ascii", newline="\n")
        return {
            "zip_path": str(zip_path),
            "zip_size_bytes": zip_path.stat().st_size,
            "zip_size_human": human_bytes(zip_path.stat().st_size),
            "zip_sha256": zip_sha256,
            "zip_checksum_path": str(checksum_path),
            "zip_top_level": dataset_root.name,
            "zip_file_count": len(files),
            "zip_split_counts": zip_counts,
            "zip_manifest_rows": manifest_count,
            "zip_validation_passed": True,
        }
    finally:
        if temporary_zip.exists():
            temporary_zip.unlink()


def write_failure_report(output: Path, reason: str, details: Any | None = None) -> Path:
    """Persist a failure report outside the reserved final dataset path."""
    report_root = output.parent / "_build_failures" / output.name
    report_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report = report_root / f"failure_{stamp}.json"
    report.write_text(
        json.dumps({"status": "FAIL", "reason": reason, "details": details}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def enable_windows_acl_inheritance(path: Path) -> None:
    """Ensure a Windows staging directory remains accessible after it is atomically published."""
    if os.name != "nt":
        return
    result = subprocess.run(
        ["icacls", str(path), "/inheritance:e"],
        check=False,
        capture_output=True,
        text=True,
        errors="replace",
    )
    if result.returncode:
        raise BuildStop(f"failed to enable inherited Windows permissions for {path}: {result.stderr.strip()}")


def build_dataset(args: argparse.Namespace) -> dict[str, Any]:
    """Validate sources, build in a temporary directory, validate, then atomically publish."""
    source = args.source.resolve()
    output = args.output.resolve()
    if args.seed < 0:
        raise BuildStop("seed must be nonnegative")
    if args.workers < 1:
        raise BuildStop("workers must be at least 1")
    if output.exists():
        raise BuildStop(f"final output already exists; refusing to overwrite: {output}")
    zip_output = args.zip_output.resolve()
    if zip_output.exists() or zip_output.with_suffix(zip_output.suffix + ".sha256").exists():
        raise BuildStop(f"ZIP output or checksum already exists; refusing to overwrite: {zip_output}")
    images_dir, labels_dir, source_yaml = source / "images", source / "labels", source / "data.yaml"
    missing = [str(path) for path in (images_dir, labels_dir, source_yaml) if not path.exists()]
    if missing:
        raise BuildStop(f"required source paths are missing: {missing}")

    names, crack_class_id, resolved_crack_name = load_class_mapping(source_yaml, args.crack_class_name)
    images, labels, pairs = discover_pairs(images_dir, labels_dir)
    marker_images = [str(path) for path in images if AUGMENTATION_MARKER.search(path.stem)]
    if marker_images:
        raise BuildStop(
            f"source filenames contain possible offline augmentation markers ({len(marker_images)}):\n"
            + "\n".join(marker_images[:200])
        )
    source_total_bytes = directory_size(source)
    disk_free_before = shutil.disk_usage(output.parent).free
    print(f"[preflight] source={source}")
    print(f"[preflight] output={output}")
    print(f"[preflight] images={len(images)} labels={len(labels)} source_size={human_bytes(source_total_bytes)}")
    print(f"[preflight] free_space={human_bytes(disk_free_before)} names={names}")
    print(f"[preflight] crack_class={crack_class_id}:{resolved_crack_name}")

    source_label_hash_rows = snapshot_source_labels(labels, labels_dir, args.workers)
    source_labels_fingerprint = source_label_snapshot_fingerprint(source_label_hash_rows)
    samples, source_errors, excluded, non_target_issues, severe_errors, source_stats = inspect_all_sources(
        pairs, images_dir, labels_dir, set(names), crack_class_id, args.workers
    )
    if source_errors:
        error_counts = Counter(error["kind"] for error in source_errors)
        report = write_failure_report(output, "source validation failed", source_errors)
        raise BuildStop(f"source validation failed: {dict(error_counts)}; detailed report: {report}")
    if severe_errors:
        report_root = output.parent / "_build_failures" / output.name
        severe_csv = report_root / "severe_crack_bbox_errors.csv"
        write_csv(severe_csv, severe_errors, SEVERE_CRACK_ERROR_FIELDS)
        report = write_failure_report(output, "severe crack bounding-box errors", severe_errors)
        raise BuildStop(f"severe crack bbox errors={len(severe_errors)}; reports: {severe_csv}, {report}")
    hash_samples(samples, args.workers)
    retained, exact_duplicates, duplicate_conflicts = remove_exact_duplicates(samples)
    if duplicate_conflicts:
        report_root = output.parent / "_build_failures" / output.name
        conflict_csv = report_root / "duplicate_label_conflicts.csv"
        write_csv(conflict_csv, duplicate_conflicts, list(duplicate_conflicts[0]))
        report = write_failure_report(output, "identical images have conflicting crack labels", duplicate_conflicts)
        raise BuildStop(f"duplicate label conflicts={len(duplicate_conflicts)}; reports: {conflict_csv}, {report}")
    if not retained:
        raise BuildStop("no valid crack samples remain after filtering and exact deduplication")

    split_counts = assign_splits(retained, args.seed)
    near_duplicates = near_duplicate_candidates(retained)
    repair_rows = retained_repair_rows(retained)
    bbox_audit = {
        "filtered_non_target_box_count": source_stats.get("filtered_non_target_boxes", 0),
        "non_target_bbox_issue_count": len(non_target_issues),
        "crack_overflow_bbox_count": source_stats.get("crack_overflow_boxes", 0),
        "repaired_crack_bbox_count": len(repair_rows),
        "severe_crack_bbox_error_count": 0,
        "repair_threshold_px": REPAIR_THRESHOLD_PX,
        "repair_threshold_ratio": REPAIR_THRESHOLD_RATIO,
        "repaired_source_rows": [
            {
                "source_label": row["source_label"],
                "line_number": row["line_number"],
                "original_yolo_bbox": row["original_yolo_bbox"],
                "repaired_yolo_bbox": row["repaired_yolo_bbox"],
            }
            for row in repair_rows
        ],
        "source_files_modified": False,
        "repair_is_offline_augmentation": False,
    }
    selected_bytes = sum(sample.image_bytes for sample in retained)
    estimated_required = math.ceil(selected_bytes * 1.10) + 512 * 1024**2
    disk_free_after_scan = shutil.disk_usage(output.parent).free
    if disk_free_after_scan < estimated_required:
        raise BuildStop(
            f"insufficient disk space: free={human_bytes(disk_free_after_scan)}, "
            f"estimated dataset requirement={human_bytes(estimated_required)}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.building-", dir=output.parent))
    enable_windows_acl_inheritance(staging)
    published = False
    try:
        copy_dataset_files(staging, retained)
        metadata = staging / "metadata"
        metadata.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_yaml, metadata / "source_data.yaml")
        rows = manifest_rows(retained, crack_class_id)
        write_csv(metadata / "split_manifest.csv", rows, MANIFEST_FIELDS)
        write_csv(
            metadata / "source_label_sha256.csv",
            source_label_hash_rows,
            ["source_label", "source_label_relative", "sha256"],
        )
        write_csv(
            metadata / "excluded_non_crack.csv",
            excluded,
            ["source_image", "source_label", "reason"],
        )
        write_csv(
            metadata / "exact_duplicates.csv",
            exact_duplicates,
            [
                "image_sha256",
                "canonical_source_image",
                "canonical_source_label",
                "excluded_source_image",
                "excluded_source_label",
                "reason",
            ],
        )
        write_csv(
            metadata / "near_duplicate_candidates.csv",
            near_duplicates,
            [
                "left_split",
                "left_source_image",
                "left_dhash",
                "right_split",
                "right_source_image",
                "right_dhash",
                "hamming_distance",
                "action",
            ],
        )
        write_csv(metadata / "bbox_repairs.csv", repair_rows, BBOX_REPAIR_FIELDS)
        write_csv(metadata / "non_target_bbox_issues.csv", non_target_issues, NON_TARGET_ISSUE_FIELDS)
        write_csv(metadata / "severe_crack_bbox_errors.csv", [], SEVERE_CRACK_ERROR_FIELDS)
        write_yaml_files(staging, output)
        fingerprint = dataset_fingerprint(rows, staging)
        (metadata / "dataset_fingerprint.sha256").write_text(
            f"{fingerprint}  {output.name}\n", encoding="ascii", newline="\n"
        )
        write_readme(staging, source, crack_class_id, args.seed, fingerprint, bbox_audit)
        audit_counts = create_audit_samples(staging, retained, args.seed)

        prepublish_validation = independent_validate(staging, output)
        if not prepublish_validation["passed"]:
            write_validation_report(metadata / "validation_report.txt", prepublish_validation)
            raise BuildStop(f"pre-publish independent validation failed: {prepublish_validation['errors'][:20]}")

        split_box_counts = {
            split: sum(sample.crack_box_count for sample in retained if sample.split == split)
            for split, _ in SPLIT_RATIOS
        }
        summary = {
            "dataset_build_status": "PASS",
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "source": str(source),
            "output": str(output),
            "seed": args.seed,
            "split_ratios_requested": {split: ratio for split, ratio in SPLIT_RATIOS},
            "source_class_mapping": names,
            "crack_original_class_id": crack_class_id,
            "crack_original_class_name": resolved_crack_name,
            "output_class_mapping": {0: "crack"},
            "raw_image_count": len(images),
            "raw_label_count": len(labels),
            "raw_source_total_bytes": source_total_bytes,
            "raw_source_total_human": human_bytes(source_total_bytes),
            "disk_free_before_bytes": disk_free_before,
            "crack_image_count_before_dedup": len(samples),
            "crack_box_count_before_dedup": sum(sample.crack_box_count for sample in samples),
            "excluded_no_crack_count": len(excluded),
            "exact_duplicate_image_count": len(exact_duplicates),
            "duplicate_label_conflict_count": 0,
            "damaged_image_count": 0,
            "missing_label_count": 0,
            "invalid_label_file_count": 0,
            "retained_image_count": len(retained),
            "retained_crack_box_count": sum(sample.crack_box_count for sample in retained),
            "filtered_non_target_box_count": bbox_audit["filtered_non_target_box_count"],
            "non_target_bbox_issue_count": bbox_audit["non_target_bbox_issue_count"],
            "crack_overflow_bbox_count": bbox_audit["crack_overflow_bbox_count"],
            "repaired_crack_bbox_count": bbox_audit["repaired_crack_bbox_count"],
            "severe_crack_bbox_error_count": 0,
            "bbox_repair_policy": bbox_audit,
            "source_labels_fingerprint_before_and_after": source_labels_fingerprint,
            "source_label_hash_file_count": len(source_label_hash_rows),
            "source_labels_modified": False,
            "splits": {
                split: {
                    "images": split_counts[split],
                    "ratio": split_counts[split] / len(retained),
                    "crack_boxes": split_box_counts[split],
                }
                for split, _ in SPLIT_RATIOS
            },
            "image_label_one_to_one": True,
            "cross_split_filename_source_sha256_disjoint": True,
            "all_output_classes_zero": True,
            "output_images_byte_identical_to_sources": True,
            "near_duplicate_candidate_count": len(near_duplicates),
            "near_duplicate_method": f"64-bit dHash, cross-split Hamming distance <= {NEAR_DUPLICATE_HAMMING_THRESHOLD}",
            "audit_sample_counts": audit_counts,
            "dataset_fingerprint": fingerprint,
            "dataset_total_bytes": 0,
            "dataset_total_human": "pending final metadata size",
            "offline_augmentation": False,
            "validation": prepublish_validation,
        }
        write_validation_report(metadata / "validation_report.txt", prepublish_validation)
        write_summary_with_stable_size(metadata / "split_summary.json", summary, staging)

        if output.exists():
            raise BuildStop(f"output appeared during build; refusing to overwrite: {output}")
        staging.rename(output)
        published = True
        final_validation = independent_validate(output)
        write_validation_report(output / "metadata" / "validation_report.txt", final_validation)
        if not final_validation["passed"]:
            quarantine = output.parent / f".{output.name}.failed-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            output.rename(quarantine)
            published = False
            raise BuildStop(f"post-publish validation failed; moved incomplete output to {quarantine}")
        summary["validation"] = final_validation
        write_summary_with_stable_size(output / "metadata" / "split_summary.json", summary, output)
        return summary
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)


def main() -> int:
    """Build a new dataset or independently verify an existing one."""
    args = parse_args()
    try:
        if args.verify_only:
            validation = independent_validate(args.output.resolve())
            print(json.dumps(validation, ensure_ascii=False, indent=2))
            print(f"DATASET_BUILD_STATUS={'PASS' if validation['passed'] else 'FAIL'}")
            return 0 if validation["passed"] else 1
        summary = build_dataset(args)
        zip_result = package_and_validate_dataset(args.output.resolve(), args.zip_output.resolve())
        print(json.dumps({"dataset": summary, "package": zip_result}, ensure_ascii=False, indent=2))
        print("DATASET_BUILD_STATUS=PASS")
        return 0
    except BuildStop as error:
        print(f"ERROR: {error}", file=sys.stderr)
        print("DATASET_BUILD_STATUS=FAIL")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
