# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Independently reconstruct and audit a tiled YOLO dataset from its unchanged source dataset."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import json
import math
import platform
import random
import re
import sys
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any

import PIL
from PIL import Image, ImageChops

SPLITS = ("train", "val", "test")
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
FLOAT_TOLERANCE = 1e-9
LABEL_TOLERANCE = 5e-10
SCHEMA_VERSION = 2
OUTPUT_FINGERPRINT_ALGORITHM = "sha256-canonical-files-v2"
SOURCE_FINGERPRINT_ALGORITHM = "sha256-canonical-source-v2"
# Canonical summary hashing excludes only generated_at and output_dataset_fingerprint.
SUMMARY_FINGERPRINT_EXCLUSIONS = (
    "generated_at",
    "output_dataset_fingerprint",
)
MAX_TILE_FILENAME_CHARS = 180
MAX_TILE_STEM_CHARS = 64
SOURCE_PATH_HASH_CHARS = 16
MANIFEST_FIELDS = (
    "split",
    "tile_file",
    "source_image",
    "source_image_hash",
    "source_image_sha256",
    "source_label",
    "source_label_sha256",
    "tile_x",
    "tile_y",
    "tile_w",
    "tile_h",
    "valid_width",
    "valid_height",
    "source_width",
    "source_height",
    "padding",
    "padding_left",
    "padding_top",
    "padding_right",
    "padding_bottom",
    "category",
    "original_intersecting_box_count",
    "retained_box_count",
    "ambiguous_box_count",
    "source_bbox_id",
    "visibility",
    "rejected_reason",
    "source_boxes",
    "retained_box_records",
    "failed_boxes",
    "output_image_sha256",
    "output_label_sha256",
)


class AuditError(RuntimeError):
    """Raised when an audit input is malformed or cannot be independently reconstructed."""


@dataclass(frozen=True)
class AuditPair:
    """One independently matched source image and label."""

    split: str
    image_path: Path
    label_path: Path
    image_relative: str
    label_relative: str


@dataclass(frozen=True)
class AuditIdentity:
    """Exact source bytes independently observed by the auditor."""

    split: str
    image_relative: str
    image_sha256: str
    label_relative: str
    label_sha256: str


@dataclass(frozen=True)
class AuditBox:
    """One independently parsed source bounding box."""

    source_bbox_id: str
    xyxy: tuple[float, float, float, float]


def stable_json(value: Any) -> str:
    """Serialize UTF-8 JSON with sorted mappings, preserved list order, and finite floats."""
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def sha256_file(path: Path) -> str:
    """Return a SHA-256 digest without modifying a file."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    """Read one required JSON object with a uniform error."""
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise AuditError(f"cannot read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise AuditError(f"expected a JSON object in {path}")
    return value


def _read_manifest(path: Path) -> list[dict[str, str]]:
    """Stream-read a strict, nonempty manifest."""
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file, strict=True)
            fieldnames = reader.fieldnames or []
            if fieldnames != list(MANIFEST_FIELDS):
                missing = sorted(set(MANIFEST_FIELDS) - set(fieldnames))
                extra = sorted(set(fieldnames) - set(MANIFEST_FIELDS))
                raise AuditError(
                    f"manifest columns do not match schema: missing={missing}, extra={extra}, order={fieldnames}"
                )
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as error:
        raise AuditError(f"cannot read manifest {path}: {error}") from error
    if not rows:
        raise AuditError(f"manifest is empty: {path}")
    for number, row in enumerate(rows, start=2):
        if None in row or any(value is None for value in row.values()):
            raise AuditError(f"malformed manifest CSV row {number}")
    return rows


def _parse_names_mapping(text: str) -> dict[int, str]:
    """Independently parse the compact YAML names section."""
    lines = text.splitlines()
    for index, raw_line in enumerate(lines):
        stripped = raw_line.split("#", 1)[0].rstrip()
        match = re.match(r"^\s*names\s*:\s*(.*?)\s*$", stripped)
        if not match:
            continue
        inline = match.group(1)
        if inline:
            try:
                value = ast.literal_eval(inline)
            except (SyntaxError, ValueError):
                return {0: "crack"} if re.fullmatch(r"\{\s*0\s*:\s*['\"]?crack['\"]?\s*\}", inline) else {}
            if isinstance(value, list):
                return {item_index: str(name) for item_index, name in enumerate(value)}
            if isinstance(value, dict):
                try:
                    return {int(key): str(name) for key, name in value.items()}
                except (TypeError, ValueError):
                    return {}
            return {}
        names: dict[int, str] = {}
        for child_line in lines[index + 1 :]:
            if not child_line.strip() or child_line.lstrip().startswith("#"):
                continue
            if not child_line[:1].isspace():
                break
            child = child_line.split("#", 1)[0].strip()
            child_match = re.fullmatch(r"(\d+)\s*:\s*['\"]?([^'\"]+?)['\"]?", child)
            if not child_match:
                return {}
            names[int(child_match.group(1))] = child_match.group(2).strip()
        return names
    return {}


def _validate_data_yaml(dataset: Path, errors: list[str]) -> bytes:
    """Validate portable paths and the exact class mapping, returning exact bytes."""
    path = dataset / "data.yaml"
    try:
        content = path.read_bytes()
        text = content.decode("utf-8-sig")
    except (OSError, UnicodeError) as error:
        errors.append(f"cannot parse data.yaml: {error}")
        return b""
    scalars: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        match = re.fullmatch(r"(path|train|val|test)\s*:\s*(.*?)\s*", line)
        if match:
            scalars[match.group(1)] = match.group(2).strip("'\"")
    if scalars.get("path", ".") not in {".", "./"}:
        errors.append(f"data.yaml path is not portable: {scalars.get('path')!r}")
    for split in SPLITS:
        if scalars.get(split, "").replace("\\", "/") != f"images/{split}":
            errors.append(f"data.yaml {split} path is invalid: {scalars.get(split)!r}")
    if _parse_names_mapping(text) != {0: "crack"}:
        errors.append("data.yaml names must be exactly {0: 'crack'}")
    return content


def _relative_key(path: Path) -> str:
    """Return a case-insensitive relative identity without its suffix."""
    return path.with_suffix("").as_posix().casefold()


def _collect_source_pairs(source: Path, split: str) -> list[AuditPair]:
    """Independently match all source images and labels in a split."""
    image_root, label_root = source / "images" / split, source / "labels" / split
    if not image_root.is_dir() or not label_root.is_dir():
        raise AuditError(f"missing required split directories: {image_root} and {label_root}")
    images = sorted(
        (
            path
            for path in image_root.rglob("*")
            if path.is_file() and path.suffix.casefold() in IMAGE_SUFFIXES
        ),
        key=lambda path: path.relative_to(image_root).as_posix().casefold(),
    )
    labels = sorted(label_root.rglob("*.txt"), key=lambda path: path.relative_to(label_root).as_posix().casefold())
    image_by_key: dict[str, Path] = {}
    label_by_key: dict[str, Path] = {}
    for image_path in images:
        key = _relative_key(image_path.relative_to(image_root))
        if key in image_by_key:
            raise AuditError(f"multiple source images map to one label: {image_by_key[key]} and {image_path}")
        image_by_key[key] = image_path
    for label_path in labels:
        key = _relative_key(label_path.relative_to(label_root))
        if key in label_by_key:
            raise AuditError(f"duplicate source label identity: {label_by_key[key]} and {label_path}")
        label_by_key[key] = label_path
    missing_labels = sorted(set(image_by_key) - set(label_by_key))
    orphan_labels = sorted(set(label_by_key) - set(image_by_key))
    if missing_labels or orphan_labels:
        raise AuditError(
            f"source image/label mismatch in {split}: missing_labels={missing_labels[:5]}, "
            f"orphan_labels={orphan_labels[:5]}"
        )
    return [
        AuditPair(
            split,
            image_by_key[key],
            label_by_key[key],
            image_by_key[key].relative_to(source).as_posix(),
            label_by_key[key].relative_to(source).as_posix(),
        )
        for key in sorted(image_by_key)
    ]


@contextmanager
def _open_rgb(path: Path) -> Iterator[Image.Image]:
    """Fully decode one image, yield one RGB copy, then deterministically close it."""
    try:
        with Image.open(path) as probe:
            probe.verify()
        with Image.open(path) as image:
            image.load()
            rgb = image.convert("RGB")
    except (OSError, ValueError) as error:
        raise AuditError(f"damaged or unreadable image {path}: {error}") from error
    try:
        yield rgb
    finally:
        rgb.close()


def _source_bbox_id(image_relative: str, line_number: int) -> str:
    """Independently derive a stable source bbox ID."""
    source_id = hashlib.sha256(image_relative.encode("utf-8")).hexdigest()[:16]
    return f"{source_id}:bbox:{line_number}"


def _load_source_labels(path: Path, image_relative: str, width: int, height: int) -> list[AuditBox]:
    """Strictly parse source YOLO labels without using generator code."""
    try:
        text = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeError) as error:
        raise AuditError(f"cannot read source label {path}: {error}") from error
    boxes: list[AuditBox] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) != 5:
            raise AuditError(f"{path}:{line_number}: expected 5 YOLO fields, got {len(fields)}")
        try:
            class_value, x_center, y_center, box_width, box_height = (float(field) for field in fields)
        except ValueError as error:
            raise AuditError(f"{path}:{line_number}: nonnumeric YOLO label {line!r}") from error
        values = (class_value, x_center, y_center, box_width, box_height)
        if not all(math.isfinite(value) for value in values):
            raise AuditError(f"{path}:{line_number}: nonfinite YOLO label")
        if not class_value.is_integer() or int(class_value) != 0:
            raise AuditError(f"{path}:{line_number}: class_id must be 0")
        if not 0 <= x_center <= 1 or not 0 <= y_center <= 1 or not 0 < box_width <= 1 or not 0 < box_height <= 1:
            raise AuditError(f"{path}:{line_number}: illegal normalized YOLO coordinates")
        x1 = (x_center - box_width / 2) * width
        y1 = (y_center - box_height / 2) * height
        x2 = (x_center + box_width / 2) * width
        y2 = (y_center + box_height / 2) * height
        if (
            x1 < -FLOAT_TOLERANCE
            or y1 < -FLOAT_TOLERANCE
            or x2 > width + FLOAT_TOLERANCE
            or y2 > height + FLOAT_TOLERANCE
            or x2 <= x1
            or y2 <= y1
        ):
            raise AuditError(f"{path}:{line_number}: source box lies outside the image")
        boxes.append(AuditBox(_source_bbox_id(image_relative, line_number), (x1, y1, x2, y2)))
    return boxes


def _axis_anchors(length: int, tile_size: int, stride: int) -> list[int]:
    """Independently generate anchors including the far edge."""
    if length <= 0 or tile_size <= 0 or stride <= 0:
        raise AuditError(f"invalid anchor geometry: length={length}, tile_size={tile_size}, stride={stride}")
    if length <= tile_size:
        return [0]
    final_anchor = length - tile_size
    anchors = list(range(0, final_anchor + 1, stride))
    if anchors[-1] != final_anchor:
        anchors.append(final_anchor)
    return anchors


def _tile_anchors(width: int, height: int, tile_size: int, stride: int) -> list[tuple[int, int]]:
    """Independently generate the complete row-major anchor grid."""
    return [
        (x, y)
        for y in _axis_anchors(height, tile_size, stride)
        for x in _axis_anchors(width, tile_size, stride)
    ]


def _select_pairs(pairs: list[AuditPair], maximum: int | None, seed: int, split: str) -> list[AuditPair]:
    """Independently reproduce deterministic smoke-subset selection."""
    if maximum is None or maximum >= len(pairs):
        return pairs
    derived = int.from_bytes(hashlib.sha256(f"{seed}:{split}".encode()).digest()[:8], "big")
    return sorted(random.Random(derived).sample(pairs, maximum), key=lambda pair: pair.image_relative.casefold())


def _rounded_box(xyxy: tuple[float, float, float, float]) -> list[float]:
    """Round manifest coordinates exactly as documented."""
    return [round(value, 10) for value in xyxy]


def _classify_tile(
    boxes: list[AuditBox],
    tile_x: int,
    tile_y: int,
    tile_size: int,
    min_visibility: float,
    min_box_size: float,
) -> dict[str, Any]:
    """Independently calculate intersections, visibility, rejection reasons, and category."""
    intersecting: list[dict[str, Any]] = []
    retained: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy
        ix1, iy1 = max(x1, tile_x), max(y1, tile_y)
        ix2, iy2 = min(x2, tile_x + tile_size), min(y2, tile_y + tile_size)
        intersection_width, intersection_height = ix2 - ix1, iy2 - iy1
        if intersection_width <= 0 or intersection_height <= 0:
            continue
        visibility = (intersection_width * intersection_height) / ((x2 - x1) * (y2 - y1))
        clipped = (ix1 - tile_x, iy1 - tile_y, ix2 - tile_x, iy2 - tile_y)
        reasons: list[str] = []
        if visibility < min_visibility:
            reasons.append("VISIBILITY_BELOW_THRESHOLD")
        if intersection_width < min_box_size:
            reasons.append("WIDTH_BELOW_MIN_BOX_SIZE")
        if intersection_height < min_box_size:
            reasons.append("HEIGHT_BELOW_MIN_BOX_SIZE")
        if (
            clipped[0] < -FLOAT_TOLERANCE
            or clipped[1] < -FLOAT_TOLERANCE
            or clipped[2] > tile_size + FLOAT_TOLERANCE
            or clipped[3] > tile_size + FLOAT_TOLERANCE
            or clipped[2] <= clipped[0]
            or clipped[3] <= clipped[1]
        ):
            reasons.append("INVALID_CLIPPED_COORDINATES")
        record = {
            "source_bbox_id": box.source_bbox_id,
            "source_xyxy": _rounded_box(box.xyxy),
            "intersection_xyxy": _rounded_box((ix1, iy1, ix2, iy2)),
            "clipped_xyxy": _rounded_box(clipped),
            "visibility": round(visibility, 10),
        }
        intersecting.append(record)
        if reasons:
            failed.append({**record, "reasons": reasons})
        else:
            output_yolo = (
                ((clipped[0] + clipped[2]) / 2) / tile_size,
                ((clipped[1] + clipped[3]) / 2) / tile_size,
                (clipped[2] - clipped[0]) / tile_size,
                (clipped[3] - clipped[1]) / tile_size,
            )
            retained.append({**record, "output_yolo": [round(value, 10) for value in output_yolo]})
    category = "ambiguous" if failed else "safe_positive" if retained else "safe_negative"
    return {"category": category, "intersecting": intersecting, "retained": retained, "failed": failed}


def _tile_name(pair: AuditPair, x: int, y: int, tile_size: int, image_format: str) -> str:
    """Independently reproduce bounded deterministic naming."""
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", pair.image_path.stem).strip("._") or "image"
    relative_hash = hashlib.sha256(pair.image_relative.encode("utf-8")).hexdigest()[
        :SOURCE_PATH_HASH_CHARS
    ]
    prefix = f"{pair.split}__"
    suffix = f"__{relative_hash}__x{x:06d}__y{y:06d}__s{tile_size}.{image_format}"
    limit = min(MAX_TILE_STEM_CHARS, MAX_TILE_FILENAME_CHARS - len(prefix) - len(suffix))
    if limit < 1:
        raise AuditError("tile filename metadata exceeds the configured component limit")
    stem = stem[:limit].rstrip("._") or "image"
    return f"{prefix}{stem}{suffix}"


def _expected_manifest_row(
    pair: AuditPair,
    identity: AuditIdentity,
    width: int,
    height: int,
    tile_x: int,
    tile_y: int,
    tile_size: int,
    tile_file: str,
    decision: dict[str, Any],
) -> dict[str, str]:
    """Build an expected manifest row exclusively from source data and recorded parameters."""
    valid_width, valid_height = min(tile_size, width - tile_x), min(tile_size, height - tile_y)
    padding_right, padding_bottom = tile_size - valid_width, tile_size - valid_height
    intersecting, retained, failed = decision["intersecting"], decision["retained"], decision["failed"]
    values: dict[str, Any] = {
        "split": pair.split,
        "tile_file": tile_file,
        "source_image": pair.image_relative,
        "source_image_hash": hashlib.sha256(pair.image_relative.encode("utf-8")).hexdigest()[:16],
        "source_image_sha256": identity.image_sha256,
        "source_label": pair.label_relative,
        "source_label_sha256": identity.label_sha256,
        "tile_x": tile_x,
        "tile_y": tile_y,
        "tile_w": tile_size,
        "tile_h": tile_size,
        "valid_width": valid_width,
        "valid_height": valid_height,
        "source_width": width,
        "source_height": height,
        "padding": stable_json({"left": 0, "top": 0, "right": padding_right, "bottom": padding_bottom}),
        "padding_left": 0,
        "padding_top": 0,
        "padding_right": padding_right,
        "padding_bottom": padding_bottom,
        "category": decision["category"],
        "original_intersecting_box_count": len(intersecting),
        "retained_box_count": len(retained),
        "ambiguous_box_count": len(failed),
        "source_bbox_id": stable_json([record["source_bbox_id"] for record in intersecting]),
        "visibility": stable_json({record["source_bbox_id"]: record["visibility"] for record in intersecting}),
        "rejected_reason": stable_json(sorted({reason for record in failed for reason in record["reasons"]})),
        "source_boxes": stable_json(intersecting),
        "retained_box_records": stable_json(retained),
        "failed_boxes": stable_json(failed),
        "output_image_sha256": "",
        "output_label_sha256": "",
    }
    return {field: str(values[field]) for field in MANIFEST_FIELDS}


def _source_fingerprint(
    identities: list[AuditIdentity], source_config_relative: str, source_config_sha256: str
) -> str:
    """Independently calculate the canonical source fingerprint."""
    records = [
        {
            "split": identity.split,
            "image": identity.image_relative,
            "image_sha256": identity.image_sha256,
            "label": identity.label_relative,
            "label_sha256": identity.label_sha256,
        }
        for identity in sorted(identities, key=lambda item: (item.split, item.image_relative.casefold()))
    ]
    payload = {
        "algorithm": SOURCE_FINGERPRINT_ALGORITHM,
        "source_config": source_config_relative,
        "source_config_sha256": source_config_sha256,
        "files": records,
    }
    return hashlib.sha256(stable_json(payload).encode("utf-8")).hexdigest()


def _normalized_summary(summary: dict[str, Any]) -> dict[str, Any]:
    """Exclude only generated_at and the output fingerprint self-reference."""
    normalized = json.loads(stable_json(summary))
    normalized.pop("generated_at", None)
    normalized.pop("output_dataset_fingerprint", None)
    return normalized


def _canonical_output_fingerprint(file_hashes: dict[str, str], summary: dict[str, Any]) -> str:
    """Independently fingerprint exact payload files and normalized summary."""
    payload = {
        "algorithm": OUTPUT_FINGERPRINT_ALGORITHM,
        "files": [
            {"path": path, "sha256": digest}
            for path, digest in sorted(file_hashes.items(), key=lambda item: item[0])
        ],
        "summary": _normalized_summary(summary),
    }
    return hashlib.sha256(stable_json(payload).encode("utf-8")).hexdigest()


def _require_int(mapping: dict[str, Any], field: str, *, minimum: int | None = None) -> int:
    """Read a strict integer field from a JSON object."""
    value = mapping.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or (minimum is not None and value < minimum):
        raise AuditError(f"summary field {field} must be an integer >= {minimum}, got {value!r}")
    return value


def _require_float(mapping: dict[str, Any], field: str) -> float:
    """Read a finite numeric field from a JSON object."""
    value = mapping.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise AuditError(f"summary field {field} must be a finite number, got {value!r}")
    return float(value)


def _validate_parameters(summary: dict[str, Any]) -> dict[str, Any]:
    """Validate every parameter needed for independent reconstruction."""
    if summary.get("schema_version") != SCHEMA_VERSION:
        raise AuditError(f"unsupported summary schema_version: {summary.get('schema_version')!r}")
    parameters = summary.get("parameters")
    if not isinstance(parameters, dict):
        raise AuditError("summary parameters must be an object")
    tile_size = _require_int(parameters, "tile_size", minimum=1)
    overlap = _require_float(parameters, "overlap")
    if not 0 <= overlap < 1:
        raise AuditError(f"summary overlap must satisfy 0 <= overlap < 1, got {overlap}")
    stride = _require_int(parameters, "stride", minimum=1)
    if stride != round(tile_size * (1 - overlap)):
        raise AuditError(
            f"summary stride is inconsistent: recorded={stride}, expected={round(tile_size * (1 - overlap))}"
        )
    min_visibility = _require_float(parameters, "min_visibility")
    min_box_size = _require_float(parameters, "min_box_size")
    if not 0 <= min_visibility <= 1 or min_box_size <= 0:
        raise AuditError("summary visibility or minimum box size is invalid")
    image_format = parameters.get("image_format")
    if image_format not in {"png", "jpg"}:
        raise AuditError(f"summary image_format is invalid: {image_format!r}")
    jpeg_quality = parameters.get("jpeg_quality")
    if image_format == "jpg":
        if isinstance(jpeg_quality, bool) or not isinstance(jpeg_quality, int) or not 1 <= jpeg_quality <= 100:
            raise AuditError("summary jpeg_quality must be an explicit integer in [1, 100]")
        expected_encoder = (0, False, False)
        actual_encoder = (
            parameters.get("jpeg_subsampling"),
            parameters.get("jpeg_optimize"),
            parameters.get("jpeg_progressive"),
        )
        if actual_encoder != expected_encoder:
            raise AuditError(f"unsupported JPEG encoder parameters: {actual_encoder!r}")
        if parameters.get("png_compress_level") is not None:
            raise AuditError("summary png_compress_level must be null for JPEG")
    elif jpeg_quality is not None:
        raise AuditError("summary jpeg_quality must be null for PNG")
    elif (
        parameters.get("png_compress_level") != 6
        or parameters.get("jpeg_subsampling") is not None
        or parameters.get("jpeg_optimize") is not None
        or parameters.get("jpeg_progressive") is not None
    ):
        raise AuditError("summary PNG/JPEG encoder parameters are inconsistent")
    seed = _require_int(parameters, "seed")
    maximum = parameters.get("max_images_per_split")
    if maximum is not None and (
        isinstance(maximum, bool) or not isinstance(maximum, int) or maximum <= 0
    ):
        raise AuditError("summary max_images_per_split must be null or a positive integer")
    if parameters.get("padding_value") != 114 or parameters.get("padding") != {
        "mode": "right_bottom_constant",
        "value": 114,
    }:
        raise AuditError("summary padding must be right/bottom constant value 114")
    if parameters.get("visibility_comparison") != (
        "retain_if_visibility_greater_than_or_equal_to_threshold"
    ):
        raise AuditError("summary visibility comparison policy is unsupported")
    if parameters.get("max_tile_filename_chars") != MAX_TILE_FILENAME_CHARS:
        raise AuditError("summary max_tile_filename_chars is inconsistent")
    if parameters.get("max_tile_stem_chars") != MAX_TILE_STEM_CHARS:
        raise AuditError("summary max_tile_stem_chars is inconsistent")
    if parameters.get("source_path_hash_chars") != SOURCE_PATH_HASH_CHARS:
        raise AuditError("summary source_path_hash_chars is inconsistent")
    if parameters.get("dry_run") is not False or summary.get("dry_run") is not False:
        raise AuditError("a materialized dataset cannot be marked as dry_run")
    return {
        "tile_size": tile_size,
        "overlap": overlap,
        "stride": stride,
        "min_visibility": min_visibility,
        "min_box_size": min_box_size,
        "image_format": image_format,
        "jpeg_quality": jpeg_quality,
        "seed": seed,
        "max_images_per_split": maximum,
    }


def _crop_with_padding(image: Image.Image, x: int, y: int, tile_size: int) -> Image.Image:
    """Independently reconstruct an expected tile with right/bottom padding."""
    valid_width, valid_height = min(tile_size, image.width - x), min(tile_size, image.height - y)
    if valid_width <= 0 or valid_height <= 0:
        raise AuditError(f"tile anchor {(x, y)} lies outside source image {image.size}")
    crop = image.crop((x, y, x + valid_width, y + valid_height))
    tile = Image.new("RGB", (tile_size, tile_size), (114, 114, 114))
    tile.paste(crop, (0, 0))
    return tile


def _encoded_jpeg(image: Image.Image, quality: int) -> bytes:
    """Reproduce the documented JPEG encoder settings in memory."""
    buffer = io.BytesIO()
    image.save(
        buffer,
        format="JPEG",
        quality=quality,
        subsampling=0,
        optimize=False,
        progressive=False,
    )
    return buffer.getvalue()


def _compare_output_image(
    path: Path,
    expected: Image.Image,
    image_format: str,
    jpeg_quality: int | None,
    errors: list[str],
) -> None:
    """Verify dimensions and content against the independently reconstructed source crop."""
    actual: Image.Image | None = None
    try:
        with Image.open(path) as actual_image:
            actual_image.load()
            actual = actual_image.convert("RGB")
        if actual.size != expected.size:
            errors.append(f"output image size mismatch: {path}")
            return
        if image_format == "png":
            if ImageChops.difference(actual, expected).getbbox() is not None:
                errors.append(f"output PNG pixels do not match source crop: {path}")
        elif path.read_bytes() != _encoded_jpeg(expected, int(jpeg_quality)):
            errors.append(f"output JPEG bytes do not match recorded encoder parameters: {path}")
    except (OSError, ValueError) as error:
        errors.append(f"damaged output image {path}: {error}")
    finally:
        if actual is not None:
            actual.close()


def _expected_label_values(decision: dict[str, Any]) -> list[list[float]]:
    """Return independent expected YOLO values for a safe tile."""
    return [record["output_yolo"] for record in decision["retained"]]


def _validate_output_label(
    path: Path, category: str, expected_values: list[list[float]], errors: list[str]
) -> None:
    """Compare an output label with independently calculated values box by box and field by field."""
    try:
        lines = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    except (OSError, UnicodeError) as error:
        errors.append(f"cannot read output label {path}: {error}")
        return
    if category == "safe_negative" and lines:
        errors.append(f"safe_negative label is not empty: {path}")
    if len(lines) != len(expected_values):
        errors.append(f"output label box count mismatch: {path}: disk={len(lines)}, expected={len(expected_values)}")
    for index, (line, expected) in enumerate(zip(lines, expected_values), start=1):
        fields = line.split()
        if len(fields) != 5:
            errors.append(f"{path}:{index}: expected 5 fields")
            continue
        try:
            values = [float(field) for field in fields]
        except ValueError:
            errors.append(f"{path}:{index}: nonnumeric output label")
            continue
        if not all(math.isfinite(value) for value in values):
            errors.append(f"{path}:{index}: nonfinite output label")
            continue
        if not values[0].is_integer() or int(values[0]) != 0:
            errors.append(f"{path}:{index}: class_id is not 0")
        for field_index, (actual, expected_value) in enumerate(zip(values[1:], expected), start=1):
            if not math.isclose(actual, expected_value, rel_tol=0, abs_tol=LABEL_TOLERANCE):
                errors.append(
                    f"{path}:{index}: YOLO field {field_index} mismatch: "
                    f"disk={actual:.10f}, expected={expected_value:.10f}"
                )


def _initial_stats(available: int, processed: int, source_label_boxes: int) -> dict[str, int]:
    """Create independently measured summary counters."""
    return {
        "source_images_available": available,
        "source_images_processed": processed,
        "source_label_boxes": source_label_boxes,
        "candidate_tiles": 0,
        "safe_positive": 0,
        "safe_negative": 0,
        "ambiguous": 0,
        "intersecting_boxes": 0,
        "retained_boxes": 0,
        "failed_boxes": 0,
        "output_label_boxes": 0,
        "empty_labels": 0,
    }


def _validate_summary_metadata(
    summary: dict[str, Any],
    expected_splits: dict[str, dict[str, int]],
    rejection_reasons: Counter[str],
    source_config_relative: str,
    source_config_sha256: str,
    full_source_fingerprint: str,
    processed_subset_fingerprint: str,
    processed_images: list[str],
    source: Path,
    errors: list[str],
) -> None:
    """Cross-check all deterministic summary metadata against independent observations."""
    expected_totals = {
        field: sum(expected_splits[split][field] for split in SPLITS)
        for field in next(iter(expected_splits.values()))
    }
    if summary.get("splits") != expected_splits:
        errors.append("summary split counters do not match independent source/manifest reconstruction")
    if summary.get("totals") != expected_totals:
        errors.append("summary total counters do not match independent split totals")
    if summary.get("rejection_reasons") != dict(sorted(rejection_reasons.items())):
        errors.append("summary rejection reasons do not match independently rejected boxes")
    comparisons = {
        "source_data_config": source_config_relative,
        "source_data_config_sha256": source_config_sha256,
        "full_source_fingerprint": full_source_fingerprint,
        "processed_subset_fingerprint": processed_subset_fingerprint,
        "processed_source_images": processed_images,
        "input_modified": False,
    }
    for field, expected in comparisons.items():
        if summary.get(field) != expected:
            errors.append(f"summary {field} does not match independent observation")
    versions = summary.get("versions")
    if versions != {"python": platform.python_version(), "pillow": PIL.__version__}:
        errors.append(f"summary versions do not match audit runtime: {versions!r}")
    generated_at = summary.get("generated_at")
    try:
        parsed_time = datetime.fromisoformat(generated_at)
        if parsed_time.tzinfo is None:
            raise ValueError("timezone is missing")
    except (TypeError, ValueError) as error:
        errors.append(f"summary generated_at is invalid: {error}")
    expected_normalization = {
        "output_algorithm": OUTPUT_FINGERPRINT_ALGORITHM,
        "source_algorithm": SOURCE_FINGERPRINT_ALGORITHM,
        "covered_output": [
            "data.yaml",
            "metadata/source_data.yaml",
            "metadata/tile_manifest.csv",
            "images/**",
            "labels/** including empty labels",
            "normalized summary.json",
        ],
        "summary_excluded_fields": list(SUMMARY_FINGERPRINT_EXCLUSIONS),
        "file_order": "UTF-8 POSIX paths in code-point order",
        "serialization": (
            "UTF-8 canonical JSON with sorted mapping keys, preserved list order, "
            "finite floats, and compact separators"
        ),
    }
    if summary.get("fingerprint_normalization") != expected_normalization:
        errors.append("summary fingerprint normalization rule is missing or inconsistent")
    recorded_path = source / "metadata" / "dataset_fingerprint.sha256"
    if recorded_path.is_file():
        try:
            token = recorded_path.read_text(encoding="utf-8-sig").strip().split()[0].casefold()
        except (OSError, UnicodeError, IndexError) as error:
            errors.append(f"cannot read recorded source fingerprint: {error}")
        else:
            if not re.fullmatch(r"[0-9a-f]{64}", token):
                errors.append("recorded source fingerprint file is invalid")
            if summary.get("recorded_source_dataset_fingerprint") != token:
                errors.append("summary recorded source fingerprint does not match source metadata")
    elif summary.get("recorded_source_dataset_fingerprint") is not None:
        errors.append("summary records a source dataset fingerprint but the source file is missing")


def audit_dataset(source: Path, dataset: Path) -> dict[str, Any]:
    """Independently reconstruct expected tiles and return a complete audit report."""
    source, dataset = source.expanduser().resolve(), dataset.expanduser().resolve()
    if not source.is_dir() or not dataset.is_dir():
        raise AuditError(f"source and dataset must exist: source={source}, dataset={dataset}")
    summary = _read_json(dataset / "metadata" / "summary.json")
    parameters = _validate_parameters(summary)
    manifest_path = dataset / "metadata" / "tile_manifest.csv"
    rows = _read_manifest(manifest_path)
    errors: list[str] = []
    _validate_data_yaml(dataset, errors)

    source_config_relative = summary.get("source_data_config")
    if (
        not isinstance(source_config_relative, str)
        or PurePosixPath(source_config_relative).is_absolute()
        or ".." in PurePosixPath(source_config_relative).parts
    ):
        raise AuditError(f"summary source_data_config is unsafe: {source_config_relative!r}")
    source_config_path = source / Path(*PurePosixPath(source_config_relative).parts)
    try:
        source_config_content = source_config_path.read_bytes()
        source_config_text = source_config_content.decode("utf-8-sig")
    except (OSError, UnicodeError) as error:
        raise AuditError(f"cannot read source data configuration {source_config_path}: {error}") from error
    if _parse_names_mapping(source_config_text) != {0: "crack"}:
        raise AuditError("source names must be exactly {0: 'crack'}")
    source_config_sha256 = hashlib.sha256(source_config_content).hexdigest()
    copied_config_path = dataset / "metadata" / "source_data.yaml"
    try:
        copied_config_content = copied_config_path.read_bytes()
    except OSError as error:
        errors.append(f"cannot read metadata/source_data.yaml: {error}")
        copied_config_content = b""
    if copied_config_content != source_config_content:
        errors.append("metadata/source_data.yaml does not exactly match the source configuration")

    all_pairs = {split: _collect_source_pairs(source, split) for split in SPLITS}
    identities: list[AuditIdentity] = []
    identities_by_image: dict[str, AuditIdentity] = {}
    source_dimensions: dict[str, tuple[int, int]] = {}
    source_boxes: dict[str, list[AuditBox]] = {}
    for split in SPLITS:
        for pair in all_pairs[split]:
            with _open_rgb(pair.image_path) as image:
                width, height = image.size
                boxes = _load_source_labels(
                    pair.label_path, pair.image_relative, width, height
                )
            identity = AuditIdentity(
                split,
                pair.image_relative,
                sha256_file(pair.image_path),
                pair.label_relative,
                sha256_file(pair.label_path),
            )
            identities.append(identity)
            identities_by_image[pair.image_relative] = identity
            source_dimensions[pair.image_relative] = (width, height)
            source_boxes[pair.image_relative] = boxes
    full_source_fingerprint = _source_fingerprint(
        identities, source_config_relative, source_config_sha256
    )
    selected_pairs = {
        split: _select_pairs(
            all_pairs[split], parameters["max_images_per_split"], parameters["seed"], split
        )
        for split in SPLITS
    }
    processed_identities = [
        identities_by_image[pair.image_relative] for split in SPLITS for pair in selected_pairs[split]
    ]
    processed_subset_fingerprint = _source_fingerprint(
        processed_identities, source_config_relative, source_config_sha256
    )
    processed_images = [identity.image_relative for identity in processed_identities]

    expected_splits: dict[str, dict[str, int]] = {
        split: _initial_stats(
            len(all_pairs[split]),
            len(selected_pairs[split]),
            sum(len(source_boxes[pair.image_relative]) for pair in selected_pairs[split]),
        )
        for split in SPLITS
    }
    actual_rows: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        key = (row["split"], row["tile_file"])
        if key in actual_rows:
            errors.append(f"duplicate manifest tile: split={key[0]}, tile={key[1]}")
        else:
            actual_rows[key] = row

    expected_keys: set[tuple[str, str]] = set()
    missing_rows: list[tuple[str, str]] = []
    expected_disk_images: set[str] = set()
    expected_disk_labels: set[str] = set()
    rejection_reasons: Counter[str] = Counter()
    for split in SPLITS:
        for pair in selected_pairs[split]:
            boxes = source_boxes[pair.image_relative]
            identity = identities_by_image[pair.image_relative]
            expected_width, expected_height = source_dimensions[pair.image_relative]
            with _open_rgb(pair.image_path) as source_image:
                if source_image.size != (expected_width, expected_height):
                    errors.append(f"source image dimensions changed during audit: {pair.image_relative}")
                for tile_x, tile_y in _tile_anchors(
                    source_image.width,
                    source_image.height,
                    parameters["tile_size"],
                    parameters["stride"],
                ):
                    decision = _classify_tile(
                        boxes,
                        tile_x,
                        tile_y,
                        parameters["tile_size"],
                        parameters["min_visibility"],
                        parameters["min_box_size"],
                    )
                    tile_file = _tile_name(
                        pair,
                        tile_x,
                        tile_y,
                        parameters["tile_size"],
                        parameters["image_format"],
                    )
                    key = (split, tile_file)
                    if key in expected_keys:
                        raise AuditError(f"independent naming collision: {tile_file}")
                    expected_keys.add(key)
                    expected_row = _expected_manifest_row(
                        pair,
                        identity,
                        source_image.width,
                        source_image.height,
                        tile_x,
                        tile_y,
                        parameters["tile_size"],
                        tile_file,
                        decision,
                    )
                    stats = expected_splits[split]
                    stats["candidate_tiles"] += 1
                    stats[decision["category"]] += 1
                    stats["intersecting_boxes"] += len(decision["intersecting"])
                    stats["retained_boxes"] += len(decision["retained"])
                    stats["failed_boxes"] += len(decision["failed"])
                    if decision["category"] == "safe_positive":
                        stats["output_label_boxes"] += len(decision["retained"])
                    elif decision["category"] == "safe_negative":
                        stats["empty_labels"] += 1
                    for failed_box in decision["failed"]:
                        rejection_reasons.update(failed_box["reasons"])

                    actual_row = actual_rows.get(key)
                    if actual_row is None:
                        missing_rows.append(key)
                    else:
                        for field in MANIFEST_FIELDS:
                            if field in {"output_image_sha256", "output_label_sha256"}:
                                continue
                            if actual_row[field] != expected_row[field]:
                                errors.append(
                                    f"manifest field mismatch for {tile_file}.{field}: "
                                    f"disk={actual_row[field]!r}, expected={expected_row[field]!r}"
                                )

                    image_relative = f"images/{split}/{tile_file}"
                    label_relative = f"labels/{split}/{Path(tile_file).stem}.txt"
                    image_path, label_path = dataset / image_relative, dataset / label_relative
                    if decision["category"] == "ambiguous":
                        if image_path.exists() or label_path.exists():
                            errors.append(f"ambiguous tile entered formal output: {tile_file}")
                        if actual_row is not None and (
                            actual_row["output_image_sha256"]
                            or actual_row["output_label_sha256"]
                        ):
                            errors.append(f"ambiguous tile records formal output hashes: {tile_file}")
                        continue

                    expected_disk_images.add(image_relative)
                    expected_disk_labels.add(label_relative)
                    if not image_path.is_file() or not label_path.is_file():
                        errors.append(f"missing formal image/label pair for {tile_file}")
                        continue
                    expected_tile = _crop_with_padding(
                        source_image,
                        tile_x,
                        tile_y,
                        parameters["tile_size"],
                    )
                    try:
                        _compare_output_image(
                            image_path,
                            expected_tile,
                            parameters["image_format"],
                            parameters["jpeg_quality"],
                            errors,
                        )
                    finally:
                        expected_tile.close()
                    _validate_output_label(
                        label_path,
                        decision["category"],
                        _expected_label_values(decision)
                        if decision["category"] == "safe_positive"
                        else [],
                        errors,
                    )
                    image_hash = sha256_file(image_path)
                    label_hash = sha256_file(label_path)
                    if actual_row is not None:
                        if actual_row["output_image_sha256"] != image_hash:
                            errors.append(f"manifest output image hash mismatch: {tile_file}")
                        if actual_row["output_label_sha256"] != label_hash:
                            errors.append(f"manifest output label hash mismatch: {tile_file}")

    if missing_rows:
        errors.append(
            f"manifest is missing independently expected tiles: {sorted(missing_rows)[:10]}"
        )
    extra_rows = sorted(set(actual_rows) - expected_keys)
    if extra_rows:
        errors.append(f"manifest contains unexpected tiles: {extra_rows[:10]}")

    disk_images = {
        path.relative_to(dataset).as_posix()
        for split in SPLITS
        for path in (dataset / "images" / split).rglob("*")
        if path.is_file() and path.suffix.casefold() in IMAGE_SUFFIXES
    }
    disk_labels = {
        path.relative_to(dataset).as_posix()
        for split in SPLITS
        for path in (dataset / "labels" / split).rglob("*.txt")
        if path.is_file()
    }
    if disk_images != expected_disk_images:
        errors.append(f"formal image set mismatch: delta={sorted(disk_images ^ expected_disk_images)[:10]}")
    if disk_labels != expected_disk_labels:
        errors.append(f"formal label set mismatch: delta={sorted(disk_labels ^ expected_disk_labels)[:10]}")
    if {Path(path).stem.casefold() for path in disk_images} != {
        Path(path).stem.casefold() for path in disk_labels
    }:
        errors.append("formal images and labels are not one-to-one")
    output_file_hashes = {
        relative: sha256_file(dataset / Path(*PurePosixPath(relative).parts))
        for relative in sorted(disk_images | disk_labels)
    }

    _validate_summary_metadata(
        summary,
        expected_splits,
        rejection_reasons,
        source_config_relative,
        source_config_sha256,
        full_source_fingerprint,
        processed_subset_fingerprint,
        processed_images,
        source,
        errors,
    )
    payload_hashes = {
        "data.yaml": sha256_file(dataset / "data.yaml")
        if (dataset / "data.yaml").is_file()
        else hashlib.sha256(b"").hexdigest(),
        "metadata/source_data.yaml": sha256_file(copied_config_path)
        if copied_config_path.is_file()
        else hashlib.sha256(b"").hexdigest(),
        "metadata/tile_manifest.csv": sha256_file(manifest_path),
        **output_file_hashes,
    }
    output_fingerprint = _canonical_output_fingerprint(payload_hashes, summary)
    fingerprint_path = dataset / "metadata" / "dataset_fingerprint.sha256"
    try:
        recorded_output = fingerprint_path.read_text(encoding="utf-8-sig").strip().split()[0].casefold()
    except (OSError, UnicodeError, IndexError) as error:
        errors.append(f"cannot read output dataset fingerprint: {error}")
        recorded_output = ""
    if not re.fullmatch(r"[0-9a-f]{64}", recorded_output):
        errors.append("output dataset fingerprint file is invalid")
    if output_fingerprint != recorded_output:
        errors.append("output dataset fingerprint does not match exact disk payload")
    if output_fingerprint != summary.get("output_dataset_fingerprint"):
        errors.append("output dataset fingerprint does not match normalized summary")

    return {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "manifest_rows": len(rows),
        "expected_tiles": len(expected_keys),
        "disk_images": len(disk_images),
        "disk_labels": len(disk_labels),
        "categories": dict(sorted(Counter(row["category"] for row in rows).items())),
        "full_source_fingerprint": full_source_fingerprint,
        "processed_subset_fingerprint": processed_subset_fingerprint,
        "output_dataset_fingerprint": output_fingerprint,
        "input_modified": summary.get("full_source_fingerprint") != full_source_fingerprint,
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    """Run the independent audit and uniformly return nonzero on every failure."""
    args = parse_args()
    try:
        report = audit_dataset(args.source, args.dataset)
    except Exception as error:
        report = {"status": "FAIL", "errors": [f"{type(error).__name__}: {error}"]}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
