# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Create a deterministic overlap-tiled YOLO detection dataset without changing its source dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import PIL
import yaml
from PIL import Image

SPLITS = ("train", "val", "test")
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
FLOAT_TOLERANCE = 1e-9
SCHEMA_VERSION = 3
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
    "source_data_config",
    "source_data_config_selection",
    "source_data_config_path",
    "source_data_config_sha256",
    "source_data_config_relative",
    "source_data_config_external",
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


class TilingError(RuntimeError):
    """Raised when source validation or safe output requirements fail."""


@dataclass(frozen=True)
class BuildConfig:
    """Validated settings for one tiled-dataset build."""

    source: Path
    output: Path
    tile_size: int = 1024
    overlap: float = 0.20
    min_visibility: float = 0.50
    min_box_size: float = 2.0
    image_format: str = "png"
    jpeg_quality: int | None = None
    seed: int = 42
    max_images_per_split: int | None = None
    dry_run: bool = False
    data_yaml: Path | None = None


@dataclass(frozen=True)
class SourceBox:
    """One strictly validated source bounding box in pixel xyxy coordinates."""

    source_bbox_id: str
    line_number: int
    class_id: int
    xyxy: tuple[float, float, float, float]


@dataclass(frozen=True)
class SourcePair:
    """One source image and its corresponding label."""

    split: str
    image_path: Path
    label_path: Path
    image_relative: str
    label_relative: str


@dataclass(frozen=True)
class SourceIdentity:
    """Stable hashes used to prove source inputs were not changed."""

    split: str
    image_relative: str
    image_sha256: str
    label_relative: str
    label_sha256: str


@dataclass(frozen=True)
class SourceLayout:
    """Selected source configuration and its strictly validated split directories."""

    config_path: Path
    config_identity: str
    config_relative: str | None
    config_external: str | None
    config_content: bytes
    selection: str
    dataset_root: Path
    image_roots: dict[str, Path]
    label_roots: dict[str, Path]
    nc: int
    names: dict[int, str]


def sha256_file(path: Path) -> str:
    """Return a SHA-256 digest without modifying a file."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_text_lf(path: Path, text: str) -> None:
    """Write deterministic UTF-8 text with LF line endings on every platform."""
    with path.open("w", encoding="utf-8", newline="\n") as file:
        file.write(text)


def stable_json(value: Any) -> str:
    """Serialize UTF-8 JSON with sorted mappings, preserved list order, and finite floats."""
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def compute_stride(tile_size: int, overlap: float) -> int:
    """Compute the rounded pixel stride for a tile size and fractional overlap."""
    if tile_size <= 0:
        raise TilingError(f"tile_size must be positive, got {tile_size}")
    if not 0 <= overlap < 1:
        raise TilingError(f"overlap must satisfy 0 <= overlap < 1, got {overlap}")
    stride = round(tile_size * (1 - overlap))
    if stride <= 0:
        raise TilingError(f"overlap {overlap} produces a nonpositive stride")
    return stride


def axis_anchors(length: int, tile_size: int, stride: int) -> list[int]:
    """Generate deterministic anchors, including the final far-edge anchor."""
    if length <= 0:
        raise TilingError(f"image dimension must be positive, got {length}")
    if tile_size <= 0 or stride <= 0:
        raise TilingError("tile_size and stride must be positive")
    if length <= tile_size:
        return [0]
    final_anchor = length - tile_size
    anchors = list(range(0, final_anchor + 1, stride))
    if anchors[-1] != final_anchor:
        anchors.append(final_anchor)
    return anchors


def generate_tile_anchors(width: int, height: int, tile_size: int, stride: int) -> list[tuple[int, int]]:
    """Return row-major (x, y) anchors that fully cover an image."""
    return [
        (x, y)
        for y in axis_anchors(height, tile_size, stride)
        for x in axis_anchors(width, tile_size, stride)
    ]


def yolo_to_xyxy(
    x_center: float, y_center: float, width: float, height: float, image_width: int, image_height: int
) -> tuple[float, float, float, float]:
    """Convert a strictly valid normalized YOLO box to source pixel xyxy."""
    values = (x_center, y_center, width, height)
    if not all(math.isfinite(value) for value in values):
        raise TilingError(f"YOLO coordinates must be finite, got {values}")
    if not 0 <= x_center <= 1 or not 0 <= y_center <= 1:
        raise TilingError(f"YOLO center must be in [0, 1], got {(x_center, y_center)}")
    if not 0 < width <= 1 or not 0 < height <= 1:
        raise TilingError(f"YOLO width and height must be in (0, 1], got {(width, height)}")
    x1 = (x_center - width / 2) * image_width
    y1 = (y_center - height / 2) * image_height
    x2 = (x_center + width / 2) * image_width
    y2 = (y_center + height / 2) * image_height
    if (
        x1 < -FLOAT_TOLERANCE
        or y1 < -FLOAT_TOLERANCE
        or x2 > image_width + FLOAT_TOLERANCE
        or y2 > image_height + FLOAT_TOLERANCE
    ):
        raise TilingError(f"YOLO box extends outside the source image: {(x1, y1, x2, y2)}")
    if x2 - x1 <= 0 or y2 - y1 <= 0:
        raise TilingError(f"YOLO box has nonpositive dimensions: {(x1, y1, x2, y2)}")
    return x1, y1, x2, y2


def xyxy_to_yolo(
    xyxy: tuple[float, float, float, float], tile_width: int, tile_height: int
) -> tuple[float, float, float, float]:
    """Convert tile-local pixel xyxy to strictly valid normalized YOLO coordinates."""
    x1, y1, x2, y2 = xyxy
    if (
        not all(math.isfinite(value) for value in xyxy)
        or x1 < -FLOAT_TOLERANCE
        or y1 < -FLOAT_TOLERANCE
        or x2 > tile_width + FLOAT_TOLERANCE
        or y2 > tile_height + FLOAT_TOLERANCE
        or x2 <= x1
        or y2 <= y1
    ):
        raise TilingError(f"invalid tile-local xyxy coordinates: {xyxy}")
    return (
        ((x1 + x2) / 2) / tile_width,
        ((y1 + y2) / 2) / tile_height,
        (x2 - x1) / tile_width,
        (y2 - y1) / tile_height,
    )


def _rounded_box(xyxy: tuple[float, float, float, float]) -> list[float]:
    """Round coordinates for stable JSON without changing written labels."""
    return [round(value, 10) for value in xyxy]


def classify_tile(
    boxes: list[SourceBox],
    tile_x: int,
    tile_y: int,
    tile_size: int,
    min_visibility: float,
    min_box_size: float,
) -> dict[str, Any]:
    """Classify a tile as safe_positive, safe_negative, or ambiguous."""
    tile_x2, tile_y2 = tile_x + tile_size, tile_y + tile_size
    intersecting: list[dict[str, Any]] = []
    retained: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []

    for box in boxes:
        x1, y1, x2, y2 = box.xyxy
        ix1, iy1 = max(x1, tile_x), max(y1, tile_y)
        ix2, iy2 = min(x2, tile_x2), min(y2, tile_y2)
        intersection_width, intersection_height = ix2 - ix1, iy2 - iy1
        if intersection_width <= 0 or intersection_height <= 0:
            continue
        source_area = (x2 - x1) * (y2 - y1)
        visibility = (intersection_width * intersection_height) / source_area
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
            retained.append(
                {
                    **record,
                    "output_yolo": [
                        round(value, 10) for value in xyxy_to_yolo(clipped, tile_size, tile_size)
                    ],
                }
            )

    if failed:
        category = "ambiguous"
    elif retained:
        category = "safe_positive"
    else:
        category = "safe_negative"
    return {"category": category, "intersecting": intersecting, "retained": retained, "failed": failed}


def _source_bbox_id(image_relative: str, line_number: int) -> str:
    """Create a stable source bbox ID from the relative image path and label line."""
    source_id = hashlib.sha256(image_relative.encode("utf-8")).hexdigest()[:16]
    return f"{source_id}:bbox:{line_number}"


def load_yolo_labels(label_path: Path, image_relative: str, image_width: int, image_height: int) -> list[SourceBox]:
    """Load one label file, failing on malformed values, nonzero classes, or out-of-bounds boxes."""
    boxes: list[SourceBox] = []
    try:
        text = label_path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeError) as error:
        raise TilingError(f"cannot read label {label_path}: {error}") from error
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) != 5:
            raise TilingError(f"{label_path}:{line_number}: expected 5 fields, got {len(fields)}")
        try:
            class_value, x_center, y_center, width, height = (float(field) for field in fields)
        except ValueError as error:
            raise TilingError(f"{label_path}:{line_number}: nonnumeric YOLO label {line!r}") from error
        if not class_value.is_integer() or int(class_value) != 0:
            raise TilingError(f"{label_path}:{line_number}: class_id must be 0, got {fields[0]}")
        xyxy = yolo_to_xyxy(x_center, y_center, width, height, image_width, image_height)
        boxes.append(SourceBox(_source_bbox_id(image_relative, line_number), line_number, 0, xyxy))
    return boxes


def _relative_key(path: Path) -> str:
    """Return a case-insensitive POSIX key without the file suffix."""
    return path.with_suffix("").as_posix().casefold()


def collect_source_pairs(
    source: Path, split: str, image_root: Path | None = None, label_root: Path | None = None
) -> list[SourcePair]:
    """Match every image and label in a split, rejecting missing or orphaned files."""
    image_root = image_root or source / "images" / split
    label_root = label_root or source / "labels" / split
    if not image_root.is_dir() or not label_root.is_dir():
        raise TilingError(f"missing required split directories: {image_root} and {label_root}")
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
    for image_path in images:
        relative = image_path.relative_to(image_root)
        key = _relative_key(relative)
        if key in image_by_key:
            raise TilingError(
                f"multiple images map to one label: {image_by_key[key]} and {image_path}"
            )
        image_by_key[key] = image_path
    label_by_key: dict[str, Path] = {}
    for label_path in labels:
        relative = label_path.relative_to(label_root)
        key = _relative_key(relative)
        if key in label_by_key:
            raise TilingError(f"duplicate label identity: {label_by_key[key]} and {label_path}")
        label_by_key[key] = label_path
    missing_labels = sorted(set(image_by_key) - set(label_by_key))
    orphan_labels = sorted(set(label_by_key) - set(image_by_key))
    if missing_labels or orphan_labels:
        raise TilingError(
            f"image/label mismatch in split {split}: missing_labels={missing_labels[:5]}, "
            f"orphan_labels={orphan_labels[:5]}"
        )
    pairs: list[SourcePair] = []
    for key in sorted(image_by_key):
        image_path, label_path = image_by_key[key], label_by_key[key]
        pairs.append(
            SourcePair(
                split=split,
                image_path=image_path,
                label_path=label_path,
                image_relative=image_path.relative_to(source).as_posix(),
                label_relative=label_path.relative_to(source).as_posix(),
            )
        )
    return pairs


def _select_pairs(pairs: list[SourcePair], maximum: int | None, seed: int, split: str) -> list[SourcePair]:
    """Select a deterministic smoke subset without changing split membership."""
    if maximum is None or maximum >= len(pairs):
        return pairs
    derived = int.from_bytes(hashlib.sha256(f"{seed}:{split}".encode()).digest()[:8], "big")
    return sorted(random.Random(derived).sample(pairs, maximum), key=lambda pair: pair.image_relative.casefold())


def _config_provenance(
    config_path: Path, source: Path, config_content: bytes
) -> tuple[str, str | None, str | None]:
    """Return a location-independent identity plus its source-relative or external form."""
    try:
        relative = config_path.relative_to(source).as_posix()
    except ValueError:
        external = f"external:sha256:{hashlib.sha256(config_content).hexdigest()}"
        return external, None, external
    return relative, relative, None


def _portable_yaml_path(value: str) -> Path:
    """Convert relative POSIX or Windows YAML syntax to native path components."""
    if PureWindowsPath(value).is_absolute() or PurePosixPath(value).is_absolute():
        return Path(value)
    pure = PureWindowsPath(value) if "\\" in value else PurePosixPath(value)
    return Path(*pure.parts)


def _parse_names(value: Any, config_path: Path) -> dict[int, str]:
    """Normalize a YAML names list or mapping to an integer-keyed mapping."""
    if isinstance(value, list):
        names = {index: str(name) for index, name in enumerate(value)}
    elif isinstance(value, dict):
        try:
            names = {int(key): str(name) for key, name in value.items()}
        except (TypeError, ValueError) as error:
            raise TilingError(f"source names keys must be integers in {config_path}") from error
    else:
        raise TilingError(f"source names must be a list or mapping in {config_path}")
    if names != {0: "crack"}:
        raise TilingError(f"source names must be exactly {{0: 'crack'}} in {config_path}")
    return names


def _resolve_yaml_path(value: Any, base: Path, field: str, config_path: Path) -> Path:
    """Resolve one required scalar YAML path against its documented base directory."""
    if not isinstance(value, str) or not value.strip():
        raise TilingError(f"source {field} must be a nonempty path string in {config_path}")
    path = _portable_yaml_path(value.strip()).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def _resolve_auto_config_candidate(source: Path, candidate: Path) -> Path:
    """Resolve an automatic YAML candidate and reject any target outside source before probing it."""
    try:
        resolved_source = source.resolve()
        resolved_candidate = candidate.resolve()
    except (OSError, RuntimeError) as error:
        raise TilingError(f"cannot resolve auto-discovered source configuration {candidate}: {error}") from error
    try:
        resolved_candidate.relative_to(resolved_source)
    except ValueError as error:
        raise TilingError(
            f"auto-discovered source configuration resolves outside --source: "
            f"{candidate} -> {resolved_candidate}"
        ) from error
    return resolved_candidate


def _load_source_config(source: Path, data_yaml: Path | None) -> SourceLayout:
    """Select one YAML, parse its layout, and prove it matches the directories that will be scanned."""
    if data_yaml is not None:
        config_path, selection = data_yaml, "explicit"
    else:
        candidates = (source / "data.yaml", source / "data_local.yaml")
        config_path = None
        for candidate in candidates:
            resolved_candidate = _resolve_auto_config_candidate(source, candidate)
            if resolved_candidate.is_file():
                config_path = resolved_candidate
                break
        selection = "auto"
        if config_path is None:
            raise TilingError(f"source data configuration is missing under {source}")
    if not config_path.is_file():
        raise TilingError(f"source data configuration is not a file: {config_path}")
    try:
        content = config_path.read_bytes()
        text = content.decode("utf-8-sig")
        document = yaml.safe_load(text)
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise TilingError(f"cannot parse source data configuration {config_path}: {error}") from error
    if not isinstance(document, dict):
        raise TilingError(f"source data configuration must be a YAML mapping: {config_path}")
    config_identity, config_relative, config_external = _config_provenance(
        config_path, source, content
    )
    names = _parse_names(document.get("names"), config_path)
    nc_value = document.get("nc")
    if nc_value is None:
        nc = len(names)
    elif isinstance(nc_value, bool) or not isinstance(nc_value, int):
        raise TilingError(f"source nc must be an integer when present in {config_path}")
    else:
        nc = nc_value
    if nc != 1 or nc != len(names):
        raise TilingError(f"source nc must be 1 and match names in {config_path}, got {nc}")

    dataset_root = _resolve_yaml_path(document.get("path", "."), config_path.parent, "path", config_path)
    image_roots: dict[str, Path] = {}
    label_roots: dict[str, Path] = {}
    for split in SPLITS:
        declared_image_root = _resolve_yaml_path(document.get(split), dataset_root, split, config_path)
        actual_image_root = (source / "images" / split).resolve()
        actual_label_root = (source / "labels" / split).resolve()
        if declared_image_root != actual_image_root:
            raise TilingError(
                f"source {split} path resolves to {declared_image_root}, "
                f"but actual scan directory is {actual_image_root}"
            )
        if not actual_image_root.is_dir() or not actual_label_root.is_dir():
            raise TilingError(
                f"missing required split directories: {actual_image_root} and {actual_label_root}"
            )
        image_roots[split] = actual_image_root
        label_roots[split] = actual_label_root
    return SourceLayout(
        config_path=config_path,
        config_identity=config_identity,
        config_relative=config_relative,
        config_external=config_external,
        config_content=content,
        selection=selection,
        dataset_root=dataset_root,
        image_roots=image_roots,
        label_roots=label_roots,
        nc=nc,
        names=names,
    )


def _validate_config(config: BuildConfig) -> BuildConfig:
    """Resolve paths and reject unsafe or underspecified arguments."""
    source, output = config.source.expanduser().resolve(), config.output.expanduser().resolve()
    if not source.is_dir():
        raise TilingError(f"source dataset does not exist: {source}")
    if output.exists():
        state = "nonempty" if output.is_dir() and next(output.iterdir(), None) is not None else "existing"
        raise TilingError(f"output path is {state}; refusing implicit overwrite: {output}")
    if source == output or source in output.parents:
        raise TilingError(f"output must not equal or be nested inside source: {output}")
    data_yaml = config.data_yaml.expanduser().resolve() if config.data_yaml is not None else None
    if data_yaml is not None and not data_yaml.is_file():
        raise TilingError(f"--data-yaml must be an existing regular file: {data_yaml}")
    compute_stride(config.tile_size, config.overlap)
    if not 0 <= config.min_visibility <= 1:
        raise TilingError(f"min_visibility must be in [0, 1], got {config.min_visibility}")
    if config.min_box_size <= 0:
        raise TilingError(f"min_box_size must be positive, got {config.min_box_size}")
    image_format = config.image_format.casefold()
    if image_format not in {"png", "jpg"}:
        raise TilingError(f"image_format must be png or jpg, got {config.image_format}")
    if image_format == "jpg" and config.jpeg_quality is None:
        raise TilingError("--jpeg-quality must be explicitly specified when --image-format jpg is used")
    if config.jpeg_quality is not None and not 1 <= config.jpeg_quality <= 100:
        raise TilingError(f"jpeg_quality must be in [1, 100], got {config.jpeg_quality}")
    if config.max_images_per_split is not None and config.max_images_per_split <= 0:
        raise TilingError("max_images_per_split must be positive")
    return BuildConfig(
        source=source,
        output=output,
        data_yaml=data_yaml,
        tile_size=config.tile_size,
        overlap=config.overlap,
        min_visibility=config.min_visibility,
        min_box_size=config.min_box_size,
        image_format=image_format,
        jpeg_quality=config.jpeg_quality,
        seed=config.seed,
        max_images_per_split=config.max_images_per_split,
        dry_run=config.dry_run,
    )


def _open_validated_image(path: Path) -> Image.Image:
    """Fully decode an image and return an independent RGB copy."""
    try:
        with Image.open(path) as probe:
            probe.verify()
        with Image.open(path) as image:
            image.load()
            return image.convert("RGB")
    except (OSError, ValueError) as error:
        raise TilingError(f"damaged or unreadable image {path}: {error}") from error


def _tile_name(pair: SourcePair, tile_x: int, tile_y: int, tile_size: int, image_format: str) -> str:
    """Create a bounded, globally unique deterministic tile filename."""
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", pair.image_path.stem).strip("._") or "image"
    relative_hash = hashlib.sha256(pair.image_relative.encode("utf-8")).hexdigest()[
        :SOURCE_PATH_HASH_CHARS
    ]
    prefix = f"{pair.split}__"
    suffix = f"__{relative_hash}__x{tile_x:06d}__y{tile_y:06d}__s{tile_size}.{image_format}"
    stem_limit = min(MAX_TILE_STEM_CHARS, MAX_TILE_FILENAME_CHARS - len(prefix) - len(suffix))
    if stem_limit < 1:
        raise TilingError("tile filename metadata exceeds the configured component limit")
    stem = stem[:stem_limit].rstrip("._") or "image"
    filename = f"{prefix}{stem}{suffix}"
    if len(filename) > MAX_TILE_FILENAME_CHARS:
        raise TilingError(f"tile filename exceeds {MAX_TILE_FILENAME_CHARS} characters: {filename}")
    return filename


def _crop_with_padding(image: Image.Image, tile_x: int, tile_y: int, tile_size: int) -> tuple[Image.Image, int, int]:
    """Crop without resizing and right/bottom pad undersized dimensions with value 114."""
    valid_width = min(tile_size, image.width - tile_x)
    valid_height = min(tile_size, image.height - tile_y)
    if valid_width <= 0 or valid_height <= 0:
        raise TilingError(f"tile anchor {(tile_x, tile_y)} is outside image size {image.size}")
    crop = image.crop((tile_x, tile_y, tile_x + valid_width, tile_y + valid_height))
    if valid_width == tile_size and valid_height == tile_size:
        return crop, valid_width, valid_height
    padded = Image.new("RGB", (tile_size, tile_size), color=(114, 114, 114))
    padded.paste(crop, (0, 0))
    return padded, valid_width, valid_height


def _save_image(image: Image.Image, path: Path, image_format: str, jpeg_quality: int | None) -> None:
    """Save with explicit deterministic encoder parameters."""
    if image_format == "png":
        image.save(path, format="PNG", compress_level=6, optimize=False)
    else:
        image.save(
            path,
            format="JPEG",
            quality=jpeg_quality,
            subsampling=0,
            optimize=False,
            progressive=False,
        )


def _label_bytes(retained: list[dict[str, Any]]) -> bytes:
    """Encode retained boxes as deterministic YOLO label bytes."""
    lines = ["0 " + " ".join(f"{value:.10f}" for value in record["output_yolo"]) for record in retained]
    return (("\n".join(lines) + "\n") if lines else "").encode("utf-8")


def _source_fingerprint(
    identities: list[SourceIdentity], source_config_relative: str, source_config_sha256: str
) -> str:
    """Fingerprint source membership, exact image/label bytes, and the selected source configuration."""
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
    """Normalize summary by excluding only generated_at and the output fingerprint self-reference."""
    normalized = json.loads(stable_json(summary))
    normalized.pop("generated_at", None)
    normalized.pop("output_dataset_fingerprint", None)
    return normalized


def _canonical_output_fingerprint(file_hashes: dict[str, str], summary: dict[str, Any]) -> str:
    """Fingerprint exact output file bytes plus the normalized deterministic summary."""
    payload = {
        "algorithm": OUTPUT_FINGERPRINT_ALGORITHM,
        "files": [
            {"path": path, "sha256": digest}
            for path, digest in sorted(file_hashes.items(), key=lambda item: item[0])
        ],
        "summary": _normalized_summary(summary),
    }
    return hashlib.sha256(stable_json(payload).encode("utf-8")).hexdigest()


def _write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    """Stream a stable UTF-8 CSV manifest directly to disk."""
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=MANIFEST_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _output_payload_hashes(rows: list[dict[str, Any]], staging: Path) -> dict[str, str]:
    """Stream-hash every exact file covered by the canonical output fingerprint."""
    hashes = {
        relative: sha256_file(staging / relative)
        for relative in (
            "data.yaml",
            "metadata/source_data.yaml",
            "metadata/tile_manifest.csv",
        )
    }
    for row in rows:
        if row["category"] == "ambiguous":
            continue
        image_path = f"images/{row['split']}/{row['tile_file']}"
        label_path = f"labels/{row['split']}/{Path(row['tile_file']).stem}.txt"
        hashes[image_path] = sha256_file(staging / image_path)
        hashes[label_path] = sha256_file(staging / label_path)
    return hashes


def _read_recorded_source_fingerprint(source: Path) -> str | None:
    """Read an existing source dataset fingerprint without recomputing or changing it."""
    path = source / "metadata" / "dataset_fingerprint.sha256"
    if not path.is_file():
        return None
    token = path.read_text(encoding="utf-8-sig").strip().split()[0]
    if not re.fullmatch(r"[0-9a-fA-F]{64}", token):
        raise TilingError(f"invalid recorded source fingerprint: {path}")
    return token.casefold()


def _manifest_row(
    pair: SourcePair,
    source_identity: SourceIdentity,
    width: int,
    height: int,
    tile_x: int,
    tile_y: int,
    tile_size: int,
    tile_file: str,
    decision: dict[str, Any],
    source_config_identity: str,
    source_config_selection: str,
    source_config_path: Path,
    source_config_sha256: str,
    source_config_relative: str | None,
    source_config_external: str | None,
) -> dict[str, Any]:
    """Build one complete manifest row."""
    valid_width, valid_height = min(tile_size, width - tile_x), min(tile_size, height - tile_y)
    padding_right, padding_bottom = tile_size - valid_width, tile_size - valid_height
    retained, failed, intersecting = decision["retained"], decision["failed"], decision["intersecting"]
    return {
        "split": pair.split,
        "tile_file": tile_file,
        "source_image": pair.image_relative,
        "source_image_hash": hashlib.sha256(pair.image_relative.encode("utf-8")).hexdigest()[:16],
        "source_image_sha256": source_identity.image_sha256,
        "source_label": pair.label_relative,
        "source_label_sha256": source_identity.label_sha256,
        "source_data_config": source_config_identity,
        "source_data_config_selection": source_config_selection,
        "source_data_config_path": str(source_config_path),
        "source_data_config_sha256": source_config_sha256,
        "source_data_config_relative": source_config_relative or "",
        "source_data_config_external": source_config_external or "",
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
        "visibility": stable_json(
            {record["source_bbox_id"]: record["visibility"] for record in intersecting}
        ),
        "rejected_reason": stable_json(
            sorted({reason for record in failed for reason in record["reasons"]})
        ),
        "source_boxes": stable_json(intersecting),
        "retained_box_records": stable_json(retained),
        "failed_boxes": stable_json(failed),
        "output_image_sha256": "",
        "output_label_sha256": "",
    }


def _initialize_split_stats(available: int, processed: int, source_label_boxes: int) -> dict[str, int]:
    """Create counters required in summary.json."""
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


def _verify_full_source_unchanged(
    all_pairs: dict[str, list[SourcePair]],
    identity_by_image: dict[str, SourceIdentity],
    source_config_path: Path,
    source_config_relative: str,
    source_config_sha256: str,
    expected_full_fingerprint: str,
) -> None:
    """Recalculate the entire source fingerprint after processing and fail on any change."""
    final_identities: list[SourceIdentity] = []
    for pair in (pair for split in SPLITS for pair in all_pairs[split]):
        expected_identity = identity_by_image[pair.image_relative]
        final_identity = SourceIdentity(
            pair.split,
            pair.image_relative,
            sha256_file(pair.image_path),
            pair.label_relative,
            sha256_file(pair.label_path),
        )
        final_identities.append(final_identity)
        if final_identity != expected_identity:
            raise TilingError(f"source input changed during build: {pair.image_relative}")
    final_config_sha256 = sha256_file(source_config_path)
    if final_config_sha256 != source_config_sha256:
        raise TilingError(f"source data configuration changed during build: {source_config_path}")
    final_fingerprint = _source_fingerprint(
        final_identities, source_config_relative, final_config_sha256
    )
    if final_fingerprint != expected_full_fingerprint:
        raise TilingError("full source dataset fingerprint changed during build")


def build_tiled_dataset(raw_config: BuildConfig) -> dict[str, Any]:
    """Validate, analyze, and optionally build a deterministic tiled dataset."""
    config = _validate_config(raw_config)
    stride = compute_stride(config.tile_size, config.overlap)
    source_layout = _load_source_config(config.source, config.data_yaml)
    source_config_path = source_layout.config_path
    source_config_content = source_layout.config_content
    source_config_relative = source_layout.config_identity
    source_config_sha256 = hashlib.sha256(source_config_content).hexdigest()
    recorded_source_fingerprint = _read_recorded_source_fingerprint(config.source)
    all_pairs = {
        split: collect_source_pairs(
            config.source,
            split,
            source_layout.image_roots[split],
            source_layout.label_roots[split],
        )
        for split in SPLITS
    }
    all_source_identities: list[SourceIdentity] = []
    source_box_counts: dict[str, int] = {}
    for split in SPLITS:
        for pair in all_pairs[split]:
            identity = SourceIdentity(
                split,
                pair.image_relative,
                sha256_file(pair.image_path),
                pair.label_relative,
                sha256_file(pair.label_path),
            )
            image = _open_validated_image(pair.image_path)
            boxes = load_yolo_labels(pair.label_path, pair.image_relative, image.width, image.height)
            all_source_identities.append(identity)
            source_box_counts[pair.image_relative] = len(boxes)
    full_source_fingerprint = _source_fingerprint(
        all_source_identities, source_config_relative, source_config_sha256
    )
    identity_by_image = {identity.image_relative: identity for identity in all_source_identities}
    selected_pairs = {
        split: _select_pairs(all_pairs[split], config.max_images_per_split, config.seed, split) for split in SPLITS
    }
    processed_source_identities = [
        identity_by_image[pair.image_relative] for split in SPLITS for pair in selected_pairs[split]
    ]
    processed_subset_fingerprint = _source_fingerprint(
        processed_source_identities, source_config_relative, source_config_sha256
    )
    split_stats = {
        split: _initialize_split_stats(
            len(all_pairs[split]),
            len(selected_pairs[split]),
            sum(source_box_counts[pair.image_relative] for pair in selected_pairs[split]),
        )
        for split in SPLITS
    }
    rejection_reasons: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    tile_filenames: set[str] = set()
    staging = config.output.parent / f".{config.output.name}.building-{os.getpid()}"
    if staging.exists():
        raise TilingError(f"staging path already exists: {staging}")
    staging_created_by_this_run = False

    try:
        if not config.dry_run:
            staging.mkdir(parents=True, exist_ok=False)
            staging_created_by_this_run = True
            for split in SPLITS:
                (staging / "images" / split).mkdir(parents=True, exist_ok=False)
                (staging / "labels" / split).mkdir(parents=True, exist_ok=False)
            (staging / "metadata").mkdir(parents=True, exist_ok=False)

        for split in SPLITS:
            for pair in selected_pairs[split]:
                identity = identity_by_image[pair.image_relative]
                image = _open_validated_image(pair.image_path)
                boxes = load_yolo_labels(pair.label_path, pair.image_relative, image.width, image.height)
                anchors = generate_tile_anchors(image.width, image.height, config.tile_size, stride)
                for tile_x, tile_y in anchors:
                    decision = classify_tile(
                        boxes,
                        tile_x,
                        tile_y,
                        config.tile_size,
                        config.min_visibility,
                        config.min_box_size,
                    )
                    tile_file = _tile_name(
                        pair, tile_x, tile_y, config.tile_size, config.image_format
                    )
                    tile_key = tile_file.casefold()
                    if tile_key in tile_filenames:
                        raise TilingError(f"duplicate output tile filename before write: {tile_file}")
                    tile_filenames.add(tile_key)
                    row = _manifest_row(
                        pair,
                        identity,
                        image.width,
                        image.height,
                        tile_x,
                        tile_y,
                        config.tile_size,
                        tile_file,
                        decision,
                        source_config_relative,
                        source_layout.selection,
                        source_config_path,
                        source_config_sha256,
                        source_layout.config_relative,
                        source_layout.config_external,
                    )
                    stats = split_stats[split]
                    stats["candidate_tiles"] += 1
                    stats[decision["category"]] += 1
                    stats["intersecting_boxes"] += len(decision["intersecting"])
                    stats["retained_boxes"] += len(decision["retained"])
                    stats["failed_boxes"] += len(decision["failed"])
                    for failed_box in decision["failed"]:
                        rejection_reasons.update(failed_box["reasons"])
                    if decision["category"] == "safe_positive":
                        stats["output_label_boxes"] += len(decision["retained"])
                    elif decision["category"] == "safe_negative":
                        stats["empty_labels"] += 1

                    if not config.dry_run and decision["category"] != "ambiguous":
                        output_image = staging / "images" / split / tile_file
                        output_label = staging / "labels" / split / f"{Path(tile_file).stem}.txt"
                        tile, _, _ = _crop_with_padding(image, tile_x, tile_y, config.tile_size)
                        _save_image(
                            tile, output_image, config.image_format, config.jpeg_quality
                        )
                        label_content = _label_bytes(decision["retained"])
                        output_label.write_bytes(label_content)
                        row["output_image_sha256"] = sha256_file(output_image)
                        row["output_label_sha256"] = hashlib.sha256(label_content).hexdigest()
                    rows.append(row)

        summary: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "parameters": {
                **asdict(config),
                "source": str(config.source),
                "output": str(config.output),
                "data_yaml": str(config.data_yaml) if config.data_yaml is not None else None,
                "stride": stride,
                "padding_value": 114,
                "png_compress_level": 6 if config.image_format == "png" else None,
                "jpeg_subsampling": 0 if config.image_format == "jpg" else None,
                "jpeg_optimize": False if config.image_format == "jpg" else None,
                "jpeg_progressive": False if config.image_format == "jpg" else None,
                "padding": {"mode": "right_bottom_constant", "value": 114},
                "visibility_comparison": "retain_if_visibility_greater_than_or_equal_to_threshold",
                "max_tile_filename_chars": MAX_TILE_FILENAME_CHARS,
                "max_tile_stem_chars": MAX_TILE_STEM_CHARS,
                "source_path_hash_chars": SOURCE_PATH_HASH_CHARS,
            },
            "versions": {
                "python": platform.python_version(),
                "pillow": PIL.__version__,
            },
            "splits": split_stats,
            "totals": {
                key: sum(stats[key] for stats in split_stats.values())
                for key in (
                    "source_images_available",
                    "source_images_processed",
                    "source_label_boxes",
                    "candidate_tiles",
                    "safe_positive",
                    "safe_negative",
                    "ambiguous",
                    "intersecting_boxes",
                    "retained_boxes",
                    "failed_boxes",
                    "output_label_boxes",
                    "empty_labels",
                )
            },
            "rejection_reasons": dict(sorted(rejection_reasons.items())),
            "source_data_config": source_config_relative,
            "source_data_config_path": str(source_config_path),
            "source_data_config_selection": source_layout.selection,
            "source_data_config_sha256": source_config_sha256,
            "source_data_config_relative": source_layout.config_relative,
            "source_data_config_external": source_layout.config_external,
            "source_data_layout": {
                "dataset_root": str(source_layout.dataset_root),
                "images": {split: str(source_layout.image_roots[split]) for split in SPLITS},
                "labels": {split: str(source_layout.label_roots[split]) for split in SPLITS},
                "nc": source_layout.nc,
                "names": {str(key): value for key, value in source_layout.names.items()},
            },
            "recorded_source_dataset_fingerprint": recorded_source_fingerprint,
            "full_source_fingerprint": full_source_fingerprint,
            "processed_subset_fingerprint": processed_subset_fingerprint,
            "processed_source_images": [
                identity.image_relative for identity in processed_source_identities
            ],
            "fingerprint_normalization": {
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
            },
            "output_dataset_fingerprint": None,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "input_modified": False,
            "dry_run": config.dry_run,
        }
        if config.dry_run:
            _verify_full_source_unchanged(
                all_pairs,
                identity_by_image,
                source_config_path,
                source_config_relative,
                source_config_sha256,
                full_source_fingerprint,
            )
            return summary

        data_yaml = (
            "path: .\n"
            "train: images/train\n"
            "val: images/val\n"
            "test: images/test\n"
            "names:\n"
            "  0: crack\n"
        ).encode("utf-8")
        (staging / "data.yaml").write_bytes(data_yaml)
        (staging / "metadata" / "source_data.yaml").write_bytes(source_config_content)
        _write_manifest(staging / "metadata" / "tile_manifest.csv", rows)
        output_fingerprint = _canonical_output_fingerprint(
            _output_payload_hashes(rows, staging), summary
        )
        summary["output_dataset_fingerprint"] = output_fingerprint
        write_text_lf(
            staging / "metadata" / "dataset_fingerprint.sha256",
            f"{output_fingerprint}  tiled-dataset-v2\n",
        )
        write_text_lf(
            staging / "metadata" / "summary.json",
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        )
        _verify_full_source_unchanged(
            all_pairs,
            identity_by_image,
            source_config_path,
            source_config_relative,
            source_config_sha256,
            full_source_fingerprint,
        )
        staging.replace(config.output)
        return summary
    except Exception as error:
        if staging_created_by_this_run:
            raise TilingError(
                f"{error}; staging retained at {staging}; "
                "manual confirmation is required before cleanup"
            ) from error
        raise


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path, help="dataset root directory")
    parser.add_argument(
        "--data-yaml",
        type=Path,
        help=(
            "explicit source dataset YAML; when provided it is used exclusively instead of "
            "automatic data.yaml/data_local.yaml discovery"
        ),
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--overlap", type=float, default=0.20)
    parser.add_argument("--min-visibility", type=float, default=0.50)
    parser.add_argument("--min-box-size", type=float, default=2.0)
    parser.add_argument("--image-format", choices=("png", "jpg"), default="png")
    parser.add_argument("--jpeg-quality", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-images-per-split", type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    """Run the command-line interface."""
    args = parse_args()
    try:
        summary = build_tiled_dataset(
            BuildConfig(
                source=args.source,
                output=args.output,
                data_yaml=args.data_yaml,
                tile_size=args.tile_size,
                overlap=args.overlap,
                min_visibility=args.min_visibility,
                min_box_size=args.min_box_size,
                image_format=args.image_format,
                jpeg_quality=args.jpeg_quality,
                seed=args.seed,
                max_images_per_split=args.max_images_per_split,
                dry_run=args.dry_run,
            )
        )
    except TilingError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
