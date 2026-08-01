# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Build a reproducible, auditable augment-first tunnel crack dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import random
import shutil
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import albumentations as A
import cv2
import numpy as np
import PIL
import yaml
from PIL import Image, ImageDraw, ImageFont


GLOBAL_SEED = 42
FULL_SOURCE_COUNT = 2404
FULL_SOURCE_BOXES = 2941
VARIANT_TYPES = ("orig", "geo", "light", "degrade", "compound")
SPLIT_RATIOS = (("train", 0.70), ("val", 0.20), ("test", 0.10))
FULL_SPLIT_COUNTS = {"train": 8414, "val": 2404, "test": 1202}
DRY_SPLIT_COUNTS = {"train": 175, "val": 50, "test": 25}
JPEG_QUALITY = 95
MAX_AUGMENT_RETRIES = 12
PREVIEWS_PER_VARIANT = 6
NEAR_DUPLICATE_PREVIEW_PAIRS = 6
MANIFEST_SCHEMA_VERSION = "3.0"
SMALL_ARRAY_VALUE_LIMIT = 256
FALLBACK_POLICY_VERSION = "adaptive_light_fallback_v1"
POLICY = {
    "split_policy": "file_level_random",
    "parent_id_grouping": False,
    "independent_test_set": False,
}

GEO_QUOTAS = (
    ("horizontal_flip", 721),
    ("vertical_flip", 361),
    ("rotate_180", 240),
    ("affine_or_perspective", 1082),
)
LIGHT_QUOTAS = (
    ("local_shadow_or_nonuniform", 721),
    ("vignette_or_directional_gradient", 601),
    ("low_light_or_gamma", 601),
    ("clahe_or_local_contrast", 481),
)
DEGRADE_QUOTAS = (
    ("motion_or_defocus_blur", 721),
    ("gaussian_or_iso_noise", 601),
    ("jpeg_or_downsample", 601),
    ("mild_mixed_degrade", 481),
)

SOURCE_MANIFEST_FIELDS = (
    "selection_index",
    "source_image",
    "source_label",
    "source_split",
    "parent_id",
    "source_image_sha256",
    "source_label_sha256",
    "bbox_count",
    "image_width",
    "image_height",
)
AUGMENTATION_MANIFEST_FIELDS = (
    "manifest_schema_version",
    "source_image",
    "source_label",
    "source_split",
    "parent_id",
    "output_image",
    "output_label",
    "variant_type",
    "variant_subtype",
    "global_seed",
    "sample_seed",
    "augmentation_parameters_json",
    "bbox_count_before",
    "bbox_count_after",
    "source_image_sha256",
    "output_image_sha256",
    "output_label_sha256",
    "image_width",
    "image_height",
)
SPLIT_MANIFEST_FIELDS = (
    "output_image",
    "parent_id",
    "variant_type",
    "pre_shuffle_index",
    "post_shuffle_index",
    "split_seed",
    "split",
)


class BuildStop(RuntimeError):
    """Signal a condition that forbids completing the dataset build."""


class AuditableGaussianNoise(A.ImageOnlyTransform):
    """Add Gaussian noise from explicitly recorded distribution parameters and seed."""

    def __init__(
        self,
        std_range: tuple[float, float],
        mean_range: tuple[float, float] = (0.0, 0.0),
        p: float = 1.0,
    ) -> None:
        super().__init__(p=p)
        self.std_range = std_range
        self.mean_range = mean_range

    def apply(
        self,
        image: np.ndarray,
        sigma_fraction: float,
        mean_fraction: float,
        random_seed: int,
        **params: Any,
    ) -> np.ndarray:
        """Apply one replayable per-channel Gaussian noise realization."""
        del params
        maximum = 255.0 if image.dtype == np.uint8 else 1.0
        generator = np.random.default_rng(random_seed)
        noise = generator.normal(
            mean_fraction * maximum,
            sigma_fraction * maximum,
            size=image.shape,
        ).astype(np.float32)
        return np.clip(image.astype(np.float32) + noise, 0.0, maximum).astype(image.dtype)

    def get_params_dependent_on_data(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
    ) -> dict[str, Any]:
        """Sample and expose every parameter required to reconstruct the noise."""
        del data
        return {
            "shape": params["shape"],
            "distribution": "gaussian",
            "sigma_fraction": self.py_random.uniform(*self.std_range),
            "mean_fraction": self.py_random.uniform(*self.mean_range),
            "random_seed": int(self.random_generator.integers(0, 2**32, dtype=np.uint32)),
        }

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        """Expose constructor values for Albumentations serialization."""
        return ("std_range", "mean_range")


@dataclass(frozen=True)
class SourceRecord:
    """One authoritative image/label pair from the current repaired NoAug dataset."""

    parent_id: str
    source_split: str
    image_path: Path
    label_path: Path
    output_image_relative: str
    output_label_relative: str
    raw_source_image: str
    raw_source_image_relative: str
    image_sha256: str
    label_sha256: str
    width: int
    height: int
    boxes: tuple[tuple[float, float, float, float], ...]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments with full generation explicitly locked."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path, help="Current repaired NoAug dataset root")
    parser.add_argument("--output", required=True, type=Path, help="Absent dry-run or full output directory")
    parser.add_argument("--seed", default=GLOBAL_SEED, type=int)
    parser.add_argument(
        "--full",
        action="store_true",
        help="Explicitly unlock all 2,404 parents and 12,020 outputs; omitted means exactly 50 parents",
    )
    parser.add_argument(
        "--preflight-collect-all",
        action="store_true",
        help="Check all 2,404 light and compound targets in memory and write only audit reports",
    )
    return parser.parse_args()


def sha256_bytes(data: bytes) -> str:
    """Return a SHA-256 digest for bytes."""
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(4 * 1024**2):
            digest.update(chunk)
    return digest.hexdigest()


def sample_seed(global_seed: int, parent_id: str, variant_type: str) -> int:
    """Derive a stable uint64 seed without Python's process-randomized hash()."""
    payload = f"{global_seed}\0{parent_id}\0{variant_type}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def attempt_seed(stable_sample_seed: int, attempt: int) -> int:
    """Derive a stable uint32 resampling seed from a sample seed and attempt number."""
    payload = f"{stable_sample_seed}\0attempt\0{attempt}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def fallback_seed(stable_sample_seed: int) -> int:
    """Derive the independent deterministic fallback seed after all normal attempts are exhausted."""
    payload = f"{stable_sample_seed}\0{FALLBACK_POLICY_VERSION}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def component_seed(stable_fallback_seed: int, component: str) -> int:
    """Derive one deterministic fallback-component seed without shared random state."""
    payload = f"{stable_fallback_seed}\0{component}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def largest_remainder_counts(total: int, ratios: tuple[tuple[str, float], ...] = SPLIT_RATIOS) -> dict[str, int]:
    """Allocate an exact total with deterministic largest remainders."""
    if total <= 0:
        raise ValueError("total must be positive")
    quotas = [(name, total * ratio) for name, ratio in ratios]
    counts = {name: math.floor(value) for name, value in quotas}
    order = sorted(range(len(quotas)), key=lambda index: (-(quotas[index][1] % 1), index))
    for index in order[: total - sum(counts.values())]:
        counts[quotas[index][0]] += 1
    return counts


def scaled_subtype_quotas(full_quotas: tuple[tuple[str, int], ...], target: int) -> dict[str, int]:
    """Scale fixed 2,404-parent quotas with exact integer largest remainders."""
    if target <= 0 or target > FULL_SOURCE_COUNT:
        raise ValueError(f"target must be in [1, {FULL_SOURCE_COUNT}]")
    floors = {name: quota * target // FULL_SOURCE_COUNT for name, quota in full_quotas}
    remainder_count = target - sum(floors.values())
    ranked = sorted(
        enumerate(full_quotas),
        key=lambda item: (-(item[1][1] * target % FULL_SOURCE_COUNT), item[0]),
    )
    for _, (name, _) in ranked[:remainder_count]:
        floors[name] += 1
    return floors


def parse_yolo_label(path: Path, *, extent_tolerance: float = 1e-8) -> tuple[tuple[float, float, float, float], ...]:
    """Read a nonempty, finite, class-0 YOLO label with valid normalized boxes."""
    try:
        lines = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    except (OSError, UnicodeError) as error:
        raise BuildStop(f"unable to read label {path}: {error}") from error
    if not lines:
        raise BuildStop(f"empty label: {path}")
    boxes: list[tuple[float, float, float, float]] = []
    for line_number, line in enumerate(lines, 1):
        tokens = line.split()
        if len(tokens) != 5 or tokens[0] != "0":
            raise BuildStop(f"expected class-0 five-column label at {path}:{line_number}: {line!r}")
        try:
            box = tuple(float(value) for value in tokens[1:])
        except ValueError as error:
            raise BuildStop(f"nonnumeric bbox at {path}:{line_number}: {line!r}") from error
        if not all(math.isfinite(value) for value in box):
            raise BuildStop(f"nonfinite bbox at {path}:{line_number}: {line!r}")
        x, y, width, height = box
        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < width <= 1 and 0 < height <= 1):
            raise BuildStop(f"invalid normalized bbox at {path}:{line_number}: {line!r}")
        left, top, right, bottom = x - width / 2, y - height / 2, x + width / 2, y + height / 2
        if min(left, top) < -extent_tolerance or max(right, bottom) > 1 + extent_tolerance:
            raise BuildStop(f"out-of-bounds bbox at {path}:{line_number}: {line!r}")
        boxes.append((x, y, width, height))
    return tuple(boxes)


def load_source_records(source: Path) -> list[SourceRecord]:
    """Load only current NoAug manifest rows and current repaired class-0 labels."""
    manifest_path = source / "metadata" / "split_manifest.csv"
    if not manifest_path.is_file():
        raise BuildStop(f"missing authoritative manifest: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    records: list[SourceRecord] = []
    parent_ids: set[str] = set()
    for row in rows:
        parent_id = row["canonical_id"]
        if parent_id.casefold() in parent_ids:
            raise BuildStop(f"duplicate canonical_id: {parent_id}")
        parent_ids.add(parent_id.casefold())
        if row["output_class_id"] != "0":
            raise BuildStop(f"source manifest contains nonzero output class: {parent_id}")
        image_path, label_path = source / row["output_image"], source / row["output_label"]
        if not image_path.is_file() or not label_path.is_file():
            raise BuildStop(f"missing current NoAug pair: {parent_id}")
        boxes = parse_yolo_label(label_path)
        if len(boxes) != int(row["crack_box_count"]):
            raise BuildStop(f"source bbox count disagrees with manifest: {parent_id}")
        records.append(
            SourceRecord(
                parent_id=parent_id,
                source_split=row["split"],
                image_path=image_path,
                label_path=label_path,
                output_image_relative=row["output_image"],
                output_label_relative=row["output_label"],
                raw_source_image=row["source_image"],
                raw_source_image_relative=row["source_image_relative"],
                image_sha256=row["image_sha256"],
                label_sha256=sha256_file(label_path),
                width=int(row["image_width"]),
                height=int(row["image_height"]),
                boxes=boxes,
            )
        )
    box_count = sum(len(record.boxes) for record in records)
    if len(records) != FULL_SOURCE_COUNT or box_count != FULL_SOURCE_BOXES:
        raise BuildStop(f"authoritative source totals changed: images={len(records)}, boxes={box_count}")
    return records


def source_dataset_fingerprint(records: list[SourceRecord]) -> str:
    """Recompute the current NoAug fingerprint from actual image and label bytes."""
    digest = hashlib.sha256()
    for record in sorted(records, key=lambda item: item.parent_id):
        actual_image_sha256 = sha256_file(record.image_path)
        actual_label_sha256 = sha256_file(record.label_path)
        if actual_image_sha256 != record.image_sha256:
            raise BuildStop(f"source image SHA-256 changed: {record.image_path}")
        if actual_label_sha256 != record.label_sha256:
            raise BuildStop(f"source label SHA-256 changed: {record.label_path}")
        fields = (
            record.parent_id,
            record.source_split,
            record.raw_source_image_relative,
            record.output_image_relative,
            actual_image_sha256,
            actual_label_sha256,
        )
        digest.update(("\0".join(fields) + "\n").encode())
    return digest.hexdigest()


def select_sources(records: list[SourceRecord], count: int, seed: int) -> list[SourceRecord]:
    """Deterministically sample parents from a stable source-relative-path ordering."""
    ordered = sorted(records, key=lambda item: item.output_image_relative.casefold())
    if count == len(ordered):
        return ordered
    selected = random.Random(seed).sample(ordered, count)
    return sorted(selected, key=lambda item: item.output_image_relative.casefold())


def assign_subtypes(
    records: list[SourceRecord], variant_type: str, full_quotas: tuple[tuple[str, int], ...], seed: int
) -> tuple[dict[str, str], dict[str, int]]:
    """Assign deterministic subtype quotas independently of the final file-level split."""
    counts = scaled_subtype_quotas(full_quotas, len(records))
    ordered = sorted(records, key=lambda item: item.output_image_relative.casefold())
    quota_seed = sample_seed(seed, "quota_assignment", variant_type)
    random.Random(quota_seed).shuffle(ordered)
    mapping: dict[str, str] = {}
    cursor = 0
    for subtype, _ in full_quotas:
        for record in ordered[cursor : cursor + counts[subtype]]:
            mapping[record.parent_id] = subtype
        cursor += counts[subtype]
    if len(mapping) != len(records):
        raise BuildStop(f"incomplete {variant_type} subtype assignment")
    return mapping, counts


def read_image(path: Path) -> np.ndarray:
    """Decode a BGR image without relying on OpenCV Windows Unicode paths."""
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise BuildStop(f"unable to decode image: {path}")
    return image


def encode_jpeg(image: np.ndarray, quality: int = JPEG_QUALITY) -> bytes:
    """Encode deterministic JPEG bytes with a fixed quality."""
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise BuildStop("unable to encode JPEG")
    return encoded.tobytes()


def json_safe(value: Any, large_array_reconstruction: dict[str, Any] | None = None) -> Any:
    """Serialize small numeric arrays exactly and summarize only image-sized arrays."""
    if isinstance(value, np.ndarray):
        if value.size <= SMALL_ARRAY_VALUE_LIMIT:
            return value.tolist()
        summary = {
            "ndarray_shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": sha256_bytes(value.tobytes()),
            "minimum": float(value.min()),
            "maximum": float(value.max()),
            "empirical_mean": float(value.mean()),
            "empirical_std": float(value.std()),
        }
        if large_array_reconstruction is None:
            raise BuildStop("large ndarray lacks public reconstruction parameters")
        summary["reconstruction"] = json_safe(large_array_reconstruction)
        return summary
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {
            str(key): json_safe(item, large_array_reconstruction=large_array_reconstruction)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [json_safe(item, large_array_reconstruction=large_array_reconstruction) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def geometry_bbox_params() -> A.BboxParams:
    """Return the single bbox policy used by geo and compound transformations."""
    return A.BboxParams(
        format="yolo",
        label_fields=["class_labels"],
        min_visibility=0.60,
        check_each_transform=True,
        clip=True,
        filter_invalid_bboxes=True,
    )


def subtle_light_transform() -> A.OneOf:
    """Return the mandatory very-light photometric change for geo outputs."""
    return A.OneOf(
        [
            A.RandomBrightnessContrast(brightness_limit=(-0.05, 0.05), contrast_limit=(-0.05, 0.05), p=1),
            A.RandomGamma(gamma_limit=(95, 105), p=1),
        ],
        p=1,
    )


def soft_shadow_parameters(image_shape: tuple[int, ...], seed: int) -> dict[str, Any]:
    """Create one deterministic, wide, continuously blurred tunnel-shadow field."""
    height, width = image_shape[:2]
    rng = random.Random(seed)
    center = (round(rng.uniform(0.20, 0.80) * width), round(rng.uniform(0.20, 0.80) * height))
    axes = (round(rng.uniform(0.22, 0.38) * width), round(rng.uniform(0.16, 0.30) * height))
    blur_sigma = rng.uniform(0.035, 0.070) * min(width, height)
    return {
        "model": "gaussian_blurred_elliptical_luminance_attenuation",
        "center_pixels": list(center),
        "center_normalized": [center[0] / width, center[1] / height],
        "semi_axes_pixels": list(axes),
        "semi_axes_normalized": [axes[0] / width, axes[1] / height],
        "rotation_degrees": rng.uniform(-80.0, 80.0),
        "attenuation": rng.uniform(0.14, 0.28),
        "blur_sigma_pixels": blur_sigma,
        "blur_sigma_normalized": blur_sigma / min(width, height),
        "coverage_threshold": 0.10,
    }


def apply_soft_shadow(image: np.ndarray, parameters: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Darken only the LAB luminance channel with a smooth, locally supported multiplicative field."""
    height, width = image.shape[:2]
    center = tuple(int(value) for value in parameters["center_pixels"])
    axes = tuple(int(value) for value in parameters["semi_axes_pixels"])
    mask = np.zeros((height, width), dtype=np.float32)
    cv2.ellipse(
        mask,
        center,
        axes,
        float(parameters["rotation_degrees"]),
        0.0,
        360.0,
        1.0,
        thickness=-1,
        lineType=cv2.LINE_AA,
    )
    mask = cv2.GaussianBlur(
        mask,
        (0, 0),
        sigmaX=float(parameters["blur_sigma_pixels"]),
        sigmaY=float(parameters["blur_sigma_pixels"]),
        borderType=cv2.BORDER_REFLECT_101,
    )
    peak = float(mask.max())
    if peak <= 0.0:
        raise BuildStop("soft shadow field is empty")
    mask /= peak
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[:, :, 0] *= 1.0 - float(parameters["attenuation"]) * mask
    output = cv2.cvtColor(np.clip(lab, 0.0, 255.0).astype(np.uint8), cv2.COLOR_LAB2BGR)
    actual = dict(parameters)
    actual["coverage_ratio"] = float(np.mean(mask >= float(parameters["coverage_threshold"])))
    actual["peak_field_value"] = float(mask.max())
    actual["luminance_operation"] = "L_out=L_source*(1-attenuation*blurred_field)"
    return output, actual


def build_pipeline(variant_type: str, subtype: str, seed: int) -> A.Compose:
    """Build one seeded Albumentations pipeline within the approved parameter ranges."""
    rng = random.Random(seed)
    bbox_params: A.BboxParams | None = None
    if variant_type == "geo":
        bbox_params = geometry_bbox_params()
        if subtype == "horizontal_flip":
            transforms = [A.HorizontalFlip(p=1), subtle_light_transform()]
        elif subtype == "vertical_flip":
            transforms = [A.VerticalFlip(p=1), subtle_light_transform()]
        elif subtype == "rotate_180":
            transforms = [A.HorizontalFlip(p=1), A.VerticalFlip(p=1), subtle_light_transform()]
        elif subtype == "affine_or_perspective":
            if rng.random() < 0.75:
                main = A.Affine(
                    scale=(0.90, 1.10),
                    translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                    rotate=(-8.0, 8.0),
                    shear={"x": (-3.0, 3.0), "y": (-3.0, 3.0)},
                    border_mode=cv2.BORDER_REFLECT_101,
                    p=1,
                )
            else:
                main = A.Perspective(
                    scale=(0.005, 0.015),
                    keep_size=True,
                    fit_output=False,
                    border_mode=cv2.BORDER_REFLECT_101,
                    p=1,
                )
            transforms = [main, subtle_light_transform()]
        else:
            raise ValueError(f"unknown geo subtype: {subtype}")
    elif variant_type == "light":
        if subtype == "local_shadow_or_nonuniform":
            raise ValueError("local shadows use the explicit soft luminance-field implementation")
        elif subtype == "vignette_or_directional_gradient":
            transforms = [
                A.Illumination(
                    mode=rng.choice(("linear", "corner", "gaussian")),
                    intensity_range=(0.08, 0.20),
                    effect_type=rng.choice(("brighten", "darken", "both")),
                    p=1,
                )
            ]
        elif subtype == "low_light_or_gamma":
            transforms = [
                A.OneOf(
                    [
                        A.RandomGamma(gamma_limit=(70, 135), p=1),
                        A.RandomBrightnessContrast(
                            brightness_limit=(-0.25, -0.08), contrast_limit=(-0.20, 0.20), p=1
                        ),
                    ],
                    p=1,
                )
            ]
        elif subtype == "clahe_or_local_contrast":
            transforms = [
                A.OneOf(
                    [
                        A.CLAHE(clip_limit=(1.5, 3.0), tile_grid_size=(8, 8), p=1),
                        A.RandomBrightnessContrast(
                            brightness_limit=(0.0, 0.0), contrast_limit=(0.08, 0.20), p=1
                        ),
                    ],
                    p=1,
                )
            ]
        else:
            raise ValueError(f"unknown light subtype: {subtype}")
    elif variant_type == "degrade":
        if subtype == "motion_or_defocus_blur":
            transforms = [
                A.OneOf(
                    [A.MotionBlur(blur_limit=(3, 7), p=1), A.Defocus(radius=(3, 5), alias_blur=(0.1, 0.35), p=1)],
                    p=1,
                )
            ]
        elif subtype == "gaussian_or_iso_noise":
            transforms = [
                A.OneOf(
                    [
                        AuditableGaussianNoise(std_range=(0.015, 0.06), mean_range=(0.0, 0.0), p=1),
                        A.ISONoise(color_shift=(0.01, 0.04), intensity=(0.10, 0.35), p=1),
                    ],
                    p=1,
                )
            ]
        elif subtype == "jpeg_or_downsample":
            transforms = [
                A.OneOf(
                    [
                        A.ImageCompression(quality_range=(50, 90), p=1),
                        A.Downscale(
                            scale_range=(0.65, 0.90),
                            interpolation_pair={"downscale": cv2.INTER_AREA, "upscale": cv2.INTER_LINEAR},
                            p=1,
                        ),
                    ],
                    p=1,
                )
            ]
        elif subtype == "mild_mixed_degrade":
            transforms = [
                A.OneOf(
                    [
                        A.GaussianBlur(blur_limit=(3, 5), p=1),
                        AuditableGaussianNoise(std_range=(0.01, 0.03), p=1),
                    ],
                    p=1,
                ),
                A.ImageCompression(quality_range=(72, 90), p=1),
            ]
        else:
            raise ValueError(f"unknown degrade subtype: {subtype}")
    elif variant_type == "compound":
        bbox_params = geometry_bbox_params()
        transforms = [
            A.OneOf(
                [
                    A.Affine(
                        scale=(0.96, 1.04),
                        translate_percent={"x": (-0.03, 0.03), "y": (-0.03, 0.03)},
                        rotate=(-5.0, 5.0),
                        shear={"x": (-1.5, 1.5), "y": (-1.5, 1.5)},
                        border_mode=cv2.BORDER_REFLECT_101,
                        p=1,
                    ),
                    A.Perspective(
                        scale=(0.003, 0.009),
                        keep_size=True,
                        fit_output=False,
                        border_mode=cv2.BORDER_REFLECT_101,
                        p=1,
                    ),
                ],
                p=1,
            ),
            A.OneOf(
                [
                    A.RandomBrightnessContrast(
                        brightness_limit=(-0.15, 0.15), contrast_limit=(-0.12, 0.12), p=1
                    ),
                    A.RandomGamma(gamma_limit=(85, 115), p=1),
                    A.Illumination(mode="linear", intensity_range=(0.05, 0.12), effect_type="both", p=1),
                ],
                p=1,
            ),
            A.OneOf(
                [
                    A.GaussianBlur(blur_limit=(3, 5), p=1),
                    AuditableGaussianNoise(std_range=(0.01, 0.035), p=1),
                    A.ImageCompression(quality_range=(75, 90), p=1),
                    A.Downscale(
                        scale_range=(0.80, 0.92),
                        interpolation_pair={"downscale": cv2.INTER_AREA, "upscale": cv2.INTER_LINEAR},
                        p=1,
                    ),
                ],
                p=1,
            ),
        ]
    else:
        raise ValueError(f"pipeline is not defined for variant: {variant_type}")
    return A.Compose(
        transforms,
        bbox_params=bbox_params,
        seed=seed,
        strict=True,
        save_applied_params=True,
    )


def canonicalize_boxes(boxes: list[Any] | tuple[Any, ...]) -> tuple[tuple[float, float, float, float], ...]:
    """Clip sub-micro numerical edge drift and return stable normalized xywh boxes."""
    result: list[tuple[float, float, float, float]] = []
    for raw_box in boxes:
        x, y, width, height = (float(value) for value in raw_box[:4])
        if not all(math.isfinite(value) for value in (x, y, width, height)):
            raise BuildStop(f"transformation produced nonfinite bbox: {raw_box}")
        left, top, right, bottom = x - width / 2, y - height / 2, x + width / 2, y + height / 2
        overflow = max(0.0, -left, -top, right - 1.0, bottom - 1.0)
        if overflow > 1e-5:
            raise BuildStop(f"transformation produced bbox overflow {overflow}: {raw_box}")
        left, top = max(0.0, left), max(0.0, top)
        right, bottom = min(1.0, right), min(1.0, bottom)
        if right <= left or bottom <= top:
            raise BuildStop(f"transformation produced degenerate bbox: {raw_box}")
        stable = ((left + right) / 2, (top + bottom) / 2, right - left, bottom - top)
        result.append(tuple(float(f"{value:.10f}") for value in stable))
    return tuple(result)


def format_yolo_labels(boxes: tuple[tuple[float, float, float, float], ...]) -> bytes:
    """Format class-0 boxes with deterministic precision."""
    lines = ["0 " + " ".join(f"{value:.10f}" for value in box) for box in boxes]
    return ("\n".join(lines) + "\n").encode()


def _quality_components(
    image: np.ndarray, boxes: tuple[tuple[float, float, float, float], ...]
) -> dict[str, Any]:
    """Measure exposure, dynamic range, and crack-box visibility primitives."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    patch_means: list[float] = []
    patch_stds: list[float] = []
    patch_ranges: list[int] = []
    patch_edges: list[float] = []
    for x, y, box_width, box_height in boxes:
        left = max(0, min(width - 1, math.floor((x - box_width / 2) * width)))
        top = max(0, min(height - 1, math.floor((y - box_height / 2) * height)))
        right = max(left + 1, min(width, math.ceil((x + box_width / 2) * width)))
        bottom = max(top + 1, min(height, math.ceil((y + box_height / 2) * height)))
        patch = gray[top:bottom, left:right]
        patch_means.append(float(patch.mean()))
        patch_stds.append(float(patch.std()))
        patch_ranges.append(int(patch.max()) - int(patch.min()))
        gradient_x = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3)
        patch_edges.append(float(np.mean(cv2.magnitude(gradient_x, gradient_y))))
    low, high = np.percentile(gray, (1.0, 99.0))
    return {
        "mean": float(gray.mean()),
        "std": float(gray.std()),
        "dark_pixel_ratio": float(np.mean(gray <= 5)),
        "bright_pixel_ratio": float(np.mean(gray >= 250)),
        "dynamic_range_p01_p99": float(high - low),
        "bbox_means": patch_means,
        "bbox_stds": patch_stds,
        "bbox_ranges": patch_ranges,
        "bbox_edge_means": patch_edges,
        "minimum_bbox_mean": min(patch_means),
        "minimum_bbox_std": min(patch_stds),
        "minimum_bbox_range": min(patch_ranges),
        "minimum_bbox_edge_mean": min(patch_edges),
    }


def image_quality(
    image: np.ndarray,
    boxes: tuple[tuple[float, float, float, float], ...],
    source_image: np.ndarray | None = None,
    source_boxes: tuple[tuple[float, float, float, float], ...] | None = None,
    variant_type: str | None = None,
    subtype: str | None = None,
) -> dict[str, Any]:
    """Enforce absolute and source-relative exposure plus bbox visibility invariants."""
    metrics = _quality_components(image, boxes)
    checks = {
        "absolute_mean": 8.0 <= metrics["mean"] <= 247.0,
        "absolute_global_contrast": metrics["std"] >= 3.0,
        "not_near_black": metrics["dark_pixel_ratio"] < 0.80,
        "not_severely_overexposed": metrics["bright_pixel_ratio"] < 0.80,
        "bbox_dynamic_range": metrics["minimum_bbox_range"] >= 3,
        "bbox_local_contrast": metrics["minimum_bbox_std"] >= 1.0,
    }
    if source_image is not None:
        if source_boxes is None or len(source_boxes) != len(boxes):
            raise BuildStop("source-relative quality requires corresponding source boxes")
        source = _quality_components(source_image, source_boxes)
        bbox_mean_ratios = [
            output / max(original, 1.0) for output, original in zip(metrics["bbox_means"], source["bbox_means"])
        ]
        bbox_std_ratios = [
            output / max(original, 1.0) for output, original in zip(metrics["bbox_stds"], source["bbox_stds"])
        ]
        bbox_edge_ratios = [
            output / max(original, 0.5)
            for output, original in zip(metrics["bbox_edge_means"], source["bbox_edge_means"])
        ]
        relative = {
            "source_mean": source["mean"],
            "source_dark_pixel_ratio": source["dark_pixel_ratio"],
            "source_dynamic_range_p01_p99": source["dynamic_range_p01_p99"],
            "mean_ratio_to_source": metrics["mean"] / max(source["mean"], 1.0),
            "dark_pixel_ratio_increase": metrics["dark_pixel_ratio"] - source["dark_pixel_ratio"],
            "dynamic_range_ratio_to_source": metrics["dynamic_range_p01_p99"]
            / max(source["dynamic_range_p01_p99"], 1.0),
            "minimum_bbox_mean_ratio_to_source": min(bbox_mean_ratios),
            "minimum_bbox_contrast_ratio_to_source": min(bbox_std_ratios),
            "minimum_bbox_edge_ratio_to_source": min(bbox_edge_ratios),
            "bbox_visibility_score": min(min(bbox_mean_ratios), min(bbox_std_ratios), min(bbox_edge_ratios)),
        }
        metrics["relative_to_source"] = relative
        if variant_type == "light":
            thresholds = {
                "local_shadow_or_nonuniform": (0.72, 1.05, 0.18, 0.60, 0.55, 0.45, 0.50),
                "vignette_or_directional_gradient": (0.60, 1.40, 0.25, 0.50, 0.45, 0.35, 0.40),
                "low_light_or_gamma": (0.42, 1.25, 0.35, 0.35, 0.35, 0.30, 0.30),
                "clahe_or_local_contrast": (0.75, 1.35, 0.15, 0.65, 0.65, 0.55, 0.55),
            }[str(subtype)]
            minimum_mean, maximum_mean, maximum_dark_delta, minimum_dynamic, bbox_mean, bbox_std, bbox_edge = (
                thresholds
            )
            checks.update(
                {
                    "relative_mean": minimum_mean <= relative["mean_ratio_to_source"] <= maximum_mean,
                    "relative_dark_pixels": relative["dark_pixel_ratio_increase"] <= maximum_dark_delta,
                    "relative_dynamic_range": relative["dynamic_range_ratio_to_source"] >= minimum_dynamic,
                    "relative_bbox_mean": relative["minimum_bbox_mean_ratio_to_source"] >= bbox_mean,
                    "relative_bbox_contrast": relative["minimum_bbox_contrast_ratio_to_source"] >= bbox_std,
                    "relative_bbox_edges": relative["minimum_bbox_edge_ratio_to_source"] >= bbox_edge,
                }
            )
        elif variant_type == "degrade":
            checks.update(
                {
                    "relative_mean": 0.65 <= relative["mean_ratio_to_source"] <= 1.35,
                    "relative_dark_pixels": relative["dark_pixel_ratio_increase"] <= 0.20,
                    "relative_dynamic_range": relative["dynamic_range_ratio_to_source"] >= 0.35,
                    "relative_bbox_mean": relative["minimum_bbox_mean_ratio_to_source"] >= 0.55,
                    "relative_bbox_edges": relative["minimum_bbox_edge_ratio_to_source"] >= 0.15,
                }
            )
        elif variant_type == "compound":
            checks.update(
                {
                    "relative_mean": 0.35 <= relative["mean_ratio_to_source"] <= 1.80,
                    "relative_dark_pixels": relative["dark_pixel_ratio_increase"] <= 0.45,
                    "relative_dynamic_range": relative["dynamic_range_ratio_to_source"] >= 0.25,
                    "relative_bbox_mean": relative["minimum_bbox_mean_ratio_to_source"] >= 0.25,
                    "relative_bbox_edges": relative["minimum_bbox_edge_ratio_to_source"] >= 0.15,
                }
            )
    metrics.pop("bbox_means")
    metrics.pop("bbox_stds")
    metrics.pop("bbox_ranges")
    metrics.pop("bbox_edge_means")
    metrics["checks"] = checks
    metrics["failed_checks"] = [name for name, passed in checks.items() if not passed]
    metrics["passed"] = not metrics["failed_checks"]
    return metrics


def apply_augmentation_attempt(
    source_image: np.ndarray,
    source_boxes: tuple[tuple[float, float, float, float], ...],
    variant_type: str,
    subtype: str,
    current_seed: int,
    recorded_shadow: dict[str, Any] | None = None,
) -> tuple[np.ndarray, tuple[tuple[float, float, float, float], ...], list[Any], dict[str, Any] | None]:
    """Apply one seed-owned attempt and expose every realized transform parameter."""
    if variant_type == "light" and subtype == "local_shadow_or_nonuniform":
        shadow = recorded_shadow or soft_shadow_parameters(source_image.shape, current_seed)
        image, actual_shadow = apply_soft_shadow(source_image, shadow)
        return image, source_boxes, [["SoftLuminanceShadow", actual_shadow]], actual_shadow
    pipeline = build_pipeline(variant_type, subtype, current_seed)
    transform_input: dict[str, Any] = {"image": source_image}
    if variant_type in {"geo", "compound"}:
        transform_input.update(bboxes=list(source_boxes), class_labels=[0] * len(source_boxes))
    transformed = pipeline(**transform_input)
    boxes = canonicalize_boxes(transformed.get("bboxes", source_boxes))
    return transformed["image"], boxes, transformed.get("applied_transforms", []), None


def serialize_applied_transforms(
    applied: list[Any], current_seed: int, variant_type: str, subtype: str
) -> list[Any]:
    """Serialize realized parameters with an explicit public replay recipe for any large arrays."""
    reconstruction = {
        "method": "rebuild_seeded_pipeline",
        "attempt_seed": current_seed,
        "variant_type": variant_type,
        "variant_subtype": subtype,
        "albumentations_version": A.__version__,
        "numpy_version": np.__version__,
        "distribution_parameters": "recorded alongside the selected transform",
    }
    return json_safe(applied, large_array_reconstruction=reconstruction)


def adaptive_light_parameters(source_metrics: dict[str, Any], stable_fallback_seed: int) -> dict[str, Any]:
    """Calculate a gentle luminance adjustment from source quality and one stable seed."""
    unit = stable_fallback_seed / (2**32 - 1)
    source_mean = float(source_metrics["mean"])
    darkness = min(1.0, max(0.0, (64.0 - source_mean) / 64.0))
    gain_bounds = (1.012, 1.028) if source_mean < 64.0 else (1.008, 1.018)
    if source_mean < 32.0:
        lift_bounds = (0.75, 1.50)
    elif source_mean < 160.0:
        lift_bounds = (0.35, 0.90)
    elif source_mean < 208.0:
        lift_bounds = (0.20, 0.55)
    else:
        lift_bounds = (-0.90, -0.35)
    gain_position = (0.35 + 0.65 * unit) * max(darkness, 0.35)
    lift_position = 0.35 + 0.65 * (1.0 - unit)
    return {
        "model": "lab_luminance_mean_anchored_contrast_and_lift",
        "policy_version": FALLBACK_POLICY_VERSION,
        "fallback_seed": stable_fallback_seed,
        "source_mean": source_mean,
        "source_dark_pixel_ratio": float(source_metrics["dark_pixel_ratio"]),
        "source_dynamic_range_p01_p99": float(source_metrics["dynamic_range_p01_p99"]),
        "source_minimum_bbox_mean": float(source_metrics["minimum_bbox_mean"]),
        "source_minimum_bbox_std": float(source_metrics["minimum_bbox_std"]),
        "source_minimum_bbox_edge_mean": float(source_metrics["minimum_bbox_edge_mean"]),
        "gain": gain_bounds[0] + (gain_bounds[1] - gain_bounds[0]) * gain_position,
        "luminance_lift": lift_bounds[0] + (lift_bounds[1] - lift_bounds[0]) * lift_position,
        "adaptive_bounds": {
            "contrast_gain": list(gain_bounds),
            "luminance_lift": list(lift_bounds),
            "darkness_score": darkness,
            "seed_fraction": unit,
        },
        "parameter_policy": (
            "source-quality-conditioned LAB luminance contrast anchored at source L mean plus bounded lift; "
            "all values derived from source metrics and fallback_seed"
        ),
    }


def apply_adaptive_light_fallback(
    image: np.ndarray,
    source_metrics: dict[str, Any],
    stable_fallback_seed: int,
    recorded_parameters: dict[str, Any] | None = None,
) -> tuple[np.ndarray, list[Any], dict[str, Any]]:
    """Apply the one shared deterministic light fallback used by light and compound."""
    parameters = recorded_parameters or adaptive_light_parameters(source_metrics, stable_fallback_seed)
    if int(parameters["fallback_seed"]) != stable_fallback_seed:
        raise BuildStop("recorded adaptive-light fallback seed disagrees with the derived seed")
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    luminance_mean = float(lab[:, :, 0].mean())
    lab[:, :, 0] = np.clip(
        (lab[:, :, 0] - luminance_mean) * float(parameters["gain"])
        + luminance_mean
        + float(parameters["luminance_lift"]),
        0.0,
        255.0,
    )
    output = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    actual = dict(parameters)
    actual["input_luminance_mean"] = luminance_mean
    actual["operation"] = "L_out=clip((L_in-mean_L)*gain+mean_L+luminance_lift,0,255)"
    return output, [["AdaptiveLuminanceContrast", json_safe(actual)]], actual


def fallback_geometry_parameters(stable_component_seed: int) -> dict[str, Any]:
    """Derive one mild, bbox-aware affine policy for a compound fallback."""
    rng = random.Random(stable_component_seed)
    return {
        "component_seed": stable_component_seed,
        "rotation_degrees": rng.choice((-1.0, 1.0)) * rng.uniform(0.75, 1.50),
        "scale": rng.uniform(0.992, 1.008),
        "translate_x": rng.uniform(-0.004, 0.004),
        "translate_y": rng.uniform(-0.004, 0.004),
        "shear_x_degrees": rng.uniform(-0.40, 0.40),
        "shear_y_degrees": rng.uniform(-0.40, 0.40),
    }


def apply_fallback_geometry(
    image: np.ndarray,
    boxes: tuple[tuple[float, float, float, float], ...],
    stable_component_seed: int,
    recorded_parameters: dict[str, Any] | None = None,
) -> tuple[np.ndarray, tuple[tuple[float, float, float, float], ...], list[Any]]:
    """Apply one deterministic, mild affine component while preserving every box."""
    policy = recorded_parameters or fallback_geometry_parameters(stable_component_seed)
    if int(policy["component_seed"]) != stable_component_seed:
        raise BuildStop("recorded compound geometry seed disagrees with the derived seed")
    transform = A.Compose(
        [
            A.Affine(
                scale=(float(policy["scale"]), float(policy["scale"])),
                translate_percent={
                    "x": (float(policy["translate_x"]), float(policy["translate_x"])),
                    "y": (float(policy["translate_y"]), float(policy["translate_y"])),
                },
                rotate=(float(policy["rotation_degrees"]), float(policy["rotation_degrees"])),
                shear={
                    "x": (float(policy["shear_x_degrees"]), float(policy["shear_x_degrees"])),
                    "y": (float(policy["shear_y_degrees"]), float(policy["shear_y_degrees"])),
                },
                border_mode=cv2.BORDER_REFLECT_101,
                p=1,
            )
        ],
        bbox_params=geometry_bbox_params(),
        seed=stable_component_seed,
        strict=True,
        save_applied_params=True,
    )
    transformed = transform(image=image, bboxes=list(boxes), class_labels=[0] * len(boxes))
    output_boxes = canonicalize_boxes(transformed["bboxes"])
    actual = dict(policy)
    actual["albumentations_applied"] = serialize_applied_transforms(
        transformed.get("applied_transforms", []), stable_component_seed, "compound", "fallback_geometry"
    )
    return transformed["image"], output_boxes, [["AdaptiveAffineGeometry", json_safe(actual)]]


def fallback_degrade_parameters(stable_component_seed: int) -> dict[str, Any]:
    """Derive one fixed-size, mild Gaussian blur for a compound fallback."""
    unit = stable_component_seed / (2**32 - 1)
    return {
        "component_seed": stable_component_seed,
        "kernel_size": [3, 3],
        "sigma": 0.30 + 0.20 * unit,
        "border_type": "BORDER_REFLECT_101",
    }


def apply_fallback_degrade(
    image: np.ndarray,
    stable_component_seed: int,
    recorded_parameters: dict[str, Any] | None = None,
) -> tuple[np.ndarray, list[Any]]:
    """Apply the deterministic, mild compound degradation component."""
    parameters = recorded_parameters or fallback_degrade_parameters(stable_component_seed)
    if int(parameters["component_seed"]) != stable_component_seed:
        raise BuildStop("recorded compound degrade seed disagrees with the derived seed")
    kernel = tuple(int(value) for value in parameters["kernel_size"])
    output = cv2.GaussianBlur(
        image,
        kernel,
        sigmaX=float(parameters["sigma"]),
        sigmaY=float(parameters["sigma"]),
        borderType=cv2.BORDER_REFLECT_101,
    )
    return output, [["AdaptiveGaussianBlur", json_safe(parameters)]]


def apply_adaptive_fallback(
    source_image: np.ndarray,
    source_boxes: tuple[tuple[float, float, float, float], ...],
    variant_type: str,
    stable_sample_seed: int,
    recorded_parameters: list[Any] | None = None,
) -> tuple[np.ndarray, tuple[tuple[float, float, float, float], ...], list[Any], dict[str, Any]]:
    """Apply the deterministic post-retry fallback without changing normal augmentation behavior."""
    stable_fallback_seed = fallback_seed(stable_sample_seed)
    source_metrics = _quality_components(source_image, source_boxes)
    if variant_type == "light":
        recorded_light = recorded_parameters[0][1] if recorded_parameters else None
        image, applied, light_parameters = apply_adaptive_light_fallback(
            source_image, source_metrics, stable_fallback_seed, recorded_light
        )
        metadata = {
            "fallback_seed": stable_fallback_seed,
            "source_quality_metrics": json_safe(source_metrics),
            "parameter_policy": light_parameters["parameter_policy"],
            "adaptive_bounds": light_parameters["adaptive_bounds"],
        }
        return image, source_boxes, applied, metadata
    if variant_type != "compound":
        raise BuildStop(f"adaptive fallback is not defined for variant {variant_type}")
    geometry_seed = component_seed(stable_fallback_seed, "geometry")
    light_seed = component_seed(stable_fallback_seed, "light")
    degrade_seed = component_seed(stable_fallback_seed, "degrade")
    recorded_geometry = recorded_parameters[0][1] if recorded_parameters else None
    recorded_light = recorded_parameters[1][1] if recorded_parameters else None
    recorded_degrade = recorded_parameters[2][1] if recorded_parameters else None
    image, boxes, geometry_applied = apply_fallback_geometry(
        source_image, source_boxes, geometry_seed, recorded_geometry
    )
    if len(boxes) != len(source_boxes):
        raise BuildStop("compound fallback geometry did not preserve every source box")
    image, light_applied, light_parameters = apply_adaptive_light_fallback(
        image, source_metrics, light_seed, recorded_light
    )
    image, degrade_applied = apply_fallback_degrade(image, degrade_seed, recorded_degrade)
    applied = geometry_applied + light_applied + degrade_applied
    metadata = {
        "fallback_seed": stable_fallback_seed,
        "component_seeds": {"geometry": geometry_seed, "light": light_seed, "degrade": degrade_seed},
        "source_quality_metrics": json_safe(source_metrics),
        "parameter_policy": light_parameters["parameter_policy"],
        "adaptive_bounds": light_parameters["adaptive_bounds"],
    }
    return image, boxes, applied, metadata


def evaluate_augmented_bytes(
    source: SourceRecord,
    variant_type: str,
    subtype: str,
    stable_sample_seed: int,
    forbidden_hashes: set[str],
) -> dict[str, Any]:
    """Evaluate normal attempts and the isolated fallback while retaining complete audit state."""
    source_image = read_image(source.image_path)
    normal_failures: list[dict[str, Any]] = []
    for attempt in range(MAX_AUGMENT_RETRIES):
        current_seed = attempt_seed(stable_sample_seed, attempt)
        try:
            image, boxes, applied, shadow = apply_augmentation_attempt(
                source_image, source.boxes, variant_type, subtype, current_seed
            )
        except (BuildStop, ValueError) as error:
            normal_failures.append(
                {"retry_index": attempt, "attempt_seed": current_seed, "failure_reasons": [str(error)]}
            )
            continue
        if len(boxes) != len(source.boxes):
            normal_failures.append(
                {
                    "retry_index": attempt,
                    "attempt_seed": current_seed,
                    "failure_reasons": [f"bbox_count_changed:{len(source.boxes)}->{len(boxes)}"],
                }
            )
            continue
        quality = image_quality(image, boxes, source_image, source.boxes, variant_type, subtype)
        if not quality["passed"]:
            normal_failures.append(
                {
                    "retry_index": attempt,
                    "attempt_seed": current_seed,
                    "failure_reasons": list(quality["failed_checks"]),
                }
            )
            continue
        image_bytes = encode_jpeg(image)
        digest = sha256_bytes(image_bytes)
        if digest in forbidden_hashes:
            normal_failures.append(
                {"retry_index": attempt, "attempt_seed": current_seed, "failure_reasons": ["forbidden_sha256"]}
            )
            continue
        serialized = serialize_applied_transforms(applied, current_seed, variant_type, subtype)
        parameters = {
            "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
            "sample_seed": stable_sample_seed,
            "retry_index": attempt,
            "attempt_seed": current_seed,
            "variant_subtype": subtype,
            "applied_transforms": serialized,
            "quality": quality,
            "fallback_used": False,
            "normal_attempts_exhausted": False,
            "fallback_seed": None,
            "source_quality_metrics": None,
            "parameter_policy": "normal_seeded_retry",
            "adaptive_bounds": None,
            "final_actual_parameters": serialized,
            "fallback_quality_metrics": None,
            "fallback_failure_reasons": [],
        }
        if shadow is not None:
            parameters["shadow"] = shadow
        if variant_type == "compound":
            if len(parameters["applied_transforms"]) != 3:
                normal_failures.append(
                    {
                        "retry_index": attempt,
                        "attempt_seed": current_seed,
                        "failure_reasons": ["compound_component_count_changed"],
                    }
                )
                continue
            parameters["required_components"] = ["geometry", "light", "degrade"]
            parameters["components"] = dict(
                zip(parameters["required_components"], parameters["applied_transforms"])
            )
        return {
            "passed": True,
            "image_bytes": image_bytes,
            "boxes": boxes,
            "parameters": parameters,
            "normal_failures": normal_failures,
        }
    if variant_type not in {"light", "compound"}:
        return {
            "passed": False,
            "image_bytes": None,
            "boxes": source.boxes,
            "parameters": None,
            "normal_failures": normal_failures,
            "failure_reasons": ["normal_attempts_exhausted_and_no_fallback_for_variant"],
        }
    stable_fallback_seed = fallback_seed(stable_sample_seed)
    fallback_failures: list[str] = []
    try:
        image, boxes, applied, fallback_metadata = apply_adaptive_fallback(
            source_image, source.boxes, variant_type, stable_sample_seed
        )
        if len(boxes) != len(source.boxes):
            fallback_failures.append(f"bbox_count_changed:{len(source.boxes)}->{len(boxes)}")
        quality = image_quality(image, boxes, source_image, source.boxes, variant_type, subtype)
        fallback_failures.extend(quality["failed_checks"])
        difference = cv2.absdiff(image, source_image)
        mean_absolute_change = float(difference.mean())
        changed_component_ratio = float(np.mean(difference > 0))
        if mean_absolute_change < 0.50:
            fallback_failures.append("fallback_change_too_small")
        if changed_component_ratio < 0.01:
            fallback_failures.append("fallback_changed_component_ratio_too_small")
        image_bytes = encode_jpeg(image)
        digest = sha256_bytes(image_bytes)
        if digest in forbidden_hashes:
            fallback_failures.append("forbidden_sha256")
    except (BuildStop, ValueError) as error:
        image_bytes, boxes, applied, quality = None, source.boxes, [], None
        fallback_metadata = {
            "fallback_seed": stable_fallback_seed,
            "source_quality_metrics": json_safe(_quality_components(source_image, source.boxes)),
            "parameter_policy": FALLBACK_POLICY_VERSION,
            "adaptive_bounds": None,
        }
        mean_absolute_change = 0.0
        changed_component_ratio = 0.0
        fallback_failures.append(str(error))
    serialized = json_safe(applied)
    parameters = {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "sample_seed": stable_sample_seed,
        "retry_index": MAX_AUGMENT_RETRIES,
        "attempt_seed": stable_fallback_seed,
        "variant_subtype": subtype,
        "applied_transforms": serialized,
        "quality": quality,
        "fallback_used": True,
        "normal_attempts_exhausted": True,
        "fallback_seed": stable_fallback_seed,
        "source_quality_metrics": fallback_metadata["source_quality_metrics"],
        "parameter_policy": fallback_metadata["parameter_policy"],
        "adaptive_bounds": fallback_metadata["adaptive_bounds"],
        "final_actual_parameters": serialized,
        "fallback_quality_metrics": quality,
        "fallback_failure_reasons": fallback_failures,
        "normal_attempt_failure_reasons": normal_failures,
        "fallback_mean_absolute_pixel_change": mean_absolute_change,
        "fallback_changed_component_ratio": changed_component_ratio,
    }
    if variant_type == "compound" and len(serialized) == 3:
        parameters["required_components"] = ["geometry", "light", "degrade"]
        parameters["components"] = dict(zip(parameters["required_components"], serialized))
        parameters["component_seeds"] = fallback_metadata["component_seeds"]
    return {
        "passed": not fallback_failures,
        "image_bytes": image_bytes if not fallback_failures else None,
        "boxes": boxes,
        "parameters": parameters,
        "normal_failures": normal_failures,
        "failure_reasons": fallback_failures,
    }


def generate_augmented_bytes(
    source: SourceRecord,
    variant_type: str,
    subtype: str,
    stable_sample_seed: int,
    forbidden_hashes: set[str],
) -> tuple[bytes, tuple[tuple[float, float, float, float], ...], dict[str, Any]]:
    """Generate one augmentation and retain fail-fast behavior after the safe fallback."""
    outcome = evaluate_augmented_bytes(source, variant_type, subtype, stable_sample_seed, forbidden_hashes)
    if not outcome["passed"]:
        raise BuildStop(
            f"unable to generate valid {variant_type}/{subtype} for {source.parent_id}; "
            f"normal attempts exhausted and fallback failed: {outcome['failure_reasons']}"
        )
    return outcome["image_bytes"], outcome["boxes"], outcome["parameters"]


def replay_augmented_bytes(
    source: SourceRecord,
    variant_type: str,
    subtype: str,
    parameters: dict[str, Any],
) -> tuple[bytes, bytes, tuple[tuple[float, float, float, float], ...], list[Any]]:
    """Replay one augmentation from its public manifest fields and authoritative source only."""
    if parameters.get("manifest_schema_version") not in {"2.0", MANIFEST_SCHEMA_VERSION}:
        raise BuildStop("unsupported manifest schema for replay")
    source_image = read_image(source.image_path)
    if parameters.get("fallback_used"):
        expected_seed = fallback_seed(int(parameters["sample_seed"]))
        if int(parameters["fallback_seed"]) != expected_seed:
            raise BuildStop("manifest fallback seed disagrees with stable derivation")
        image, boxes, applied, _ = apply_adaptive_fallback(
            source_image,
            source.boxes,
            variant_type,
            int(parameters["sample_seed"]),
            recorded_parameters=parameters["final_actual_parameters"],
        )
        serialized = json_safe(applied)
    else:
        image, boxes, applied, _ = apply_augmentation_attempt(
            source_image,
            source.boxes,
            variant_type,
            subtype,
            int(parameters["attempt_seed"]),
            recorded_shadow=parameters.get("shadow"),
        )
        serialized = serialize_applied_transforms(applied, int(parameters["attempt_seed"]), variant_type, subtype)
    label_bytes = (
        format_yolo_labels(boxes)
        if variant_type in {"geo", "compound"}
        else source.label_path.read_bytes()
    )
    return encode_jpeg(image), label_bytes, boxes, serialized


def build_rows(
    records: list[SourceRecord],
    geo_subtypes: dict[str, str],
    light_subtypes: dict[str, str],
    degrade_subtypes: dict[str, str],
    seed: int,
) -> list[dict[str, Any]]:
    """Create five auditable, collision-free output records per parent."""
    rows: list[dict[str, Any]] = []
    for source in records:
        subtypes = {
            "orig": "byte_exact_copy",
            "geo": geo_subtypes[source.parent_id],
            "light": light_subtypes[source.parent_id],
            "degrade": degrade_subtypes[source.parent_id],
            "compound": "geometry_light_degrade",
        }
        for variant_type in VARIANT_TYPES:
            extension = source.image_path.suffix.casefold() if variant_type == "orig" else ".jpg"
            sample_id = f"{source.parent_id}__{variant_type}"
            rows.append(
                {
                    "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
                    "sample_id": sample_id,
                    "parent_id": source.parent_id,
                    "source_image": source.output_image_relative,
                    "source_label": source.output_label_relative,
                    "source_split": source.source_split,
                    "variant_type": variant_type,
                    "variant_subtype": subtypes[variant_type],
                    "global_seed": seed,
                    "sample_seed": sample_seed(seed, source.parent_id, variant_type),
                    "pool_image": f"pool/images/{sample_id}{extension}",
                    "pool_label": f"pool/labels/{sample_id}.txt",
                    "output_image": "",
                    "output_label": "",
                    "augmentation_parameters_json": "",
                    "bbox_count_before": len(source.boxes),
                    "bbox_count_after": 0,
                    "source_image_sha256": source.image_sha256,
                    "output_image_sha256": "",
                    "output_label_sha256": "",
                    "image_width": source.width,
                    "image_height": source.height,
                    "pre_shuffle_index": -1,
                    "post_shuffle_index": -1,
                    "split_seed": seed,
                    "split": "",
                }
            )
    sample_ids = [row["sample_id"].casefold() for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise BuildStop("output sample_id collision")
    return rows


def generate_pool(staging: Path, rows: list[dict[str, Any]], source_by_parent: dict[str, SourceRecord]) -> None:
    """Generate the complete unsplit file pool before the one global split shuffle."""
    (staging / "pool" / "images").mkdir(parents=True)
    (staging / "pool" / "labels").mkdir(parents=True)
    family_hashes: defaultdict[str, set[str]] = defaultdict(set)
    for index, row in enumerate(sorted(rows, key=lambda item: item["pool_image"].casefold()), 1):
        source = source_by_parent[row["parent_id"]]
        image_path, label_path = staging / row["pool_image"], staging / row["pool_label"]
        if row["variant_type"] == "orig":
            shutil.copy2(source.image_path, image_path)
            shutil.copy2(source.label_path, label_path)
            parameters = {
                "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
                "sample_seed": row["sample_seed"],
                "operation": "byte_exact_copy",
                "source_extension": source.image_path.suffix.casefold(),
                "fallback_used": False,
                "normal_attempts_exhausted": False,
                "retry_index": None,
                "attempt_seed": None,
                "fallback_seed": None,
                "source_quality_metrics": None,
                "parameter_policy": "byte_exact_copy",
                "adaptive_bounds": None,
                "final_actual_parameters": {"operation": "byte_exact_copy"},
                "fallback_quality_metrics": None,
                "fallback_failure_reasons": [],
            }
            boxes = source.boxes
        else:
            image_bytes, boxes, parameters = generate_augmented_bytes(
                source,
                row["variant_type"],
                row["variant_subtype"],
                row["sample_seed"],
                family_hashes[source.parent_id] | {source.image_sha256},
            )
            image_path.write_bytes(image_bytes)
            label_path.write_bytes(
                source.label_path.read_bytes()
                if row["variant_type"] in {"light", "degrade"}
                else format_yolo_labels(boxes)
            )
        image_digest, label_digest = sha256_file(image_path), sha256_file(label_path)
        if row["variant_type"] == "orig" and image_digest != source.image_sha256:
            raise BuildStop(f"orig is not byte-identical: {row['sample_id']}")
        if row["variant_type"] != "orig" and image_digest == source.image_sha256:
            raise BuildStop(f"augmentation equals its source: {row['sample_id']}")
        if image_digest in family_hashes[source.parent_id]:
            raise BuildStop(f"same-family exact duplicate: {row['sample_id']}")
        family_hashes[source.parent_id].add(image_digest)
        row["augmentation_parameters_json"] = json.dumps(
            parameters, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        row["bbox_count_after"] = len(boxes)
        row["output_image_sha256"] = image_digest
        row["output_label_sha256"] = label_digest
        if index % 25 == 0 or index == len(rows):
            print(f"generated pool {index}/{len(rows)}", flush=True)


def assign_file_level_splits(rows: list[dict[str, Any]], seed: int) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Sort by stable output path and perform exactly one global Random(seed).shuffle."""
    ordered = sorted(rows, key=lambda item: item["pool_image"].casefold())
    for index, row in enumerate(ordered):
        row["pre_shuffle_index"] = index
    shuffled = list(ordered)
    random.Random(seed).shuffle(shuffled)
    counts = largest_remainder_counts(len(rows))
    cursor = 0
    for split, _ in SPLIT_RATIOS:
        for row in shuffled[cursor : cursor + counts[split]]:
            row["split"] = split
        cursor += counts[split]
    for index, row in enumerate(shuffled):
        row["post_shuffle_index"] = index
    return shuffled, counts


def materialize_splits(staging: Path, rows: list[dict[str, Any]]) -> None:
    """Move the already-generated pool into final split directories."""
    for split, _ in SPLIT_RATIOS:
        (staging / "images" / split).mkdir(parents=True)
        (staging / "labels" / split).mkdir(parents=True)
    for row in rows:
        image_name, label_name = Path(row["pool_image"]).name, Path(row["pool_label"]).name
        output_image = Path("images") / row["split"] / image_name
        output_label = Path("labels") / row["split"] / label_name
        (staging / row["pool_image"]).replace(staging / output_image)
        (staging / row["pool_label"]).replace(staging / output_label)
        row["output_image"], row["output_label"] = output_image.as_posix(), output_label.as_posix()
    (staging / "pool" / "images").rmdir()
    (staging / "pool" / "labels").rmdir()
    (staging / "pool").rmdir()


def write_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    """Write a stable CSV with explicit fields."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows({field: row[field] for field in fieldnames} for row in rows)


def write_source_manifest(path: Path, records: list[SourceRecord]) -> None:
    """Write selected-parent provenance."""
    rows = [
        {
            "selection_index": index,
            "source_image": record.output_image_relative,
            "source_label": record.output_label_relative,
            "source_split": record.source_split,
            "parent_id": record.parent_id,
            "source_image_sha256": record.image_sha256,
            "source_label_sha256": record.label_sha256,
            "bbox_count": len(record.boxes),
            "image_width": record.width,
            "image_height": record.height,
        }
        for index, record in enumerate(records)
    ]
    write_csv(path, SOURCE_MANIFEST_FIELDS, rows)


def difference_hash(path: Path) -> str:
    """Return a 64-bit dHash for audit-only near-duplicate statistics."""
    with Image.open(path) as image:
        gray = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
        pixels = np.asarray(gray, dtype=np.int16)
    bits = pixels[:, 1:] > pixels[:, :-1]
    value = 0
    for bit in bits.flat:
        value = (value << 1) | int(bit)
    return f"{value:016x}"


def hamming_distance(left: str, right: str) -> int:
    """Return hexadecimal bitwise Hamming distance."""
    return (int(left, 16) ^ int(right, 16)).bit_count()


def read_existing_near_duplicate_pairs(
    source: Path, records: list[SourceRecord]
) -> list[dict[str, Any]]:
    """Map all existing audit-only dHash candidate pairs to parent IDs."""
    by_raw_path = {
        os.path.normcase(os.path.normpath(record.raw_source_image)): record.parent_id for record in records
    }
    path = source / "metadata" / "near_duplicate_candidates.csv"
    pairs: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            left = by_raw_path.get(os.path.normcase(os.path.normpath(row["left_source_image"])))
            right = by_raw_path.get(os.path.normcase(os.path.normpath(row["right_source_image"])))
            if left and right:
                pairs.append(
                    {
                        "left_parent_id": left,
                        "right_parent_id": right,
                        "left_dhash": row["left_dhash"],
                        "right_dhash": row["right_dhash"],
                        "hamming_distance": int(row["hamming_distance"]),
                    }
                )
    return pairs


def output_duplicate_audit(staging: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute exact and perceptual duplicate statistics for every dry-run output."""
    hash_groups: defaultdict[str, list[str]] = defaultdict(list)
    dhashes: dict[str, str] = {}
    row_by_image = {row["output_image"]: row for row in rows}
    for row in rows:
        hash_groups[row["output_image_sha256"]].append(row["output_image"])
        dhashes[row["output_image"]] = difference_hash(staging / row["output_image"])
    exact_groups = {digest: paths for digest, paths in hash_groups.items() if len(paths) > 1}
    images = sorted(dhashes)
    near_pairs: list[dict[str, Any]] = []
    for left_index, left in enumerate(images):
        for right in images[left_index + 1 :]:
            distance = hamming_distance(dhashes[left], dhashes[right])
            if distance <= 5:
                left_row, right_row = row_by_image[left], row_by_image[right]
                near_pairs.append(
                    {
                        "left": left,
                        "right": right,
                        "hamming_distance": distance,
                        "same_parent": left_row["parent_id"] == right_row["parent_id"],
                        "cross_split": left_row["split"] != right_row["split"],
                    }
                )
    return {
        "method": "64-bit dHash Hamming distance <= 5; audit-only, no deletion",
        "exact_duplicate_groups": len(exact_groups),
        "exact_duplicate_files": sum(len(paths) for paths in exact_groups.values()),
        "near_duplicate_pairs": len(near_pairs),
        "near_duplicate_same_parent_pairs": sum(pair["same_parent"] for pair in near_pairs),
        "near_duplicate_cross_split_pairs": sum(pair["cross_split"] for pair in near_pairs),
        "near_duplicate_examples": near_pairs[:100],
    }


def parent_split_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Report parent leakage without changing split assignment."""
    memberships: defaultdict[str, set[str]] = defaultdict(set)
    split_parents: defaultdict[str, set[str]] = defaultdict(set)
    for row in rows:
        memberships[row["parent_id"]].add(row["split"])
        split_parents[row["split"]].add(row["parent_id"])
    crossing = {parent: sorted(splits) for parent, splits in memberships.items() if len(splits) > 1}
    return {
        "parent_count": len(memberships),
        "parents_crossing_splits": len(crossing),
        "parents_in_all_three_splits": sum(len(splits) == 3 for splits in memberships.values()),
        "train_val_parent_overlap": len(split_parents["train"] & split_parents["val"]),
        "train_test_parent_overlap": len(split_parents["train"] & split_parents["test"]),
        "val_test_parent_overlap": len(split_parents["val"] & split_parents["test"]),
        "crossing_parent_membership": dict(sorted(crossing.items())),
    }


def validate_outputs(
    staging: Path, rows: list[dict[str, Any]], source_by_parent: dict[str, SourceRecord]
) -> dict[str, Any]:
    """Validate every generated image, label, bbox, hash, split, and family invariant."""
    errors: list[str] = []
    split_images: Counter[str] = Counter()
    split_labels: Counter[str] = Counter()
    split_boxes: Counter[str] = Counter()
    variant_counts: Counter[str] = Counter()
    subtype_counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
    disk_images = {path.relative_to(staging).as_posix() for path in (staging / "images").rglob("*") if path.is_file()}
    disk_labels = {path.relative_to(staging).as_posix() for path in (staging / "labels").rglob("*.txt")}
    expected_images = {row["output_image"] for row in rows}
    expected_labels = {row["output_label"] for row in rows}
    if disk_images != expected_images or disk_labels != expected_labels:
        errors.append(
            f"image/label manifest mismatch: images={len(disk_images)}, labels={len(disk_labels)}, rows={len(rows)}"
        )
    family_hashes: defaultdict[str, set[str]] = defaultdict(set)
    for row in rows:
        image_path, label_path = staging / row["output_image"], staging / row["output_label"]
        try:
            image = read_image(image_path)
            if (image.shape[1], image.shape[0]) != (int(row["image_width"]), int(row["image_height"])):
                errors.append(f"dimension mismatch: {row['output_image']}")
            boxes = parse_yolo_label(label_path)
        except BuildStop as error:
            errors.append(str(error))
            continue
        if len(boxes) != int(row["bbox_count_before"]) or len(boxes) != int(row["bbox_count_after"]):
            errors.append(f"bbox count changed: {row['output_image']}")
        image_digest, label_digest = sha256_file(image_path), sha256_file(label_path)
        if image_digest != row["output_image_sha256"] or label_digest != row["output_label_sha256"]:
            errors.append(f"hash mismatch: {row['output_image']}")
        if row["variant_type"] == "orig" and image_digest != row["source_image_sha256"]:
            errors.append(f"orig not byte-identical: {row['output_image']}")
        if row["variant_type"] != "orig" and image_digest == row["source_image_sha256"]:
            errors.append(f"augmentation equals source: {row['output_image']}")
        if image_digest in family_hashes[row["parent_id"]]:
            errors.append(f"same-family exact duplicate: {row['output_image']}")
        family_hashes[row["parent_id"]].add(image_digest)
        source = source_by_parent[row["parent_id"]]
        source_image = read_image(source.image_path)
        quality = image_quality(
            image,
            boxes,
            source_image,
            source.boxes,
            row["variant_type"],
            row["variant_subtype"],
        )
        if not quality["passed"]:
            errors.append(f"image quality failed: {row['output_image']}: {quality}")
        split_images[row["split"]] += 1
        split_labels[row["split"]] += 1
        split_boxes[row["split"]] += len(boxes)
        variant_counts[row["variant_type"]] += 1
        subtype_counts[row["variant_type"]][row["variant_subtype"]] += 1
    return {
        "passed": not errors,
        "errors": errors,
        "image_label_one_to_one": disk_images == expected_images and disk_labels == expected_labels,
        "split_image_counts": dict(split_images),
        "split_label_counts": dict(split_labels),
        "split_bbox_counts": dict(split_boxes),
        "variant_counts": dict(variant_counts),
        "variant_subtype_counts": {variant: dict(counts) for variant, counts in subtype_counts.items()},
        "total_images": len(disk_images),
        "total_labels": len(disk_labels),
        "total_bboxes": sum(split_boxes.values()),
    }


def draw_boxes(image_path: Path, label_path: Path, width: int = 700) -> Image.Image:
    """Render one RGB thumbnail with current class-0 boxes."""
    with Image.open(image_path) as image:
        image.load()
        rgb = image.convert("RGB")
    height = round(rgb.height * width / rgb.width)
    preview = rgb.resize((width, height), Image.Resampling.LANCZOS)
    rgb.close()
    draw = ImageDraw.Draw(preview)
    for x, y, box_width, box_height in parse_yolo_label(label_path):
        draw.rectangle(
            (
                (x - box_width / 2) * width,
                (y - box_height / 2) * height,
                (x + box_width / 2) * width,
                (y + box_height / 2) * height,
            ),
            outline=(255, 64, 64),
            width=3,
        )
    return preview


def comparison_preview(
    source: SourceRecord, output_image: Path, output_label: Path, title: str
) -> Image.Image:
    """Create an original/augmented side-by-side boxed comparison."""
    source_preview = draw_boxes(source.image_path, source.label_path)
    output_preview = draw_boxes(output_image, output_label)
    sheet = Image.new("RGB", (1400, source_preview.height + 34), "white")
    sheet.paste(source_preview, (0, 34))
    sheet.paste(output_preview, (700, 34))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    draw.text((8, 9), f"original: {source.parent_id}", fill="black", font=font)
    draw.text((708, 9), title, fill="black", font=font)
    source_preview.close()
    output_preview.close()
    return sheet


def create_previews(
    staging: Path,
    rows: list[dict[str, Any]],
    source_by_parent: dict[str, SourceRecord],
    existing_pairs: list[dict[str, Any]],
    all_sources_by_parent: dict[str, SourceRecord],
    seed: int,
) -> dict[str, list[str]]:
    """Create six boxed previews per variant plus existing dHash candidate comparisons."""
    preview_paths: dict[str, list[str]] = {}
    for variant_type in VARIANT_TYPES:
        variant_rows = sorted(
            (row for row in rows if row["variant_type"] == variant_type),
            key=lambda item: item["output_image"].casefold(),
        )[:PREVIEWS_PER_VARIANT]
        output_dir = staging / "previews" / variant_type
        output_dir.mkdir(parents=True)
        paths: list[str] = []
        for index, row in enumerate(variant_rows, 1):
            sheet = comparison_preview(
                source_by_parent[row["parent_id"]],
                staging / row["output_image"],
                staging / row["output_label"],
                f"{variant_type}/{row['variant_subtype']} / {row['split']}",
            )
            relative = Path("previews") / variant_type / f"preview_{index:02d}_{row['parent_id']}.jpg"
            sheet.save(staging / relative, quality=90)
            sheet.close()
            paths.append(relative.as_posix())
        preview_paths[variant_type] = paths
    near_dir = staging / "previews" / "near_duplicates"
    near_dir.mkdir(parents=True)
    ordered_pairs = sorted(
        existing_pairs,
        key=lambda pair: (pair["hamming_distance"], pair["left_parent_id"], pair["right_parent_id"]),
    )
    chosen_pairs = random.Random(seed).sample(
        ordered_pairs, min(NEAR_DUPLICATE_PREVIEW_PAIRS, len(ordered_pairs))
    )
    near_paths: list[str] = []
    for index, pair in enumerate(chosen_pairs, 1):
        left, right = all_sources_by_parent[pair["left_parent_id"]], all_sources_by_parent[pair["right_parent_id"]]
        left_preview = draw_boxes(left.image_path, left.label_path)
        right_preview = draw_boxes(right.image_path, right.label_path)
        sheet = Image.new("RGB", (1400, left_preview.height + 34), "white")
        sheet.paste(left_preview, (0, 34))
        sheet.paste(right_preview, (700, 34))
        draw = ImageDraw.Draw(sheet)
        draw.text((8, 9), left.parent_id, fill="black", font=ImageFont.load_default())
        draw.text(
            (708, 9),
            f"{right.parent_id}; dHash distance={pair['hamming_distance']}",
            fill="black",
            font=ImageFont.load_default(),
        )
        left_preview.close()
        right_preview.close()
        relative = Path("previews") / "near_duplicates" / f"pair_{index:02d}.jpg"
        sheet.save(staging / relative, quality=90)
        sheet.close()
        near_paths.append(relative.as_posix())
    preview_paths["near_duplicates"] = near_paths
    return preview_paths


def dataset_fingerprint(rows: list[dict[str, Any]]) -> str:
    """Fingerprint final membership and all derived image/label bytes."""
    digest = hashlib.sha256()
    for row in sorted(rows, key=lambda item: item["output_image"].casefold()):
        fields = (
            row["output_image"],
            row["output_image_sha256"],
            row["output_label"],
            row["output_label_sha256"],
            row["parent_id"],
            row["variant_type"],
            row["variant_subtype"],
            row["split"],
            str(row["sample_seed"]),
        )
        digest.update(("\0".join(fields) + "\n").encode())
    return digest.hexdigest()


def dependency_versions() -> dict[str, str]:
    """Record exact reproducibility-relevant dependency versions."""
    return {
        "albumentations": A.__version__,
        "opencv-python": importlib.metadata.version("opencv-python"),
        "opencv-python-headless": importlib.metadata.version("opencv-python-headless"),
        "numpy": np.__version__,
        "Pillow": PIL.__version__,
        "PyYAML": yaml.__version__,
    }


def write_json(path: Path, value: dict[str, Any]) -> None:
    """Write stable UTF-8 JSON."""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_data_yaml(staging: Path) -> dict[str, Any]:
    """Write and locally resolve a path-free portable dataset YAML."""
    data = {
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": 1,
        "names": {0: "crack"},
    }
    path = staging / "data.yaml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8", newline="\n")
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    resolved = {split: (path.parent / loaded[split]).resolve() for split, _ in SPLIT_RATIOS}
    if not all(value.is_dir() for value in resolved.values()):
        raise BuildStop(f"portable data.yaml paths do not resolve locally: {resolved}")
    return {split: loaded[split] for split, _ in SPLIT_RATIOS}


def write_readme(staging: Path, summary: dict[str, Any]) -> None:
    """Write required policy, leakage, quota, and audit disclosures."""
    validation = summary["validation"]
    parent_audit = summary["parent_split_audit"]
    duplicate_audit = summary["duplicate_audit"]
    text = f"""# Tunnel Crack AugFirst Diverse5x RandomSplit 7:2:1 seed42

This dataset was built only from the current repaired NoAug dataset and its class-0 labels.

- split_policy=file_level_random
- parent_id_grouping=false
- independent_test_set=false
- manifest_schema_version={summary['manifest_schema_version']}
- global_seed={summary['global_seed']}
- mode={summary['mode']}
- source_parents={summary['source_parent_count']}
- output_images={validation['total_images']}
- output_labels={validation['total_labels']}
- output_bboxes={validation['total_bboxes']}
- split_counts={summary['split_counts']}
- dataset_fingerprint={summary['dataset_fingerprint']}
- source_fingerprint_before={summary['source_fingerprint_before']}
- source_fingerprint_after={summary['source_fingerprint_after']}
- parents_crossing_splits={parent_audit['parents_crossing_splits']}
- parents_in_all_three_splits={parent_audit['parents_in_all_three_splits']}
- exact_duplicate_groups={duplicate_audit['exact_duplicate_groups']}
- output_dhash_near_duplicate_pairs={duplicate_audit['near_duplicate_pairs']}
- existing_source_dhash_candidate_pairs={summary['existing_near_duplicate_audit']['full_candidate_pairs']}

The five variants are `orig`, `geo`, `light`, `degrade`, and `compound`. All output paths were sorted, then shuffled
once with `random.Random(42)` and split at file level without parent grouping or variant stratification. Variants from
the same parent may cross train, validation, and test. This reproduces and diagnoses the legacy random-split protocol;
the test split is not leakage-free, parent-independent, or a strict independent test set.

`metadata/source_manifest.csv` records selected current-NoAug parents. `metadata/augmentation_manifest.csv` records
stable sample seeds, actual transform parameters, hashes, dimensions, and box counts. `metadata/split_manifest.csv`
records pre/post shuffle indices. Existing and newly computed dHash near-duplicates are audit-only and were not
deleted. Manual previews are under `previews/`, including original/augmented boxed comparisons and source dHash pairs.
"""
    (staging / "README.md").write_text(text, encoding="utf-8", newline="\n")


def collect_all_preflight(source: Path, output: Path, seed: int) -> dict[str, Any]:
    """Check all full light/compound targets through the production path without persisting dataset JPEGs."""
    records = load_source_records(source)
    recorded_fingerprint = (source / "metadata" / "dataset_fingerprint.sha256").read_text(
        encoding="ascii"
    ).split()[0]
    print("recomputing source fingerprint before collect-all", flush=True)
    fingerprint_before = source_dataset_fingerprint(records)
    if fingerprint_before != recorded_fingerprint:
        raise BuildStop(f"source fingerprint disagrees with metadata: {fingerprint_before} != {recorded_fingerprint}")
    geo_subtypes, _ = assign_subtypes(records, "geo", GEO_QUOTAS, seed)
    light_subtypes, _ = assign_subtypes(records, "light", LIGHT_QUOTAS, seed)
    degrade_subtypes, _ = assign_subtypes(records, "degrade", DEGRADE_QUOTAS, seed)
    planned = build_rows(records, geo_subtypes, light_subtypes, degrade_subtypes, seed)
    targets = sorted(
        (row for row in planned if row["variant_type"] in {"light", "compound"}),
        key=lambda row: row["pool_image"].casefold(),
    )
    if len(targets) != FULL_SOURCE_COUNT * 2:
        raise BuildStop(f"collect-all target count changed: {len(targets)}")
    source_by_parent = {record.parent_id: record for record in records}
    family_hashes: defaultdict[str, set[str]] = defaultdict(set)
    details: list[dict[str, Any]] = []
    variant_counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
    retry_distribution: defaultdict[str, Counter[str]] = defaultdict(Counter)
    normal_failure_rules: Counter[str] = Counter()
    final_failure_rules: Counter[str] = Counter()
    with staged_output(output) as staging:
        for index, row in enumerate(targets, 1):
            source_record = source_by_parent[row["parent_id"]]
            outcome = evaluate_augmented_bytes(
                source_record,
                row["variant_type"],
                row["variant_subtype"],
                int(row["sample_seed"]),
                family_hashes[row["parent_id"]] | {source_record.image_sha256},
            )
            parameters = outcome["parameters"]
            fallback_used = bool(parameters and parameters["fallback_used"])
            if outcome["passed"]:
                family_hashes[row["parent_id"]].add(sha256_bytes(outcome["image_bytes"]))
            variant_counts[row["variant_type"]]["targets"] += 1
            variant_counts[row["variant_type"]]["passed" if outcome["passed"] else "failed"] += 1
            if fallback_used:
                variant_counts[row["variant_type"]]["normal_attempts_exhausted"] += 1
                variant_counts[row["variant_type"]]["fallback_started"] += 1
                variant_counts[row["variant_type"]]["fallback_passed" if outcome["passed"] else "fallback_failed"] += 1
            else:
                variant_counts[row["variant_type"]]["normal_success"] += 1
            retry_key = str(parameters["retry_index"] if parameters else MAX_AUGMENT_RETRIES)
            retry_distribution[row["variant_type"]][retry_key] += 1
            for normal_failure in outcome["normal_failures"]:
                normal_failure_rules.update(normal_failure["failure_reasons"])
            final_failure_rules.update(outcome.get("failure_reasons", []))
            details.append(
                {
                    "parent_id": row["parent_id"],
                    "source_image": row["source_image"],
                    "variant_type": row["variant_type"],
                    "variant_subtype": row["variant_subtype"],
                    "sample_seed": int(row["sample_seed"]),
                    "passed": bool(outcome["passed"]),
                    "fallback_used": fallback_used,
                    "normal_attempts_exhausted": fallback_used,
                    "retry_index": int(parameters["retry_index"] if parameters else MAX_AUGMENT_RETRIES),
                    "fallback_seed": (
                        parameters["fallback_seed"] if parameters else fallback_seed(int(row["sample_seed"]))
                    ),
                    "normal_attempt_failure_reasons_json": json.dumps(
                        outcome["normal_failures"], ensure_ascii=False, sort_keys=True, separators=(",", ":")
                    ),
                    "source_quality_metrics_json": json.dumps(
                        parameters["source_quality_metrics"] if parameters else None,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    "parameter_policy": parameters["parameter_policy"] if parameters else FALLBACK_POLICY_VERSION,
                    "adaptive_bounds_json": json.dumps(
                        parameters["adaptive_bounds"] if parameters else None,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    "final_actual_parameters_json": json.dumps(
                        parameters["final_actual_parameters"] if parameters else None,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    "fallback_quality_metrics_json": json.dumps(
                        parameters["fallback_quality_metrics"] if parameters else None,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    "fallback_failure_reasons_json": json.dumps(
                        outcome.get("failure_reasons", []),
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                }
            )
            if index % 25 == 0 or index == len(targets):
                print(f"collect-all {index}/{len(targets)}", flush=True)
        print("recomputing source fingerprint after collect-all", flush=True)
        fingerprint_after = source_dataset_fingerprint(records)
        if fingerprint_after != fingerprint_before:
            raise BuildStop(f"source fingerprint changed: {fingerprint_before} -> {fingerprint_after}")
        fallback_details = [row for row in details if row["fallback_used"]]
        final_failures = [row for row in details if not row["passed"]]
        retry_highest = sorted(
            details, key=lambda row: (-int(row["retry_index"]), row["variant_type"], row["parent_id"])
        )[:100]
        known_parent = "sample_7eae2feee5741ef9cfde"
        report = {
            **POLICY,
            "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
            "mode": "preflight_collect_all_light_compound",
            "global_seed": seed,
            "processed_targets": len(details),
            "variant_statistics": {variant: dict(counts) for variant, counts in sorted(variant_counts.items())},
            "normal_success_count": sum(not row["fallback_used"] and row["passed"] for row in details),
            "normal_attempts_exhausted_count": len(fallback_details),
            "fallback_started_count": len(fallback_details),
            "fallback_passed_count": sum(row["passed"] for row in fallback_details),
            "fallback_failed_count": sum(not row["passed"] for row in fallback_details),
            "unique_fallback_parent_count": len({row["parent_id"] for row in fallback_details}),
            "fallback_trigger_rate": len(fallback_details) / len(details),
            "retry_index_distribution": {
                variant: dict(sorted(counts.items(), key=lambda item: int(item[0])))
                for variant, counts in sorted(retry_distribution.items())
            },
            "normal_attempt_failure_rule_distribution": dict(normal_failure_rules.most_common()),
            "final_failure_rule_distribution": dict(final_failure_rules.most_common()),
            "retry_highest_samples": retry_highest,
            "fallback_samples": fallback_details,
            "final_failure_samples": final_failures,
            "known_failed_parent_passed": any(
                row["parent_id"] == known_parent and row["variant_type"] == "light" and row["passed"]
                for row in details
            ),
            "source_fingerprint_before": fingerprint_before,
            "source_fingerprint_after": fingerprint_after,
            "source_unchanged": fingerprint_before == fingerprint_after,
        }
        report["formal_rerun_gate_passed"] = all(
            (
                len(details) == FULL_SOURCE_COUNT * 2,
                not final_failures,
                report["known_failed_parent_passed"],
                len(fallback_details) <= 5,
                report["fallback_trigger_rate"] <= 5 / (FULL_SOURCE_COUNT * 2),
                report["source_unchanged"],
            )
        )
        write_json(staging / "collect_all_report.json", report)
        fieldnames = tuple(details[0])
        write_csv(staging / "collect_all_details.csv", fieldnames, details)
    summary = {
        key: value
        for key, value in report.items()
        if key not in {"fallback_samples", "final_failure_samples", "retry_highest_samples"}
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return report


def validate_output_target(output: Path) -> Path:
    """Reject overwrite and return the fixed same-parent staging path."""
    if output.exists():
        raise BuildStop(f"output already exists; refusing overwrite: {output}")
    staging = output.with_name(f".{output.name}.building")
    if staging.exists():
        raise BuildStop(f"staging path already exists; refusing reuse: {staging}")
    return staging


@contextmanager
def staged_output(output: Path) -> Iterator[Path]:
    """Build beside the target, clean only this run's staging on failure, and rename atomically on success."""
    staging = validate_output_target(output)
    staging.mkdir(parents=True)
    try:
        yield staging
        staging.replace(output)
    except Exception:
        if staging.is_dir():
            shutil.rmtree(staging)
        raise


def main() -> int:
    """Build and validate exactly one dry-run or explicitly unlocked full dataset."""
    args = parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if args.seed != GLOBAL_SEED:
        raise BuildStop(f"this experiment requires seed={GLOBAL_SEED}")
    if args.full and args.preflight_collect_all:
        raise BuildStop("--full and --preflight-collect-all are mutually exclusive")
    if args.preflight_collect_all:
        collect_all_preflight(source, output, args.seed)
        return 0
    source_count = FULL_SOURCE_COUNT if args.full else 50
    expected_outputs = source_count * len(VARIANT_TYPES)
    records = load_source_records(source)
    recorded_source_fingerprint = (
        source / "metadata" / "dataset_fingerprint.sha256"
    ).read_text(encoding="ascii").split()[0]
    print("recomputing source fingerprint before build", flush=True)
    fingerprint_before = source_dataset_fingerprint(records)
    if fingerprint_before != recorded_source_fingerprint:
        raise BuildStop(
            f"source fingerprint disagrees with metadata: {fingerprint_before} != {recorded_source_fingerprint}"
        )
    selected = select_sources(records, source_count, args.seed)
    geo_subtypes, geo_counts = assign_subtypes(selected, "geo", GEO_QUOTAS, args.seed)
    light_subtypes, light_counts = assign_subtypes(selected, "light", LIGHT_QUOTAS, args.seed)
    degrade_subtypes, degrade_counts = assign_subtypes(selected, "degrade", DEGRADE_QUOTAS, args.seed)
    rows = build_rows(selected, geo_subtypes, light_subtypes, degrade_subtypes, args.seed)
    if len(rows) != expected_outputs:
        raise BuildStop(f"unexpected planned output count: {len(rows)} != {expected_outputs}")
    source_by_parent = {record.parent_id: record for record in selected}
    all_sources_by_parent = {record.parent_id: record for record in records}
    existing_pairs = read_existing_near_duplicate_pairs(source, records)
    if len(existing_pairs) != 8149:
        raise BuildStop(f"existing dHash candidate count changed: {len(existing_pairs)}")
    selected_parent_ids = set(source_by_parent)
    selected_existing_pairs = [
        pair
        for pair in existing_pairs
        if pair["left_parent_id"] in selected_parent_ids and pair["right_parent_id"] in selected_parent_ids
    ]
    with staged_output(output) as staging:
        generate_pool(staging, rows, source_by_parent)
        shuffled, split_counts = assign_file_level_splits(rows, args.seed)
        expected_split_counts = FULL_SPLIT_COUNTS if args.full else DRY_SPLIT_COUNTS
        if split_counts != expected_split_counts:
            raise BuildStop(f"split allocation changed: {split_counts} != {expected_split_counts}")
        materialize_splits(staging, rows)
        write_source_manifest(staging / "metadata" / "source_manifest.csv", selected)
        write_csv(
            staging / "metadata" / "augmentation_manifest.csv",
            AUGMENTATION_MANIFEST_FIELDS,
            sorted(rows, key=lambda item: item["output_image"].casefold()),
        )
        write_csv(
            staging / "metadata" / "split_manifest.csv",
            SPLIT_MANIFEST_FIELDS,
            sorted(shuffled, key=lambda item: item["post_shuffle_index"]),
        )
        portable_yaml_paths = write_data_yaml(staging)
        validation = validate_outputs(staging, rows, source_by_parent)
        duplicate_audit = output_duplicate_audit(staging, rows)
        parent_audit = parent_split_audit(rows)
        previews = create_previews(
            staging,
            rows,
            source_by_parent,
            existing_pairs,
            all_sources_by_parent,
            args.seed,
        )
        print("recomputing source fingerprint after build", flush=True)
        fingerprint_after = source_dataset_fingerprint(records)
        if fingerprint_after != fingerprint_before:
            raise BuildStop(f"source fingerprint changed: {fingerprint_before} -> {fingerprint_after}")
        if not validation["passed"]:
            raise BuildStop(f"output validation failed: {validation['errors'][:20]}")
        if duplicate_audit["exact_duplicate_groups"]:
            raise BuildStop(f"exact output duplicates found: {duplicate_audit['exact_duplicate_groups']}")
        fingerprint = dataset_fingerprint(rows)
        versions = dependency_versions()
        fingerprint_report = {
            **POLICY,
            "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
            "algorithm": (
                "sha256(sorted(output_image, output_image_sha256, output_label, output_label_sha256, "
                "parent_id, variant_type, variant_subtype, split, sample_seed) joined with NUL and LF)"
            ),
            "dataset_fingerprint": fingerprint,
            "source_fingerprint": fingerprint_before,
            "global_seed": args.seed,
            "dependencies": versions,
        }
        audit_report = {
            **POLICY,
            "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
            "mode": "full" if args.full else "dry_run_50",
            "source_fingerprint_before": fingerprint_before,
            "source_fingerprint_after": fingerprint_after,
            "source_unchanged": fingerprint_before == fingerprint_after,
            "validation": validation,
            "duplicate_audit": duplicate_audit,
            "parent_split_audit": parent_audit,
            "existing_near_duplicate_audit": {
                "method": "existing 64-bit dHash candidates; audit-only, no deletion",
                "full_candidate_pairs": len(existing_pairs),
                "selected_parent_candidate_pairs": len(selected_existing_pairs),
                "selected_parent_candidate_examples": selected_existing_pairs[:100],
            },
            "preview_paths": previews,
            "portable_yaml_paths": portable_yaml_paths,
        }
        summary = {
            **POLICY,
            "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
            "mode": "full" if args.full else "dry_run_50",
            "global_seed": args.seed,
            "source_dataset": source.name,
            "source_parent_count": len(selected),
            "source_bbox_count": sum(len(record.boxes) for record in selected),
            "output_image_count": len(rows),
            "output_label_count": len(rows),
            "expected_full_output_count": FULL_SOURCE_COUNT * len(VARIANT_TYPES),
            "expected_full_bbox_count": FULL_SOURCE_BOXES * len(VARIANT_TYPES),
            "variant_counts": {
                variant: sum(row["variant_type"] == variant for row in rows) for variant in VARIANT_TYPES
            },
            "subtype_quotas": {"geo": geo_counts, "light": light_counts, "degrade": degrade_counts},
            "split_counts": split_counts,
            "source_selection": "stable output-image path sort, then random.Random(42).sample",
            "split_algorithm": "stable pool output path sort, then one global random.Random(42).shuffle",
            "source_fingerprint_before": fingerprint_before,
            "source_fingerprint_after": fingerprint_after,
            "dataset_fingerprint": fingerprint,
            "dependencies": versions,
            "validation": validation,
            "duplicate_audit": duplicate_audit,
            "parent_split_audit": parent_audit,
            "existing_near_duplicate_audit": audit_report["existing_near_duplicate_audit"],
            "preview_paths": previews,
        }
        write_json(staging / "metadata" / "dataset_fingerprint.json", fingerprint_report)
        write_json(staging / "metadata" / "audit_report.json", audit_report)
        write_json(staging / "metadata" / "augmentation_summary.json", summary)
        write_readme(staging, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
