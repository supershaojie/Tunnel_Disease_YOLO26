# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Tests for the tunnel-crack augment-first file-level random-split builder."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "datasets" / "Tunnel_Crack_Original_NoAug_7_2_1_seed42"
SCRIPT = ROOT / "tunnel_project" / "scripts" / "02_build_crack_augfirst_diverse5x_randomsplit.py"
SPEC = importlib.util.spec_from_file_location("crack_augfirst_builder", SCRIPT)
assert SPEC and SPEC.loader
BUILDER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BUILDER
SPEC.loader.exec_module(BUILDER)


def synthetic_source(tmp_path: Path) -> BUILDER.SourceRecord:
    """Create one textured class-0 source sample for deterministic unit tests."""
    height, width = 240, 320
    x_ramp = np.tile(np.linspace(30, 140, width, dtype=np.uint8), (height, 1))
    image = np.dstack((x_ramp, np.flip(x_ramp, axis=1), np.full_like(x_ramp, 85)))
    cv2.line(image, (105, 70), (215, 170), (230, 230, 230), 3)
    image_path, label_path = tmp_path / "source.jpg", tmp_path / "source.txt"
    image_path.write_bytes(BUILDER.encode_jpeg(image))
    boxes = ((0.50, 0.50, 0.45, 0.55),)
    label_path.write_bytes(BUILDER.format_yolo_labels(boxes))
    return BUILDER.SourceRecord(
        parent_id="synthetic_parent",
        source_split="train",
        image_path=image_path,
        label_path=label_path,
        output_image_relative="images/train/source.jpg",
        output_label_relative="labels/train/source.txt",
        raw_source_image="unused",
        raw_source_image_relative="raw/source.jpg",
        image_sha256=BUILDER.sha256_file(image_path),
        label_sha256=BUILDER.sha256_file(label_path),
        width=width,
        height=height,
        boxes=boxes,
    )


def test_authoritative_source_pairs_and_boxes() -> None:
    """The builder reads all current repaired NoAug pairs and no legacy labels."""
    records = BUILDER.load_source_records(SOURCE)
    assert len(records) == 2404
    assert sum(len(record.boxes) for record in records) == 2941
    assert len({record.parent_id for record in records}) == 2404
    assert all(record.image_path.is_file() and record.label_path.is_file() for record in records)


def test_fixed_subtype_quotas_use_largest_remainders() -> None:
    """The dry quotas are an exact deterministic scale-down of the fixed full quotas."""
    assert BUILDER.scaled_subtype_quotas(BUILDER.GEO_QUOTAS, 50) == {
        "horizontal_flip": 15,
        "vertical_flip": 8,
        "rotate_180": 5,
        "affine_or_perspective": 22,
    }
    assert BUILDER.scaled_subtype_quotas(BUILDER.LIGHT_QUOTAS, 50) == {
        "local_shadow_or_nonuniform": 15,
        "vignette_or_directional_gradient": 13,
        "low_light_or_gamma": 12,
        "clahe_or_local_contrast": 10,
    }
    assert BUILDER.scaled_subtype_quotas(BUILDER.DEGRADE_QUOTAS, 50) == {
        "motion_or_defocus_blur": 15,
        "gaussian_or_iso_noise": 13,
        "jpeg_or_downsample": 12,
        "mild_mixed_degrade": 10,
    }
    assert BUILDER.scaled_subtype_quotas(BUILDER.GEO_QUOTAS, 2404) == dict(BUILDER.GEO_QUOTAS)


def test_sample_seed_is_stable_and_variant_specific() -> None:
    """SHA-256 seed derivation is repeatable and does not use Python hash()."""
    first = BUILDER.sample_seed(42, "sample_abc", "geo")
    assert first == BUILDER.sample_seed(42, "sample_abc", "geo")
    assert first != BUILDER.sample_seed(42, "sample_abc", "light")
    assert first != BUILDER.sample_seed(43, "sample_abc", "geo")


def test_file_level_shuffle_is_deterministic_and_exact() -> None:
    """A sorted 250-file pool receives one global random shuffle and a 175/50/25 split."""
    rows = [
        {"pool_image": f"pool/images/sample_{index:03d}.jpg", "parent_id": f"p{index // 5:02d}"}
        for index in reversed(range(250))
    ]
    first_rows, second_rows = copy.deepcopy(rows), copy.deepcopy(rows)
    first, first_counts = BUILDER.assign_file_level_splits(first_rows, 42)
    second, second_counts = BUILDER.assign_file_level_splits(second_rows, 42)
    assert first_counts == second_counts == {"train": 175, "val": 50, "test": 25}
    assert [row["pool_image"] for row in first] == [row["pool_image"] for row in second]
    assert [row["pre_shuffle_index"] for row in sorted(first, key=lambda row: row["pre_shuffle_index"])] == list(
        range(250)
    )
    assert {row["split"] for row in first} == {"train", "val", "test"}


def test_bbox_round_trip_and_geometry_preserve_all_boxes(tmp_path: Path) -> None:
    """YOLO box serialization and bbox-aware geometry retain every class-0 box."""
    boxes = ((0.50, 0.50, 0.20, 0.30), (0.30, 0.25, 0.10, 0.10))
    label = tmp_path / "sample.txt"
    label.write_bytes(BUILDER.format_yolo_labels(boxes))
    assert BUILDER.parse_yolo_label(label) == boxes

    height, width = 240, 320
    ramp = np.tile(np.arange(width, dtype=np.uint8), (height, 1))
    image = np.dstack((ramp, np.flip(ramp, axis=1), np.full_like(ramp, 100)))
    pipeline = BUILDER.build_pipeline("geo", "horizontal_flip", 42)
    transformed = pipeline(image=image, bboxes=list(boxes), class_labels=[0, 0])
    output_boxes = BUILDER.canonicalize_boxes(transformed["bboxes"])
    assert len(output_boxes) == len(boxes)
    assert output_boxes[0][0] == pytest.approx(0.50)
    assert output_boxes[1][0] == pytest.approx(0.70)
    assert all(0 <= value <= 1 for box in output_boxes for value in box)


def test_every_declared_augmentation_pipeline_constructs() -> None:
    """Installed Albumentations accepts every dry/full subtype parameterization."""
    declared = {
        "geo": [name for name, _ in BUILDER.GEO_QUOTAS],
        "light": [name for name, _ in BUILDER.LIGHT_QUOTAS],
        "degrade": [name for name, _ in BUILDER.DEGRADE_QUOTAS],
        "compound": ["geometry_light_degrade"],
    }
    for variant_type, subtypes in declared.items():
        for subtype in subtypes:
            if subtype == "local_shadow_or_nonuniform":
                continue
            pipeline = BUILDER.build_pipeline(variant_type, subtype, 42)
            assert pipeline.transforms


def test_required_manifest_fields_and_policy_are_explicit() -> None:
    """Audit manifests expose every requested provenance and split field."""
    required_augmentation = {
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
        "manifest_schema_version",
    }
    required_split = {
        "output_image",
        "parent_id",
        "variant_type",
        "pre_shuffle_index",
        "post_shuffle_index",
        "split_seed",
        "split",
    }
    assert required_augmentation <= set(BUILDER.AUGMENTATION_MANIFEST_FIELDS)
    assert required_split <= set(BUILDER.SPLIT_MANIFEST_FIELDS)
    assert BUILDER.POLICY == {
        "split_policy": "file_level_random",
        "parent_id_grouping": False,
        "independent_test_set": False,
    }


def test_existing_output_is_refused_and_failure_is_atomic(tmp_path: Path) -> None:
    """Existing formal outputs are never overwritten and failed staging is removed."""
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(BUILDER.BuildStop, match="refusing overwrite"):
        BUILDER.validate_output_target(existing)

    output = tmp_path / "dataset"
    staging = output.with_name(f".{output.name}.building")
    with pytest.raises(RuntimeError, match="deliberate"):
        with BUILDER.staged_output(output) as build_dir:
            (build_dir / "partial.txt").write_text("partial", encoding="utf-8")
            raise RuntimeError("deliberate failure")
    assert not output.exists()
    assert not staging.exists()


def test_successful_staging_is_atomically_renamed(tmp_path: Path) -> None:
    """A fully successful build appears only at the formal target."""
    output = tmp_path / "dataset"
    with BUILDER.staged_output(output) as staging:
        (staging / "complete.txt").write_text("complete", encoding="utf-8")
        assert not output.exists()
    assert (output / "complete.txt").read_text(encoding="utf-8") == "complete"
    assert not output.with_name(f".{output.name}.building").exists()


def test_source_fingerprint_is_stable_and_detects_byte_changes(tmp_path: Path) -> None:
    """The before/after source fingerprint is byte-sensitive for images and labels."""
    image = tmp_path / "image.jpg"
    label = tmp_path / "label.txt"
    image.write_bytes(b"image-bytes")
    label.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    record = BUILDER.SourceRecord(
        parent_id="sample_parent",
        source_split="train",
        image_path=image,
        label_path=label,
        output_image_relative="images/train/sample.jpg",
        output_label_relative="labels/train/sample.txt",
        raw_source_image="unused",
        raw_source_image_relative="raw/sample.jpg",
        image_sha256=hashlib.sha256(image.read_bytes()).hexdigest(),
        label_sha256=hashlib.sha256(label.read_bytes()).hexdigest(),
        width=10,
        height=10,
        boxes=((0.5, 0.5, 0.2, 0.2),),
    )
    fingerprint = BUILDER.source_dataset_fingerprint([record])
    assert fingerprint == BUILDER.source_dataset_fingerprint([record])
    image.write_bytes(b"changed")
    with pytest.raises(BUILDER.BuildStop, match="source image SHA-256 changed"):
        BUILDER.source_dataset_fingerprint([record])


def test_soft_shadow_replaces_hard_polygon_and_has_continuous_transition() -> None:
    """The local shadow is a smooth luminance field, never Albumentations' hard polygon shadow."""
    assert "A.RandomShadow(" not in SCRIPT.read_text(encoding="utf-8")
    image = np.full((240, 320, 3), 180, dtype=np.uint8)
    parameters = {
        "model": "gaussian_blurred_elliptical_luminance_attenuation",
        "center_pixels": [160, 120],
        "center_normalized": [0.5, 0.5],
        "semi_axes_pixels": [100, 60],
        "semi_axes_normalized": [0.3125, 0.25],
        "rotation_degrees": 27.0,
        "attenuation": 0.24,
        "blur_sigma_pixels": 24.0,
        "blur_sigma_normalized": 0.1,
        "coverage_threshold": 0.10,
    }
    output, actual = BUILDER.apply_soft_shadow(image, parameters)
    darkness = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.int16) - cv2.cvtColor(
        output, cv2.COLOR_BGR2GRAY
    ).astype(np.int16)
    assert 0.05 < actual["coverage_ratio"] < 0.80
    assert darkness.max() > 20
    assert np.max(np.abs(np.diff(darkness[120].astype(np.float32)))) <= 3
    assert actual["model"].startswith("gaussian_blurred_elliptical")


def test_near_black_light_is_rejected_and_retry_seed_is_deterministic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A source-relative near-black result is rejected before a deterministic second attempt succeeds."""
    source = synthetic_source(tmp_path)
    source_image = BUILDER.read_image(source.image_path)
    observed_seeds: list[int] = []

    def fake_attempt(
        image: np.ndarray,
        boxes: tuple[tuple[float, float, float, float], ...],
        variant_type: str,
        subtype: str,
        current_seed: int,
        recorded_shadow: dict | None = None,
    ) -> tuple[np.ndarray, tuple, list, None]:
        del variant_type, subtype, recorded_shadow
        observed_seeds.append(current_seed)
        output = np.zeros_like(image) if len(observed_seeds) == 1 else np.clip(image.astype(np.int16) + 10, 0, 255).astype(np.uint8)
        return output, boxes, [["MockLocalContrast", {"offset": 10}]], None

    monkeypatch.setattr(BUILDER, "apply_augmentation_attempt", fake_attempt)
    stable_seed = BUILDER.sample_seed(42, source.parent_id, "light")
    _, _, parameters = BUILDER.generate_augmented_bytes(
        source, "light", "clahe_or_local_contrast", stable_seed, {source.image_sha256}
    )
    assert parameters["retry_index"] == 1
    assert observed_seeds == [BUILDER.attempt_seed(stable_seed, 0), BUILDER.attempt_seed(stable_seed, 1)]


def test_bbox_visibility_rejects_locally_erased_crack() -> None:
    """Global exposure cannot conceal loss of contrast and edges inside the crack box."""
    image = np.full((160, 200, 3), 55, dtype=np.uint8)
    cv2.line(image, (75, 45), (125, 115), (220, 220, 220), 3)
    boxes = ((0.5, 0.5, 0.4, 0.6),)
    erased = image.copy()
    erased[32:128, 60:140] = 55
    quality = BUILDER.image_quality(
        erased, boxes, image, boxes, "light", "clahe_or_local_contrast"
    )
    assert not quality["passed"]
    assert {"relative_bbox_contrast", "relative_bbox_edges"} & set(quality["failed_checks"])


def test_json_safe_preserves_small_matrices_and_shadow_values() -> None:
    """Geometry matrices and all compact shadow parameters remain inspectable numeric values."""
    matrix = np.arange(9, dtype=np.float64).reshape(3, 3)
    assert BUILDER.json_safe(matrix) == matrix.tolist()
    shadow = BUILDER.soft_shadow_parameters((240, 320, 3), 42)
    serialized = BUILDER.json_safe(shadow)
    assert isinstance(serialized["center_pixels"], list)
    assert isinstance(serialized["semi_axes_pixels"], list)
    assert isinstance(serialized["rotation_degrees"], float)
    assert isinstance(serialized["attenuation"], float)
    assert isinstance(serialized["blur_sigma_pixels"], float)


def test_large_noise_map_has_digest_and_public_reconstruction_parameters() -> None:
    """Large arrays retain a digest while publishing seed and distribution needed for reconstruction."""
    noise = np.arange(400, dtype=np.float32).reshape(20, 20)
    reconstruction = {
        "method": "rebuild_seeded_pipeline",
        "attempt_seed": 1234,
        "distribution": "gaussian",
        "sigma_fraction": 0.03,
        "mean_fraction": 0.0,
    }
    serialized = BUILDER.json_safe(noise, reconstruction)
    assert serialized["ndarray_shape"] == [20, 20]
    assert serialized["sha256"] == hashlib.sha256(noise.tobytes()).hexdigest()
    assert serialized["reconstruction"] == reconstruction
    assert serialized["empirical_std"] > 0


def test_auditable_gaussian_noise_records_distribution_and_repeats() -> None:
    """Noise is reconstructed from its recorded realization seed and distribution parameters."""
    image = np.full((48, 64, 3), 100, dtype=np.uint8)

    def run() -> dict:
        pipeline = BUILDER.A.Compose(
            [BUILDER.AuditableGaussianNoise((0.02, 0.04), p=1)], seed=42, save_applied_params=True
        )
        return pipeline(image=image)

    first, second = run(), run()
    assert np.array_equal(first["image"], second["image"])
    name, parameters = first["applied_transforms"][0]
    assert name == "AuditableGaussianNoise"
    assert parameters["distribution"] == "gaussian"
    assert 0.02 <= parameters["sigma_fraction"] <= 0.04
    assert isinstance(parameters["random_seed"], int)


def test_compound_manifest_components_and_public_replay_match_hashes(tmp_path: Path) -> None:
    """Compound exposes three actual components and replays image and labels byte-for-byte."""
    source = synthetic_source(tmp_path)
    stable_seed = BUILDER.sample_seed(42, source.parent_id, "compound")
    image_bytes, boxes, parameters = BUILDER.generate_augmented_bytes(
        source, "compound", "geometry_light_degrade", stable_seed, {source.image_sha256}
    )
    assert list(parameters["components"]) == ["geometry", "light", "degrade"]
    assert len(parameters["applied_transforms"]) == 3
    geometry_parameters = parameters["components"]["geometry"][1]
    assert isinstance(geometry_parameters["matrix"], list)
    replay_image, replay_label, replay_boxes, replay_applied = BUILDER.replay_augmented_bytes(
        source, "compound", "geometry_light_degrade", parameters
    )
    assert hashlib.sha256(replay_image).hexdigest() == hashlib.sha256(image_bytes).hexdigest()
    assert hashlib.sha256(replay_label).hexdigest() == hashlib.sha256(BUILDER.format_yolo_labels(boxes)).hexdigest()
    assert replay_boxes == boxes
    assert replay_applied == parameters["applied_transforms"]


def test_same_seed_repeats_geo_image_labels_and_boxes(tmp_path: Path) -> None:
    """The same source, seed, and parameters preserve all boxes and exact output hashes."""
    source = synthetic_source(tmp_path)
    stable_seed = BUILDER.sample_seed(42, source.parent_id, "geo")
    first = BUILDER.generate_augmented_bytes(source, "geo", "horizontal_flip", stable_seed, {source.image_sha256})
    second = BUILDER.generate_augmented_bytes(source, "geo", "horizontal_flip", stable_seed, {source.image_sha256})
    assert hashlib.sha256(first[0]).hexdigest() == hashlib.sha256(second[0]).hexdigest()
    assert first[1] == second[1]
    assert len(first[1]) == len(source.boxes)
    assert hashlib.sha256(BUILDER.format_yolo_labels(first[1])).hexdigest() == hashlib.sha256(
        BUILDER.format_yolo_labels(second[1])
    ).hexdigest()
