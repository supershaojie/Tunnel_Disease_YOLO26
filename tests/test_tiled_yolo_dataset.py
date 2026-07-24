# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Unit tests for deterministic overlap-tiled YOLO dataset tooling."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import platform
import shutil
import subprocess
import tempfile
import unittest
from contextlib import contextmanager, redirect_stdout
from pathlib import Path
from unittest import mock

import PIL
from PIL import Image

from tools.tiling import audit_tiled_dataset as auditor
from tools.tiling import create_tiled_yolo_dataset as creator
from tools.tiling import visualize_tiled_annotations as visualizer
from tools.tiling.audit_tiled_dataset import audit_dataset, main as audit_main
from tools.tiling.create_tiled_yolo_dataset import (
    BuildConfig,
    SourceBox,
    TilingError,
    axis_anchors,
    build_tiled_dataset,
    classify_tile,
    compute_stride,
    generate_tile_anchors,
    xyxy_to_yolo,
    yolo_to_xyxy,
)
from tools.tiling.visualize_tiled_annotations import _render_ambiguous, failure_annotation_lines

LEGACY_SCHEMA2_MANIFEST_FIELDS = (
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
LEGACY_SCHEMA2_SUMMARY_FIELDS = (
    "schema_version",
    "parameters",
    "versions",
    "splits",
    "totals",
    "rejection_reasons",
    "source_data_config",
    "source_data_config_sha256",
    "recorded_source_dataset_fingerprint",
    "full_source_fingerprint",
    "processed_subset_fingerprint",
    "processed_source_images",
    "fingerprint_normalization",
    "output_dataset_fingerprint",
    "generated_at",
    "input_modified",
    "dry_run",
)
LEGACY_SCHEMA2_PARAMETER_FIELDS = (
    "source",
    "output",
    "tile_size",
    "overlap",
    "min_visibility",
    "min_box_size",
    "image_format",
    "jpeg_quality",
    "seed",
    "max_images_per_split",
    "dry_run",
    "stride",
    "padding_value",
    "png_compress_level",
    "jpeg_subsampling",
    "jpeg_optimize",
    "jpeg_progressive",
    "padding",
    "visibility_comparison",
    "max_tile_filename_chars",
    "max_tile_stem_chars",
    "source_path_hash_chars",
)


class TestTiledYoloDataset(unittest.TestCase):
    """Exercise tiling geometry, strict validation, provenance, and safety."""

    def setUp(self) -> None:
        """Create an isolated synthetic workspace."""
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)

    def tearDown(self) -> None:
        """Remove only synthetic files created by this test."""
        self.temporary_directory.cleanup()

    def create_source(self, name: str = "source") -> Path:
        """Create an empty three-split single-class YOLO dataset."""
        source = self.root / name
        for split in ("train", "val", "test"):
            (source / "images" / split).mkdir(parents=True)
            (source / "labels" / split).mkdir(parents=True)
        (source / "data.yaml").write_text(
            "path: .\n"
            "train: images/train\n"
            "val: images/val\n"
            "test: images/test\n"
            "nc: 1\n"
            "names:\n"
            "  0: crack\n",
            encoding="utf-8",
        )
        return source

    @staticmethod
    def write_source_config(
        path: Path,
        *,
        dataset_path: str = ".",
        train: str = "images/train",
        val: str = "images/val",
        test: str = "images/test",
        nc: int | None = 1,
        names: dict[int, str] | None = None,
        comment: str | None = None,
    ) -> bytes:
        """Write one synthetic source YAML and return its exact bytes."""
        names = {0: "crack"} if names is None else names
        lines = [
            f"path: {dataset_path}",
            f"train: {train}",
            f"val: {val}",
            f"test: {test}",
        ]
        if nc is not None:
            lines.append(f"nc: {nc}")
        lines.append("names:")
        lines.extend(f"  {class_id}: {name}" for class_id, name in names.items())
        if comment is not None:
            lines.append(f"# {comment}")
        content = ("\n".join(lines) + "\n").encode()
        path.write_bytes(content)
        return content

    def add_sample(
        self,
        source: Path,
        split: str,
        relative: str,
        size: tuple[int, int],
        labels: list[tuple[int, float, float, float, float]],
    ) -> None:
        """Add one synthetic image and its YOLO label."""
        image_path = source / "images" / split / relative
        label_path = source / "labels" / split / f"{Path(relative).with_suffix('')}.txt"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", size, (20, 40, 60)).save(image_path, format="PNG")
        label_path.write_text(
            "".join(
                f"{class_id} {x:.10f} {y:.10f} {width:.10f} {height:.10f}\n"
                for class_id, x, y, width, height in labels
            ),
            encoding="utf-8",
        )

    @staticmethod
    def manifest_rows(dataset: Path) -> list[dict[str, str]]:
        """Read generated manifest rows."""
        with (dataset / "metadata" / "tile_manifest.csv").open(
            "r", encoding="utf-8", newline=""
        ) as file:
            return list(csv.DictReader(file))

    @staticmethod
    def write_manifest_rows(dataset: Path, rows: list[dict[str, str]]) -> None:
        """Rewrite a synthetic manifest for adversarial audit tests."""
        path = dataset / "metadata" / "tile_manifest.csv"
        with path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def read_summary(dataset: Path) -> dict:
        """Read one synthetic output summary."""
        return json.loads((dataset / "metadata" / "summary.json").read_text(encoding="utf-8"))

    @staticmethod
    def write_summary(dataset: Path, summary: dict) -> None:
        """Rewrite a synthetic summary for adversarial audit tests."""
        (dataset / "metadata" / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    @staticmethod
    def normalized_summary(summary: dict) -> dict:
        """Exclude only the timestamp and output-fingerprint self-reference."""
        normalized = json.loads(json.dumps(summary))
        normalized.pop("generated_at", None)
        normalized.pop("output_dataset_fingerprint", None)
        return normalized

    @classmethod
    def refresh_output_fingerprint(cls, dataset: Path) -> str:
        """Make a deliberately tampered synthetic output cryptographically self-consistent."""
        summary = cls.read_summary(dataset)
        file_hashes = {
            relative: creator.sha256_file(dataset / relative)
            for relative in (
                "data.yaml",
                "metadata/source_data.yaml",
                "metadata/tile_manifest.csv",
            )
        }
        for root_name in ("images", "labels"):
            for path in sorted((dataset / root_name).rglob("*")):
                if path.is_file():
                    relative = path.relative_to(dataset).as_posix()
                    file_hashes[relative] = creator.sha256_file(path)
        fingerprint = creator._canonical_output_fingerprint(file_hashes, summary)
        summary["output_dataset_fingerprint"] = fingerprint
        cls.write_summary(dataset, summary)
        (dataset / "metadata" / "dataset_fingerprint.sha256").write_text(
            f"{fingerprint}  tiled-dataset-v2\n", encoding="utf-8"
        )
        return fingerprint

    @staticmethod
    def audit_exit_code(
        source: Path, dataset: Path, data_yaml: Path | None = None
    ) -> tuple[int, dict]:
        """Run the audit CLI entry point and capture its structured result."""
        stdout = io.StringIO()
        arguments = [
            "audit_tiled_dataset.py",
            "--source",
            str(source),
            "--dataset",
            str(dataset),
        ]
        if data_yaml is not None:
            arguments.extend(("--data-yaml", str(data_yaml)))
        with mock.patch("sys.argv", arguments), redirect_stdout(stdout):
            exit_code = audit_main()
        return exit_code, json.loads(stdout.getvalue())

    @staticmethod
    def fixture_sha256_file(path: Path) -> str:
        """Hash fixture bytes with only the standard library."""
        digest = hashlib.sha256()
        with path.open("rb") as file:
            for block in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    @staticmethod
    def fixture_stable_json(value) -> str:
        """Apply the frozen Schema 2 canonical JSON rule independently."""
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

    @classmethod
    def create_frozen_schema2_fixture(cls, root: Path) -> tuple[Path, Path]:
        """Construct a complete literal Schema 2 source and output without production helpers."""
        source, dataset = root / "schema2-source", root / "schema2-output"
        for split in ("train", "val", "test"):
            (source / "images" / split).mkdir(parents=True)
            (source / "labels" / split).mkdir(parents=True)
            (dataset / "images" / split).mkdir(parents=True)
            (dataset / "labels" / split).mkdir(parents=True)
        (dataset / "metadata").mkdir()

        source_config = (
            "path: .\n"
            "train: images/train\n"
            "val: images/val\n"
            "test: images/test\n"
            "nc: 1\n"
            "names:\n"
            "  0: crack\n"
        ).encode()
        source_config_path = source / "data.yaml"
        source_config_path.write_bytes(source_config)
        source_image_path = source / "images" / "train" / "image.png"
        source_label_path = source / "labels" / "train" / "image.txt"
        with Image.new("RGB", (32, 32), (20, 40, 60)) as source_image:
            source_image.save(source_image_path, format="PNG")
            with Image.new("RGB", (64, 64), (114, 114, 114)) as tile:
                tile.paste(source_image, (0, 0))
                image_relative = "images/train/image.png"
                relative_hash = hashlib.sha256(image_relative.encode()).hexdigest()[:16]
                tile_file = f"train__image__{relative_hash}__x000000__y000000__s64.png"
                output_image_path = dataset / "images" / "train" / tile_file
                tile.save(output_image_path, format="PNG", compress_level=6, optimize=False)
        source_label_path.write_bytes(b"")
        output_label_path = dataset / "labels" / "train" / f"{Path(tile_file).stem}.txt"
        output_label_path.write_bytes(b"")

        output_data = (
            "path: .\n"
            "train: images/train\n"
            "val: images/val\n"
            "test: images/test\n"
            "names:\n"
            "  0: crack\n"
        ).encode()
        (dataset / "data.yaml").write_bytes(output_data)
        (dataset / "metadata" / "source_data.yaml").write_bytes(source_config)

        source_image_sha256 = cls.fixture_sha256_file(source_image_path)
        source_label_sha256 = cls.fixture_sha256_file(source_label_path)
        output_image_sha256 = cls.fixture_sha256_file(output_image_path)
        output_label_sha256 = cls.fixture_sha256_file(output_label_path)
        source_config_sha256 = hashlib.sha256(source_config).hexdigest()
        manifest_row = {
            "split": "train",
            "tile_file": tile_file,
            "source_image": image_relative,
            "source_image_hash": relative_hash,
            "source_image_sha256": source_image_sha256,
            "source_label": "labels/train/image.txt",
            "source_label_sha256": source_label_sha256,
            "tile_x": "0",
            "tile_y": "0",
            "tile_w": "64",
            "tile_h": "64",
            "valid_width": "32",
            "valid_height": "32",
            "source_width": "32",
            "source_height": "32",
            "padding": '{"bottom":32,"left":0,"right":32,"top":0}',
            "padding_left": "0",
            "padding_top": "0",
            "padding_right": "32",
            "padding_bottom": "32",
            "category": "safe_negative",
            "original_intersecting_box_count": "0",
            "retained_box_count": "0",
            "ambiguous_box_count": "0",
            "source_bbox_id": "[]",
            "visibility": "{}",
            "rejected_reason": "[]",
            "source_boxes": "[]",
            "retained_box_records": "[]",
            "failed_boxes": "[]",
            "output_image_sha256": output_image_sha256,
            "output_label_sha256": output_label_sha256,
        }
        if tuple(manifest_row) != LEGACY_SCHEMA2_MANIFEST_FIELDS:
            raise AssertionError("literal Schema 2 manifest fixture order changed")
        manifest_path = dataset / "metadata" / "tile_manifest.csv"
        with manifest_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=LEGACY_SCHEMA2_MANIFEST_FIELDS,
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerow(manifest_row)

        source_fingerprint_payload = {
            "algorithm": "sha256-canonical-source-v2",
            "source_config": "data.yaml",
            "source_config_sha256": source_config_sha256,
            "files": [
                {
                    "split": "train",
                    "image": image_relative,
                    "image_sha256": source_image_sha256,
                    "label": "labels/train/image.txt",
                    "label_sha256": source_label_sha256,
                }
            ],
        }
        source_fingerprint = hashlib.sha256(
            cls.fixture_stable_json(source_fingerprint_payload).encode()
        ).hexdigest()
        train_stats = {
            "source_images_available": 1,
            "source_images_processed": 1,
            "source_label_boxes": 0,
            "candidate_tiles": 1,
            "safe_positive": 0,
            "safe_negative": 1,
            "ambiguous": 0,
            "intersecting_boxes": 0,
            "retained_boxes": 0,
            "failed_boxes": 0,
            "output_label_boxes": 0,
            "empty_labels": 1,
        }
        empty_stats = {
            "source_images_available": 0,
            "source_images_processed": 0,
            "source_label_boxes": 0,
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
        summary = {
            "schema_version": 2,
            "parameters": {
                "source": str(source.resolve()),
                "output": str(dataset.resolve()),
                "tile_size": 64,
                "overlap": 0.0,
                "min_visibility": 0.5,
                "min_box_size": 2.0,
                "image_format": "png",
                "jpeg_quality": None,
                "seed": 42,
                "max_images_per_split": None,
                "dry_run": False,
                "stride": 64,
                "padding_value": 114,
                "png_compress_level": 6,
                "jpeg_subsampling": None,
                "jpeg_optimize": None,
                "jpeg_progressive": None,
                "padding": {"mode": "right_bottom_constant", "value": 114},
                "visibility_comparison": (
                    "retain_if_visibility_greater_than_or_equal_to_threshold"
                ),
                "max_tile_filename_chars": 180,
                "max_tile_stem_chars": 64,
                "source_path_hash_chars": 16,
            },
            "versions": {
                "python": platform.python_version(),
                "pillow": PIL.__version__,
            },
            "splits": {
                "train": train_stats,
                "val": dict(empty_stats),
                "test": dict(empty_stats),
            },
            "totals": dict(train_stats),
            "rejection_reasons": {},
            "source_data_config": "data.yaml",
            "source_data_config_sha256": source_config_sha256,
            "recorded_source_dataset_fingerprint": None,
            "full_source_fingerprint": source_fingerprint,
            "processed_subset_fingerprint": source_fingerprint,
            "processed_source_images": [image_relative],
            "fingerprint_normalization": {
                "output_algorithm": "sha256-canonical-files-v2",
                "source_algorithm": "sha256-canonical-source-v2",
                "covered_output": [
                    "data.yaml",
                    "metadata/source_data.yaml",
                    "metadata/tile_manifest.csv",
                    "images/**",
                    "labels/** including empty labels",
                    "normalized summary.json",
                ],
                "summary_excluded_fields": [
                    "generated_at",
                    "output_dataset_fingerprint",
                ],
                "file_order": "UTF-8 POSIX paths in code-point order",
                "serialization": (
                    "UTF-8 canonical JSON with sorted mapping keys, preserved list order, "
                    "finite floats, and compact separators"
                ),
            },
            "output_dataset_fingerprint": None,
            "generated_at": "2026-01-01T00:00:00+00:00",
            "input_modified": False,
            "dry_run": False,
        }
        if tuple(summary) != LEGACY_SCHEMA2_SUMMARY_FIELDS:
            raise AssertionError("literal Schema 2 summary fixture order changed")
        if tuple(summary["parameters"]) != LEGACY_SCHEMA2_PARAMETER_FIELDS:
            raise AssertionError("literal Schema 2 parameter fixture order changed")

        covered_files = (
            "data.yaml",
            f"images/train/{tile_file}",
            f"labels/train/{Path(tile_file).stem}.txt",
            "metadata/source_data.yaml",
            "metadata/tile_manifest.csv",
        )
        file_hashes = {
            relative: cls.fixture_sha256_file(dataset / relative)
            for relative in covered_files
        }
        normalized_summary = dict(summary)
        normalized_summary.pop("generated_at")
        normalized_summary.pop("output_dataset_fingerprint")
        output_fingerprint_payload = {
            "algorithm": "sha256-canonical-files-v2",
            "files": [
                {"path": relative, "sha256": digest}
                for relative, digest in sorted(file_hashes.items())
            ],
            "summary": normalized_summary,
        }
        output_fingerprint = hashlib.sha256(
            cls.fixture_stable_json(output_fingerprint_payload).encode()
        ).hexdigest()
        summary["output_dataset_fingerprint"] = output_fingerprint
        with (dataset / "metadata" / "dataset_fingerprint.sha256").open(
            "w", encoding="utf-8", newline="\n"
        ) as file:
            file.write(f"{output_fingerprint}  tiled-dataset-v2\n")
        with (dataset / "metadata" / "summary.json").open(
            "w", encoding="utf-8", newline="\n"
        ) as file:
            file.write(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
        return source, dataset

    def test_01_exact_tile_image_has_one_anchor(self) -> None:
        """A 1024-square image produces exactly one tile."""
        self.assertEqual(generate_tile_anchors(1024, 1024, 1024, 819), [(0, 0)])

    def test_02_large_image_anchors_cover_right_and_bottom(self) -> None:
        """A 4096x2168 image receives explicit far-edge anchors."""
        x_anchors = axis_anchors(4096, 1024, 819)
        y_anchors = axis_anchors(2168, 1024, 819)
        self.assertEqual(x_anchors, [0, 819, 1638, 2457, 3072])
        self.assertEqual(y_anchors, [0, 819, 1144])
        self.assertEqual(x_anchors[-1] + 1024, 4096)
        self.assertEqual(y_anchors[-1] + 1024, 2168)
        self.assertEqual(len(generate_tile_anchors(4096, 2168, 1024, 819)), 15)

    def test_03_stride_is_rounded_to_819(self) -> None:
        """The fixed 20% overlap uses the required rounded stride."""
        self.assertEqual(compute_stride(1024, 0.20), 819)

    def test_04_small_image_is_right_bottom_padded_with_114(self) -> None:
        """An undersized source is padded, never stretched."""
        source, output = self.create_source(), self.root / "output"
        self.add_sample(source, "train", "small.png", (40, 30), [])
        build_tiled_dataset(BuildConfig(source, output, tile_size=64, overlap=0))
        row = self.manifest_rows(output)[0]
        self.assertEqual((row["valid_width"], row["valid_height"]), ("40", "30"))
        self.assertEqual((row["padding_right"], row["padding_bottom"]), ("24", "34"))
        with Image.open(output / "images" / "train" / row["tile_file"]) as tile:
            self.assertEqual(tile.size, (64, 64))
            self.assertEqual(tile.getpixel((63, 63)), (114, 114, 114))

    def test_05_fully_contained_bbox_is_safe_positive(self) -> None:
        """A fully contained valid box is retained."""
        box = SourceBox("box-1", 1, 0, (10.0, 12.0, 30.0, 32.0))
        decision = classify_tile([box], 0, 0, 64, 0.5, 2.0)
        self.assertEqual(decision["category"], "safe_positive")
        self.assertEqual(len(decision["retained"]), 1)
        self.assertAlmostEqual(decision["retained"][0]["visibility"], 1.0)

    def test_06_boundary_bbox_is_clipped_correctly(self) -> None:
        """A sufficiently visible boundary box is clipped to the tile."""
        box = SourceBox("box-1", 1, 0, (50.0, 10.0, 70.0, 30.0))
        decision = classify_tile([box], 0, 0, 64, 0.5, 2.0)
        self.assertEqual(decision["category"], "safe_positive")
        retained = decision["retained"][0]
        self.assertEqual(retained["clipped_xyxy"], [50.0, 10.0, 64, 30.0])
        converted = yolo_to_xyxy(*retained["output_yolo"], 64, 64)
        for actual, expected in zip(converted, (50.0, 10.0, 64.0, 30.0)):
            self.assertAlmostEqual(actual, expected)

    def test_07_visibility_equal_to_threshold_is_retained(self) -> None:
        """Visibility exactly equal to 0.50 passes."""
        box = SourceBox("box-1", 1, 0, (50.0, 0.0, 78.0, 10.0))
        decision = classify_tile([box], 0, 0, 64, 0.5, 2.0)
        self.assertEqual(decision["category"], "safe_positive")
        self.assertAlmostEqual(decision["retained"][0]["visibility"], 0.5)

    def test_08_visibility_below_threshold_is_ambiguous(self) -> None:
        """Any intersecting box below visibility threshold makes the tile ambiguous."""
        box = SourceBox("box-1", 1, 0, (50.0, 0.0, 80.0, 10.0))
        decision = classify_tile([box], 0, 0, 64, 0.5, 2.0)
        self.assertEqual(decision["category"], "ambiguous")
        self.assertIn("VISIBILITY_BELOW_THRESHOLD", decision["failed"][0]["reasons"])

    def test_09_valid_and_invalid_boxes_together_are_ambiguous(self) -> None:
        """A retained box cannot rescue a tile containing one failed intersection."""
        boxes = [
            SourceBox("valid", 1, 0, (10.0, 10.0, 20.0, 20.0)),
            SourceBox("invalid", 2, 0, (60.0, 30.0, 100.0, 40.0)),
        ]
        decision = classify_tile(boxes, 0, 0, 64, 0.5, 2.0)
        self.assertEqual(decision["category"], "ambiguous")
        self.assertEqual(len(decision["retained"]), 1)
        self.assertEqual(len(decision["failed"]), 1)

    def test_10_no_intersection_creates_safe_negative_empty_label(self) -> None:
        """A tile with no intersecting source boxes gets an empty label file."""
        source, output = self.create_source(), self.root / "output"
        self.add_sample(source, "train", "wide.png", (128, 64), [(0, 0.75, 0.5, 0.125, 0.25)])
        build_tiled_dataset(BuildConfig(source, output, tile_size=64, overlap=0))
        row = next(row for row in self.manifest_rows(output) if row["tile_x"] == "0")
        self.assertEqual(row["category"], "safe_negative")
        label = output / "labels" / "train" / f"{Path(row['tile_file']).stem}.txt"
        self.assertEqual(label.read_bytes(), b"")

    def test_11_yolo_xyxy_conversion_round_trip(self) -> None:
        """Normalized YOLO and pixel xyxy conversions are inverse operations."""
        xyxy = yolo_to_xyxy(0.5, 0.4, 0.25, 0.2, 800, 600)
        normalized = xyxy_to_yolo(xyxy, 800, 600)
        for actual, expected in zip(normalized, (0.5, 0.4, 0.25, 0.2)):
            self.assertAlmostEqual(actual, expected)

    def test_12_splits_are_inherited_and_never_cross(self) -> None:
        """Manifest and independent audit preserve every source split."""
        source, output = self.create_source(), self.root / "output"
        for split in ("train", "val", "test"):
            self.add_sample(source, split, f"{split}.png", (32, 32), [])
        build_tiled_dataset(BuildConfig(source, output, tile_size=64, overlap=0))
        rows = self.manifest_rows(output)
        self.assertEqual({row["split"] for row in rows}, {"train", "val", "test"})
        for row in rows:
            self.assertTrue(row["source_image"].startswith(f"images/{row['split']}/"))
        self.assertEqual(audit_dataset(source, output)["status"], "PASS")

    def test_13_same_stem_in_different_directories_does_not_collide(self) -> None:
        """Relative-path hashes distinguish identical stems in separate directories."""
        source, output = self.create_source(), self.root / "output"
        self.add_sample(source, "train", "a/same.png", (32, 32), [])
        self.add_sample(source, "train", "b/same.png", (32, 32), [])
        build_tiled_dataset(BuildConfig(source, output, tile_size=64, overlap=0))
        names = [row["tile_file"] for row in self.manifest_rows(output)]
        self.assertEqual(len(names), 2)
        self.assertEqual(len(set(names)), 2)

    def test_14_nonzero_class_fails_immediately(self) -> None:
        """A source class other than zero aborts without creating output."""
        source, output = self.create_source(), self.root / "output"
        self.add_sample(source, "train", "bad.png", (32, 32), [(1, 0.5, 0.5, 0.25, 0.25)])
        with self.assertRaisesRegex(TilingError, "class_id must be 0"):
            build_tiled_dataset(BuildConfig(source, output, tile_size=64, overlap=0))
        self.assertFalse(output.exists())

    def test_15_missing_image_or_label_fails_immediately(self) -> None:
        """Both a missing label and an orphan label are fatal."""
        source = self.create_source("missing_label")
        Image.new("RGB", (32, 32)).save(source / "images" / "train" / "image.png")
        with self.subTest(case="missing_label"):
            with self.assertRaisesRegex(TilingError, "image/label mismatch"):
                build_tiled_dataset(BuildConfig(source, self.root / "output-a", tile_size=64))

        orphan_source = self.create_source("missing_image")
        (orphan_source / "labels" / "train" / "orphan.txt").write_text("", encoding="utf-8")
        with self.subTest(case="missing_image"):
            with self.assertRaisesRegex(TilingError, "image/label mismatch"):
                build_tiled_dataset(BuildConfig(orphan_source, self.root / "output-b", tile_size=64))

    def test_16_nonempty_output_stops_without_overwrite(self) -> None:
        """An existing output directory is never overwritten."""
        source, output = self.create_source(), self.root / "output"
        self.add_sample(source, "train", "image.png", (32, 32), [])
        output.mkdir()
        marker = output / "keep.txt"
        marker.write_text("keep", encoding="utf-8")
        with self.assertRaisesRegex(TilingError, "refusing implicit overwrite"):
            build_tiled_dataset(BuildConfig(source, output, tile_size=64))
        self.assertEqual(marker.read_text(encoding="utf-8"), "keep")

    def test_17_dry_run_creates_no_output_dataset(self) -> None:
        """Dry-run performs validation and statistics without creating output."""
        source, output = self.create_source(), self.root / "output"
        self.add_sample(source, "train", "image.png", (32, 32), [])
        summary = build_tiled_dataset(BuildConfig(source, output, tile_size=64, dry_run=True))
        self.assertTrue(summary["dry_run"])
        self.assertEqual(summary["totals"]["candidate_tiles"], 1)
        self.assertFalse(output.exists())
        self.assertFalse(any(path.name.startswith(".output.building-") for path in self.root.iterdir()))

    def test_18_identical_inputs_and_parameters_are_deterministic(self) -> None:
        """Rebuilding at the same output path preserves every deterministic payload."""
        source, output = self.create_source(), self.root / "deterministic-output"
        self.add_sample(source, "train", "image.png", (96, 64), [(0, 0.5, 0.5, 0.25, 0.25)])
        config = BuildConfig(source, output, tile_size=64, overlap=0.25, seed=7)
        summary_a = build_tiled_dataset(config)
        normalized_a = self.normalized_summary(self.read_summary(output))
        payload_a = {
            path.relative_to(output).as_posix(): path.read_bytes()
            for path in output.rglob("*")
            if path.is_file() and path.name != "summary.json"
        }
        self.assertEqual(audit_dataset(source, output)["status"], "PASS")

        shutil.rmtree(output)
        summary_b = build_tiled_dataset(config)
        normalized_b = self.normalized_summary(self.read_summary(output))
        payload_b = {
            path.relative_to(output).as_posix(): path.read_bytes()
            for path in output.rglob("*")
            if path.is_file() and path.name != "summary.json"
        }
        self.assertEqual(summary_a["parameters"]["source"], summary_b["parameters"]["source"])
        self.assertEqual(summary_a["parameters"]["output"], summary_b["parameters"]["output"])
        self.assertEqual(normalized_a, normalized_b)
        self.assertEqual(
            summary_a["output_dataset_fingerprint"], summary_b["output_dataset_fingerprint"]
        )
        self.assertEqual(payload_a, payload_b)
        self.assertEqual(audit_dataset(source, output)["status"], "PASS")

    def test_19_self_consistent_manifest_and_label_tampering_fails_independent_audit(self) -> None:
        """Manifest, label, summary, and refreshed fingerprint cannot collude against source truth."""
        source, output = self.create_source(), self.root / "output"
        self.add_sample(source, "train", "positive.png", (64, 64), [(0, 0.5, 0.5, 0.25, 0.25)])
        build_tiled_dataset(BuildConfig(source, output, tile_size=64, overlap=0))
        rows = self.manifest_rows(output)
        row = rows[0]
        self.assertEqual(row["category"], "safe_positive")
        row.update(
            {
                "category": "safe_negative",
                "original_intersecting_box_count": "0",
                "retained_box_count": "0",
                "ambiguous_box_count": "0",
                "source_bbox_id": "[]",
                "visibility": "{}",
                "rejected_reason": "[]",
                "source_boxes": "[]",
                "retained_box_records": "[]",
                "failed_boxes": "[]",
                "output_label_sha256": hashlib.sha256(b"").hexdigest(),
            }
        )
        label = output / "labels" / "train" / f"{Path(row['tile_file']).stem}.txt"
        label.write_bytes(b"")
        self.write_manifest_rows(output, rows)
        summary = self.read_summary(output)
        for section in (summary["splits"]["train"], summary["totals"]):
            section["safe_positive"] -= 1
            section["safe_negative"] += 1
            section["intersecting_boxes"] -= 1
            section["retained_boxes"] -= 1
            section["output_label_boxes"] -= 1
            section["empty_labels"] += 1
        self.write_summary(output, summary)
        refreshed = self.refresh_output_fingerprint(output)
        exit_code, report = self.audit_exit_code(source, output)
        self.assertNotEqual(exit_code, 0)
        self.assertEqual(report["status"], "FAIL")
        self.assertTrue(any(".category" in error for error in report["errors"]))
        self.assertEqual(self.read_summary(output)["output_dataset_fingerprint"], refreshed)
        self.assertFalse(any("fingerprint does not match" in error for error in report["errors"]))

    def test_20_visibility_and_rejection_reason_tampering_fails(self) -> None:
        """Independent geometry catches altered visibility and rejection metadata."""
        source, output = self.create_source(), self.root / "output"
        self.add_sample(
            source,
            "train",
            "ambiguous.png",
            (128, 64),
            [(0, 65 / 128, 10 / 64, 30 / 128, 10 / 64)],
        )
        build_tiled_dataset(BuildConfig(source, output, tile_size=64, overlap=0))
        rows = self.manifest_rows(output)
        row = next(item for item in rows if item["category"] == "ambiguous")
        source_bbox_id = json.loads(row["source_bbox_id"])[0]
        row["visibility"] = json.dumps({source_bbox_id: 0.999}, separators=(",", ":"))
        row["rejected_reason"] = "[]"
        self.write_manifest_rows(output, rows)
        errors = audit_dataset(source, output)["errors"]
        self.assertTrue(any(".visibility" in error for error in errors))
        self.assertTrue(any(".rejected_reason" in error for error in errors))

    def test_21_missing_anchor_fails_even_when_surface_counts_and_files_are_updated(self) -> None:
        """Deleting a tile cannot be hidden by adjusting summary counts and formal files."""
        source, output = self.create_source(), self.root / "output"
        self.add_sample(source, "train", "wide.png", (128, 64), [])
        build_tiled_dataset(BuildConfig(source, output, tile_size=64, overlap=0))
        rows = self.manifest_rows(output)
        removed = rows.pop()
        (output / "images" / "train" / removed["tile_file"]).unlink()
        (output / "labels" / "train" / f"{Path(removed['tile_file']).stem}.txt").unlink()
        self.write_manifest_rows(output, rows)
        summary = self.read_summary(output)
        for section in (summary["splits"]["train"], summary["totals"]):
            section["candidate_tiles"] -= 1
            section["safe_negative"] -= 1
            section["empty_labels"] -= 1
        self.write_summary(output, summary)
        report = audit_dataset(source, output)
        self.assertEqual(report["status"], "FAIL")
        self.assertTrue(any("missing independently expected tiles" in error for error in report["errors"]))

    def test_22_replaced_tile_pixels_fail_source_content_validation(self) -> None:
        """An output image with valid dimensions but unrelated pixels is rejected."""
        source, output = self.create_source(), self.root / "output"
        self.add_sample(source, "train", "image.png", (64, 64), [])
        build_tiled_dataset(BuildConfig(source, output, tile_size=64, overlap=0))
        row = self.manifest_rows(output)[0]
        Image.new("RGB", (64, 64), (255, 0, 0)).save(
            output / "images" / "train" / row["tile_file"], format="PNG"
        )
        errors = audit_dataset(source, output)["errors"]
        self.assertTrue(any("PNG pixels do not match source crop" in error for error in errors))

    def test_23_data_yaml_change_is_covered_by_output_fingerprint(self) -> None:
        """Changing otherwise valid data.yaml bytes invalidates the canonical fingerprint."""
        source, output = self.create_source(), self.root / "output"
        self.add_sample(source, "train", "image.png", (32, 32), [])
        build_tiled_dataset(BuildConfig(source, output, tile_size=64))
        with (output / "data.yaml").open("a", encoding="utf-8") as file:
            file.write("# tampered\n")
        errors = audit_dataset(source, output)["errors"]
        self.assertIn("output dataset fingerprint does not match exact disk payload", errors)

    def test_24_source_data_copy_change_fails_content_and_fingerprint_checks(self) -> None:
        """Changing metadata/source_data.yaml is detected independently and cryptographically."""
        source, output = self.create_source(), self.root / "output"
        self.add_sample(source, "train", "image.png", (32, 32), [])
        build_tiled_dataset(BuildConfig(source, output, tile_size=64))
        with (output / "metadata" / "source_data.yaml").open("a", encoding="utf-8") as file:
            file.write("# tampered\n")
        errors = audit_dataset(source, output)["errors"]
        self.assertIn("metadata/source_data.yaml does not exactly match the source configuration", errors)
        self.assertIn("output dataset fingerprint does not match exact disk payload", errors)

    def test_25_summary_parameters_counts_and_reasons_are_independently_checked(self) -> None:
        """Each class of deterministic summary field is protected."""
        source = self.create_source()
        self.add_sample(
            source,
            "train",
            "ambiguous.png",
            (128, 64),
            [(0, 65 / 128, 10 / 64, 30 / 128, 10 / 64)],
        )
        cases = {
            "parameter": lambda value: value["parameters"].__setitem__("min_box_size", 3.0),
            "count": lambda value: value["splits"]["train"].__setitem__(
                "candidate_tiles", value["splits"]["train"]["candidate_tiles"] + 1
            ),
            "reason": lambda value: value.__setitem__("rejection_reasons", {"FAKE_REASON": 99}),
        }
        for index, (case, mutate) in enumerate(cases.items()):
            with self.subTest(case=case):
                output = self.root / f"output-{index}"
                build_tiled_dataset(BuildConfig(source, output, tile_size=64, overlap=0))
                summary = self.read_summary(output)
                mutate(summary)
                self.write_summary(output, summary)
                report = audit_dataset(source, output)
                self.assertEqual(report["status"], "FAIL")
                if case == "count":
                    self.assertTrue(any("split counters" in error for error in report["errors"]))
                elif case == "reason":
                    self.assertTrue(any("rejection reasons" in error for error in report["errors"]))
                else:
                    self.assertTrue(any("fingerprint" in error for error in report["errors"]))

    def test_26_full_source_fingerprint_covers_unselected_files(self) -> None:
        """A max-images smoke build still protects bytes of every unselected source pair."""
        source, output = self.create_source(), self.root / "output"
        for index in range(3):
            self.add_sample(source, "train", f"{index}.png", (32, 32), [])
        summary = build_tiled_dataset(
            BuildConfig(source, output, tile_size=64, max_images_per_split=1, seed=9)
        )
        self.assertEqual(len(summary["processed_source_images"]), 1)
        all_images = {f"images/train/{index}.png" for index in range(3)}
        unselected = sorted(all_images - set(summary["processed_source_images"]))[0]
        (source / unselected.replace("images/", "labels/")).with_suffix(".txt").write_text(
            "0 0.5 0.5 0.25 0.25\n", encoding="utf-8"
        )
        report = audit_dataset(source, output)
        self.assertEqual(report["status"], "FAIL")
        self.assertTrue(any("full_source_fingerprint" in error for error in report["errors"]))

    def test_27_unselected_source_change_during_build_aborts(self) -> None:
        """The post-build full snapshot detects a concurrent change to an unselected label."""
        source, output = self.create_source(), self.root / "output"
        for index in range(3):
            self.add_sample(source, "train", f"{index}.png", (32, 32), [])
        dry_summary = build_tiled_dataset(
            BuildConfig(
                source,
                self.root / "unused-output",
                tile_size=64,
                max_images_per_split=1,
                seed=11,
                dry_run=True,
            )
        )
        all_images = {f"images/train/{index}.png" for index in range(3)}
        unselected = sorted(all_images - set(dry_summary["processed_source_images"]))[0]
        unselected_label = (source / unselected.replace("images/", "labels/")).with_suffix(".txt")
        original_save = creator._save_image
        changed = False

        def save_then_mutate(*args, **kwargs):
            nonlocal changed
            original_save(*args, **kwargs)
            if not changed:
                unselected_label.write_text("0 0.5 0.5 0.25 0.25\n", encoding="utf-8")
                changed = True

        with mock.patch.object(creator, "_save_image", side_effect=save_then_mutate):
            with self.assertRaisesRegex(TilingError, "source input changed during build"):
                build_tiled_dataset(
                    BuildConfig(source, output, tile_size=64, max_images_per_split=1, seed=11)
                )
        self.assertFalse(output.exists())

    def test_28_long_same_stem_and_chinese_names_are_bounded_unique_and_deterministic(self) -> None:
        """Windows-safe naming truncates long stems while retaining stable path hashes."""
        source = self.create_source()
        long_name = "very_long_" + "a" * 140 + ".png"
        self.add_sample(source, "train", f"目录一/{long_name}", (32, 32), [])
        self.add_sample(source, "train", f"目录二/{long_name}", (32, 32), [])
        self.add_sample(source, "train", f"中文目录/{'裂缝' * 35}.png", (32, 32), [])
        output_a, output_b = self.root / "output-a", self.root / "output-b"
        build_tiled_dataset(BuildConfig(source, output_a, tile_size=64))
        build_tiled_dataset(BuildConfig(source, output_b, tile_size=64))
        names_a = [row["tile_file"] for row in self.manifest_rows(output_a)]
        names_b = [row["tile_file"] for row in self.manifest_rows(output_b)]
        self.assertEqual(names_a, names_b)
        self.assertEqual(len(names_a), len(set(name.casefold() for name in names_a)))
        self.assertTrue(all(len(name) <= 180 for name in names_a))
        path_hashes = [name.split("__")[2] for name in names_a]
        self.assertTrue(all(len(value) == 16 for value in path_hashes))
        self.assertTrue(all(set(value) <= set("0123456789abcdef") for value in path_hashes))

    def test_29_multiple_ambiguous_boxes_preserve_per_box_visualization_details(self) -> None:
        """Every failed box remains recoverable and individually labeled for visualization."""
        source, output = self.create_source(), self.root / "output"
        self.add_sample(
            source,
            "train",
            "multi.png",
            (128, 64),
            [
                (0, 65 / 128, 10 / 64, 30 / 128, 10 / 64),
                (0, 80 / 128, 35 / 64, 40 / 128, 10 / 64),
            ],
        )
        build_tiled_dataset(BuildConfig(source, output, tile_size=64, overlap=0))
        row = next(item for item in self.manifest_rows(output) if item["category"] == "ambiguous")
        failed = json.loads(row["failed_boxes"])
        annotations = failure_annotation_lines(failed)
        self.assertEqual(len(failed), 2)
        self.assertEqual(len(annotations), 2)
        for record, annotation in zip(failed, annotations):
            self.assertIn(record["source_bbox_id"], annotation)
            self.assertIn("visibility=", annotation)
            self.assertIn("reasons=", annotation)
            self.assertIn("source=[", annotation)
            self.assertIn("intersection=[", annotation)
        rendered = _render_ambiguous(row, source)
        self.assertGreater(rendered.height, 64)

    def test_30_malformed_summary_csv_and_image_return_nonzero_cli_status(self) -> None:
        """Malformed numbers, missing CSV columns, and damaged images produce structured CLI failures."""
        source = self.create_source()
        self.add_sample(source, "train", "image.png", (32, 32), [])
        for case in ("number", "csv", "image"):
            with self.subTest(case=case):
                output = self.root / f"output-{case}"
                build_tiled_dataset(BuildConfig(source, output, tile_size=64))
                if case == "number":
                    summary = self.read_summary(output)
                    summary["parameters"]["overlap"] = "not-a-number"
                    self.write_summary(output, summary)
                elif case == "csv":
                    manifest = output / "metadata" / "tile_manifest.csv"
                    lines = manifest.read_text(encoding="utf-8").splitlines()
                    lines[0] = ",".join(lines[0].split(",")[:-1])
                    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
                else:
                    row = self.manifest_rows(output)[0]
                    (output / "images" / "train" / row["tile_file"]).write_bytes(b"not an image")
                exit_code, report = self.audit_exit_code(source, output)
                self.assertNotEqual(exit_code, 0)
                self.assertEqual(report["status"], "FAIL")
                self.assertTrue(report["errors"])

    def test_31_jpeg_content_is_reencoded_and_verified(self) -> None:
        """JPEG audit reproduces quality and encoder settings instead of skipping content checks."""
        source, output = self.create_source(), self.root / "output"
        self.add_sample(source, "train", "image.png", (64, 64), [])
        build_tiled_dataset(
            BuildConfig(source, output, tile_size=64, image_format="jpg", jpeg_quality=92)
        )
        self.assertEqual(audit_dataset(source, output)["status"], "PASS")
        row = self.manifest_rows(output)[0]
        image_path = output / "images" / "train" / row["tile_file"]
        with Image.open(image_path) as image:
            image.convert("RGB").save(
                image_path,
                format="JPEG",
                quality=80,
                subsampling=0,
                optimize=False,
                progressive=False,
            )
        errors = audit_dataset(source, output)["errors"]
        self.assertTrue(any("JPEG bytes do not match" in error for error in errors))

    def test_32_visibility_just_below_threshold_is_not_hidden_by_epsilon(self) -> None:
        """A mathematically sub-threshold visibility is ambiguous even when very close to 0.50."""
        box = SourceBox("box-1", 1, 0, (50.0, 0.0, 78.000000000056, 10.0))
        decision = classify_tile([box], 0, 0, 64, 0.5, 2.0)
        self.assertLess(14.0 / 28.000000000056, 0.5)
        self.assertEqual(decision["category"], "ambiguous")
        self.assertIn("VISIBILITY_BELOW_THRESHOLD", decision["failed"][0]["reasons"])

    def test_33_output_label_value_tampering_fails_field_comparison(self) -> None:
        """A legal-looking but geometrically wrong output label is rejected field by field."""
        source, output = self.create_source(), self.root / "output"
        self.add_sample(source, "train", "positive.png", (64, 64), [(0, 0.5, 0.5, 0.25, 0.25)])
        build_tiled_dataset(BuildConfig(source, output, tile_size=64))
        row = self.manifest_rows(output)[0]
        label = output / "labels" / "train" / f"{Path(row['tile_file']).stem}.txt"
        label.write_text("0 0.6000000000 0.5000000000 0.2500000000 0.2500000000\n", encoding="utf-8")
        errors = audit_dataset(source, output)["errors"]
        self.assertTrue(any("YOLO field 1 mismatch" in error for error in errors))

    def test_34_summary_timestamp_and_historical_versions_are_validated(self) -> None:
        """Invalid generation time and dependency-version provenance cannot pass validation."""
        source, output = self.create_source(), self.root / "output"
        self.add_sample(source, "train", "image.png", (32, 32), [])
        build_tiled_dataset(BuildConfig(source, output, tile_size=64))
        summary = self.read_summary(output)
        summary["generated_at"] = "not-a-time"
        summary["versions"]["pillow"] = "0.0"
        self.write_summary(output, summary)
        errors = audit_dataset(source, output)["errors"]
        self.assertTrue(any("generated_at is invalid" in error for error in errors))
        self.assertTrue(any("versions are invalid" in error for error in errors))

    def test_35_summary_source_parameter_tampering_changes_fingerprint(self) -> None:
        """The source path remains inside canonical summary fingerprint coverage."""
        source, output = self.create_source(), self.root / "output"
        self.add_sample(source, "train", "image.png", (32, 32), [])
        build_tiled_dataset(BuildConfig(source, output, tile_size=64))
        summary = self.read_summary(output)
        original_fingerprint = summary["output_dataset_fingerprint"]
        summary["parameters"]["source"] = str(self.root / "forged-source")
        self.write_summary(output, summary)
        report = audit_dataset(source, output)
        self.assertEqual(report["status"], "FAIL")
        self.assertNotEqual(report["output_dataset_fingerprint"], original_fingerprint)
        self.assertIn("output dataset fingerprint does not match exact disk payload", report["errors"])

    def test_36_summary_output_parameter_tampering_changes_fingerprint(self) -> None:
        """The output path remains inside canonical summary fingerprint coverage."""
        source, output = self.create_source(), self.root / "output"
        self.add_sample(source, "train", "image.png", (32, 32), [])
        build_tiled_dataset(BuildConfig(source, output, tile_size=64))
        summary = self.read_summary(output)
        original_fingerprint = summary["output_dataset_fingerprint"]
        summary["parameters"]["output"] = str(self.root / "forged-output")
        self.write_summary(output, summary)
        report = audit_dataset(source, output)
        self.assertEqual(report["status"], "FAIL")
        self.assertNotEqual(report["output_dataset_fingerprint"], original_fingerprint)
        self.assertIn("output dataset fingerprint does not match exact disk payload", report["errors"])

    def test_37_forged_extra_anchor_fails_after_surface_state_is_refingerprinted(self) -> None:
        """An extra image, label, manifest row, summary count, and fresh fingerprint still fail."""
        source, output = self.create_source(), self.root / "output"
        self.add_sample(source, "train", "wide.png", (128, 64), [])
        build_tiled_dataset(BuildConfig(source, output, tile_size=64, overlap=0))
        rows = self.manifest_rows(output)
        original = rows[0]
        forged = dict(original)
        forged_name = "train__forged__0123456789abcdef__x000001_y000000_t64.png"
        forged.update({"tile_file": forged_name, "tile_x": "1"})
        original_image = output / "images" / "train" / original["tile_file"]
        forged_image = output / "images" / "train" / forged_name
        forged_image.write_bytes(original_image.read_bytes())
        forged_label = output / "labels" / "train" / f"{Path(forged_name).stem}.txt"
        forged_label.write_bytes(b"")
        forged["output_image_sha256"] = creator.sha256_file(forged_image)
        forged["output_label_sha256"] = creator.sha256_file(forged_label)
        rows.append(forged)
        self.write_manifest_rows(output, rows)
        summary = self.read_summary(output)
        for section in (summary["splits"]["train"], summary["totals"]):
            section["candidate_tiles"] += 1
            section["safe_negative"] += 1
            section["empty_labels"] += 1
        self.write_summary(output, summary)
        refreshed = self.refresh_output_fingerprint(output)
        exit_code, report = self.audit_exit_code(source, output)
        self.assertNotEqual(exit_code, 0)
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(self.read_summary(output)["output_dataset_fingerprint"], refreshed)
        self.assertTrue(any("manifest contains unexpected tiles" in error for error in report["errors"]))
        self.assertFalse(any("fingerprint does not match" in error for error in report["errors"]))

    def test_38_source_images_are_streamed_and_closed_during_audit(self) -> None:
        """Auditing several source images never holds more than one source image context."""
        source, output = self.create_source(), self.root / "output"
        for index in range(4):
            self.add_sample(source, "train", f"{index}.png", (96, 64), [])
        build_tiled_dataset(BuildConfig(source, output, tile_size=64, overlap=0))
        original_open_rgb = auditor._open_rgb
        active = opened = closed = maximum_active = 0

        @contextmanager
        def counted_open_rgb(path: Path):
            nonlocal active, opened, closed, maximum_active
            opened += 1
            active += 1
            maximum_active = max(maximum_active, active)
            try:
                with original_open_rgb(path) as image:
                    yield image
            finally:
                active -= 1
                closed += 1

        with mock.patch.object(auditor, "_open_rgb", counted_open_rgb):
            report = audit_dataset(source, output)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(maximum_active, 1)
        self.assertEqual(active, 0)
        self.assertEqual(opened, closed)
        self.assertEqual(opened, 8)

    def test_39_duplicate_generated_filename_aborts_before_overwrite(self) -> None:
        """A constructed naming collision is detected before the second tile can overwrite."""
        source, output = self.create_source(), self.root / "output"
        self.add_sample(source, "train", "one.png", (32, 32), [])
        self.add_sample(source, "train", "two.png", (32, 32), [])
        collision = "train__collision__0123456789abcdef__x000000_y000000_t64.png"
        staging = self.root / f".output.building-{creator.os.getpid()}"
        with mock.patch.object(creator, "_tile_name", return_value=collision):
            with self.assertRaises(TilingError) as context:
                build_tiled_dataset(BuildConfig(source, output, tile_size=64))
        self.assertIn("duplicate output tile filename before write", str(context.exception))
        self.assertIn(f"staging retained at {staging}", str(context.exception))
        self.assertFalse(output.exists())
        self.assertTrue(staging.is_dir())
        self.assertTrue((staging / "images" / "train" / collision).is_file())

    def test_40_explicit_data_yaml_strictly_selects_data_local(self) -> None:
        """An explicit data_local.yaml wins even when data.yaml exists beside it."""
        source, output = self.create_source(), self.root / "output"
        self.add_sample(source, "train", "image.png", (32, 32), [])
        local_yaml = source / "data_local.yaml"
        local_content = self.write_source_config(
            local_yaml,
            dataset_path=source.as_posix(),
            nc=None,
            comment="explicit local configuration",
        )
        summary = build_tiled_dataset(
            BuildConfig(source, output, data_yaml=local_yaml, tile_size=64)
        )
        self.assertEqual(summary["source_data_config"], "data_local.yaml")
        self.assertEqual(summary["source_data_config_path"], str(local_yaml.resolve()))
        self.assertEqual(summary["source_data_config_selection"], "explicit")
        self.assertEqual(
            (output / "metadata" / "source_data.yaml").read_bytes(), local_content
        )
        self.assertNotEqual(
            (output / "metadata" / "source_data.yaml").read_bytes(),
            (source / "data.yaml").read_bytes(),
        )

    def test_41_invalid_server_data_yaml_does_not_interfere_with_explicit_local_yaml(self) -> None:
        """Automatic-candidate server paths are never consulted after an explicit selection."""
        source, output = self.create_source(), self.root / "output"
        self.add_sample(source, "train", "image.png", (32, 32), [])
        self.write_source_config(
            source / "data.yaml", dataset_path="/root/nonexistent/server/dataset"
        )
        local_yaml = source / "data_local.yaml"
        self.write_source_config(local_yaml, dataset_path=source.as_posix(), nc=None)
        summary = build_tiled_dataset(
            BuildConfig(source, output, data_yaml=local_yaml, tile_size=64, dry_run=True)
        )
        self.assertTrue(summary["dry_run"])
        self.assertEqual(summary["source_data_config"], "data_local.yaml")
        self.assertEqual(summary["totals"]["source_images_processed"], 1)
        self.assertFalse(output.exists())

    def test_42_automatic_source_yaml_discovery_remains_backward_compatible(self) -> None:
        """Without --data-yaml, data.yaml remains the first automatic candidate."""
        source, output = self.create_source(), self.root / "output"
        self.add_sample(source, "train", "image.png", (32, 32), [])
        automatic_content = self.write_source_config(
            source / "data.yaml", comment="automatic first candidate"
        )
        self.write_source_config(
            source / "data_local.yaml",
            dataset_path=source.as_posix(),
            comment="automatic second candidate",
        )
        summary = build_tiled_dataset(BuildConfig(source, output, tile_size=64))
        self.assertEqual(summary["parameters"]["data_yaml"], None)
        self.assertEqual(summary["source_data_config"], "data.yaml")
        self.assertEqual(summary["source_data_config_selection"], "auto")
        self.assertEqual(
            (output / "metadata" / "source_data.yaml").read_bytes(), automatic_content
        )
        report = audit_dataset(source, output)
        self.assertEqual(report["status"], "PASS", report["errors"])

    def test_43_missing_explicit_data_yaml_fails_before_output_write(self) -> None:
        """A nonexistent explicit YAML is rejected before creating any output."""
        source, output = self.create_source(), self.root / "output"
        missing = source / "missing.yaml"
        with self.assertRaisesRegex(TilingError, "--data-yaml must be an existing regular file"):
            build_tiled_dataset(BuildConfig(source, output, data_yaml=missing, tile_size=64))
        self.assertFalse(output.exists())
        self.assertFalse(any(path.name.startswith(".output.building-") for path in self.root.iterdir()))

    def test_44_explicit_data_yaml_directory_fails_before_output_write(self) -> None:
        """A directory cannot be accepted as an explicit dataset configuration."""
        source, output = self.create_source(), self.root / "output"
        with self.assertRaisesRegex(TilingError, "--data-yaml must be an existing regular file"):
            build_tiled_dataset(BuildConfig(source, output, data_yaml=source, tile_size=64))
        self.assertFalse(output.exists())
        self.assertFalse(any(path.name.startswith(".output.building-") for path in self.root.iterdir()))

    def test_45_declared_split_path_mismatch_fails_before_scanning(self) -> None:
        """YAML split declarations must resolve to the exact source directories that will be scanned."""
        source, output = self.create_source(), self.root / "output"
        local_yaml = source / "data_local.yaml"
        self.write_source_config(local_yaml, train="images/not-train")
        with self.assertRaisesRegex(TilingError, "train path resolves to .*actual scan directory"):
            build_tiled_dataset(BuildConfig(source, output, data_yaml=local_yaml, tile_size=64))
        self.assertFalse(output.exists())

    def test_46_invalid_names_or_nc_fails_before_output_write(self) -> None:
        """The effective class count and class mapping must both be exactly one crack class."""
        for case in ("names", "nc"):
            with self.subTest(case=case):
                source = self.create_source(f"source-{case}")
                output = self.root / f"output-{case}"
                local_yaml = source / "data_local.yaml"
                if case == "names":
                    self.write_source_config(
                        local_yaml, nc=2, names={0: "crack", 1: "other"}
                    )
                    expected = "source names must be exactly"
                else:
                    self.write_source_config(local_yaml, nc=2)
                    expected = "source nc must be 1"
                with self.assertRaisesRegex(TilingError, expected):
                    build_tiled_dataset(
                        BuildConfig(source, output, data_yaml=local_yaml, tile_size=64)
                    )
                self.assertFalse(output.exists())

    def test_47_explicit_config_dry_run_creates_no_output_or_staging(self) -> None:
        """Explicit YAML selection does not weaken the no-write dry-run guarantee."""
        source, output = self.create_source(), self.root / "output"
        self.add_sample(source, "train", "image.png", (32, 32), [])
        local_yaml = source / "data_local.yaml"
        self.write_source_config(local_yaml, dataset_path=source.as_posix(), nc=None)
        summary = build_tiled_dataset(
            BuildConfig(source, output, data_yaml=local_yaml, tile_size=64, dry_run=True)
        )
        self.assertTrue(summary["dry_run"])
        self.assertEqual(summary["parameters"]["data_yaml"], str(local_yaml.resolve()))
        self.assertEqual(summary["source_data_config_selection"], "explicit")
        self.assertFalse(output.exists())
        self.assertFalse(any(path.name.startswith(".output.building-") for path in self.root.iterdir()))

    def test_48_explicit_config_provenance_reaches_manifest_summary_and_fingerprints(self) -> None:
        """Selected YAML identity and bytes are preserved and independently audited everywhere."""
        source, output = self.create_source(), self.root / "output"
        self.add_sample(source, "train", "image.png", (32, 32), [])
        local_yaml = source / "data_local.yaml"
        local_content = self.write_source_config(
            local_yaml,
            dataset_path=source.as_posix(),
            nc=None,
            comment="fingerprinted explicit configuration",
        )
        summary = build_tiled_dataset(
            BuildConfig(source, output, data_yaml=local_yaml, tile_size=64)
        )
        config_sha256 = hashlib.sha256(local_content).hexdigest()
        rows = self.manifest_rows(output)
        self.assertTrue(rows)
        self.assertTrue(all(row["source_data_config"] == "data_local.yaml" for row in rows))
        self.assertTrue(
            all(row["source_data_config_sha256"] == config_sha256 for row in rows)
        )
        self.assertEqual(summary["source_data_config_sha256"], config_sha256)
        self.assertRegex(summary["full_source_fingerprint"], r"^[0-9a-f]{64}$")
        self.assertRegex(summary["processed_subset_fingerprint"], r"^[0-9a-f]{64}$")
        self.assertEqual(
            (output / "metadata" / "source_data.yaml").read_bytes(), local_content
        )
        self.assertEqual(audit_dataset(source, output)["status"], "PASS")
        local_yaml.write_bytes(local_content + b"# changed after generation\n")
        report = audit_dataset(source, output)
        self.assertEqual(report["status"], "FAIL")
        self.assertNotEqual(
            report["full_source_fingerprint"], summary["full_source_fingerprint"]
        )
        self.assertNotEqual(
            report["processed_subset_fingerprint"], summary["processed_subset_fingerprint"]
        )
        self.assertTrue(
            any("full_source_fingerprint" in error for error in report["errors"])
        )
        self.assertTrue(
            any("processed_subset_fingerprint" in error for error in report["errors"])
        )
        self.assertTrue(
            any("does not exactly match" in error for error in report["errors"])
        )

    def test_49_create_help_documents_explicit_data_yaml_precedence(self) -> None:
        """CLI help exposes --data-yaml and states that explicit selection disables discovery."""
        stdout = io.StringIO()
        with mock.patch("sys.argv", ["create_tiled_yolo_dataset.py", "--help"]), redirect_stdout(
            stdout
        ), self.assertRaises(SystemExit) as context:
            creator.parse_args()
        help_text = stdout.getvalue()
        normalized_help = " ".join(help_text.split())
        self.assertEqual(context.exception.code, 0)
        self.assertIn("--data-yaml", help_text)
        self.assertIn("used exclusively", normalized_help)
        self.assertIn("automatic data.yaml/data_local.yaml discovery", normalized_help)

    def test_50_cli_forwards_explicit_data_yaml_without_writing_output(self) -> None:
        """The CLI forwards --data-yaml to the build while preserving dry-run no-write behavior."""
        source, output = self.create_source(), self.root / "output"
        self.add_sample(source, "train", "image.png", (32, 32), [])
        self.write_source_config(
            source / "data.yaml", dataset_path="/root/nonexistent/server/dataset"
        )
        local_yaml = source / "data_local.yaml"
        self.write_source_config(local_yaml, dataset_path=source.as_posix(), nc=None)
        stdout = io.StringIO()
        arguments = [
            "create_tiled_yolo_dataset.py",
            "--source",
            str(source),
            "--data-yaml",
            str(local_yaml),
            "--output",
            str(output),
            "--tile-size",
            "64",
            "--overlap",
            "0",
            "--dry-run",
        ]
        with mock.patch("sys.argv", arguments), redirect_stdout(stdout):
            exit_code = creator.main()
        summary = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(summary["parameters"]["data_yaml"], str(local_yaml.resolve()))
        self.assertEqual(summary["source_data_config"], "data_local.yaml")
        self.assertEqual(summary["source_data_config_selection"], "explicit")
        self.assertEqual(summary["totals"]["source_images_processed"], 1)
        self.assertFalse(output.exists())

    def test_51_schema3_source_and_dataset_migration_audits_from_current_source(self) -> None:
        """Portable config identity lets a complete Schema 3 pair move to a new root."""
        source, output = self.create_source(), self.root / "output"
        self.add_sample(source, "train", "image.png", (32, 32), [])
        build_tiled_dataset(BuildConfig(source, output, tile_size=64))
        migrated = self.root / "迁移后"
        migrated.mkdir()
        new_source = Path(shutil.move(str(source), str(migrated / "source")))
        new_output = Path(shutil.move(str(output), str(migrated / "output")))
        report = audit_dataset(new_source, new_output)
        self.assertEqual(report["status"], "PASS", report["errors"])
        self.assertFalse(source.exists())
        self.assertFalse(output.exists())

    def test_52_invalid_historical_absolute_paths_do_not_break_migrated_fingerprints(self) -> None:
        """Historical paths may disappear while all three recorded fingerprints remain stable."""
        source, output = self.create_source(), self.root / "output"
        self.add_sample(source, "train", "image.png", (32, 32), [])
        original = build_tiled_dataset(BuildConfig(source, output, tile_size=64))
        historical_config = Path(original["source_data_config_path"])
        migrated = self.root / "new-root"
        migrated.mkdir()
        new_source = Path(shutil.move(str(source), str(migrated / "source")))
        new_output = Path(shutil.move(str(output), str(migrated / "output")))
        self.assertFalse(historical_config.exists())
        report = audit_dataset(new_source, new_output)
        self.assertEqual(report["status"], "PASS", report["errors"])
        for field in (
            "full_source_fingerprint",
            "processed_subset_fingerprint",
            "output_dataset_fingerprint",
        ):
            self.assertEqual(report[field], original[field])

    def test_53_migrated_external_config_passes_with_current_audit_data_yaml(self) -> None:
        """An external config is relocated explicitly and never through its historical path."""
        bundle = self.root / "bundle"
        source = self.create_source("bundle/source")
        self.add_sample(source, "train", "image.png", (32, 32), [])
        external = bundle / "config" / "external.yaml"
        external.parent.mkdir()
        self.write_source_config(external, dataset_path="../source")
        output = bundle / "output"
        summary = build_tiled_dataset(
            BuildConfig(source, output, data_yaml=external, tile_size=64)
        )
        migrated_bundle = Path(shutil.move(str(bundle), str(self.root / "migrated-bundle")))
        new_source = migrated_bundle / "source"
        new_output = migrated_bundle / "output"
        new_external = migrated_bundle / "config" / "external.yaml"
        self.assertFalse(Path(summary["source_data_config_path"]).exists())
        report = audit_dataset(new_source, new_output, new_external)
        self.assertEqual(report["status"], "PASS", report["errors"])
        self.assertEqual(report["full_source_fingerprint"], summary["full_source_fingerprint"])
        row = self.manifest_rows(new_output)[0]
        self.assertEqual(summary["source_data_config_relative"], None)
        self.assertEqual(
            summary["source_data_config_external"], summary["source_data_config"]
        )
        self.assertEqual(summary["source_data_config_selection"], "explicit")
        self.assertEqual(summary["source_data_config_path"], str(external.resolve()))
        self.assertEqual(
            summary["source_data_config_sha256"],
            hashlib.sha256(new_external.read_bytes()).hexdigest(),
        )
        self.assertEqual(row["source_data_config_relative"], "")
        self.assertEqual(
            row["source_data_config_external"], summary["source_data_config_external"]
        )

    def test_54_external_config_without_current_data_yaml_fails_clearly(self) -> None:
        """An external identity cannot be guessed from the current source tree."""
        source = self.create_source("bundle/source")
        self.add_sample(source, "train", "image.png", (32, 32), [])
        external = self.root / "bundle" / "config" / "external.yaml"
        external.parent.mkdir()
        self.write_source_config(external, dataset_path="../source")
        output = self.root / "bundle" / "output"
        build_tiled_dataset(BuildConfig(source, output, data_yaml=external, tile_size=64))
        with self.assertRaisesRegex(
            auditor.AuditError, "external to --source.*--data-yaml"
        ):
            audit_dataset(source, output)
        self.assertEqual(audit_dataset(source, output, external)["status"], "PASS")

    def test_55_current_yaml_hash_mismatch_still_fails_after_portable_selection(self) -> None:
        """Location independence does not weaken exact current-config byte validation."""
        source, output = self.create_source(), self.root / "output"
        self.add_sample(source, "train", "image.png", (32, 32), [])
        build_tiled_dataset(BuildConfig(source, output, tile_size=64))
        (source / "data.yaml").write_bytes((source / "data.yaml").read_bytes() + b"# changed\n")
        report = audit_dataset(source, output)
        self.assertEqual(report["status"], "FAIL")
        self.assertTrue(
            any("source_data_config_sha256" in error for error in report["errors"]),
            report["errors"],
        )
        self.assertTrue(
            any("does not exactly match" in error for error in report["errors"]),
            report["errors"],
        )

    def test_56_source_image_label_and_split_changes_remain_detectable(self) -> None:
        """Migration support still detects independent image, label, and split changes."""
        for mutation in ("image", "label", "split"):
            with self.subTest(mutation=mutation):
                source = self.create_source(f"source-{mutation}")
                output = self.root / f"output-{mutation}"
                self.add_sample(
                    source,
                    "train",
                    "image.png",
                    (32, 32),
                    [(0, 0.5, 0.5, 0.5, 0.5)],
                )
                build_tiled_dataset(BuildConfig(source, output, tile_size=64))
                image = source / "images" / "train" / "image.png"
                label = source / "labels" / "train" / "image.txt"
                if mutation == "image":
                    Image.new("RGB", (32, 32), (1, 2, 3)).save(image)
                elif mutation == "label":
                    label.write_text("0 0.4 0.4 0.25 0.25\n", encoding="utf-8")
                else:
                    shutil.move(str(image), str(source / "images" / "val" / image.name))
                    shutil.move(str(label), str(source / "labels" / "val" / label.name))
                report = audit_dataset(source, output)
                self.assertEqual(report["status"], "FAIL")
                self.assertTrue(report["input_modified"])

    def test_57_real_schema2_field_structure_remains_auditable(self) -> None:
        """A literal pre-upgrade fixture passes without calling current production builders."""
        with mock.patch.object(
            creator, "build_tiled_dataset", side_effect=AssertionError("generator called")
        ) as generator_build, mock.patch(
            f"{__name__}.build_tiled_dataset", side_effect=AssertionError("imported generator called")
        ) as imported_build, mock.patch.object(
            creator, "_manifest_row", side_effect=AssertionError("generator manifest called")
        ) as generator_manifest, mock.patch.object(
            creator, "_source_fingerprint", side_effect=AssertionError("generator fingerprint called")
        ) as generator_source_fingerprint, mock.patch.object(
            creator,
            "_canonical_output_fingerprint",
            side_effect=AssertionError("generator output fingerprint called"),
        ) as generator_output_fingerprint, mock.patch.object(
            auditor, "_source_fingerprint", side_effect=AssertionError("auditor fingerprint called")
        ) as auditor_source_fingerprint, mock.patch.object(
            auditor,
            "_canonical_output_fingerprint",
            side_effect=AssertionError("auditor output fingerprint called"),
        ) as auditor_output_fingerprint:
            source, output = self.create_frozen_schema2_fixture(self.root)
        for production_helper in (
            generator_build,
            imported_build,
            generator_manifest,
            generator_source_fingerprint,
            generator_output_fingerprint,
            auditor_source_fingerprint,
            auditor_output_fingerprint,
        ):
            production_helper.assert_not_called()
        summary = self.read_summary(output)
        rows = self.manifest_rows(output)
        self.assertEqual(summary["schema_version"], 2)
        self.assertNotIn("data_yaml", summary["parameters"])
        self.assertEqual(tuple(rows[0]), LEGACY_SCHEMA2_MANIFEST_FIELDS)
        self.assertEqual(tuple(summary), LEGACY_SCHEMA2_SUMMARY_FIELDS)
        self.assertEqual(
            tuple(summary["parameters"]), LEGACY_SCHEMA2_PARAMETER_FIELDS
        )
        report = audit_dataset(source, output)
        self.assertEqual(report["status"], "PASS", report["errors"])

    def test_58_unknown_schema_version_fails_before_manifest_trust(self) -> None:
        """Only genuine Schema 2 and current Schema 3 are accepted."""
        source, output = self.create_source(), self.root / "output"
        self.add_sample(source, "train", "image.png", (32, 32), [])
        build_tiled_dataset(BuildConfig(source, output, tile_size=64))
        summary = self.read_summary(output)
        summary["schema_version"] = 99
        self.write_summary(output, summary)
        with self.assertRaisesRegex(auditor.AuditError, "unsupported summary schema_version"):
            audit_dataset(source, output)

    def test_59_schema3_manifest_contains_complete_config_provenance(self) -> None:
        """Every Schema 3 row records identity, mode, history, hash, and location kind."""
        source, output = self.create_source(), self.root / "output"
        self.add_sample(source, "train", "image.png", (32, 32), [])
        summary = build_tiled_dataset(BuildConfig(source, output, tile_size=64))
        row = self.manifest_rows(output)[0]
        expected = {
            "source_data_config": "data.yaml",
            "source_data_config_selection": "auto",
            "source_data_config_path": summary["source_data_config_path"],
            "source_data_config_sha256": summary["source_data_config_sha256"],
            "source_data_config_relative": "data.yaml",
            "source_data_config_external": "",
        }
        for field, value in expected.items():
            self.assertIn(field, auditor.MANIFEST_FIELDS)
            self.assertEqual(row[field], value)

    def test_60_manifest_summary_config_provenance_conflict_fails(self) -> None:
        """A fresh output fingerprint cannot hide manifest/summary config-mode disagreement."""
        source, output = self.create_source(), self.root / "output"
        self.add_sample(source, "train", "image.png", (32, 32), [])
        build_tiled_dataset(BuildConfig(source, output, tile_size=64))
        rows = self.manifest_rows(output)
        rows[0]["source_data_config_selection"] = "explicit"
        self.write_manifest_rows(output, rows)
        self.refresh_output_fingerprint(output)
        report = audit_dataset(source, output)
        self.assertEqual(report["status"], "FAIL")
        self.assertTrue(
            any(
                ".source_data_config_selection" in error
                for error in report["errors"]
            ),
            report["errors"],
        )
        self.assertFalse(
            any("fingerprint does not match" in error for error in report["errors"])
        )

    def test_61_dry_run_never_deletes_or_modifies_preexisting_staging(self) -> None:
        """A preexisting same-name staging sentinel survives a dry-run failure byte-for-byte."""
        source, output = self.create_source(), self.root / "output"
        self.add_sample(source, "train", "image.png", (32, 32), [])
        staging = self.root / f".output.building-{creator.os.getpid()}"
        staging.mkdir()
        sentinel = staging / "sentinel.bin"
        sentinel.write_bytes(b"third-party")
        with self.assertRaisesRegex(TilingError, "staging path already exists"):
            build_tiled_dataset(BuildConfig(source, output, tile_size=64, dry_run=True))
        self.assertEqual(sentinel.read_bytes(), b"third-party")
        self.assertTrue(staging.is_dir())
        self.assertFalse(output.exists())

    def test_62_formal_failure_does_not_delete_unowned_preexisting_staging(self) -> None:
        """Formal generation refuses but never cleans a staging directory it did not create."""
        source, output = self.create_source(), self.root / "output"
        self.add_sample(source, "train", "image.png", (32, 32), [])
        staging = self.root / f".output.building-{creator.os.getpid()}"
        staging.mkdir()
        sentinel = staging / "sentinel.txt"
        sentinel.write_text("owned elsewhere", encoding="utf-8")
        with self.assertRaisesRegex(TilingError, "staging path already exists"):
            build_tiled_dataset(BuildConfig(source, output, tile_size=64))
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "owned elsewhere")
        self.assertTrue(staging.is_dir())

    def test_63_formal_failure_retains_owned_staging_without_recursive_cleanup(self) -> None:
        """A controlled write failure preserves its staging content and never invokes rmtree."""
        source, output = self.create_source(), self.root / "output"
        self.add_sample(source, "train", "image.png", (32, 32), [])
        staging = self.root / f".output.building-{creator.os.getpid()}"
        partial_files: list[Path] = []

        def save_partial_then_fail(_image, path, *_args) -> None:
            path.write_bytes(b"partial output retained for inspection")
            partial_files.append(path)
            raise RuntimeError("synthetic write failure")

        with mock.patch.object(
            creator, "_save_image", side_effect=save_partial_then_fail
        ), mock.patch("shutil.rmtree") as recursive_delete:
            with self.assertRaises(TilingError) as context:
                build_tiled_dataset(BuildConfig(source, output, tile_size=64))
        self.assertIn("synthetic write failure", str(context.exception))
        self.assertIn(f"staging retained at {staging}", str(context.exception))
        self.assertIn("manual confirmation is required before cleanup", str(context.exception))
        recursive_delete.assert_not_called()
        self.assertTrue(staging.is_dir())
        self.assertEqual(len(partial_files), 1)
        self.assertEqual(
            partial_files[0].read_bytes(), b"partial output retained for inspection"
        )
        self.assertFalse(output.exists())
        self.assertTrue((source / "images" / "train" / "image.png").is_file())

    def test_64_absolute_yaml_split_paths_are_validated_and_used(self) -> None:
        """Absolute train/val/test declarations resolve to the exact scanned directories."""
        source, output = self.create_source(), self.root / "output"
        self.add_sample(source, "train", "image.png", (32, 32), [])
        explicit = source / "absolute.yaml"
        self.write_source_config(
            explicit,
            train=(source / "images" / "train").as_posix(),
            val=(source / "images" / "val").as_posix(),
            test=(source / "images" / "test").as_posix(),
        )
        summary = build_tiled_dataset(
            BuildConfig(source, output, data_yaml=explicit, tile_size=64, dry_run=True)
        )
        self.assertTrue(summary["dry_run"])
        self.assertEqual(summary["totals"]["source_images_processed"], 1)
        self.assertFalse(output.exists())

    def test_65_windows_backslash_yaml_paths_remain_supported(self) -> None:
        """Windows-style relative separators map to native source split paths."""
        source, output = self.create_source(), self.root / "output"
        self.add_sample(source, "train", "image.png", (32, 32), [])
        explicit = source / "backslashes.yaml"
        self.write_source_config(
            explicit,
            train=r"images\train",
            val=r"images\val",
            test=r"images\test",
        )
        summary = build_tiled_dataset(
            BuildConfig(source, output, data_yaml=explicit, tile_size=64, dry_run=True)
        )
        self.assertEqual(summary["source_data_config"], "backslashes.yaml")
        self.assertEqual(summary["totals"]["source_images_processed"], 1)
        self.assertFalse(output.exists())

    def test_66_chinese_and_long_migrated_paths_audit_successfully(self) -> None:
        """Portable provenance survives Unicode and long directory components."""
        source = self.create_source("中文_" + "长目录" * 20)
        output = self.root / ("输出_" + "长目录" * 15)
        self.add_sample(source, "train", "裂缝图像.png", (32, 32), [])
        build_tiled_dataset(BuildConfig(source, output, tile_size=64))
        migrated = self.root / ("迁移_" + "新目录" * 15)
        migrated.mkdir()
        new_source = Path(shutil.move(str(source), str(migrated / source.name)))
        new_output = Path(shutil.move(str(output), str(migrated / output.name)))
        report = audit_dataset(new_source, new_output)
        self.assertEqual(report["status"], "PASS", report["errors"])

    def test_67_audit_help_documents_current_data_yaml_migration_rule(self) -> None:
        """Audit help exposes exclusive current-machine config selection for migration."""
        stdout = io.StringIO()
        with mock.patch("sys.argv", ["audit_tiled_dataset.py", "--help"]), redirect_stdout(
            stdout
        ), self.assertRaises(SystemExit) as context:
            auditor.parse_args()
        normalized = " ".join(stdout.getvalue().split())
        self.assertEqual(context.exception.code, 0)
        self.assertIn("--data-yaml", normalized)
        self.assertIn("external to --source", normalized)
        self.assertIn("used exclusively", normalized)

    def test_68_replaced_same_name_staging_is_not_deleted_by_exception_cleanup(self) -> None:
        """The no-cleanup failure path preserves a concurrent replacement directory."""
        source, output = self.create_source(), self.root / "output"
        self.add_sample(source, "train", "image.png", (32, 32), [])
        staging = self.root / f".output.building-{creator.os.getpid()}"
        sentinel = staging / "third-party.txt"

        def replace_staging_then_fail(*_args, **_kwargs) -> None:
            shutil.rmtree(staging)
            staging.mkdir()
            sentinel.write_text("replacement", encoding="utf-8")
            raise RuntimeError("concurrent replacement")

        with mock.patch.object(creator, "_save_image", side_effect=replace_staging_then_fail):
            with self.assertRaises(TilingError) as context:
                build_tiled_dataset(BuildConfig(source, output, tile_size=64))
        self.assertIn("concurrent replacement", str(context.exception))
        self.assertIn(f"staging retained at {staging}", str(context.exception))
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "replacement")
        self.assertTrue(staging.is_dir())
        self.assertFalse(output.exists())

    def test_69_schema3_audit_does_not_require_generation_runtime_versions(self) -> None:
        """Valid historical runtime versions remain provenance after cross-machine migration."""
        source, output = self.create_source(), self.root / "output"
        self.add_sample(source, "train", "image.png", (32, 32), [])
        build_tiled_dataset(BuildConfig(source, output, tile_size=64))
        summary = self.read_summary(output)
        summary["versions"] = {"python": "3.8.0", "pillow": "9.0.0"}
        self.write_summary(output, summary)
        self.refresh_output_fingerprint(output)
        report = audit_dataset(source, output)
        self.assertEqual(report["status"], "PASS", report["errors"])

    def test_70_portable_config_identity_rejects_cross_platform_escape_syntax(self) -> None:
        """Portable provenance rejects roots, drives, UNC paths, backslashes, and parent traversal."""
        unsafe_values = (
            r"C:\outside\data.yaml",
            "C:/outside/data.yaml",
            r"\\server\share\data.yaml",
            "//server/share/data.yaml",
            "/outside/data.yaml",
            "../outside/data.yaml",
            r"..\outside\data.yaml",
            "safe/../outside/data.yaml",
            r"safe\..\outside\data.yaml",
            r"safe\outside/data.yaml",
            r"safe/outside\data.yaml",
            "",
            ".",
            "./",
            "safe//data.yaml",
            "safe/./data.yaml",
            "safe/nested/../../outside/data.yaml",
        )
        for value in unsafe_values:
            with self.subTest(value=value), self.assertRaises(auditor.AuditError) as context:
                auditor._safe_source_relative(value, "synthetic identity")
            self.assertRegex(str(context.exception), r"unsafe|nonempty")
        for value in ("configs/data.yaml", "nested/configs/data.yaml"):
            with self.subTest(value=value):
                self.assertEqual(
                    auditor._safe_source_relative(value, "synthetic identity"),
                    value,
                )

    def test_71_nested_internal_config_migrates_and_audits_from_new_root(self) -> None:
        """A canonical nested relative config remains discoverable after whole-bundle migration."""
        source, output = self.create_source(), self.root / "output"
        self.add_sample(source, "train", "image.png", (32, 32), [])
        nested_config = source / "configs" / "data_local.yaml"
        nested_config.parent.mkdir()
        self.write_source_config(nested_config, dataset_path="..")
        summary = build_tiled_dataset(
            BuildConfig(source, output, data_yaml=nested_config, tile_size=64)
        )
        self.assertEqual(
            summary["source_data_config_relative"], "configs/data_local.yaml"
        )
        migrated = self.root / "migrated"
        migrated.mkdir()
        new_source = Path(shutil.move(str(source), str(migrated / "source")))
        new_output = Path(shutil.move(str(output), str(migrated / "output")))
        report = audit_dataset(new_source, new_output)
        self.assertEqual(report["status"], "PASS", report["errors"])
        self.assertFalse(Path(summary["source_data_config_path"]).exists())

    def test_72_symlinked_internal_identity_cannot_resolve_outside_source(self) -> None:
        """A source-contained lexical name cannot traverse an outward file symlink."""
        source = self.create_source()
        external_config = self.root / "outside-data.yaml"
        self.write_source_config(external_config)
        link = source / "linked-data.yaml"
        try:
            link.symlink_to(external_config)
        except OSError as error:
            resolved_source = source.resolve()
            resolved_external = external_config.resolve()
            path_type = type(source)
            original_resolve = path_type.resolve

            def simulate_outward_symlink(path, *args, **kwargs):
                if path == link:
                    return resolved_external
                return original_resolve(path, *args, **kwargs)

            with mock.patch.object(
                path_type, "resolve", autospec=True, side_effect=simulate_outward_symlink
            ), self.assertRaisesRegex(
                auditor.AuditError, "resolves outside --source"
            ):
                auditor._resolve_source_relative(
                    resolved_source, "linked-data.yaml", "synthetic identity"
                )
            self.assertIsInstance(error, OSError)
            return
        with self.assertRaisesRegex(auditor.AuditError, "resolves outside --source"):
            auditor._resolve_source_relative(
                source, "linked-data.yaml", "synthetic identity"
            )

    def test_73_windows_junction_cannot_resolve_outside_source(self) -> None:
        """Resolved containment rejects an outward junction when Windows can create one."""
        if creator.os.name != "nt":
            self.skipTest("Windows junctions are not available on this platform")
        source = self.create_source()
        external_directory = self.root / "outside-junction-target"
        external_directory.mkdir()
        self.write_source_config(external_directory / "data.yaml")
        junction = source / "junction"
        result = subprocess.run(
            [
                "cmd.exe",
                "/d",
                "/c",
                "mklink",
                "/J",
                str(junction),
                str(external_directory),
            ],
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            self.skipTest(
                f"junction creation is unavailable: {result.stderr.decode(errors='replace')}"
            )
        try:
            with self.assertRaisesRegex(
                auditor.AuditError, "resolves outside --source"
            ):
                auditor._resolve_source_relative(
                    source, "junction/data.yaml", "synthetic identity"
                )
        finally:
            junction.rmdir()

    def test_74_unsafe_identity_fails_before_any_file_probe_or_config_read(self) -> None:
        """Lexically unsafe provenance is rejected before filesystem content is consulted."""
        summary = {
            "source_data_config_relative": r"C:\outside\data.yaml",
            "source_data_config_external": None,
        }
        with mock.patch.object(
            Path, "is_file", side_effect=AssertionError("unexpected file probe")
        ) as file_probe, mock.patch.object(
            Path, "read_bytes", side_effect=AssertionError("unexpected config read")
        ) as content_read, mock.patch.object(
            auditor.yaml,
            "safe_load",
            side_effect=AssertionError("unexpected YAML parse"),
        ) as yaml_parse:
            with self.assertRaisesRegex(auditor.AuditError, "unsafe"):
                auditor._select_current_config(self.root / "source", summary, 3, None)
        file_probe.assert_not_called()
        content_read.assert_not_called()
        yaml_parse.assert_not_called()

    def test_75_successful_formal_build_still_atomically_publishes_without_rmtree(self) -> None:
        """Successful generation replaces staging with output and never requests recursive cleanup."""
        source, output = self.create_source(), self.root / "output"
        self.add_sample(source, "train", "image.png", (32, 32), [])
        staging = self.root / f".output.building-{creator.os.getpid()}"
        with mock.patch("shutil.rmtree") as recursive_delete:
            summary = build_tiled_dataset(BuildConfig(source, output, tile_size=64))
        recursive_delete.assert_not_called()
        self.assertFalse(staging.exists())
        self.assertTrue(output.is_dir())
        self.assertFalse(summary["dry_run"])
        report = audit_dataset(source, output)
        self.assertEqual(report["status"], "PASS", report["errors"])

    def test_76_visualization_failure_retains_staging_without_rmtree(self) -> None:
        """Visualization uses the same conservative no-delete failure semantics."""
        source, dataset = self.create_source(), self.root / "dataset"
        self.add_sample(source, "train", "image.png", (32, 32), [])
        build_tiled_dataset(BuildConfig(source, dataset, tile_size=64))
        output = self.root / "review"
        staging = self.root / f".review.building-{visualizer.os.getpid()}"
        with mock.patch.object(
            visualizer, "_render_safe", side_effect=RuntimeError("render failure")
        ), mock.patch("shutil.rmtree") as recursive_delete:
            with self.assertRaises(visualizer.VisualizationError) as context:
                visualizer.visualize(
                    source, dataset, output, ["safe_negative"], 1, seed=42
                )
        recursive_delete.assert_not_called()
        self.assertIn("render failure", str(context.exception))
        self.assertIn(f"staging retained at {staging}", str(context.exception))
        self.assertTrue(staging.is_dir())
        self.assertTrue((staging / "safe_negative").is_dir())
        self.assertFalse(output.exists())
        self.assertTrue(source.is_dir())
        self.assertTrue(dataset.is_dir())

    def test_77_visualization_success_still_atomically_publishes(self) -> None:
        """Successful visualization publishes staging without recursive deletion."""
        source, dataset = self.create_source(), self.root / "dataset"
        self.add_sample(source, "train", "image.png", (32, 32), [])
        build_tiled_dataset(BuildConfig(source, dataset, tile_size=64))
        output = self.root / "review"
        staging = self.root / f".review.building-{visualizer.os.getpid()}"
        with mock.patch("shutil.rmtree") as recursive_delete:
            counts = visualizer.visualize(
                source, dataset, output, ["safe_negative"], 1, seed=42
            )
        recursive_delete.assert_not_called()
        self.assertEqual(counts, {"safe_negative": 1})
        self.assertTrue(output.is_dir())
        self.assertFalse(staging.exists())
        self.assertEqual(len(list((output / "safe_negative").glob("*.png"))), 1)

    def test_78_auto_internal_nested_config_alias_builds_and_audits(self) -> None:
        """An automatic candidate resolving to a nested in-source YAML remains portable."""
        source, output = self.create_source(), self.root / "output"
        self.add_sample(source, "train", "image.png", (32, 32), [])
        automatic_candidate = source / "data.yaml"
        automatic_candidate.unlink()
        nested_config = source / "configs" / "data.yaml"
        nested_config.parent.mkdir()
        self.write_source_config(nested_config, dataset_path="..")
        resolved_nested = nested_config.resolve()
        path_type = type(source)
        original_resolve = path_type.resolve

        def simulate_internal_alias(path, *args, **kwargs):
            if path == automatic_candidate:
                return resolved_nested
            return original_resolve(path, *args, **kwargs)

        with mock.patch.object(
            path_type,
            "resolve",
            autospec=True,
            side_effect=simulate_internal_alias,
        ):
            summary = build_tiled_dataset(BuildConfig(source, output, tile_size=64))
        self.assertEqual(summary["source_data_config_selection"], "auto")
        self.assertEqual(summary["source_data_config_relative"], "configs/data.yaml")
        self.assertEqual(summary["source_data_config_path"], str(resolved_nested))
        report = audit_dataset(source, output)
        self.assertEqual(report["status"], "PASS", report["errors"])

    def test_79_auto_outward_file_symlink_is_rejected_when_supported(self) -> None:
        """A real automatic file symlink cannot escape source or fall back to data_local.yaml."""
        source, output = self.create_source(), self.root / "output"
        automatic_candidate = source / "data.yaml"
        automatic_candidate.unlink()
        self.write_source_config(source / "data_local.yaml")
        external_config = self.root / "outside.yaml"
        self.write_source_config(external_config)
        try:
            automatic_candidate.symlink_to(external_config)
        except OSError as error:
            self.skipTest(f"real file symlink creation is unavailable: {error}")
        with self.assertRaisesRegex(TilingError, "resolves outside --source"):
            build_tiled_dataset(BuildConfig(source, output, tile_size=64, dry_run=True))
        self.assertFalse(output.exists())
        self.assertTrue((source / "data_local.yaml").is_file())

    def test_80_auto_escape_fails_before_probe_read_parse_hash_or_fallback(self) -> None:
        """Resolved auto containment fails before all content access and without mode fallback."""
        source = self.create_source()
        self.write_source_config(source / "data_local.yaml", comment="must not be selected")
        automatic_candidate = source / "data.yaml"
        resolved_source = source.resolve()
        resolved_external = (self.root / "outside.yaml").resolve()
        path_type = type(source)
        original_resolve = path_type.resolve

        def simulate_outward_alias(path, *args, **kwargs):
            if path == automatic_candidate:
                return resolved_external
            return original_resolve(path, *args, **kwargs)

        with mock.patch.object(
            path_type,
            "resolve",
            autospec=True,
            side_effect=simulate_outward_alias,
        ), mock.patch.object(
            Path,
            "is_file",
            side_effect=AssertionError("unexpected file probe"),
        ) as file_probe, mock.patch.object(
            Path,
            "read_bytes",
            side_effect=AssertionError("unexpected config read"),
        ) as content_read, mock.patch.object(
            creator.yaml,
            "safe_load",
            side_effect=AssertionError("unexpected YAML parse"),
        ) as yaml_parse, mock.patch.object(
            creator.hashlib,
            "sha256",
            side_effect=AssertionError("unexpected config hash"),
        ) as hash_probe:
            with self.assertRaisesRegex(TilingError, "resolves outside --source"):
                creator._load_source_config(resolved_source, None)
        file_probe.assert_not_called()
        content_read.assert_not_called()
        yaml_parse.assert_not_called()
        hash_probe.assert_not_called()

    def test_81_auto_outward_windows_junction_is_rejected_before_fallback(self) -> None:
        """A real Windows junction at an automatic candidate cannot escape source."""
        if creator.os.name != "nt":
            self.skipTest("Windows junctions are not available on this platform")
        source = self.create_source()
        automatic_candidate = source / "data.yaml"
        automatic_candidate.unlink()
        local_fallback = source / "data_local.yaml"
        local_content = self.write_source_config(local_fallback)
        external_directory = self.root / "outside-auto-junction"
        external_directory.mkdir()
        result = subprocess.run(
            [
                "cmd.exe",
                "/d",
                "/c",
                "mklink",
                "/J",
                str(automatic_candidate),
                str(external_directory),
            ],
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            self.skipTest(
                f"junction creation is unavailable: {result.stderr.decode(errors='replace')}"
            )
        try:
            with self.assertRaisesRegex(TilingError, "resolves outside --source"):
                creator._load_source_config(source.resolve(), None)
            self.assertEqual(local_fallback.read_bytes(), local_content)
        finally:
            automatic_candidate.rmdir()


if __name__ == "__main__":
    unittest.main()
