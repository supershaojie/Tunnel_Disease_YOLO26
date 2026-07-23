# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Unit tests for deterministic overlap-tiled YOLO dataset tooling."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import shutil
import tempfile
import unittest
from contextlib import contextmanager, redirect_stdout
from pathlib import Path
from unittest import mock

from PIL import Image

from tools.tiling import audit_tiled_dataset as auditor
from tools.tiling import create_tiled_yolo_dataset as creator
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
            "names:\n"
            "  0: crack\n",
            encoding="utf-8",
        )
        return source

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
    def audit_exit_code(source: Path, dataset: Path) -> tuple[int, dict]:
        """Run the audit CLI entry point and capture its structured result."""
        stdout = io.StringIO()
        with mock.patch(
            "sys.argv", ["audit_tiled_dataset.py", "--source", str(source), "--dataset", str(dataset)]
        ), redirect_stdout(stdout):
            exit_code = audit_main()
        return exit_code, json.loads(stdout.getvalue())

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

    def test_34_summary_timestamp_and_runtime_versions_are_validated(self) -> None:
        """Invalid generation time and dependency versions cannot pass summary validation."""
        source, output = self.create_source(), self.root / "output"
        self.add_sample(source, "train", "image.png", (32, 32), [])
        build_tiled_dataset(BuildConfig(source, output, tile_size=64))
        summary = self.read_summary(output)
        summary["generated_at"] = "not-a-time"
        summary["versions"]["pillow"] = "0.0"
        self.write_summary(output, summary)
        errors = audit_dataset(source, output)["errors"]
        self.assertTrue(any("generated_at is invalid" in error for error in errors))
        self.assertTrue(any("versions do not match" in error for error in errors))

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
        with mock.patch.object(creator, "_tile_name", return_value=collision):
            with self.assertRaisesRegex(TilingError, "duplicate output tile filename before write"):
                build_tiled_dataset(BuildConfig(source, output, tile_size=64))
        self.assertFalse(output.exists())
        self.assertFalse(any(path.name.startswith(".output.building-") for path in self.root.iterdir()))


if __name__ == "__main__":
    unittest.main()
