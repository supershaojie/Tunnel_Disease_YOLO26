"""Audit a raw YOLO detection dataset without modifying it."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
CLASS_ID_PATTERN = re.compile(r"^[+-]?\d+$")
PLANNED_CRACK_CLASS_ID = 0
BOUNDARY_TOLERANCE = 1e-6


def parse_nonnegative_int(value: str) -> int:
    """Parse a nonnegative integer command-line value."""
    if not CLASS_ID_PATTERN.fullmatch(value):
        raise argparse.ArgumentTypeError(f"expected a nonnegative integer, got {value!r}")
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError(f"expected a nonnegative integer, got {value!r}")
    return parsed


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root", required=True, type=Path, help="Root containing images/, labels/, and data.yaml"
    )
    parser.add_argument("--target-class-id", required=True, type=parse_nonnegative_int, help="Raw crack class ID")
    return parser.parse_args()


def relative_path(path: Path, root: Path) -> str:
    """Return a stable POSIX-style path relative to the dataset root."""
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def stem_key(path: Path, base: Path) -> str:
    """Build a case-insensitive relative stem used to pair images and labels."""
    relative = path.relative_to(base)
    return (relative.parent / relative.stem).as_posix().casefold()


def find_files(root: Path, suffixes: set[str]) -> list[Path]:
    """Find files recursively with case-insensitive suffix matching."""
    return sorted(
        (path for path in root.rglob("*") if path.is_file() and path.suffix.casefold() in suffixes),
        key=lambda path: path.as_posix().casefold(),
    )


def basic_stats(values: list[float | int]) -> dict[str, int | float | None]:
    """Return count, minimum, maximum, mean, and median for numeric values."""
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None, "median": None}
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
    }


def human_bytes(value: int) -> str:
    """Format a byte count for the Markdown report."""
    amount = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if amount < 1024 or unit == "TiB":
            return f"{amount:.2f} {unit}"
        amount /= 1024
    return f"{amount:.2f} TiB"


def add_invalid_entry(
    entries: list[dict[str, Any]],
    label_path: str,
    line_number: int | None,
    error_codes: list[str],
    details: list[str],
    line_text: str = "",
) -> None:
    """Append one label validation entry."""
    entries.append(
        {
            "label_path": label_path,
            "line_number": line_number,
            "error_codes": error_codes,
            "details": details,
            "line_text": line_text,
        }
    )


def scan_labels(
    label_files: list[Path], dataset_root: Path
) -> tuple[dict[Path, set[int]], Counter[int], dict[str, Any]]:
    """Scan all labels while retaining errors instead of failing fast."""
    classes_by_label: dict[Path, set[int]] = {}
    class_box_counts: Counter[int] = Counter()
    invalid_entries: list[dict[str, Any]] = []
    empty_files: list[str] = []
    read_errors: list[dict[str, str]] = []
    error_code_counts: Counter[str] = Counter()
    total_nonblank_rows = 0
    parseable_box_rows = 0
    strictly_valid_rows = 0
    duplicate_rows = 0

    for label_path in label_files:
        display_path = relative_path(label_path, dataset_root)
        classes: set[int] = set()
        classes_by_label[label_path] = classes
        try:
            text = label_path.read_text(encoding="utf-8-sig")
        except (OSError, UnicodeError) as error:
            details = [f"Unable to read label as UTF-8: {error}"]
            add_invalid_entry(invalid_entries, display_path, None, ["LABEL_READ_ERROR"], details)
            error_code_counts["LABEL_READ_ERROR"] += 1
            read_errors.append({"path": display_path, "error": str(error)})
            continue

        nonblank_lines = [(number, line.strip()) for number, line in enumerate(text.splitlines(), 1) if line.strip()]
        if not nonblank_lines:
            empty_files.append(display_path)
            add_invalid_entry(
                invalid_entries,
                display_path,
                None,
                ["EMPTY_LABEL_FILE"],
                ["The label file contains no nonblank rows."],
            )
            error_code_counts["EMPTY_LABEL_FILE"] += 1
            continue

        first_line_for_canonical: dict[str, int] = {}
        for line_number, line_text in nonblank_lines:
            total_nonblank_rows += 1
            tokens = line_text.split()
            errors: list[str] = []
            details: list[str] = []
            class_id: int | None = None
            coordinates: list[float] | None = None

            if len(tokens) != 5:
                errors.append("COLUMN_COUNT_NOT_5")
                details.append(f"Expected 5 columns, found {len(tokens)}.")

            if tokens:
                if CLASS_ID_PATTERN.fullmatch(tokens[0]):
                    class_id = int(tokens[0])
                    if class_id < 0:
                        errors.append("NEGATIVE_CLASS_ID")
                        details.append(f"class_id must be nonnegative, found {tokens[0]!r}.")
                        class_id = None
                else:
                    errors.append("CLASS_ID_NOT_INTEGER")
                    details.append(f"class_id must be an integer, found {tokens[0]!r}.")

            if len(tokens) >= 5:
                parsed_coordinates: list[float] = []
                coordinate_names = ("x_center", "y_center", "width", "height")
                for name, token in zip(coordinate_names, tokens[1:5]):
                    try:
                        coordinate = float(token)
                    except ValueError:
                        errors.append("COORDINATE_NOT_NUMBER")
                        details.append(f"{name} is not numeric: {token!r}.")
                        continue
                    if not math.isfinite(coordinate):
                        errors.append("COORDINATE_NOT_FINITE")
                        details.append(f"{name} must be finite, found {token!r}.")
                    parsed_coordinates.append(coordinate)
                if len(parsed_coordinates) == 4 and all(math.isfinite(value) for value in parsed_coordinates):
                    coordinates = parsed_coordinates

            if coordinates is not None:
                x_center, y_center, width, height = coordinates
                for name, value in (("x_center", x_center), ("y_center", y_center)):
                    if not 0 <= value <= 1:
                        errors.append("CENTER_OUT_OF_RANGE")
                        details.append(f"{name} must be in [0, 1], found {value}.")
                for name, value in (("width", width), ("height", height)):
                    if not 0 < value <= 1:
                        errors.append("SIZE_OUT_OF_RANGE")
                        details.append(f"{name} must be in (0, 1], found {value}.")
                if 0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 < width <= 1 and 0 < height <= 1:
                    left, right = x_center - width / 2, x_center + width / 2
                    top, bottom = y_center - height / 2, y_center + height / 2
                    if (
                        left < -BOUNDARY_TOLERANCE
                        or top < -BOUNDARY_TOLERANCE
                        or right > 1 + BOUNDARY_TOLERANCE
                        or bottom > 1 + BOUNDARY_TOLERANCE
                    ):
                        errors.append("BBOX_OUT_OF_BOUNDS")
                        details.append(
                            f"Normalized edges are left={left:.8g}, top={top:.8g}, "
                            f"right={right:.8g}, bottom={bottom:.8g}."
                        )

            canonical = " ".join(tokens)
            if canonical in first_line_for_canonical:
                errors.append("DUPLICATE_LABEL_LINE")
                details.append(f"Duplicates line {first_line_for_canonical[canonical]} after whitespace normalization.")
                duplicate_rows += 1
            else:
                first_line_for_canonical[canonical] = line_number

            # Raw class/box totals include structurally parseable rows, even if their range is invalid or duplicated.
            if len(tokens) == 5 and class_id is not None and coordinates is not None:
                class_box_counts[class_id] += 1
                classes.add(class_id)
                parseable_box_rows += 1

            if errors:
                unique_errors = list(dict.fromkeys(errors))
                for error_code in unique_errors:
                    error_code_counts[error_code] += 1
                add_invalid_entry(
                    invalid_entries, display_path, line_number, unique_errors, details, line_text=line_text
                )
            else:
                strictly_valid_rows += 1

    blocking_entries = [
        entry for entry in invalid_entries if any(code != "EMPTY_LABEL_FILE" for code in entry["error_codes"])
    ]
    files_with_blocking_errors = sorted({entry["label_path"] for entry in blocking_entries})
    return classes_by_label, class_box_counts, {
        "total_nonblank_rows": total_nonblank_rows,
        "parseable_box_rows": parseable_box_rows,
        "strictly_valid_rows": strictly_valid_rows,
        "invalid_line_count": sum(entry["line_number"] is not None for entry in invalid_entries),
        "invalid_entry_count": len(invalid_entries),
        "blocking_entry_count": len(blocking_entries),
        "duplicate_row_count": duplicate_rows,
        "empty_label_file_count": len(empty_files),
        "empty_label_files": empty_files,
        "read_error_count": len(read_errors),
        "read_errors": read_errors,
        "files_with_blocking_errors_count": len(files_with_blocking_errors),
        "files_with_blocking_errors": files_with_blocking_errors,
        "error_code_counts": dict(sorted(error_code_counts.items())),
        "invalid_entries": invalid_entries,
    }


def scan_images(image_files: list[Path], dataset_root: Path) -> tuple[dict[Path, int], dict[str, Any]]:
    """Read every image fully and collect dimensions and file sizes."""
    file_sizes: dict[Path, int] = {}
    widths: list[int] = []
    heights: list[int] = []
    aspect_ratios: list[float] = []
    size_values: list[int] = []
    errors_by_path: dict[Path, list[str]] = defaultdict(list)

    for image_path in image_files:
        try:
            size = image_path.stat().st_size
            file_sizes[image_path] = size
            size_values.append(size)
        except OSError as error:
            errors_by_path[image_path].append(f"Unable to stat file: {error}")

        try:
            with Image.open(image_path) as image:
                image.load()
                width, height = image.size
            if width <= 0 or height <= 0:
                raise ValueError(f"invalid image dimensions {width}x{height}")
            widths.append(width)
            heights.append(height)
            aspect_ratios.append(width / height)
        except Exception as error:  # Pillow can raise several decoder-specific exception types.
            errors_by_path[image_path].append(f"Unable to decode image: {type(error).__name__}: {error}")

    corrupt_images = [
        {"path": relative_path(path, dataset_root), "errors": errors}
        for path, errors in sorted(errors_by_path.items(), key=lambda item: item[0].as_posix().casefold())
    ]
    return file_sizes, {
        "readable_image_count": len(image_files) - len(errors_by_path),
        "corrupt_or_unreadable_image_count": len(errors_by_path),
        "corrupt_or_unreadable_images": corrupt_images,
        "width_pixels": basic_stats(widths),
        "height_pixels": basic_stats(heights),
        "aspect_ratio_width_over_height": basic_stats(aspect_ratios),
        "file_size_bytes": {**basic_stats(size_values), "total": sum(size_values)},
    }


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    """Write a UTF-8 CSV with a header even when there are no data rows."""
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def render_markdown(report: dict[str, Any]) -> str:
    """Render the human-readable audit report."""
    files = report["files"]
    pairing = report["pairing"]
    labels = report["label_validation"]
    crack = report["crack_statistics"]
    images = report["image_statistics"]
    readiness = report["readiness"]
    lines = [
        "# 原始隧道裂缝数据集审查报告",
        "",
        f"- 生成时间：`{report['generated_at']}`",
        f"- 数据集根目录：`{report['dataset']['root']}`",
        "- 审查方式：**只读扫描；本次没有修改原始数据**。",
        f"- 原始裂缝类别 ID：`{report['dataset']['target_class_id']}`。",
        f"- 未来新数据集裂缝类别 ID：计划映射为 `{report['dataset']['planned_class_id']}`（`crack`）。",
        "",
        "## 文件与配对",
        "",
        "| 指标 | 数量 |",
        "| --- | ---: |",
        f"| 图片文件 | {files['image_count']} |",
        f"| 标签文件 | {files['label_count']} |",
        f"| 唯一的一对一配对 | {pairing['unambiguous_pair_count']} |",
        f"| 有图片无标签 | {pairing['image_without_label_count']} |",
        f"| 有标签无图片 | {pairing['label_without_image_count']} |",
        f"| 图片 stem 冲突 | {pairing['image_stem_conflict_count']} |",
        f"| 标签 stem 冲突 | {pairing['label_stem_conflict_count']} |",
        f"| 是否完全一一对应 | {'是' if pairing['is_one_to_one'] else '否'} |",
        "",
        "## 标签检查",
        "",
        "| 指标 | 数量 |",
        "| --- | ---: |",
        f"| 非空标签行 | {labels['total_nonblank_rows']} |",
        f"| 可解析的 5 列框行 | {labels['parseable_box_rows']} |",
        f"| 严格有效标签行 | {labels['strictly_valid_rows']} |",
        f"| 含阻断性错误的条目 | {labels['blocking_entry_count']} |",
        f"| 空标签文件 | {labels['empty_label_file_count']} |",
        f"| 重复标签行 | {labels['duplicate_row_count']} |",
        f"| 标签读取失败 | {labels['read_error_count']} |",
        "",
        "### 原始类别统计",
        "",
        "框数量统计包含所有格式可解析的 5 列行，包括重复或越界行；这些异常另行列出。图片数量按物理图片计数。",
        "",
        "| 原始类别 ID | 图片数量 | 标注框数量 |",
        "| ---: | ---: | ---: |",
    ]
    for class_id, values in report["class_statistics"].items():
        lines.append(f"| {class_id} | {values['image_count']} | {values['box_count']} |")
    if not report["class_statistics"]:
        lines.append("| _无_ | 0 | 0 |")

    lines.extend(
        [
            "",
            "## 裂缝类别统计",
            "",
            "| 指标 | 数量 |",
            "| --- | ---: |",
            f"| 包含裂缝的图片 | {crack['images_with_target_class']} |",
            f"| 裂缝标注框 | {crack['target_box_count']} |",
            f"| 仅包含裂缝的图片 | {crack['images_with_only_target_class']} |",
            f"| 同时包含裂缝和其他类别的图片 | {crack['images_with_target_and_other_classes']} |",
            f"| 不包含裂缝的图片 | {crack['images_without_target_class']} |",
            f"| 包含裂缝图片的总文件大小 | {crack['target_image_total_size_bytes']} B ({crack['target_image_total_size_human']}) |",
            "",
            "## 图片统计",
            "",
            f"- 可正常读取：{images['readable_image_count']}；损坏或无法读取：{images['corrupt_or_unreadable_image_count']}。",
            f"- 宽度（像素）：`{images['width_pixels']}`",
            f"- 高度（像素）：`{images['height_pixels']}`",
            f"- 宽高比（宽/高）：`{images['aspect_ratio_width_over_height']}`",
            f"- 文件大小（字节）：`{images['file_size_bytes']}`",
            "",
            "## 是否可以继续构建裂缝数据集",
            "",
            f"**结论：{'满足条件，可以继续' if readiness['can_continue'] else '暂不满足条件'}。**",
            "",
        ]
    )
    if readiness["blockers"]:
        lines.append("必须先解决的问题：")
        lines.append("")
        lines.extend(f"- {issue}" for issue in readiness["blockers"])
        lines.append("")
    else:
        lines.extend(["未发现必须先解决的阻断问题。", ""])
    if readiness["warnings"]:
        lines.append("需要关注但不阻断后续筛选的问题：")
        lines.append("")
        lines.extend(f"- {warning}" for warning in readiness["warnings"])
        lines.append("")
    lines.extend(
        [
            "详细错误请查看 `invalid_labels.csv`，配对与 stem 冲突请查看 `unmatched_files.csv`；完整明细保存在 JSON 报告中。",
            "",
        ]
    )
    return "\n".join(lines)


def audit(dataset_root: Path, target_class_id: int, report_dir: Path) -> dict[str, Any]:
    """Run the read-only audit and return a JSON-serializable report."""
    images_dir = dataset_root / "images"
    labels_dir = dataset_root / "labels"
    data_yaml = dataset_root / "data.yaml"

    missing = [
        str(path)
        for path in (dataset_root, images_dir, labels_dir)
        if not path.exists() or not path.is_dir()
    ]
    if missing:
        raise FileNotFoundError(f"required dataset directories are missing: {', '.join(missing)}")

    try:
        report_dir.relative_to(dataset_root)
    except ValueError:
        pass
    else:
        raise ValueError("report directory must not be inside the raw dataset root")

    image_files = find_files(images_dir, IMAGE_SUFFIXES)
    label_files = find_files(labels_dir, {".txt"})
    image_map: dict[str, list[Path]] = defaultdict(list)
    label_map: dict[str, list[Path]] = defaultdict(list)
    for image_path in image_files:
        image_map[stem_key(image_path, images_dir)].append(image_path)
    for label_path in label_files:
        label_map[stem_key(label_path, labels_dir)].append(label_path)

    classes_by_label, class_box_counts, label_validation = scan_labels(label_files, dataset_root)
    image_file_sizes, image_statistics = scan_images(image_files, dataset_root)

    unmatched_rows: list[dict[str, Any]] = []
    for key, paths in image_map.items():
        if key not in label_map:
            for path in paths:
                unmatched_rows.append(
                    {
                        "issue_type": "IMAGE_WITHOUT_LABEL",
                        "stem": key,
                        "image_path": relative_path(path, dataset_root),
                        "label_path": "",
                        "details": "No label with the same relative stem.",
                    }
                )
    for key, paths in label_map.items():
        if key not in image_map:
            for path in paths:
                unmatched_rows.append(
                    {
                        "issue_type": "LABEL_WITHOUT_IMAGE",
                        "stem": key,
                        "image_path": "",
                        "label_path": relative_path(path, dataset_root),
                        "details": "No supported image with the same relative stem.",
                    }
                )

    image_stem_conflicts = {key: paths for key, paths in image_map.items() if len(paths) > 1}
    label_stem_conflicts = {key: paths for key, paths in label_map.items() if len(paths) > 1}
    for key, paths in image_stem_conflicts.items():
        joined = "; ".join(relative_path(path, dataset_root) for path in paths)
        for path in paths:
            unmatched_rows.append(
                {
                    "issue_type": "IMAGE_STEM_CONFLICT",
                    "stem": key,
                    "image_path": relative_path(path, dataset_root),
                    "label_path": "",
                    "details": f"Multiple image extensions share this relative stem: {joined}",
                }
            )
    for key, paths in label_stem_conflicts.items():
        joined = "; ".join(relative_path(path, dataset_root) for path in paths)
        for path in paths:
            unmatched_rows.append(
                {
                    "issue_type": "LABEL_STEM_CONFLICT",
                    "stem": key,
                    "image_path": "",
                    "label_path": relative_path(path, dataset_root),
                    "details": f"Multiple label files share this relative stem: {joined}",
                }
            )
    unmatched_rows.sort(key=lambda row: (row["issue_type"], row["stem"], row["image_path"], row["label_path"]))

    class_image_counts: Counter[int] = Counter()
    images_with_target = 0
    images_with_only_target = 0
    images_with_target_and_others = 0
    images_without_target = 0
    target_image_total_size = 0
    target_images_without_size = 0
    for key, paths in image_map.items():
        class_ids: set[int] = set()
        for label_path in label_map.get(key, []):
            class_ids.update(classes_by_label.get(label_path, set()))
        for image_path in paths:
            for class_id in class_ids:
                class_image_counts[class_id] += 1
            if target_class_id in class_ids:
                images_with_target += 1
                if class_ids == {target_class_id}:
                    images_with_only_target += 1
                else:
                    images_with_target_and_others += 1
                if image_path in image_file_sizes:
                    target_image_total_size += image_file_sizes[image_path]
                else:
                    target_images_without_size += 1
            else:
                images_without_target += 1

    all_class_ids = sorted(set(class_box_counts) | set(class_image_counts))
    class_statistics = {
        str(class_id): {"image_count": class_image_counts[class_id], "box_count": class_box_counts[class_id]}
        for class_id in all_class_ids
    }

    image_without_label_count = sum(len(paths) for key, paths in image_map.items() if key not in label_map)
    label_without_image_count = sum(len(paths) for key, paths in label_map.items() if key not in image_map)
    unambiguous_pair_count = sum(
        len(image_map[key]) == 1 and len(label_map.get(key, [])) == 1 for key in image_map
    )
    is_one_to_one = (
        unambiguous_pair_count == len(image_files) == len(label_files)
        and not image_stem_conflicts
        and not label_stem_conflicts
    )

    data_yaml_exists = data_yaml.is_file()
    data_yaml_readable = False
    data_yaml_error = ""
    if data_yaml_exists:
        try:
            with data_yaml.open("rb") as file:
                file.read(1)
            data_yaml_readable = True
        except OSError as error:
            data_yaml_error = str(error)

    blockers: list[str] = []
    warnings: list[str] = []
    if not data_yaml_exists or not data_yaml_readable:
        blockers.append("data.yaml 缺失或不可读。")
    if image_statistics["corrupt_or_unreadable_image_count"]:
        blockers.append(f"有 {image_statistics['corrupt_or_unreadable_image_count']} 张图片损坏或无法读取。")
    if image_without_label_count:
        blockers.append(f"有 {image_without_label_count} 张图片没有同 stem 标签。")
    if label_without_image_count:
        blockers.append(f"有 {label_without_image_count} 个标签没有同 stem 图片。")
    if image_stem_conflicts:
        blockers.append(f"有 {len(image_stem_conflicts)} 个图片 stem 存在不同扩展名冲突。")
    if label_stem_conflicts:
        blockers.append(f"有 {len(label_stem_conflicts)} 个标签 stem 冲突。")
    if label_validation["blocking_entry_count"]:
        blockers.append(f"标签中有 {label_validation['blocking_entry_count']} 个需要修复的格式或数值错误条目。")
    if images_with_target == 0 or class_box_counts[target_class_id] == 0:
        blockers.append(f"未找到原始类别 ID {target_class_id} 的可解析裂缝图片或标注框。")
    if label_validation["empty_label_file_count"]:
        warnings.append(f"有 {label_validation['empty_label_file_count']} 个空标签文件；需确认它们是否为预期背景样本。")
    if images_with_target_and_others:
        warnings.append(f"有 {images_with_target_and_others} 张图片同时包含裂缝和其他类别，后续筛选时需只保留并映射裂缝框。")
    if images_without_target:
        warnings.append(f"有 {images_without_target} 张图片不包含裂缝，后续构建单类别数据集时需按既定策略处理。")

    report = {
        "schema_version": 1,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "audit_guarantees": {
            "raw_dataset_modified": False,
            "mode": "read-only scan",
            "report_output_directory": str(report_dir),
        },
        "dataset": {
            "root": str(dataset_root),
            "images_directory": str(images_dir),
            "labels_directory": str(labels_dir),
            "data_yaml": str(data_yaml),
            "data_yaml_exists": data_yaml_exists,
            "data_yaml_readable": data_yaml_readable,
            "data_yaml_error": data_yaml_error,
            "target_class_id": target_class_id,
            "planned_class_id": PLANNED_CRACK_CLASS_ID,
            "planned_class_name": "crack",
            "supported_image_extensions": sorted(IMAGE_SUFFIXES),
        },
        "files": {"image_count": len(image_files), "label_count": len(label_files)},
        "pairing": {
            "is_one_to_one": is_one_to_one,
            "unambiguous_pair_count": unambiguous_pair_count,
            "image_without_label_count": image_without_label_count,
            "label_without_image_count": label_without_image_count,
            "image_stem_conflict_count": len(image_stem_conflicts),
            "label_stem_conflict_count": len(label_stem_conflicts),
            "image_stem_conflicts": {
                key: [relative_path(path, dataset_root) for path in paths]
                for key, paths in sorted(image_stem_conflicts.items())
            },
            "label_stem_conflicts": {
                key: [relative_path(path, dataset_root) for path in paths]
                for key, paths in sorted(label_stem_conflicts.items())
            },
            "unmatched_and_conflict_entries": unmatched_rows,
        },
        "label_validation": label_validation,
        "class_statistics": class_statistics,
        "crack_statistics": {
            "target_class_id": target_class_id,
            "images_with_target_class": images_with_target,
            "target_box_count": class_box_counts[target_class_id],
            "images_with_only_target_class": images_with_only_target,
            "images_with_target_and_other_classes": images_with_target_and_others,
            "images_without_target_class": images_without_target,
            "target_image_total_size_bytes": target_image_total_size,
            "target_image_total_size_human": human_bytes(target_image_total_size),
            "target_images_without_readable_file_size": target_images_without_size,
        },
        "image_statistics": image_statistics,
        "readiness": {"can_continue": not blockers, "blockers": blockers, "warnings": warnings},
    }

    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / "raw_crack_dataset_audit.json"
    markdown_path = report_dir / "raw_crack_dataset_audit.md"
    invalid_csv_path = report_dir / "invalid_labels.csv"
    unmatched_csv_path = report_dir / "unmatched_files.csv"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    invalid_csv_rows = [
        {
            **entry,
            "line_number": "" if entry["line_number"] is None else entry["line_number"],
            "error_codes": ";".join(entry["error_codes"]),
            "details": " ".join(entry["details"]),
        }
        for entry in label_validation["invalid_entries"]
    ]
    write_csv(
        invalid_csv_path,
        ["label_path", "line_number", "error_codes", "details", "line_text"],
        invalid_csv_rows,
    )
    write_csv(
        unmatched_csv_path,
        ["issue_type", "stem", "image_path", "label_path", "details"],
        unmatched_rows,
    )
    return report


def main() -> int:
    """Run the CLI."""
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    report_dir = Path(__file__).resolve().parents[1] / "reports"
    try:
        report = audit(dataset_root, args.target_class_id, report_dir)
    except (FileNotFoundError, PermissionError, ValueError) as error:
        print(f"Audit failed before scanning: {error}", file=sys.stderr)
        return 2

    print(f"Read-only audit complete: {report['files']['image_count']} images, {report['files']['label_count']} labels")
    print(f"Reports written to: {report_dir}")
    print(f"Ready to continue: {report['readiness']['can_continue']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
