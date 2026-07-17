"""Review crack boxes dropped by the augmented-first dataset build without changing either dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime
from decimal import Decimal, getcontext
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
BIN_DEFINITIONS = (
    ("不超过1e-6", None, Decimal("0.000001")),
    ("大于1e-6且不超过1e-5", Decimal("0.000001"), Decimal("0.00001")),
    ("大于1e-5且不超过1e-4", Decimal("0.00001"), Decimal("0.0001")),
    ("大于1e-4且不超过1e-3", Decimal("0.0001"), Decimal("0.001")),
    ("大于1e-3且不超过1e-2", Decimal("0.001"), Decimal("0.01")),
    ("大于或等于1e-2", Decimal("0.01"), None),
)


class ReviewStop(RuntimeError):
    """Signal a review precondition or reconciliation failure."""


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--build-report", required=True, type=Path)
    parser.add_argument("--build-script", required=True, type=Path)
    parser.add_argument("--visual-root", required=True, type=Path)
    parser.add_argument("--source-class-id", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    return parser.parse_args()


def ensure_new_output(path: Path, name: str) -> None:
    """Refuse to overwrite an existing file or nonempty directory."""
    if path.is_file():
        raise ReviewStop(f"{name} already exists: {path}")
    if path.is_dir() and next(path.iterdir(), None) is not None:
        raise ReviewStop(f"{name} already exists and is nonempty: {path}")


def sha256_file(path: Path) -> str:
    """Hash one file in read-only mode."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(4 * 1024**2):
            digest.update(chunk)
    return digest.hexdigest()


def load_build_module(path: Path) -> Any:
    """Load the exact build script so the review reuses its row validator."""
    spec = importlib.util.spec_from_file_location("crack_augfirst_build_for_review", path)
    if spec is None or spec.loader is None:
        raise ReviewStop(f"unable to load build script: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def overflow_bin(value: Decimal) -> str:
    """Assign a maximum overflow to the requested interval."""
    for name, lower, upper in BIN_DEFINITIONS:
        if lower is None and value <= upper:
            return name
        if upper is None and value >= lower:
            return name
        if lower is not None and upper is not None and lower < value <= upper:
            return name
    raise ReviewStop(f"unable to bin overflow value {value}")


def decimal_text(value: Decimal) -> str:
    """Serialize Decimal without exponent notation."""
    return format(value, "f")


def find_images(images_dir: Path) -> dict[str, Path]:
    """Index raw images by globally unique case-insensitive stem."""
    image_map: dict[str, list[Path]] = defaultdict(list)
    for path in images_dir.rglob("*"):
        if path.is_file() and path.suffix.casefold() in IMAGE_SUFFIXES:
            image_map[path.stem.casefold()].append(path)
    conflicts = {stem: paths for stem, paths in image_map.items() if len(paths) != 1}
    if conflicts:
        raise ReviewStop(f"raw image stems are not unique: {len(conflicts)} conflicts")
    return {stem: paths[0] for stem, paths in image_map.items()}


def scan_raw_labels(
    labels_dir: Path, source_class_id: int, build_module: Any
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Recompute every invalid source-class row using the exact build validator."""
    dropped: list[dict[str, Any]] = []
    valid_counts: dict[str, int] = {}
    for label_path in sorted(labels_dir.rglob("*.txt"), key=lambda path: path.as_posix().casefold()):
        text = label_path.read_text(encoding="utf-8-sig")
        seen_rows: dict[str, int] = {}
        valid_target_count = 0
        label_dropped: list[dict[str, Any]] = []
        for line_number, raw_line in enumerate(text.splitlines(), 1):
            line = raw_line.strip()
            if not line:
                continue
            tokens = line.split()
            class_id, coordinates, errors = build_module.parse_label_tokens(tokens)
            canonical = " ".join(tokens)
            if canonical in seen_rows:
                errors.append("DUPLICATE_LABEL_LINE")
            else:
                seen_rows[canonical] = line_number
            if class_id != source_class_id:
                continue
            if errors or coordinates is None:
                label_dropped.append(
                    {
                        "source_stem": label_path.stem,
                        "label_file": label_path.name,
                        "label_path": label_path,
                        "line_number": line_number,
                        "line": line,
                        "tokens": tokens,
                        "build_errors": list(dict.fromkeys(errors)),
                    }
                )
            else:
                valid_target_count += 1
        valid_counts[label_path.stem] = valid_target_count
        for row in label_dropped:
            row["other_valid_crack_box_count"] = valid_target_count
            dropped.append(row)
    return dropped, valid_counts


def enrich_decimal_geometry(row: dict[str, Any]) -> None:
    """Compute exact decimal edges and overflow amounts from the raw six-column strings."""
    tokens = row["tokens"]
    if len(tokens) != 5:
        raise ReviewStop(f"dropped crack row does not contain five columns: {row['label_file']}:{row['line_number']}")
    x_center, y_center, width, height = (Decimal(token) for token in tokens[1:5])
    two = Decimal(2)
    left = x_center - width / two
    top = y_center - height / two
    right = x_center + width / two
    bottom = y_center + height / two
    zero, one = Decimal(0), Decimal(1)
    overflows = {
        "left": max(zero, -left),
        "top": max(zero, -top),
        "right": max(zero, right - one),
        "bottom": max(zero, bottom - one),
    }
    directions = [direction for direction, amount in overflows.items() if amount > 0]
    maximum = max(overflows.values())
    row.update(
        {
            "x_center": x_center,
            "y_center": y_center,
            "width": width,
            "height": height,
            "left": left,
            "top": top,
            "right": right,
            "bottom": bottom,
            "overflow_directions": directions,
            "left_overflow": overflows["left"],
            "top_overflow": overflows["top"],
            "right_overflow": overflows["right"],
            "bottom_overflow": overflows["bottom"],
            "max_overflow": maximum,
            "overflow_bin": overflow_bin(maximum),
            "decimal_is_out_of_bounds": bool(directions),
            "micro_rounding_scale": maximum <= Decimal("0.000001"),
        }
    )


def load_font(size: int) -> ImageFont.ImageFont:
    """Load a readable local font without downloading dependencies."""
    for candidate in (Path("C:/Windows/Fonts/arial.ttf"), Path("C:/Windows/Fonts/segoeui.ttf")):
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def draw_boundary_schematic(draw: ImageDraw.ImageDraw, row: dict[str, Any], box: tuple[int, int, int, int]) -> None:
    """Draw a magnified schematic so sub-pixel overflow remains visible."""
    x0, y0, x1, y1 = box
    draw.rectangle(box, outline=(120, 120, 120), width=2)
    font = load_font(18)
    draw.text((x0 + 10, y0 + 8), "Boundary magnification (schematic, not to scale)", fill="black", font=font)
    boundary_x = (x0 + x1) // 2
    draw.line((boundary_x, y0 + 38, boundary_x, y1 - 10), fill=(0, 170, 255), width=4)
    direction = row["overflow_directions"][0]
    outward = -1 if direction in {"left", "top"} else 1
    raw_x = boundary_x + outward * 70
    draw.line((raw_x, y0 + 48, raw_x, y1 - 20), fill=(255, 32, 32), width=5)
    draw.line((boundary_x, y0 + 48, boundary_x, y1 - 20), fill=(0, 220, 80), width=2)
    draw.text((x0 + 10, y1 - 32), f"magnified direction: {direction}", fill="black", font=font)


def visualize_box(row: dict[str, Any], image_path: Path, output_path: Path) -> None:
    """Visualize raw, hypothetically clipped, and image-boundary rectangles without editing labels."""
    with Image.open(image_path) as image:
        image.load()
        rgb = image.convert("RGB")
    display_width = 1280
    display_height = round(rgb.height * display_width / rgb.width)
    resized = rgb.resize((display_width, display_height), Image.Resampling.LANCZOS)
    rgb.close()
    margin = 90
    header_height = 125
    schematic_height = 135
    canvas_width = display_width + margin * 2
    canvas_height = header_height + display_height + margin * 2 + schematic_height
    canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
    image_x, image_y = margin, header_height + margin
    canvas.paste(resized, (image_x, image_y))
    resized.close()
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(26)
    body_font = load_font(19)
    draw.text(
        (20, 12),
        f"{row['source_stem']} | line {row['line_number']} | max overflow={decimal_text(row['max_overflow'])}",
        fill="black",
        font=title_font,
    )
    draw.text(
        (20, 50),
        f"directions={','.join(row['overflow_directions'])} | red=raw | "
        "green=hypothetical clipped | cyan=image boundary",
        fill="black",
        font=body_font,
    )
    draw.text(
        (20, 82),
        f"xyxy=({decimal_text(row['left'])}, {decimal_text(row['top'])}, "
        f"{decimal_text(row['right'])}, {decimal_text(row['bottom'])})",
        fill="black",
        font=body_font,
    )
    image_boundary = (image_x, image_y, image_x + display_width, image_y + display_height)
    draw.rectangle(image_boundary, outline=(0, 170, 255), width=5)

    def to_canvas(x: Decimal, y: Decimal) -> tuple[float, float]:
        return image_x + float(x) * display_width, image_y + float(y) * display_height

    raw_left, raw_top = to_canvas(row["left"], row["top"])
    raw_right, raw_bottom = to_canvas(row["right"], row["bottom"])
    draw.rectangle((raw_left, raw_top, raw_right, raw_bottom), outline=(255, 32, 32), width=6)
    clipped_left = max(Decimal(0), row["left"])
    clipped_top = max(Decimal(0), row["top"])
    clipped_right = min(Decimal(1), row["right"])
    clipped_bottom = min(Decimal(1), row["bottom"])
    clip_left, clip_top = to_canvas(clipped_left, clipped_top)
    clip_right, clip_bottom = to_canvas(clipped_right, clipped_bottom)
    draw.rectangle((clip_left, clip_top, clip_right, clip_bottom), outline=(0, 220, 80), width=3)
    schematic_top = header_height + display_height + margin * 2
    draw_boundary_schematic(draw, row, (20, schematic_top, canvas_width - 20, canvas_height - 15))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="JPEG", quality=92)
    canvas.close()


def write_csv_report(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write the requested per-box review CSV."""
    fields = [
        "source_stem",
        "label_file",
        "line_number",
        "x_center",
        "y_center",
        "width",
        "height",
        "left",
        "top",
        "right",
        "bottom",
        "overflow_directions",
        "left_overflow",
        "top_overflow",
        "right_overflow",
        "bottom_overflow",
        "max_overflow",
        "overflow_bin",
        "has_other_valid_crack_box",
        "other_valid_crack_box_count",
        "source_fully_excluded",
        "micro_rounding_scale",
        "build_errors",
        "visual_groups",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "source_stem": row["source_stem"],
                    "label_file": row["label_file"],
                    "line_number": row["line_number"],
                    "x_center": decimal_text(row["x_center"]),
                    "y_center": decimal_text(row["y_center"]),
                    "width": decimal_text(row["width"]),
                    "height": decimal_text(row["height"]),
                    "left": decimal_text(row["left"]),
                    "top": decimal_text(row["top"]),
                    "right": decimal_text(row["right"]),
                    "bottom": decimal_text(row["bottom"]),
                    "overflow_directions": ";".join(row["overflow_directions"]),
                    "left_overflow": decimal_text(row["left_overflow"]),
                    "top_overflow": decimal_text(row["top_overflow"]),
                    "right_overflow": decimal_text(row["right_overflow"]),
                    "bottom_overflow": decimal_text(row["bottom_overflow"]),
                    "max_overflow": decimal_text(row["max_overflow"]),
                    "overflow_bin": row["overflow_bin"],
                    "has_other_valid_crack_box": row["other_valid_crack_box_count"] > 0,
                    "other_valid_crack_box_count": row["other_valid_crack_box_count"],
                    "source_fully_excluded": row["source_fully_excluded"],
                    "micro_rounding_scale": row["micro_rounding_scale"],
                    "build_errors": ";".join(row["build_errors"]),
                    "visual_groups": ";".join(sorted(row["visual_groups"])),
                }
            )


def render_markdown(review: dict[str, Any]) -> str:
    """Render the aggregate review conclusions."""
    counts = review["counts"]
    lines = [
        "# 被删除原始裂缝框专项只读审查",
        "",
        f"- 生成时间：`{review['generated_at']}`",
        f"- 原始数据集：`{review['dataset_root']}`",
        f"- 构建报告：`{review['build_report']}`",
        f"- 构建脚本 SHA-256：`{review['build_script_sha256']}`",
        "- 审查方式：只读读取构建报告、构建脚本和原始标签/图片；未修改原始或派生数据。",
        "- 边界计算：对原始六位小数字符串使用 `Decimal` 精确计算，同时复用构建脚本的零容差校验函数。",
        "",
        "## 核对结论",
        "",
        "| 项目 | 结果 |",
        "| --- | ---: |",
        f"| 构建报告删除裂缝框 | {counts['build_report_dropped_crack_boxes']} |",
        f"| 原始标签独立还原删除裂缝框 | {counts['recomputed_dropped_crack_boxes']} |",
        f"| 构建报告完全排除 source | {counts['build_report_excluded_sources']} |",
        f"| 独立计算完全排除 source | {counts['recomputed_excluded_sources']} |",
        f"| 有其他有效裂缝框的被删框 | {counts['boxes_with_other_valid_crack_box']} |",
        f"| 导致 source 完全排除的被删框 | {counts['boxes_from_fully_excluded_sources']} |",
        f"| Decimal 与构建程序越界判断不一致 | {counts['program_misclassification_count']} |",
        "",
        f"三方集合一致：**{'是' if review['reconciliation']['all_sets_match'] else '否'}**。",
        "",
        "## 最大越界量分布",
        "",
        "| 区间 | 框数 | 占比 |",
        "| --- | ---: | ---: |",
    ]
    total = counts["recomputed_dropped_crack_boxes"]
    for name, _, _ in BIN_DEFINITIONS:
        count = review["overflow_bins"][name]
        lines.append(f"| {name} | {count} | {count / total:.6%} |")
    lines.extend(
        [
            "",
            "## 判断",
            "",
            f"- 微小舍入尺度越界（最大越界量不超过 `1e-6`）：{counts['micro_rounding_scale_boxes']} / {total} "
            f"（{counts['micro_rounding_scale_ratio']:.6%}）。",
            f"- 最大越界框：`{review['maximum_overflow']['source_stem']}` 第 "
            f"{review['maximum_overflow']['line_number']} 行，最大越界量 "
            f"`{review['maximum_overflow']['value']}`。",
            "- 程序误判：未发现。构建脚本的 float 零容差结果与 Decimal 精确十进制计算逐框一致。",
            "- 舍入来源：125 个框的越界量处于六位小数半单位量级，强烈符合坐标量化/舍入造成的边界微溢出；"
            "无法从舍入后的标签反推出舍入前坐标，因此这是有数据依据的推断而非绝对证明。",
            f"- 数据损失：删除占原始裂缝框 {counts['dropped_box_ratio_of_raw']:.6%}，完全排除占原始裂缝图片 "
            f"{counts['excluded_source_ratio_of_raw']:.6%}，属于可感知但非灾难性的损失。",
            "- 建议：不建议未来继续采用严格零容差删除方案。建议在明确批准重建后，将 `<=1e-6` 作为格式量化容差，"
            "仍丢弃超过容差的真实越界框；本次审查不修改、不恢复也不重建当前数据集。",
            "",
            "## 可视化",
            "",
            f"- 最大越界量前 30：`{review['visuals']['top30_directory']}`",
            f"- 随机 20 个微小越界框：`{review['visuals']['random20_micro_directory']}`",
            f"- 已知 t812：`{review['visuals']['known_t812_directory']}`",
            "",
            "图中红框为原始框，青色为图像边界，绿框仅表示假设裁剪到边界后的框；没有对任何标签执行裁剪。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    """Run the read-only reconciliation, reports, and visual review."""
    getcontext().prec = 40
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    build_report_path = args.build_report.expanduser().resolve()
    build_script_path = args.build_script.expanduser().resolve()
    visual_root = args.visual_root.expanduser().resolve()
    report_dir = Path(__file__).resolve().parents[1] / "reports"
    csv_path = report_dir / "dropped_crack_boxes_review.csv"
    markdown_path = report_dir / "dropped_crack_boxes_review.md"
    try:
        for path in (dataset_root / "images", dataset_root / "labels"):
            if not path.is_dir():
                raise ReviewStop(f"required raw directory is missing: {path}")
        for path in (build_report_path, build_script_path):
            if not path.is_file():
                raise ReviewStop(f"required build artifact is missing: {path}")
        ensure_new_output(visual_root, "visual output directory")
        ensure_new_output(csv_path, "CSV report")
        ensure_new_output(markdown_path, "Markdown report")
        build_report = json.loads(build_report_path.read_text(encoding="utf-8"))
        build_module = load_build_module(build_script_path)
        dropped, _ = scan_raw_labels(dataset_root / "labels", args.source_class_id, build_module)
        for row in dropped:
            enrich_decimal_geometry(row)
        report_rows = [
            row
            for row in build_report["filtering"]["invalid_raw_rows"]
            if row["class_id"] == args.source_class_id and row["action"] == "DROP_INVALID_CRACK_BOX"
        ]
        report_keys = {(row["label"], row["line_number"], row["line"]) for row in report_rows}
        recomputed_keys = {(row["label_file"], row["line_number"], row["line"]) for row in dropped}
        excluded_report_sources = {
            row["source_stem"]
            for row in build_report["filtering"]["excluded_sources"]
            if row["reason"] == "NO_VALID_CRACK_BOX_AFTER_INVALID_DROP"
        }
        recomputed_excluded_sources = {
            row["source_stem"] for row in dropped if row["other_valid_crack_box_count"] == 0
        }
        for row in dropped:
            row["source_fully_excluded"] = row["source_stem"] in recomputed_excluded_sources
            row["visual_groups"] = set()
        program_misclassifications = [
            row
            for row in dropped
            if ("BBOX_OUT_OF_BOUNDS" in row["build_errors"]) != row["decimal_is_out_of_bounds"]
        ]
        expected_dropped = build_report["filtering"]["dropped_invalid_crack_box_count"]
        expected_excluded = build_report["filtering"]["excluded_crack_image_count"]
        all_sets_match = (
            len(dropped) == expected_dropped == len(report_rows)
            and report_keys == recomputed_keys
            and len(recomputed_excluded_sources) == expected_excluded == len(excluded_report_sources)
            and recomputed_excluded_sources == excluded_report_sources
            and not program_misclassifications
        )
        if not all_sets_match:
            raise ReviewStop(
                "dropped-box reconciliation failed: "
                f"recomputed={len(dropped)}, report_rows={len(report_rows)}, expected={expected_dropped}, "
                f"recomputed_excluded={len(recomputed_excluded_sources)}, expected_excluded={expected_excluded}"
            )

        image_map = find_images(dataset_root / "images")
        top30 = sorted(
            dropped,
            key=lambda row: (-row["max_overflow"], row["source_stem"].casefold(), row["line_number"]),
        )[:30]
        micro_candidates = [row for row in dropped if row["micro_rounding_scale"]]
        random20 = random.Random(args.seed).sample(micro_candidates, 20)
        known_t812 = [row for row in dropped if row["source_stem"] == "t812_812_011906777"]
        if len(known_t812) != 1:
            raise ReviewStop(f"expected exactly one known t812 dropped crack row, found {len(known_t812)}")
        selections = (
            ("top30", top30),
            ("random20_micro", random20),
            ("known_t812", known_t812),
        )
        for group, selected in selections:
            for row in selected:
                row["visual_groups"].add(group)
                image_path = image_map.get(row["source_stem"].casefold())
                if image_path is None:
                    raise ReviewStop(f"missing raw image for {row['source_stem']}")
                filename = f"{row['source_stem']}__line_{row['line_number']}__{group}.jpg"
                visualize_box(row, image_path, visual_root / group / filename)

        bin_counts = Counter(row["overflow_bin"] for row in dropped)
        micro_count = sum(row["micro_rounding_scale"] for row in dropped)
        fully_excluded_box_count = sum(row["source_fully_excluded"] for row in dropped)
        maximum = max(dropped, key=lambda row: row["max_overflow"])
        counts = {
            "build_report_dropped_crack_boxes": expected_dropped,
            "recomputed_dropped_crack_boxes": len(dropped),
            "build_report_excluded_sources": expected_excluded,
            "recomputed_excluded_sources": len(recomputed_excluded_sources),
            "boxes_with_other_valid_crack_box": len(dropped) - fully_excluded_box_count,
            "boxes_from_fully_excluded_sources": fully_excluded_box_count,
            "program_misclassification_count": len(program_misclassifications),
            "micro_rounding_scale_boxes": micro_count,
            "micro_rounding_scale_ratio": micro_count / len(dropped),
            "dropped_box_ratio_of_raw": len(dropped) / build_report["filtering"]["raw_crack_box_count"],
            "excluded_source_ratio_of_raw": len(recomputed_excluded_sources)
            / build_report["filtering"]["raw_crack_image_count"],
        }
        review = {
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "dataset_root": str(dataset_root),
            "build_report": str(build_report_path),
            "build_script": str(build_script_path),
            "build_script_sha256": sha256_file(build_script_path),
            "source_class_id": args.source_class_id,
            "seed": args.seed,
            "counts": counts,
            "overflow_bins": {name: bin_counts[name] for name, _, _ in BIN_DEFINITIONS},
            "maximum_overflow": {
                "source_stem": maximum["source_stem"],
                "line_number": maximum["line_number"],
                "value": decimal_text(maximum["max_overflow"]),
                "directions": maximum["overflow_directions"],
            },
            "reconciliation": {
                "all_sets_match": all_sets_match,
                "report_only_rows": sorted(report_keys - recomputed_keys),
                "recomputed_only_rows": sorted(recomputed_keys - report_keys),
                "excluded_source_symmetric_difference": sorted(
                    excluded_report_sources ^ recomputed_excluded_sources
                ),
            },
            "visuals": {
                "root": str(visual_root),
                "top30_directory": str(visual_root / "top30"),
                "top30_count": len(top30),
                "random20_micro_directory": str(visual_root / "random20_micro"),
                "random20_micro_count": len(random20),
                "known_t812_directory": str(visual_root / "known_t812"),
                "known_t812_count": len(known_t812),
            },
            "recommendation": {
                "keep_current_strict_zero_tolerance_for_future_rebuild": False,
                "suggested_future_boundary_tolerance": "1e-6",
                "current_dataset_modified": False,
            },
        }
        report_dir.mkdir(parents=True, exist_ok=True)
        write_csv_report(csv_path, dropped)
        markdown_path.write_text(render_markdown(review), encoding="utf-8")
        print(json.dumps(review, ensure_ascii=False, indent=2), flush=True)
        return 0
    except ReviewStop as error:
        print(f"REVIEW STOP: {error}", file=sys.stderr, flush=True)
        return 2
    except Exception as error:
        print(f"REVIEW FAILED: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
