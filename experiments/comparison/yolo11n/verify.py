"""Bounded real-image SMOKE of native training, fresh-process evaluation and core packaging."""

import argparse
import importlib.metadata
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

from _entry import HERE, setup


def main():
    """Use two train images and one image per evaluation split; never launch formal training."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Existing crack dataset YAML; original data remains read-only")
    parser.add_argument("--pretrained-cache", required=True)
    parser.add_argument("--output", required=True, help="New development-only output directory")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    common = setup()
    sys.path.insert(0, str(HERE.parent))
    from shared.data import label_path, split_files

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "RUNNING",
        "formal_training": "NOT_STARTED",
        "server": "UNVERIFIED",
        "data": "REAL_IMAGE_SUBSET_SMOKE",
        "data_counts": {"train": 2, "val": 1, "test": 1},
        "source_data": str(Path(args.data).resolve()),
        "device_requested": args.device,
        "checks": {},
    }

    def command(name, entry, extra=()):
        argv = [sys.executable, "-B", "-W", "ignore::FutureWarning", "-u", str(HERE / (entry + ".py")), *base, *extra]
        with (output / (name + ".log")).open("w", encoding="utf-8") as log:
            result = subprocess.run(
                argv,
                cwd=common.ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                env={
                    **os.environ,
                    "PYTHONUTF8": "1",
                    "YOLO_COMPARISON_SMOKE": "1",
                    "OMP_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                },
                timeout=300,
            )
        if result.returncode:
            raise RuntimeError(f"{name} exited {result.returncode}; see {output / (name + '.log')}")

    try:
        source_yaml = Path(args.data).resolve()
        source = common.read_yaml(source_yaml)
        source_root = Path(source.get("path", source_yaml.parent))
        source_root = source_root if source_root.is_absolute() else source_yaml.parent / source_root
        subset = output / "data"
        selected = {}
        for split, count in [("train", 2), ("val", 1), ("test", 1)]:
            images = split_files(source[split], source_root.resolve())[:count]
            if len(images) != count:
                raise ValueError(f"Need {count} real images in {split}")
            selected[split] = []
            for image in images:
                label = label_path(image)
                if not label.is_file() or not label.read_text(encoding="utf-8").strip():
                    raise ValueError(f"SMOKE requires a labeled detection image: {image}")
                for src, kind in [(image, "images"), (label, "labels")]:
                    dest = subset / kind / split / src.name
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(src, dest)
                selected[split].append({"source": str(image), "sha256": common.file_hash(image)})
        common.write_json(output / "selected_images.json", selected)
        import yaml

        (subset / "data.yaml").write_text(
            yaml.safe_dump(
                {
                    "path": str(subset),
                    "train": "images/train",
                    "val": "images/val",
                    "test": "images/test",
                    "names": {0: "crack"},
                    "nc": 1,
                }
            ),
            encoding="utf-8",
        )
        base = [
            "--data",
            str(subset / "data.yaml"),
            "--pretrained-cache",
            str(Path(args.pretrained_cache).resolve()),
            "--cache-dir",
            str(output / "cache"),
            "--project",
            str(output / "runs"),
            "--name",
            "SMOKE",
            "--device",
            args.device,
        ]
        for key, value in {
            "train.epochs": 1,
            "train.batch": 1,
            "train.imgsz": 640,
            "train.workers": 0,
            "train.amp": False,
            "train.warmup_epochs": 0,
            "train.nbs": 1,
            "train.plots": False,
            "augment.profile": "none",
            "eval.batch": 1,
            "eval.workers": 0,
        }.items():
            base.extend(["--set", f"{key}={json.dumps(value)}"])
        command("prepare", "prepare")
        command("train", "train")
        run = output / "runs/SMOKE"
        initialization = json.loads((run / "initialization.json").read_text(encoding="utf-8"))
        epoch = json.loads((run / "optimizer_epochs.jsonl").read_text(encoding="utf-8").splitlines()[0])
        if initialization["nc"] != 1 or not initialization["verified_equal_keys"]:
            raise AssertionError("Official COCO-to-nc1 transfer was not verified")
        if epoch["EMA_updates"] != 2 or not all(math.isfinite(x) and x > 0 for x in epoch["loss_items"]):
            raise AssertionError(f"Expected two finite native detection updates: {epoch}")
        report["checks"]["official_pretrained_nc1_native_updates"] = {
            "status": "PASSED",
            "optimizer_updates": epoch["EMA_updates"],
            "loss_items_box_cls_dfl": epoch["loss_items"],
            "inherited_equal_keys": initialization["verified_equal_keys"],
            "nc": initialization["nc"],
            "reg_max": initialization["reg_max"],
            "scale": initialization["model_yaml"].get("scale"),
            "modules": initialization["modules"],
            "pretrained": initialization["pretrained"],
        }
        report["environment"] = json.loads((run / "environment.json").read_text(encoding="utf-8"))
        report["environment"]["scipy"] = importlib.metadata.version("scipy")
        weight = run / "weights/best.pt"
        for split in ["val", "test"]:
            extra = ["--weights", str(weight), "--split", split]
            if split == "test":
                extra.extend(["--val-metrics", str(run / "evaluate_val/metrics.json")])
            command("evaluate_" + split, "evaluate", extra)
            metrics = json.loads((run / ("evaluate_" + split) / "metrics.json").read_text(encoding="utf-8"))
            if metrics["images"] != 1 or metrics["formal_result"] != "SMOKE":
                raise AssertionError("Fresh-process evaluation was not the bounded SMOKE")
            report["checks"]["fresh_process_" + split] = {
                "status": "PASSED",
                "images": metrics["images"],
                "geometry": metrics["geometry"],
                "parameters": metrics["parameters"],
                "weight_sha256": metrics["weight_sha256"],
            }
        command(
            "benchmark",
            "benchmark",
            ["--weights", str(weight), "--set", "benchmark.warmup=0", "--set", "benchmark.iterations=2"],
        )
        benchmark = json.loads((run / "benchmark/benchmark.json").read_text(encoding="utf-8"))
        if not benchmark["fused_actual"] or benchmark["FLOPs"]["status"] != "MEASURED_PARTIAL":
            raise AssertionError("Native fusion and the 640 THOP profile were not both measured")
        report["checks"]["fused_nc1_640_profile"] = {
            "status": "PASSED",
            "formal_result": benchmark["formal_result"],
            "parameters": benchmark["parameters"],
            "FLOPs": benchmark["FLOPs"],
            "timing_note": "Two-iteration development SMOKE only; not formal FPS",
        }
        archive = output / "yolo11n_smoke_core.zip"
        command("package", "package", ["--result-dir", str(run / "evaluate_test"), "--output", str(archive)])
        report["checks"]["core_package"] = {"status": "PASSED", "path": str(archive), "bytes": archive.stat().st_size}
        report["status"] = "PASSED"
    except BaseException as error:
        report.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        common.write_json(output / "validation.json", report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "report": str(output / "validation.json"),
                "package": report["checks"]["core_package"],
            }
        )
    )


if __name__ == "__main__":
    main()
