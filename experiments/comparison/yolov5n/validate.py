"""Finite local validation only: synthetic labels, real native loss updates, fresh-process exports."""

import argparse
import ast
import copy
import json
import subprocess
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

from common import AUG, DEFAULT, HERE, bootstrap, configuration, load_yaml, parser, verify_sources
from comparison_io import read_json, write_json


def run(command, output, expected=0):
    result = subprocess.run(
        [sys.executable, *command], capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    Path(output).write_text(result.stdout + "\n" + result.stderr, encoding="utf-8")
    if result.returncode != expected:
        raise AssertionError(
            f"Exit {result.returncode}, expected {expected}: {command}; see {output}\n{result.stdout[-1500:]}\n{result.stderr[-2500:]}"
        )
    return result.stdout


def fixture(directory):
    import cv2

    directory.mkdir(parents=True)
    for split in ("train", "val", "test"):
        (directory / "images" / split).mkdir(parents=True)
        (directory / "labels" / split).mkdir(parents=True)
        for number in range(2):
            pixels = np.full((128, 128, 3), (35, 85, 180) if number == 0 else (170, 130, 30), dtype=np.uint8)
            cv2.line(pixels, (12, 20), (105, 110), (255, 255, 255), 3)
            cv2.imencode(".png", pixels)[1].tofile(directory / "images" / split / f"{number}.png")
            text = (
                ""
                if split == "test" and number == 1
                else "0 0.1875 0.1875 0.1875 0.1875\n"
                if number == 0
                else "0 0.78125 0.78125 0.1875 0.1875\n"
            )
            (directory / "labels" / split / f"{number}.txt").write_text(text, encoding="utf-8")
    (directory / "data.yaml").write_text(
        yaml.safe_dump(
            {
                "path": str(directory),
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "nc": 1,
                "names": {0: "crack"},
            }
        ),
        encoding="utf-8",
    )
    return directory / "data.yaml"


def config_checks(directory, cfg_path):
    entry = str(HERE / "train.py")
    for name in (
        "prepare",
        "preflight",
        "train",
        "evaluate",
        "benchmark",
        "package",
        "aggregate",
        "freeze_sources",
        "validate",
    ):
        run([str(HERE / f"{name}.py"), "--help"], directory / f"help_{name}.txt")
    settings = [
        "train.epochs=150",
        "train.batch=16",
        "train.optimizer=AdamW",
        "train.lr0=0.001",
        "augment.overrides.mixup=0.05",
        "augment.overrides.cutmix=0.0",
        "augment.overrides.close_mosaic=5",
    ]
    command = [entry, "--dry-run"]
    for setting in settings:
        command.extend(["--set", setting])
    plan = json.loads(run(command, directory / "dry_run.json"))
    assert plan["config"]["train"]["optimizer"] == "AdamW" and plan["config"]["train"]["lr0"] == 0.001
    assert plan["config"]["train"]["batch"] == 16 and plan["config"]["augment"]["resolved"]["close_mosaic"] == 5
    assert plan["config"]["name"].endswith("e150_i640_b16_adamw_s42")
    for number, extra in enumerate(
        (
            ["--set", "train.typo=3"],
            ["--set", "augment.overrides.mixup=1.1"],
            ["--set", "train.batch=2", "--set", "train.batch=3"],
            ["--set", "train.optimizer=MuSGD"],
            ["--set", "train.batch=-1"],
            ["--set", "train.amp=1"],
            ["--set", "augment.profile=none", "--set", "augment.overrides.mosaic=1.0"],
            ["--set", "augment.profile=none", "--set", "augment.albumentations=frozen_default"],
        )
    ):
        run([entry, "--dry-run", *extra], directory / f"bad_config_{number}.txt", expected=1)
    duplicate = directory / "duplicate.yaml"
    duplicate.write_text("train:\n  batch: 2\n  batch: 3\n", encoding="utf-8")
    run([entry, "--config", str(duplicate), "--dry-run"], directory / "bad_duplicate.txt", expected=1)
    run(
        [str(HERE / "evaluate.py"), "--dry-run", "--split", "val", "--split", "test"],
        directory / "contradictory_split.txt",
        expected=2,
    )
    foreign = directory / "foreign_namespace/models"
    foreign.mkdir(parents=True)
    (foreign / "__init__.py").write_text("raise RuntimeError('foreign models imported')\n", encoding="utf-8")
    run(
        [
            "-c",
            f"import sys; sys.path.insert(0, {str(HERE)!r}); from common import bootstrap, VENDOR; "
            f"sys.path.insert(0, {str(foreign.parent)!r}); sys.path.append(str(VENDOR)); "
            f"result = bootstrap({str(directory / 'isolated_cache')!r}); "
            "assert result['models'] == str(VENDOR / 'models/yolo.py'); assert sys.path[0] == str(VENDOR)",
        ],
        directory / "adversarial_import_order.txt",
    )
    before = set((directory / "runs").glob("*")) if (directory / "runs").exists() else set()
    run([entry, "--config", str(cfg_path), "--dry-run"], directory / "dry_smoke.json")
    after = set((directory / "runs").glob("*")) if (directory / "runs").exists() else set()
    assert before == after


def metric_checks(directory):
    from shared.b19.metrics import ap_per_class
    from reporting import counts, matches, metric_values

    predictions = [
        np.array(
            [[10, 10, 30, 30, 0.9, 0], [60, 60, 80, 80, 0.85, 0], [10, 10, 30, 30, 0.8, 0], [90, 90, 110, 110, 0.7, 0]],
            np.float32,
        ),
        np.array([[10, 10, 30, 30, 0.6, 0]], np.float32),
        np.zeros((0, 6), np.float32),
    ]
    targets = [
        np.array([[10, 10, 30, 30, 0], [60, 60, 80, 80, 0]], np.float32),
        np.zeros((0, 5), np.float32),
        np.array([[1, 1, 5, 5, 0]], np.float32),
    ]
    result = counts(predictions, targets, 0.25)
    assert (result["TP"], result["FP"], result["FN"]) == (2, 3, 1)
    assert np.isclose(result["P"], 0.4) and np.isclose(result["R"], 2 / 3) and np.isclose(result["F1"], 0.5)
    values, curves = metric_values(predictions, targets)
    assert values["AP50"] == values["AP75"] and 0 < values["mAP50-95"] < 1
    partial = np.array([[13, 10, 33, 30, 0.9, 0]], dtype=np.float32)
    one_gt = targets[0][:1]
    partial_values, _ = metric_values([partial], [one_gt])
    assert partial_values["AP50"] > 0 and partial_values["AP75"] == 0
    competing = np.concatenate([partial, np.array([[10, 10, 30, 30, 0.4, 0]], dtype=np.float32)])
    assert counts([competing], [one_gt], 0.5)["TP"] == 1  # rematch after filtering, not a sliced AP mask
    empty, _ = metric_values([np.zeros((0, 6), np.float32)], [np.zeros((0, 5), np.float32)])
    assert empty["AP50"] is None and empty["fixed_confidence"]["0.25"]["TP"] == 0
    # Compile only the source numerical functions; do not import the B19 framework or another models package.
    source = (HERE.parents[2] / "ultralytics/utils/metrics.py").read_text(encoding="utf-8")
    functions = [
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name in ("smooth", "compute_ap", "ap_per_class")
    ]
    scope = {
        "np": np,
        "Path": Path,
        "checks": SimpleNamespace(check_version=lambda current, required: hasattr(np, "trapezoid")),
    }
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[])), "<B19 reference metrics>", "exec"
        ),
        scope,
    )
    correct = np.concatenate([matches(p, t) for p, t in zip(predictions, targets)])
    conf = np.concatenate([p[:, 4] for p in predictions])
    classes = np.concatenate([p[:, 5] for p in predictions])
    gt = np.concatenate([t[:, 4] for t in targets])
    reference, frozen = scope["ap_per_class"](correct, conf, classes, gt), ap_per_class(correct, conf, classes, gt)
    for left, right in zip(reference, frozen):
        np.testing.assert_allclose(left, right, rtol=0, atol=0)
    write_json(
        directory / "metric_fixture.json",
        {"exact_counts": result, "metrics": values, "B19_reference_arrays": "EXACT_EQUAL"},
    )


def augmentation_checks(directory, cfg_path, data):
    import random
    import torch

    from shared.b19.augment import CutMix, MixUp, Mosaic, RandomPerspective
    from augmentation import schedule
    from data import ComparisonDataset, loader

    cfg, audit = configuration(parser("check").parse_args(["--config", str(cfg_path), "--dry-run"]))
    cfg["train"]["imgsz"] = 128
    cfg["train"]["epochs"] = 20
    cfg["augment"]["profile"] = "custom"
    cfg["augment"]["resolved"] = {
        **AUG,
        "degrees": 0.0,
        "translate": 0.0,
        "scale": 0.0,
        "shear": 0.0,
        "perspective": 0.0,
        "close_mosaic": 5,
    }
    dataset = ComparisonDataset(data["splits"]["train"]["records"], cfg, "train")
    first = dataset.get_image_and_label(0)
    second = dataset.get_image_and_label(1)
    affine = RandomPerspective(degrees=0, translate=0, scale=0, shear=0, perspective=0, size=(128, 128))
    random.seed(42)
    np.random.seed(42)
    mosaic = Mosaic(dataset, 128, p=1.0)
    output = mosaic(copy.deepcopy(first))
    assert output["img"].shape == (256, 256, 3) and len(output["instances"]) >= 2
    mix = MixUp(dataset, pre_transform=affine, p=1.0)
    mix.get_indexes = lambda: 1
    mixed = mix(affine(copy.deepcopy(first)))
    assert len(mixed["instances"]) == 2 and not np.array_equal(mixed["img"], first["img"])
    cut = CutMix(dataset, pre_transform=affine, p=1.0)
    cut.get_indexes = lambda: 1
    cut._rand_bbox = lambda width, height: (72, 72, 112, 112)
    cut_result = cut(affine(copy.deepcopy(first)))
    assert len(cut_result["instances"]) == 2
    assert np.array_equal(cut_result["img"][80:110, 80:110], second["img"][80:110, 80:110])
    before, _ = schedule(cfg["augment"]["resolved"], 14, 20)
    after, closed = schedule(cfg["augment"]["resolved"], 15, 20)
    assert before["cutmix"] > 0 and closed and all(after[k] == 0 for k in ("mosaic", "copy_paste", "mixup", "cutmix"))
    assert after["degrees"] == before["degrees"]
    image, target, _, _ = dataset[0]
    assert image.shape == (3, 128, 128) and target.shape[1] == 6 and torch.isfinite(target).all()
    assert ((target[:, 2:] >= 0) & (target[:, 2:] <= 1)).all()
    assert dataset.set_epoch(15) and dataset.current_values["cutmix"] == 0
    none = copy.deepcopy(cfg)
    none["augment"]["profile"] = "none"
    plain = ComparisonDataset(data["splits"]["train"]["records"], none, "train")
    state = random.getstate()
    a = plain[0][0]
    b = plain[0][0]
    assert torch.equal(a, b) and state == random.getstate()
    worker_cfg = copy.deepcopy(cfg)
    worker_cfg["train"]["workers"] = 1
    worker_loader, worker_dataset = loader(data["splits"]["train"]["records"], worker_cfg, "train")
    worker_dataset.set_epoch(15)
    image, target, _, _ = next(iter(worker_loader))
    assert image.shape[:2] == (2, 3) and target.shape[1] == 6
    write_json(
        directory / "augmentation_fixture.json",
        {
            "Mosaic": "FORCED_PASSED",
            "MixUp": "FORCED_PASSED",
            "CutMix": "FORCED_LABEL_AND_PIXEL_CHANGE_PASSED",
            "close_mosaic": "epoch 15/20 disables Mosaic/CopyPaste/MixUp/CutMix",
            "none": "deterministic; Python RNG unchanged",
            "worker_spawn": "PASSED",
            "coordinates": "valid normalized xywh; one native collate encoding",
        },
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True, help="New development-only temporary directory")
    p.add_argument("--no-cuda", action="store_true")
    args = p.parse_args()
    directory = args.output.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    report = {
        "purpose": "SMOKE",
        "server_training": "NOT_STARTED",
        "server_environment": "UNVERIFIED",
        "real_dataset_test": "NOT_RUN",
        "checks": {},
    }
    try:
        import torch

        verify_sources()
        bootstrap(directory / "cache")
        data_yaml = fixture(directory / "synthetic_data")
        cfg = load_yaml(DEFAULT)
        cfg.update(
            data=str(data_yaml),
            weights=str(args.weights.resolve()),
            project=str(directory / "runs"),
            cache=str(directory / "cache"),
            name="SMOKE_cpu",
        )
        cfg["train"].update(epochs=2, imgsz=64, batch=2, workers=0, device="cpu", amp=False, autoanchor=False)
        cfg["augment"].update(profile="none")
        cfg["eval"].update(batch=2, workers=0)
        cfg["benchmark"].update(imgsz=64, warmup=1, iterations=2)
        cfg_path = directory / "smoke.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        from data import manifest

        data = manifest(data_yaml)
        config_checks(directory, cfg_path)
        report["checks"]["interfaces_configuration"] = "PASSED"
        metric_checks(directory)
        report["checks"]["metrics_reference_and_counts"] = "PASSED"
        augmentation_checks(directory, cfg_path, data)
        report["checks"]["forced_augmentations_schedule_workers"] = "PASSED"
        entry = [str(HERE / "train.py"), "--config", str(cfg_path)]
        run(entry, directory / "CPU_train.txt")
        cpu = directory / "runs/SMOKE_cpu"
        result = read_json(cpu / "training_summary.json")
        assert result["optimizer_updates"] == 2 and result["completed_epochs"] == 2
        transfer = read_json(cpu / "pretrained_transfer.json")
        assert transfer["nc"] == 1 and transfer["missing_keys"] and not transfer["fused_before_training"]
        report["checks"]["CPU_FP32_two_native_loss_updates"] = {
            "state": "PASSED",
            "summary": result,
            "untransferred_keys": transfer["missing_keys"],
        }
        weight = cpu / "weights/best.pt"
        run(
            [
                str(HERE / "evaluate.py"),
                "--config",
                str(cfg_path),
                "--weights",
                str(weight),
                "--split",
                "val",
                "--set",
                "augment.profile=native",
                "--result-dir",
                str(directory / "wrong_training_identity"),
            ],
            directory / "wrong_training_identity.txt",
            expected=1,
        )
        assert not (directory / "wrong_training_identity").exists()
        report["checks"]["evaluation_preserves_training_augmentation"] = "PASSED"
        for split in ("val", "test"):
            run(
                [str(HERE / "evaluate.py"), "--config", str(cfg_path), "--weights", str(weight), "--split", split],
                directory / f"CPU_evaluate_{split}.txt",
            )
        metrics = read_json(cpu / "evaluation/test/metrics.json")
        assert (
            metrics["actual"]["model_dtype"] == "torch.float32"
            and metrics["actual"]["input_dtype"] == "torch.float32"
            and metrics["actual"]["fused"]
        )
        assert metrics["images"] == 2 and metrics["purpose"] == "SMOKE"
        report["checks"]["fresh_process_evaluate_FP32_fuse_NMS_exports"] = "PASSED"
        run(
            [
                str(HERE / "benchmark.py"),
                "--config",
                str(cfg_path),
                "--weights",
                str(weight),
                "--image",
                str(directory / "synthetic_data/images/test/0.png"),
            ],
            directory / "CPU_benchmark.txt",
        )
        run([str(HERE / "package.py"), "--result-dir", str(cpu)], directory / "CPU_package.txt")
        package = read_json(str(cpu.with_name(cpu.name + ".core.zip")) + ".manifest.json")
        assert package["bytes"] < 12000000 and not any(item["name"].endswith(".pt") for item in package["members"])
        report["checks"]["compact_package"] = {
            "state": "PASSED",
            "bytes": package["bytes"],
            "members": len(package["members"]),
        }
        if torch.cuda.is_available() and not args.no_cuda:
            gpu = copy.deepcopy(cfg)
            gpu["name"] = "SMOKE_cuda"
            gpu["train"].update(device="0", amp=True)
            gpu_path = directory / "smoke_cuda.yaml"
            gpu_path.write_text(yaml.safe_dump(gpu, sort_keys=False), encoding="utf-8")
            run([str(HERE / "train.py"), "--config", str(gpu_path)], directory / "CUDA_train.txt")
            gpu_run = directory / "runs/SMOKE_cuda"
            updates = read_json(gpu_run / "training_summary.json")
            actual = read_json(gpu_run / "amp_check.json")
            assert updates["optimizer_updates"] == 2 and actual["actual"]
            report["checks"]["CUDA_AMP_two_native_loss_updates"] = {"state": "PASSED", "amp": actual}
        else:
            report["checks"]["CUDA_AMP_two_native_loss_updates"] = "UNVERIFIED"
        # Explicit resume extends one epoch; protocol remains unchanged and optimizer/EMA state reloads.
        resume_cmd = [
            str(HERE / "train.py"),
            "--config",
            str(cfg_path),
            "--resume",
            str(cpu / "weights/last.pt"),
            "--set",
            "train.epochs=3",
        ]
        run(resume_cmd, directory / "CPU_resume.txt")
        resumed = read_json(cpu / "training_summary.json")
        assert resumed["completed_epochs"] == 3 and resumed["optimizer_updates"] == 3
        report["checks"]["explicit_resume_state"] = "PASSED"
        # No new run can silently overwrite an existing output directory.
        run(entry, directory / "existing_run_rejected.txt", expected=1)
        report["checks"]["duplicate_run_refused"] = "PASSED"
        report["state"] = "PASSED"
    except BaseException as error:
        report["state"] = "FAILED"
        report["error"] = f"{type(error).__name__}: {error}"
        traceback.print_exc()
        raise
    finally:
        write_json(directory / "validation_report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
