"""Finite checks of configuration, frozen operators, real detection updates and reload/package paths."""

import argparse
import ast
import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from common import HERE, ROOT, parser, protocol, resolve, write_json


class WorkerSamples:
    """Expose a worker's augmentation state across the native loader/resume lifecycle."""

    mosaic = True

    def __len__(self):
        return 4

    def __getitem__(self, index):
        return int(self.mosaic)

    def close_mosaic(self, hyp):
        self.mosaic = False


def main():
    """Run a bounded synthetic SMOKE, never formal training or the original test split."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True, help="New development-only SMOKE directory")
    p.add_argument("--pretrained-cache", required=True)
    p.add_argument("--skip-cuda", action="store_true")
    args = p.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    results = {"formal_training": "NOT_STARTED", "server": "UNVERIFIED", "data": "SYNTHETIC_SMOKE", "checks": {}}

    def command(name, argv, expect=0):
        result = subprocess.run(
            [sys.executable, "-B", *map(str, argv)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env={**os.environ, "PYTHONUTF8": "1", "YOLO_COMPARISON_SMOKE": "1"},
        )
        (output / (name + ".log")).write_text(result.stdout + result.stderr, encoding="utf-8")
        if result.returncode != expect:
            raise AssertionError(f"{name}: exit={result.returncode}, expected={expect}; see {output / (name + '.log')}")
        return result

    try:
        for entry in ["prepare", "preflight", "train", "evaluate", "benchmark", "package", "launch"]:
            command("help_" + entry, [HERE / (entry + ".py"), "--help"])
        results["checks"]["entry_help"] = "PASSED"
        demo = [
            "--dry-run",
            "--set",
            "train.epochs=150",
            "--set",
            "train.batch=16",
            "--set",
            "train.optimizer=AdamW",
            "--set",
            "train.lr0=0.001",
            "--set",
            "augment.overrides.mixup=0.05",
            "--set",
            "augment.overrides.cutmix=0.0",
            "--set",
            "augment.overrides.close_mosaic=5",
        ]
        preview = json.loads(command("dry_run", [HERE / "train.py", *demo]).stdout)
        assert preview["requested_train"]["optimizer"] == "AdamW" and preview["requested_train"]["lr0"] == 0.001
        assert preview["requested_train"]["batch"] == 16 and preview["effective_augment"]["effective"]["mixup"] == 0.05
        assert (
            preview["effective_augment"]["effective"]["cutmix"] == 0.0
            and not Path(preview["project"], preview["name"]).exists()
        )
        command("dry_auto", [HERE / "train.py", "--dry-run", "--set", "train.optimizer=auto"])
        command("dry_none", [HERE / "train.py", "--dry-run", "--set", "augment.profile=none"])
        for name, setting in [
            ("unknown", ["train.unrecognized=1"]),
            ("probability", ["augment.overrides.cutmix=1.1"]),
            ("duplicate", ["train.batch=16", "train.batch=32"]),
            ("boolean", ["train.amp=yes_invalid"]),
            ("hybrid_gt", ["train.save_hybrid=true"]),
            ("mapping_typo", ["augment.overrides={mixupp: 0.05}"]),
        ]:
            command(
                "reject_" + name,
                [HERE / "train.py", "--dry-run", *[x for v in setting for x in ["--set", v]]],
                expect=1,
            )
        results["checks"]["config_overrides_and_rejections"] = "PASSED"
        command("reject_repeated_device", [HERE / "train.py", "--dry-run", "--device", "0", "--device", "1"], expect=2)
        cfg = resolve(parser("verify").parse_args(["--dry-run"]))
        other = deepcopy(cfg)
        other["train"]["optimizer"] = "AdamW"
        other["name"] = "different"
        other["project"] = "/different"
        assert (
            protocol(cfg, "synthetic-content")["protocol_sha256"]
            == protocol(other, "synthetic-content")["protocol_sha256"]
        )
        results["checks"]["protocol_excludes_model_recipe_paths"] = "PASSED"

        sys.path.insert(0, str(HERE.parent))
        from shared import evaluation, provider
        from shared.b19.augment import CutMix, MixUp, Mosaic, RandomPerspective
        from shared.b19.instance import Instances
        import torch

        torch.set_num_threads(2)

        class Samples:
            cache = "ram"
            buffer = []
            data = {"names": {0: "crack"}}

            def __len__(self):
                return 4

            def get_image_and_label(self, index):
                box = [40, 40, 56, 56] if index else [2, 2, 10, 10]
                return {
                    "img": np.full((64, 64, 3), index * 50, dtype=np.uint8),
                    "cls": np.zeros((1, 1), dtype=np.float32),
                    "instances": Instances(
                        np.array([box], dtype=np.float32),
                        np.zeros((0, 1000, 2), dtype=np.float32),
                        bbox_format="xyxy",
                        normalized=False,
                    ),
                    "im_file": str(index),
                    "ori_shape": (64, 64),
                    "resized_shape": (64, 64),
                }

        samples = Samples()
        np.random.seed(42)
        __import__("random").seed(42)
        mosaic = Mosaic(samples, imgsz=64, p=1.0)(samples.get_image_and_label(0))
        assert mosaic["img"].shape == (128, 128, 3) and len(mosaic["cls"]) > 1
        affine = RandomPerspective(degrees=0, translate=0, scale=0, shear=0, perspective=0, size=(64, 64))
        cropped = affine(mosaic)
        assert cropped["img"].shape == (64, 64, 3)
        assert ((cropped["instances"].bboxes >= 0) & (cropped["instances"].bboxes <= 64)).all()
        mixup = MixUp(samples, p=1.0)
        with patch.object(mixup, "get_indexes", return_value=1):
            mixed = mixup(samples.get_image_and_label(0))
        assert len(mixed["cls"]) == 2 and not np.array_equal(mixed["img"], samples.get_image_and_label(0)["img"])
        cutmix = CutMix(samples, p=1.0)
        with patch.object(cutmix, "get_indexes", return_value=1), patch.object(
            cutmix, "_rand_bbox", return_value=(32, 32, 64, 64)
        ):
            cut = cutmix(samples.get_image_and_label(0))
        assert len(cut["cls"]) == 2 and (cut["img"][40:56, 40:56] == 50).all()
        closed = provider.closed_values(cfg["augment"]["effective"])
        assert all(closed[k] == 0 for k in provider.CLOSED_OPERATIONS)
        assert closed["hsv_h"] == cfg["augment"]["effective"]["hsv_h"]
        results["checks"]["forced_mosaic_mixup_cutmix_cropping_schedule"] = "PASSED"

        records = [
            {
                "image_id": "positive",
                "gt": np.array([[0, 0, 10, 10, 0], [20, 20, 30, 30, 0]], dtype=np.float32),
                "predictions": np.array(
                    [
                        [0, 0, 10, 10, 0.9, 0],
                        [0, 0, 10, 10, 0.8, 0],
                        [20, 20, 30, 30, 0.4, 0],
                        [45, 45, 50, 50, 0.3, 0],
                    ],
                    dtype=np.float32,
                ),
            },
            {
                "image_id": "empty",
                "gt": np.zeros((0, 5), dtype=np.float32),
                "predictions": np.zeros((0, 6), dtype=np.float32),
            },
            {
                "image_id": "false_positive",
                "gt": np.zeros((0, 5), dtype=np.float32),
                "predictions": np.array([[0, 0, 10, 10, 0.6, 0]], dtype=np.float32),
            },
        ]
        metrics, curves = evaluation.compute(records, "val")
        assert metrics["fixed_confidence"]["0.25"]["TP"] == 2 and metrics["fixed_confidence"]["0.25"]["FP"] == 3
        assert metrics["fixed_confidence"]["0.5"]["TP"] == 1 and metrics["fixed_confidence"]["0.5"]["FN"] == 1
        assert metrics["AP75"] == float(curves["AP"][:, 5].mean()) and metrics["units"] == "fraction"
        # Execute original B19 functions independently from the anchor worktree for equivalence.
        source = (ROOT / "ultralytics/utils/metrics.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        nodes = [n for n in tree.body if getattr(n, "name", None) in ["smooth", "compute_ap", "ap_per_class"]]
        namespace = {
            "np": np,
            "Path": Path,
            "checks": SimpleNamespace(check_version=lambda current, required: int(current.split(".")[0]) >= 2),
        }
        module = ast.Module(
            body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes],
            type_ignores=[],
        )
        exec(compile(ast.fix_missing_locations(module), "<B19-reference>", "exec"), namespace)
        pred = np.concatenate([r["predictions"] for r in records])
        gt = np.concatenate([r["gt"] for r in records])
        tp = np.concatenate([evaluation.correct(r["predictions"], r["gt"]) for r in records])
        ref = namespace["ap_per_class"](tp, pred[:, 4], pred[:, 5], gt[:, 4])
        assert np.array_equal(ref[5], curves["AP"])
        results["checks"]["frozen_reference_ap_matching_counts_empty_images"] = "PASSED"

        from PIL import Image, ImageDraw
        import yaml

        data_root = output / "SMOKE_data"
        for split, n in [("train", 4), ("val", 2), ("test", 3)]:
            (data_root / "images" / split).mkdir(parents=True)
            (data_root / "labels" / split).mkdir(parents=True)
            for index in range(n):
                im = Image.new("RGB", (96, 64 + (index % 2) * 32), (30 + index * 30,) * 3)
                ImageDraw.Draw(im).rectangle((25, 16, 62, 47), fill=(230, 230, 230))
                im.save(data_root / "images" / split / f"{split}_{index}.png")
                (data_root / "labels" / split / f"{split}_{index}.txt").write_text(
                    "" if split == "test" and index == 2 else "0 0.45 0.45 0.30 0.30\n", encoding="utf-8"
                )
        data_yaml = data_root / "data.yaml"
        data_yaml.write_text(
            yaml.safe_dump(
                {
                    "path": str(data_root),
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
            data_yaml,
            "--baseline-args",
            HERE / "reference/b19_archived_args.yaml",
            "--pretrained-cache",
            Path(args.pretrained_cache).resolve(),
            "--cache-dir",
            output / "cache",
            "--project",
            output / "SMOKE_runs",
            "--set",
            "train.epochs=1",
            "--set",
            "train.imgsz=64",
            "--set",
            "train.batch=2",
            "--set",
            "train.nbs=2",
            "--set",
            "train.workers=0",
            "--set",
            "train.warmup_epochs=0",
            "--set",
            "train.optimizer=AdamW",
            "--set",
            "train.lr0=0.001",
            "--set",
            "train.plots=false",
            "--set",
            "eval.batch=2",
            "--set",
            "eval.workers=0",
            "--set",
            "augment.overrides.close_mosaic=0",
        ]
        command(
            "cpu_detection_updates",
            [HERE / "train.py", *base, "--name", "SMOKE_cpu", "--device", "cpu", "--set", "train.amp=false"],
        )
        run = output / "SMOKE_runs/SMOKE_cpu"
        summary = json.loads((run / "training_summary.json").read_text(encoding="utf-8"))
        assert summary["optimizer_updates_ema"] >= 2
        assert summary["server_training_status"] == "SMOKE"
        init = json.loads((run / "initialization.json").read_text(encoding="utf-8"))
        assert len(init["inherited_keys"]) > 100 and len(init["reinitialized_keys"]) > 0
        epoch = json.loads((run / "optimizer_epochs.jsonl").read_text(encoding="utf-8").splitlines()[0])
        assert np.isfinite(epoch["loss_items"]).all() and epoch["loss_items"][0] > 0 and epoch["loss_items"][2] > 0
        assert not list(data_root.rglob("*.cache")) and not list(data_root.rglob("*.npy"))
        results["checks"]["cpu_fp32_two_native_detection_optimizer_updates"] = {
            "status": "PASSED",
            "updates": summary["optimizer_updates_ema"],
        }
        results["checks"]["nc1_pretrained_inheritance_and_cache_isolation"] = "PASSED"
        weight = run / "weights/best.pt"
        # Inspect the real native auto rule without further training or a parameter sweep.
        from common import bootstrap
        from runtime import verified_pretrained

        local_cfg = resolve(
            parser("verify").parse_args(
                list(map(str, [*base, "--name", "SMOKE_probe", "--device", "cpu", "--set", "train.amp=false"]))
            )
        )
        pretrained, _ = verified_pretrained(local_cfg)
        backend = bootstrap(local_cfg, pretrained)
        native_model = backend.YOLO(str(weight)).model
        from ultralytics.engine.trainer import BaseTrainer

        auto_results = {}
        for budget in [26400, 20]:
            probe = SimpleNamespace(args=SimpleNamespace(lr0=0.123, momentum=0.7, warmup_bias_lr=0.1))
            optimizer = BaseTrainer.build_optimizer(
                probe, native_model, name="auto", lr=0.123, momentum=0.7, decay=0.0005, iterations=budget
            )
            auto_results[str(budget)] = {
                "class": type(optimizer).__name__,
                "lr": optimizer.param_groups[0]["lr"],
                "momentum_or_betas": optimizer.param_groups[0].get("momentum", optimizer.param_groups[0].get("betas")),
                "warmup_bias_lr": probe.args.warmup_bias_lr,
            }
        assert auto_results["26400"]["class"] == "SGD" and auto_results["26400"]["lr"] == 0.01
        assert auto_results["20"]["class"] == "AdamW" and auto_results["20"]["lr"] == 0.002
        results["checks"]["locked_native_auto_requested_effective"] = auto_results
        # Exercise the adapter's actual normalized sample and the native dataloader's worker initialization.
        from shared.data import audit, backend_paths
        from adapter import ComparisonTrainer, build_dataset
        from ultralytics.cfg import get_cfg
        from ultralytics.data import build_dataloader
        from ultralytics.data.build import seed_worker

        audited, data_report = audit(data_yaml)
        paths = backend_paths(audited, output / "probe_splits")
        native_args = get_cfg(overrides={k: v for k, v in local_cfg["train"].items() if k != "betas"})
        dataset = build_dataset(local_cfg, native_args, paths, "train", 2)
        sample = dataset[0]
        assert sample["img"].shape == (3, 64, 64)
        assert ((sample["bboxes"] >= 0) & (sample["bboxes"] <= 1)).all() and (sample["bboxes"][:, 2:] > 0).all()
        assert (sample["cls"] == 0).all()
        dataset.close_mosaic(native_args)
        assert all(dataset.augmentation_values[k] == 0 for k in provider.CLOSED_OPERATIONS)
        loader = build_dataloader(dataset, 2, 0, shuffle=True, rank=-1)
        assert loader.worker_init_fn is seed_worker
        results["checks"]["native_adapter_format_close_worker_seed"] = "PASSED"
        from ultralytics.data.build import InfiniteDataLoader

        workers = InfiniteDataLoader(WorkerSamples(), batch_size=2, num_workers=1)
        try:
            assert next(iter(workers)).tolist() == [1, 1]
            ancestor_cfg = deepcopy(local_cfg)
            ancestor_cfg["train"].update(epochs=5, close_mosaic=4)
            ancestor_cfg["augment"]["effective"]["close_mosaic"] = 4
            ancestor = protocol(ancestor_cfg, data_report["data_sha256"])["config"]["augment"]
            resume_cfg = deepcopy(ancestor_cfg)
            resume_cfg["train"].update(batch=16, imgsz=128)
            fresh = protocol(resume_cfg, data_report["data_sha256"])
            resumed = ComparisonTrainer.__new__(ComparisonTrainer)
            resumed.resume, resumed.ema, resumed.epochs = True, None, 5
            resumed.args = SimpleNamespace(model="SMOKE", close_mosaic=4)
            resumed.train_loader = workers
            resumed.save_dir = output / "SMOKE_resume_policy"
            resumed.save_dir.mkdir()
            resumed.context = {
                "config": resume_cfg,
                "data_report": data_report,
                "protocol": deepcopy(fresh),
                "resume_augmentation": ancestor,
            }
            resumed._close_dataloader_mosaic = lambda: BaseTrainer._close_dataloader_mosaic(resumed)
            resumed.resume_training({"epoch": 1, "optimizer": None})
            assert resumed.start_epoch == 2 and next(iter(workers)).tolist() == [0, 0]
            inherited = deepcopy(resumed.context["protocol"])
            history = inherited["config"]["augment"]["history"]
            assert history[0]["policy"] == ancestor and history[1]["start_epoch_zero_based"] == 2
            assert history[1]["policy"] == fresh["config"]["augment"]
            assert inherited["protocol_sha256"] != fresh["protocol_sha256"]
            resumed.context.update(protocol=deepcopy(fresh), resume_augmentation=inherited["config"]["augment"])
            resumed.resume_training({"epoch": 1, "optimizer": None})
            assert resumed.context["protocol"] == inherited
        finally:
            workers.iterator._shutdown_workers()
        results["checks"]["resume_after_close_refreshes_real_loader_workers"] = "PASSED"
        results["checks"]["resume_retains_training_history_without_duplicate_segments"] = "PASSED"
        native_cfg = resolve(parser("verify").parse_args(["--dry-run", "--set", "augment.profile=native"]))
        native_protocol = protocol(native_cfg, "synthetic-content")
        assert native_protocol["augment_core_sha256"] != protocol(cfg, "synthetic-content")["augment_core_sha256"]
        assert isinstance(native_protocol["config"]["augment"]["extras"]["albumentations"]["enabled"], bool)
        results["checks"]["native_operator_and_optional_pipeline_identity"] = "PASSED"
        rect_policies = []
        for rectangular in [False, True]:
            policy_cfg = resolve(
                parser("verify").parse_args(
                    list(
                        map(
                            str,
                            [
                                *base,
                                "--set",
                                "augment.profile=native",
                                "--set",
                                "augment.overrides.mixup=0.05",
                                "--set",
                                f"train.rect={str(rectangular).lower()}",
                            ],
                        )
                    )
                )
            )
            recipe = protocol(policy_cfg, "synthetic-content")
            effective = recipe["config"]["augment"]["values"]
            real_args = get_cfg(overrides={k: v for k, v in policy_cfg["train"].items() if k != "betas"})
            real_dataset = build_dataset(policy_cfg, real_args, paths, "train", 2)
            assert effective == real_dataset.augmentation_values
            assert effective["mosaic"] == (0.0 if rectangular else 1.0)
            assert effective["mixup"] == (0.0 if rectangular else 0.05)
            rect_policies.append(recipe["protocol_sha256"])
        assert rect_policies[0] != rect_policies[1]
        results["checks"]["native_rect_pipeline_owned_effective_identity"] = "PASSED"
        command(
            "reload_val",
            [
                HERE / "evaluate.py",
                *base,
                "--weights",
                weight,
                "--device",
                "cpu",
                "--split",
                "val",
                "--result-dir",
                output / "SMOKE_val",
            ],
        )
        command(
            "reload_test",
            [
                HERE / "evaluate.py",
                *base,
                "--weights",
                weight,
                "--device",
                "cpu",
                "--split",
                "test",
                "--result-dir",
                output / "SMOKE_test",
                "--val-metrics",
                output / "SMOKE_val/metrics.json",
                "--set",
                "augment.profile=none",
            ],
        )
        tested = json.loads((output / "SMOKE_test/metrics.json").read_text(encoding="utf-8"))
        assert tested["images"] == 3 and tested["val_frozen"] is not None
        assert tested["formal_result"] == "SMOKE"
        assert tested["training_augmentation"]["profile"] == "b19"
        assert json.loads((output / "SMOKE_test/run_config.json").read_text())["augment"]["profile"] == "none"
        assert tested["geometry"]["model_input_dtypes"] == [["torch.float32", "torch.float32"]]
        with np.load(output / "SMOKE_test/predictions.npz") as stored:
            assert stored["boxes"].dtype == np.float32 and len(stored["image_offsets"]) == 4
        results["checks"]["fresh_process_native_fp32_fuse_nms_complete_exports"] = "PASSED"
        command(
            "small_benchmark",
            [
                HERE / "benchmark.py",
                *base,
                "--weights",
                weight,
                "--device",
                "cpu",
                "--result-dir",
                output / "SMOKE_custom_benchmark",
                "--set",
                "augment.profile=native",
                "--set",
                "benchmark.imgsz=64",
                "--set",
                "benchmark.warmup=1",
                "--set",
                "benchmark.iterations=2",
            ],
        )
        command("core_package", [HERE / "package.py", "--dry-run", "--result-dir", output / "SMOKE_test"])
        command(
            "core_package_real",
            [
                HERE / "package.py",
                "--result-dir",
                output / "SMOKE_test",
                "--baseline-args",
                HERE / "reference/b19_archived_args.yaml",
                "--output",
                output / "SMOKE_core.zip",
            ],
        )
        assert (output / "SMOKE_core.zip").stat().st_size < 12000000
        import zipfile

        with zipfile.ZipFile(output / "SMOKE_core.zip") as z:
            measured = json.loads(z.read("benchmark/benchmark.json"))
            assert measured["status"] == "MEASURED" and measured["formal_result"] == "SMOKE"
            assert measured["weight_sha256"] == tested["weight_sha256"]
            assert measured["training_augmentation"] == tested["training_augmentation"]
        results["checks"]["evaluation_benchmark_keep_checkpoint_training_identity"] = "PASSED"
        results["checks"]["core_package_members_and_strict_bytes"] = {
            "status": "PASSED",
            "bytes": (output / "SMOKE_core.zip").stat().st_size,
        }
        if torch.cuda.is_available() and not args.skip_cuda:
            command(
                "cuda_amp_detection_updates",
                [HERE / "train.py", *base, "--name", "SMOKE_cuda", "--device", "0", "--set", "train.amp=true"],
            )
            effective = json.loads((output / "SMOKE_runs/SMOKE_cuda/effective_config.json").read_text(encoding="utf-8"))
            assert effective["amp_effective"]
            results["checks"]["cuda_native_amp_updates"] = "PASSED"
        else:
            results["checks"]["cuda_native_amp_updates"] = "UNVERIFIED"
        results["status"] = "PASSED"
    except BaseException as e:
        results["status"] = "FAILED"
        results["reason"] = f"{type(e).__name__}: {e}"
        write_json(output / "verification.json", results)
        raise
    write_json(output / "verification.json", results)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
