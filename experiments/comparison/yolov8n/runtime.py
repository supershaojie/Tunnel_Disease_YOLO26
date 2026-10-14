"""Preparation, preflight, formal training and independently reloaded evaluation."""

import json
import os
import shutil
import sys
import time
from pathlib import Path

from common import (
    HERE,
    SHARED,
    bootstrap,
    digest,
    environment,
    file_hash,
    parser,
    protocol,
    read_yaml,
    resolve,
    result_directory,
    status,
    verify_backend,
    write_json,
)


def weight_path(cfg):
    """Choose an explicit weight or the verified per-model official cache."""
    return Path(cfg["weights"] or Path(cfg["pretrained_cache"]) / "yolov8n.pt").resolve()


def verified_pretrained(cfg):
    """Verify both official provenance and computed integrity before unsafe legacy pickle loading."""
    path = weight_path(cfg)
    manifest_path = path.with_suffix(".source.json")
    if not path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"Unprepared official weight: {path}; run prepare.py first")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    asset = verify_backend()["pretrained"]
    if manifest["url"] != asset["url"] or manifest["size"] != asset["size"] or path.stat().st_size != asset["size"]:
        raise ValueError("Official pretrained source/size mismatch; existing files are never overwritten")
    if manifest["sha256"] != file_hash(path):
        raise ValueError(f"Pretrained SHA256 mismatch: {path}")
    return path, manifest


def prepare(cfg, args):
    """Prepare only the official locked model, atomically and without changing installed packages."""
    lock = verify_backend()
    path = weight_path(cfg)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "backend": lock["commit"],
                    "official_default_optimizer": cfg["train"]["optimizer"],
                    "weight": str(path),
                    "exists": path.is_file(),
                    "source": lock["pretrained"],
                },
                indent=2,
            )
        )
        return
    if path.is_file() and path.with_suffix(".source.json").is_file():
        _, manifest = verified_pretrained(cfg)
        print(json.dumps({"status": "REUSED_VERIFIED", "path": str(path), "sha256": manifest["sha256"]}))
        return
    import requests

    asset = lock["pretrained"]
    response = requests.get(asset["api"], timeout=30)
    response.raise_for_status()
    candidates = [a for a in response.json()["assets"] if a["name"] == asset["name"]]
    if (
        len(candidates) != 1
        or candidates[0]["browser_download_url"] != asset["url"]
        or candidates[0]["size"] != asset["size"]
    ):
        raise RuntimeError("Official release asset differs from the checked source lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(".prepare.lock")
    # Atomic per-weight short-lived exclusion; no GPU waiting and no directory-wide lock.
    with lock_path.open("x", encoding="utf-8") as f:
        f.write(str(os.getpid()))
    temp = path.with_name(path.name + f".{os.getpid()}.part")
    try:
        with requests.get(asset["url"], stream=True, timeout=(30, 120)) as response:
            response.raise_for_status()
            with temp.open("xb") as f:
                for chunk in response.iter_content(1048576):
                    f.write(chunk)
        if temp.stat().st_size != asset["size"]:
            raise ValueError("Downloaded official weight size mismatch")
        sha = file_hash(temp)
        if path.exists():
            if file_hash(path) != sha:
                raise ValueError(f"Existing same-name weight differs; preserved {path}; official SHA256={sha}")
            temp.unlink()
        else:
            os.replace(temp, path)
        manifest = {
            **asset,
            "sha256": sha,
            "sha256_authority": "locally computed, not officially published",
            "release_verified_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        write_json(path.with_suffix(".source.json"), manifest)
    finally:
        temp.unlink(missing_ok=True)
        lock_path.unlink(missing_ok=True)
    print(json.dumps({"status": "PREPARED_VERIFIED", "path": str(path), "sha256": sha}))


def preflight(cfg, args):
    """Validate effective configuration and original data with honest missing-file status."""
    lock = verify_backend()
    sys.path.insert(0, str(HERE.parent))
    from shared.data import audit

    if args.dry_run:
        report = {
            "status": "DRY_RUN",
            "formal_training": "NOT_STARTED",
            "backend_sha": lock["commit"],
            "expected_import_path": str(HERE / "vendor/ultralytics_source/ultralytics/__init__.py"),
            "data_status": "EXISTS_NOT_SCANNED" if Path(cfg["data"]).is_file() else "UNVERIFIED_MISSING",
            "weights_status": "EXISTS_NOT_LOADED" if weight_path(cfg).is_file() else "UNVERIFIED_NOT_PREPARED",
            "baseline_args_status": cfg["_provenance"]["baseline_args_status"],
            "name": cfg["name"],
            "project": cfg["project"],
            "requested_train": cfg["train"],
            "effective_augment": cfg["augment"],
            "auto_note": "auto may override lr0/momentum; effective groups recorded after native optimizer build",
            "run_config_sha256": digest(cfg),
        }
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return None
    path, provenance = verified_pretrained(cfg)
    dataset, data_report = audit(cfg["data"], read_yaml)
    backend = bootstrap(cfg, path, data_identity=data_report["data_sha256"])
    model = backend.YOLO(str(path), task="detect").model
    from adapter import verify_model

    verify_model(model, nc=80)
    if model.is_fused():
        raise ValueError("Official formal initialization must be unfused")
    report = {
        "status": "PASSED",
        "server_training": "NOT_STARTED",
        "environment": environment(),
        "data": data_report,
        "pretrained": provenance,
        "protocol": protocol(cfg, data_report["data_sha256"]),
    }
    if args.output:
        write_json(args.output, report)
    print(json.dumps({k: v for k, v in report.items() if k != "data"}, indent=2))
    return report


def checkpoint_info(path):
    """Verify the experiment-side checkpoint hash before native trusted deserialization."""
    path = Path(path).resolve()
    run = path.parent.parent
    manifest_path = run / "checkpoint_manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"Checkpoint has no experiment provenance: {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("backend_sha") != verify_backend()["commit"] or manifest.get("model") != "yolov8n":
        raise ValueError("Checkpoint belongs to another model/backend")
    if manifest["weights"].get(path.name) != file_hash(path):
        raise ValueError(f"Experiment checkpoint hash mismatch: {path}")
    return run, manifest


def train(cfg, args):
    """Run the actual production Trainer registered with native YOLO.train."""
    if args.dry_run:
        return preflight(cfg, args)
    sys.path.insert(0, str(HERE.parent))
    from shared.data import audit, backend_paths

    dataset, data_report = audit(cfg["data"], read_yaml)
    path, provenance = verified_pretrained(cfg)
    planned = Path(cfg["project"]) / cfg["name"]
    resumed = cfg["train"]["resume"]
    if resumed:
        run, checkpoint = checkpoint_info(resumed)
        saved = json.loads((run / "run_config.json").read_text(encoding="utf-8"))
        if checkpoint["data_sha256"] != data_report["data_sha256"]:
            raise ValueError("Resume data contents differ from original run")
        # Native resume restores the original recipe; reject changes it would silently discard.
        diffs = {
            k: {"original": saved["train"].get(k), "requested": v}
            for k, v in cfg["train"].items()
            if saved["train"].get(k) != v and k not in ["resume", "batch", "imgsz", "device", "close_mosaic"]
        }
        if diffs or saved["augment"] != cfg["augment"]:
            raise ValueError(f"Resume recipe differs; restart explicitly instead: {diffs or 'augment'}")
        if saved["_provenance"]["baseline_args_sha256"] != cfg["_provenance"]["baseline_args_sha256"]:
            raise ValueError("Resume B19 reference changed")
        directory = run
        cfg["_execution_kind"] = saved["_execution_kind"]
    else:
        if planned.exists():
            raise FileExistsError(f"New run directory already exists: {planned}; use explicit --resume last.pt")
        directory = planned
        directory.mkdir(parents=True)
        cfg["_execution_kind"] = "SMOKE" if os.environ.get("YOLO_COMPARISON_SMOKE") == "1" else "FORMAL"
    status(
        directory,
        "train",
        "RUNNING",
        formal_training="SMOKE" if cfg["_execution_kind"] == "SMOKE" else "STARTED",
    )
    try:
        backend = bootstrap(cfg, path, data_identity=data_report["data_sha256"])
        from adapter import ComparisonTrainer

        data_paths = backend_paths(dataset, Path(cfg["cache_dir"]) / ("splits-" + data_report["data_sha256"]))
        public = protocol(cfg, data_report["data_sha256"])
        ComparisonTrainer.context = {
            "config": cfg,
            "dataset": data_paths,
            "data_report": data_report,
            "pretrained_provenance": provenance,
            "protocol": public,
            "backend_sha": verify_backend()["commit"],
            "smoke": cfg["_execution_kind"] == "SMOKE",
        }
        if not resumed:
            write_json(directory / "run_config.json", cfg)
            write_json(directory / "user_config.json", cfg["_provenance"]["user_config"])
            write_json(directory / "official_defaults.json", cfg["_provenance"]["official_defaults"])
            write_json(directory / "parameter_sources.json", cfg["_provenance"]["sources"])
            write_json(directory / "requested_config.json", cfg["train"])
            write_json(directory / "data_manifest.json", data_report)
            write_json(directory / "protocol.json", public)
            shutil.copyfile(HERE / "backend.lock.json", directory / "backend.lock.json")
        else:
            write_json(directory / ("resume_request_" + str(os.getpid()) + ".json"), cfg)
        native = {k: v for k, v in cfg["train"].items() if k != "betas"}
        native.update(
            data=str(Path(cfg["data"]).resolve()),
            project=str(directory.parent),
            name=directory.name,
            exist_ok=True,
            model=str(path),
            task="detect",
            mode="train",
        )
        model = backend.YOLO(str(resumed or path), task="detect")
        model.train(trainer=ComparisonTrainer, **native)
        status(
            directory,
            "train",
            "COMPLETED",
            completed_epochs=model.trainer.epoch + 1,
            formal_training="SMOKE" if ComparisonTrainer.context["smoke"] else "COMPLETED",
        )
    except BaseException as e:
        # This records failure without suppressing it or changing batch/device/AMP.
        status(directory, "train", "FAILED", reason=f"{type(e).__name__}: {e}")
        raise


def evaluate(cfg, args):
    """Reload the learned checkpoint in a fresh process, with native DFL decoding and NMS."""
    if args.dry_run:
        return preflight(cfg, args)
    if not cfg["weights"]:
        raise ValueError("evaluate requires --weights <this-experiment-best.pt>")
    run, checkpoint = checkpoint_info(cfg["weights"])
    sys.path.insert(0, str(HERE.parent))
    from shared.data import audit, backend_paths
    from shared import evaluation

    dataset, data_report = audit(cfg["data"], read_yaml)
    if checkpoint["data_sha256"] != data_report["data_sha256"]:
        raise ValueError("Evaluation data differs from audited training data")
    split = cfg["eval"]["split"]
    directory = result_directory(run, args, "evaluate", split)
    if directory.exists():
        raise FileExistsError(f"Evaluation directory exists: {directory}; choose --result-dir")
    directory.mkdir(parents=True)
    status(directory, "evaluate", "RUNNING")
    try:
        backend = bootstrap(cfg, data_identity=data_report["data_sha256"])
        public = protocol(cfg, data_report["data_sha256"])
        import torch
        from ultralytics.cfg import get_cfg
        from ultralytics.data import build_dataloader
        from ultralytics.utils import ops
        from ultralytics.utils.torch_utils import select_device
        from adapter import ComparisonValidator, build_dataset, verify_model

        device = select_device(cfg["train"]["device"], cfg["eval"]["batch"])
        if cfg["eval"]["half"] and device.type == "cpu":
            raise ValueError("FP16 evaluation on CPU is unsupported; choose half=false explicitly")
        yolo = backend.YOLO(str(Path(cfg["weights"]).resolve()), task="detect")
        model = yolo.model
        verify_model(model)
        training_params = sum(p.numel() for p in model.parameters())
        if cfg["eval"]["fuse"]:
            model.fuse()
        model.to(device).eval()
        model.half() if cfg["eval"]["half"] else model.float()
        native = get_cfg(
            overrides={
                "imgsz": cfg["train"]["imgsz"],
                "batch": cfg["eval"]["batch"],
                "rect": cfg["eval"]["rect"],
                "workers": cfg["eval"]["workers"],
                "conf": cfg["eval"]["conf"],
                "iou": cfg["eval"]["iou"],
                "max_det": cfg["eval"]["max_det"],
                "half": cfg["eval"]["half"],
                "plots": False,
                "split": split,
            }
        )
        paths = backend_paths(dataset, Path(cfg["cache_dir"]) / ("splits-" + data_report["data_sha256"]))
        ds = build_dataset(cfg, native, {**paths, "val": paths[split]}, "val", cfg["eval"]["batch"])
        loader = build_dataloader(ds, cfg["eval"]["batch"], cfg["eval"]["workers"], shuffle=False, rank=-1)
        validator = ComparisonValidator(loader, save_dir=directory, args=native)
        validator.data = paths
        validator.device = device
        validator.training = False
        validator.init_metrics(model)
        observed = set()
        with torch.inference_mode():
            for index, batch in enumerate(loader):
                validator.batch_i = index
                batch = validator.preprocess(batch)
                observed.add((str(next(model.parameters()).dtype), str(batch["img"].dtype)))
                raw = model(batch["img"])
                # v8 tuple=(decoded tensor, feature maps); native NMS selects decoded tensor once.
                preds = ops.non_max_suppression(
                    raw, cfg["eval"]["conf"], cfg["eval"]["iou"], multi_label=True, max_det=cfg["eval"]["max_det"]
                )
                validator.update_metrics(preds, batch)
        root = Path(data_report["root"])
        for record in validator.records:
            record["image_id"] = Path(record["image_id"]).relative_to(root).as_posix()
        threshold = None
        if args.val_metrics:
            val = json.loads(Path(args.val_metrics).read_text(encoding="utf-8"))
            if (
                val["split"] != "val"
                or val["weight_sha256"] != file_hash(cfg["weights"])
                or val["protocol_sha256"] != public["protocol_sha256"]
            ):
                raise ValueError("Frozen threshold source is not val of the same weight and protocol")
            threshold = val["threshold_selection"]["confidence"]
        metrics, curves = evaluation.compute(validator.records, split, threshold, collection_conf=cfg["eval"]["conf"])
        metrics.update(
            model="yolov8n",
            weight=str(Path(cfg["weights"]).resolve()),
            weight_sha256=file_hash(cfg["weights"]),
            **{k: public[k] for k in ["protocol_sha256", "run_config_sha256"]},
            augment_core_sha256=public["augment_core_sha256"],
            adapter_sha256=public["adapter_sha256"],
            metrics_sha256=file_hash(SHARED / "b19/metrics.py"),
            data_sha256=data_report["data_sha256"],
            formal_result="SMOKE" if checkpoint["execution_kind"] == "SMOKE" else "MEASURED",
            geometry={
                "input_shapes": sorted(validator.observed_shapes),
                "rect": cfg["eval"]["rect"],
                "letterbox": {
                    "auto": False,
                    "scaleup": False,
                    "center": True,
                    "padding_value": 114,
                    "stride": 32,
                    "rectangle_pad": 0.5,
                },
                "decoded": "native v8 DFL, class score, no objectness multiplication",
                "NMS_iou": cfg["eval"]["iou"],
                "max_det": cfg["eval"]["max_det"],
                "fuse_requested": cfg["eval"]["fuse"],
                "fused_actual": model.is_fused(),
                "model_input_dtypes": sorted(observed),
                "TTA": False,
            },
            parameters={"unfused": training_params, "inference": sum(p.numel() for p in model.parameters())},
        )
        write_json(directory / "metrics.json", metrics)
        write_json(directory / "protocol.json", public)
        write_json(directory / "run_config.json", cfg)
        write_json(directory / "data_manifest.json", data_report)
        write_json(directory / "environment.json", environment())
        evaluation.summary(directory / "summary.csv", metrics)
        evaluation.save_predictions(directory, validator.records)
        import numpy as np

        np.savez_compressed(directory / "curves.npz", **curves)
        from artifacts import figures, comparisons

        figures(directory, metrics, curves, run)
        comparisons(directory, validator.records, cfg, data_report)
        status(
            directory,
            "evaluate",
            "COMPLETED",
            split=split,
            images=metrics["images"],
            formal_result=metrics["formal_result"],
        )
        print(
            json.dumps(
                {k: metrics[k] for k in ["split", "images", "boxes", "AP50", "AP75", "mAP50-95", "formal_result"]}
            )
        )
    except BaseException as e:
        status(directory, "evaluate", "FAILED", reason=f"{type(e).__name__}: {e}")
        raise


def main(mode):
    """Dispatch a real entry point after common strict parsing."""
    args = parser(mode).parse_args()
    cfg = resolve(args)
    if mode in ["prepare", "preflight", "train", "evaluate"]:
        return globals()[mode](cfg, args)
    if mode == "benchmark":
        from measure import benchmark

        return benchmark(cfg, args)
    if mode == "package":
        from artifacts import package

        return package(cfg, args)
    raise ValueError(mode)
