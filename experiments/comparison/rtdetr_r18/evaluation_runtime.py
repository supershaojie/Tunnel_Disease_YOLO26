"""Independent RT-DETR query inference with the existing comparison statistics and exports."""

import json
from pathlib import Path

from common import (
    MODEL,
    SHARED,
    bootstrap,
    environment,
    file_hash,
    protocol,
    read_yaml,
    result_directory,
    status,
    write_json,
)
from runtime import checkpoint_info


def evaluate(cfg, args):
    """Reload the selected checkpoint and retain every collected query prediction in original coordinates."""
    if args.dry_run:
        print(json.dumps({"mode": "evaluate", "weights": cfg["weights"], "settings": cfg["eval"]}, indent=2))
        return
    if not cfg["weights"]:
        raise ValueError("evaluate requires --weights <this-experiment-best.pt>")
    from shared import evaluation
    from shared.data import audit, backend_paths

    run, checkpoint = checkpoint_info(cfg["weights"])
    dataset, data_report = audit(cfg["data"], read_yaml)
    if checkpoint["data_sha256"] != data_report["data_sha256"]:
        raise ValueError("Evaluation data differs from training data")
    split = cfg["eval"]["split"]
    directory = result_directory(run, args, "evaluate", split)
    directory.mkdir(parents=True, exist_ok=False)
    status(directory, "evaluate", "RUNNING")
    try:
        captured_weight = directory / "weight_snapshot.pt"
        _, checkpoint = checkpoint_info(cfg["weights"], captured_weight)
        weight_name = Path(cfg["weights"]).name
        weight_sha = checkpoint["weights"][weight_name]
        backend = bootstrap(cfg, data_identity=data_report["data_sha256"])
        public = protocol(cfg, data_report["data_sha256"], checkpoint["training_augmentation"][weight_name])
        import numpy as np
        import torch
        from adapter import build_dataset, decode, records_from_batch, verify_model

        from ultralytics.cfg import get_cfg
        from ultralytics.data import build_dataloader
        from ultralytics.utils.torch_utils import select_device

        device = select_device(cfg["train"]["device"], cfg["eval"]["batch"])
        if cfg["eval"]["half"] and device.type != "cuda":
            raise ValueError("FP16 evaluation requires CUDA")
        model = backend.RTDETR(str(captured_weight)).model
        verify_model(model)
        training_params = sum(parameter.numel() for parameter in model.parameters())
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
                "plots": False,
            }
        )
        paths = backend_paths(dataset, Path(cfg["cache_dir"]) / ("splits-" + data_report["data_sha256"]))
        ds = build_dataset(cfg, native, {**paths, "val": paths[split]}, "val", cfg["eval"]["batch"])
        loader = build_dataloader(ds, cfg["eval"]["batch"], cfg["eval"]["workers"], shuffle=False, rank=-1)
        records, shapes, dtypes = [], set(), set()
        with torch.inference_mode():
            for batch in loader:
                images = batch["img"].to(device, non_blocking=True)
                batch["img"] = (images.half() if cfg["eval"]["half"] else images.float()) / 255.0
                shapes.add(tuple(batch["img"].shape))
                dtypes.add((str(next(model.parameters()).dtype), str(batch["img"].dtype)))
                predictions = decode(
                    model(batch["img"]), batch["img"].shape[-2:], cfg["eval"]["conf"], cfg["eval"]["max_det"]
                )
                records.extend(records_from_batch(predictions, batch))
        for record in records:
            record["image_id"] = Path(record["image_id"]).relative_to(data_report["root"]).as_posix()
        threshold = None
        if args.val_metrics:
            val = json.loads(Path(args.val_metrics).read_text(encoding="utf-8"))
            if (val["split"], val["weight_sha256"], val["protocol_sha256"]) != (
                "val",
                weight_sha,
                public["protocol_sha256"],
            ):
                raise ValueError("Frozen threshold source must be independent val of the same checkpoint and protocol")
            threshold = val["threshold_selection"]["confidence"]
        metrics, curves = evaluation.compute(records, split, threshold, collection_conf=cfg["eval"]["conf"])
        metrics.update(
            model=MODEL,
            paper_model="RT-DETR-R18 (LCBR C2)",
            P_R_F1_rule="legacy_curve_peak",
            P_R_display_rule="B19 smoothed F1 peak: P/R/F1 from the same curve index; val_frozen is separate",
            weight=str(Path(cfg["weights"]).resolve()),
            weight_sha256=weight_sha,
            **{
                key: public[key]
                for key in ["protocol_sha256", "run_config_sha256", "augment_core_sha256", "adapter_sha256"]
            },
            training_augmentation=public["config"]["augment"],
            metrics_sha256=file_hash(SHARED / "b19/metrics.py"),
            data_sha256=data_report["data_sha256"],
            formal_result="SMOKE" if checkpoint["execution_kind"] == "SMOKE" else "MEASURED",
            initialization=checkpoint["training_provenance"][weight_name]["initialization"],
            geometry={
                "input_shapes": sorted(shapes),
                "rect": cfg["eval"]["rect"],
                "letterbox": {
                    "auto": False,
                    "scaleup": False,
                    "center": True,
                    "padding_value": 114,
                    "stride": 32,
                    "rectangle_pad": 0.5,
                },
                "decoded": "native normalized xywh times actual input width/height, then undo resize/padding",
                "scores": "native query probabilities, no additional sigmoid or objectness",
                "NMS": "not applicable to native RT-DETR queries",
                "max_det": cfg["eval"]["max_det"],
                "fuse_requested": cfg["eval"]["fuse"],
                "fused_actual": model.is_fused(),
                "model_input_dtypes": sorted(dtypes),
                "TTA": False,
            },
            parameters={
                "unfused": training_params,
                "inference": sum(parameter.numel() for parameter in model.parameters()),
            },
        )
        for name, value in {
            "metrics": metrics,
            "protocol": public,
            "run_config": cfg,
            "data_manifest": data_report,
            "environment": environment(),
        }.items():
            write_json(directory / (name + ".json"), value)
        evaluation.summary(directory / "summary.csv", metrics, rule="legacy_curve_peak")
        evaluation.save_predictions(directory, records)
        np.savez_compressed(directory / "curves.npz", **curves)
        from shared.artifacts import comparisons, figures, freeze_training

        freeze_training(directory, checkpoint, weight_name)
        figures(directory, metrics, curves)
        comparisons(directory, records, cfg, data_report)
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
                {
                    "directory": str(directory),
                    **{
                        key: metrics[key]
                        for key in ["split", "images", "boxes", "legacy_curve_peak", "AP50", "mAP50-95"]
                    },
                },
                indent=2,
            )
        )
    except BaseException as error:
        status(directory, "evaluate", "FAILED", reason=f"{type(error).__name__}: {error}")
        raise
