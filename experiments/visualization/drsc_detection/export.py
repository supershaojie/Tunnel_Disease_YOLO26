"""Thin native evaluator capture, executed by exactly one model's existing Python environment."""

import argparse
import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np


def load(path):
    """Read saved experiment settings."""
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save(directory, records, request, actual):
    """Persist the established portable NPZ layout and one source record after successful inference."""
    ids = np.array([r["image_id"] for r in records])
    for name, field, columns in (("predictions", "predictions", 6), ("ground_truth", "gt", 5)):
        arrays = [np.asarray(record[field], dtype=np.float32).reshape(-1, columns) for record in records]
        values = np.concatenate(arrays)
        payload = {
            "boxes": values[:, :4],
            "classes": values[:, -1].astype(np.int32),
            "image_ids": ids,
            "image_offsets": np.cumsum([0] + [len(a) for a in arrays], dtype=np.int64),
        }
        if columns == 6:
            payload["scores"] = values[:, 4]
        np.savez_compressed(directory / (name + ".npz"), **payload)
    profile = request["profile"]
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=profile["source"], capture_output=True, text=True, check=True
    ).stdout.strip()
    metadata = {
        "model": request["key"],
        "weight_sha256": profile["sha256"],
        "weights": profile["weights"],
        "source": profile["source"],
        "code_sha": revision,
        "python": sys.executable,
        "collection_conf": 0.001,
        "coordinate_space": "original_xyxy",
        "split": "test",
        "geometry": actual,
    }
    (directory / "visualization_source.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")


def comparison_config(request):
    """Use this run's actual saved evaluator configuration instead of new backend defaults."""
    profile = request["profile"]
    run = Path(profile["run"])
    candidates = [
        run / "evaluate_test/run_config.json",
        run / "evaluation/test/requested_config.json",
        run / "run_config.json",
        run / "requested_config.json",
    ]
    if "cache" in profile:
        candidates[:0] = [Path(profile["cache"]) / "run_config.json", Path(profile["cache"]) / "requested_config.json"]
    path = next((p for p in candidates if p.is_file()), None)
    if path is None:
        raise FileNotFoundError(f"Missing run-owned evaluator config for {request['key']}: {candidates}")
    cfg = load(path)
    cfg["data"], cfg["weights"] = request["data"], profile["weights"]
    cfg["train"]["device"] = request["device"]
    cfg["eval"]["split"], cfg["eval"]["conf"] = "test", 0.001
    if cfg["eval"].get("augment", False):
        raise ValueError(f"Formal no-TTA evaluation expected: {path}")
    return cfg, str(path)


def export_v5(request, manifest):
    """Reuse official anchor-v7 loader, EMA selection and objectness-aware native inference."""
    source = Path(request["profile"]["source"])
    sys.path.insert(0, str(source / "experiments/comparison/yolov5n"))
    from common import bootstrap
    from comparison_io import trusted_load

    cfg, cfg_path = comparison_config(request)
    bootstrap(cfg["cache"])
    import torch
    from data import loader
    from runtime import infer
    from tqdm import tqdm
    from utils.torch_utils import select_device

    checkpoint = trusted_load(request["profile"]["weights"])
    model = deepcopy(checkpoint["ema"] if checkpoint.get("ema") is not None else checkpoint["model"])
    model = model.float().to(select_device(request["device"], batch_size=cfg["eval"]["batch"])).eval()
    if cfg["eval"]["fuse"]:
        model.fuse()
    if model.model[-1].nc != 1 or model.model[-1].__class__.__name__ != "Detect":
        raise ValueError("Expected original v5 single-class anchor Detect")
    records = []
    for image in manifest["images"]:
        left, _, right = image["path"].replace("\\", "/").rpartition("/images/")
        records.append(
            {
                "image_id": image["id"],
                "image": image["path"],
                "label": str(Path(left + "/labels/" + right).with_suffix(".txt")),
                "shape_hw": [image["height"], image["width"]],
            }
        )
    dataloader, _ = loader(records, cfg, "test")
    with torch.inference_mode():
        _, _, predictions, targets, _, actual = infer(
            model, tqdm(dataloader, desc="YOLOv5n export", unit="batch"), cfg, records
        )
    actual["configuration"] = cfg_path
    return [
        {"image_id": row["image_id"], "predictions": p, "gt": g} for row, p, g in zip(records, predictions, targets)
    ], actual


def export_comparison(request, manifest, directory):
    """Capture the existing B19 dataset/decoder boundary for v8, v11, v13 and R18."""
    key = request["key"]
    source = Path(request["profile"]["source"])
    experiment = source / "experiments/comparison" / key
    if key == "yolo11n":
        sys.path.insert(0, str(experiment))
        from _entry import setup

        common = setup()
    else:
        sys.path.insert(0, str(experiment))
        import common
    cfg, cfg_path = comparison_config(request)
    run = Path(request["profile"]["run"])
    data_manifest = next(
        (p for p in (run / "evaluate_test/data_manifest.json", run / "data_manifest.json") if p.is_file()), None
    )
    if data_manifest is None:
        raise FileNotFoundError(f"Run-owned data identity missing: {run}")
    backend = common.bootstrap(cfg, data_identity=load(data_manifest)["data_sha256"])
    import torch
    from adapter import build_dataset, verify_model

    from ultralytics.cfg import get_cfg
    from ultralytics.data import build_dataloader
    from ultralytics.utils import TQDM as tqdm
    from ultralytics.utils.torch_utils import select_device

    device = select_device(request["device"], cfg["eval"]["batch"])
    if cfg["eval"]["half"] and device.type != "cuda":
        raise ValueError("The saved FP16 evaluator requires CUDA")
    model = (
        backend.RTDETR(request["profile"]["weights"])
        if key == "rtdetr_r18"
        else backend.YOLO(request["profile"]["weights"], task="detect")
    ).model
    verify_model(model)
    if cfg["eval"]["fuse"]:
        model.fuse()
    model.to(device).eval()
    model.half() if cfg["eval"]["half"] else model.float()
    options = {name: cfg["eval"][name] for name in ("batch", "rect", "workers", "conf", "iou", "max_det", "half")}
    native = get_cfg(overrides={**options, "imgsz": cfg["train"]["imgsz"], "plots": False, "split": "test"})
    listing = directory / "test.txt"
    listing.write_text("\n".join(image["path"] for image in manifest["images"]) + "\n", encoding="utf-8")
    paths = {"val": str(listing), "test": str(listing), "names": {0: "crack"}, "nc": 1, "path": manifest["root"]}
    ds = build_dataset(cfg, native, paths, "val", cfg["eval"]["batch"])
    dataloader = build_dataloader(ds, cfg["eval"]["batch"], cfg["eval"]["workers"], shuffle=False, rank=-1)
    records, shapes = [], set()
    if key != "rtdetr_r18":
        from adapter import ComparisonValidator

        validator = ComparisonValidator(dataloader, save_dir=directory, args=native)
        validator.data, validator.device, validator.training = paths, device, False
        validator.init_metrics(model)
    with torch.inference_mode():
        for i, batch in enumerate(tqdm(dataloader, desc=key + " export", unit="batch")):
            if key == "rtdetr_r18":
                from adapter import decode, records_from_batch

                images = batch["img"].to(device, non_blocking=True)
                batch["img"] = (images.half() if cfg["eval"]["half"] else images.float()) / 255
                predictions = decode(
                    model(batch["img"]), batch["img"].shape[-2:], cfg["eval"]["conf"], cfg["eval"]["max_det"]
                )
                records.extend(records_from_batch(predictions, batch))
            else:
                from ultralytics.utils import ops

                validator.batch_i = i
                batch = validator.preprocess(batch)
                predictions = ops.non_max_suppression(
                    model(batch["img"]),
                    cfg["eval"]["conf"],
                    cfg["eval"]["iou"],
                    multi_label=True,
                    max_det=cfg["eval"]["max_det"],
                )
                validator.update_metrics(predictions, batch)
            shapes.add(tuple(batch["img"].shape))
    if key != "rtdetr_r18":
        records = validator.records
    for record in records:
        record["image_id"] = Path(record["image_id"]).relative_to(manifest["root"]).as_posix()
    return records, {
        "configuration": cfg_path,
        "input_shapes": sorted(shapes),
        "eval": cfg["eval"],
        "model_dtype": str(next(model.parameters()).dtype),
        "TTA": False,
        "postprocess": "native queries, actual input WH, one letterbox inverse; no NMS/sigmoid/objectness"
        if key == "rtdetr_r18"
        else "native evaluator DFL/class scores/NMS, one original-coordinate inverse",
    }


def export_drsc(request):
    """Capture the compatible FP32 native validator at its original-coordinate output boundary."""
    sys.path.insert(0, request["profile"]["source"])
    from experiments.dtr_sce.validate import FP32Validator, load_for_evaluation, numerical_settings
    from ultralytics.utils import ops

    numerical_settings()
    model = load_for_evaluation(request["profile"]["weights"], "b19" if request["key"] == "yolo26n" else "dtr_sce")
    model.model.fuse(verbose=False)

    class Capture(FP32Validator):
        """Observe native detections without altering preprocessing, forward or postprocessing."""

        def init_metrics(self, model):
            super().init_metrics(model)
            self.records, self.shapes = [], set()

        def update_metrics(self, predictions, batch):
            self.shapes.add(tuple(batch["img"].shape))
            for i, prediction in enumerate(predictions):
                sample = self._prepare_batch(i, batch)
                scaled = self.scale_preds(prediction, sample)
                gt_boxes = ops.scale_boxes(
                    sample["imgsz"], sample["bboxes"].clone(), sample["ori_shape"], ratio_pad=sample["ratio_pad"]
                )
                import torch

                self.records.append(
                    {
                        "image_id": sample["im_file"],
                        "predictions": torch.cat((scaled["bboxes"], scaled["conf"][:, None], scaled["cls"][:, None]), 1)
                        .float()
                        .cpu()
                        .numpy(),
                        "gt": torch.cat((gt_boxes, sample["cls"][:, None]), 1).float().cpu().numpy(),
                    }
                )
            super().update_metrics(predictions, batch)

    output = Path(request["output"])
    validator = Capture(
        args={
            "data": request["data"],
            "device": request["device"],
            "split": "test",
            "imgsz": 640,
            "batch": 32,
            "workers": 8,
            "conf": 0.001,
            "iou": 0.7,
            "max_det": 300,
            "rect": True,
            "augment": False,
            "quantize": None,
            "end2end": True,
            "plots": False,
            "save_json": False,
            "project": str(output),
            "name": "native",
            "exist_ok": True,
            "task": "detect",
            "mode": "val",
            "seed": 42,
            "deterministic": True,
        }
    )
    validator(model=model.model)
    return validator.records, {
        "evaluator": "experiments.dtr_sce.validate.FP32Validator",
        "input_shapes": sorted(validator.shapes),
        "actual_configuration": vars(validator.args),
        "model_dtype": validator.actual_model_dtype,
        "input_dtype": validator.actual_input_dtype,
        "postprocess": "native end-to-end Detect; one original-coordinate inverse",
    }


def main():
    """Dispatch one existing environment per subprocess; no training backend installs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request")
    request = load(parser.parse_args().request)
    os.environ["YOLO_AUTOINSTALL"] = "false"
    directory = Path(request["output"])
    # A failed retry cannot make a stale metadata file certify new partial arrays.
    (directory / "visualization_source.json").unlink(missing_ok=True)
    manifest = load(request["manifest"])
    if request["key"] in {"yolo26n", "drsc"}:
        records, actual = export_drsc(request)
    elif request["key"] == "yolov5n":
        records, actual = export_v5(request, manifest)
    else:
        records, actual = export_comparison(request, manifest, directory)
    save(directory, records, request, actual)


if __name__ == "__main__":
    main()
