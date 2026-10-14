"""Small integration hooks around the locked native YOLOv5 training loop."""

import copy
import csv
import importlib.metadata
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from common import HERE, backend_train, bootstrap, flatten, protocol
from comparison_io import checksum, digest, read_json, trusted_load, write_json
from data import read_labels
from reporting import metric_values, training_figure

HOOKS = None


def environment(imports):
    packages = {}
    for name in (
        "torch",
        "torchvision",
        "numpy",
        "opencv-python",
        "PyYAML",
        "matplotlib",
        "scipy",
        "pandas",
        "seaborn",
        "tqdm",
        "ultralytics-thop",
        "albumentations",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "NOT_INSTALLED"
    return {
        "sys_executable": sys.executable,
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "imports": imports,
        "backend_commit": read_json(HERE / "backend.lock.json")["commit"],
        "server_validation": "UNVERIFIED",
    }


def gpu_processes():
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv,noheader,nounits"],
            text=True,
            timeout=10,
        )
        lines = [line.strip() for line in output.splitlines() if line.strip()]
        foreign = [line for line in lines if line.split(",")[0].strip() != str(os.getpid())]
        return {
            "state": "CONTENDED" if foreign else "NO_OTHER_COMPUTE_PROCESS_OBSERVED",
            "foreign_processes": foreign,
            "scope": "read-only snapshot; no waiting, stopping or scheduling; driver monitoring may miss activity",
        }
    except (OSError, subprocess.SubprocessError) as error:
        return {"state": "UNVERIFIED", "reason": str(error)}


def atomic_torch_save(value, path):
    path = Path(path)
    temporary = path.with_name(path.name + f".{os.getpid()}.partial")
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def original_targets(record):
    from shared.b19.ops import xywh2xyxy

    labels = read_labels(Path(record["label"]))
    boxes = xywh2xyxy(labels[:, 1:])
    boxes[:, [0, 2]] *= record["shape_hw"][1]
    boxes[:, [1, 3]] *= record["shape_hw"][0]
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, record["shape_hw"][1])
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, record["shape_hw"][0])
    return np.concatenate([boxes, labels[:, :1]], axis=1).astype(np.float32)


@torch.inference_mode()
def infer(model, dataloader, cfg, records, compute_loss=None):
    from utils.general import non_max_suppression

    model.eval()
    model.half() if cfg["eval"]["half"] else model.float()
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    if cfg["eval"]["half"] and device.type != "cuda":
        raise ValueError("half evaluation requires an explicit CUDA device")
    by_path = {}
    losses = torch.zeros(3, device=device)
    batches = 0
    input_shapes = []
    for images, targets, paths, shapes in dataloader:
        images = images.to(device, non_blocking=True).to(dtype) / 255
        decoded, raw = model(images, augment=False)
        if decoded.shape[-1] != 6:
            raise ValueError(f"Expected original nc=1 v5 anchor output (xywh,obj,cls), got {decoded.shape}")
        if compute_loss is not None:
            losses += compute_loss([part.float() for part in raw], targets.to(device))[1]
        nms = non_max_suppression(
            decoded.float(),
            conf_thres=cfg["eval"]["conf"],
            iou_thres=cfg["eval"]["iou"],
            max_det=cfg["eval"]["max_det"],
            multi_label=False,
        )
        input_shapes.append(list(images.shape))
        for prediction, path, shape in zip(nms, paths, shapes):
            original, transform = shape
            if transform is None:
                raise ValueError("Evaluation loader unexpectedly applied random augmentation")
            (gain_y, gain_x), (pad_x, pad_y) = transform
            prediction[:, [0, 2]] = (prediction[:, [0, 2]] - pad_x) / gain_x
            prediction[:, [1, 3]] = (prediction[:, [1, 3]] - pad_y) / gain_y
            prediction[:, [0, 2]] = prediction[:, [0, 2]].clamp(0, original[1])
            prediction[:, [1, 3]] = prediction[:, [1, 3]].clamp(0, original[0])
            by_path[str(Path(path).resolve())] = prediction.cpu().numpy().astype(np.float32)
        batches += 1
    if len(by_path) != len(records):
        raise ValueError("Prediction export did not cover the complete evaluated split")
    predictions = [by_path[str(Path(r["image"]).resolve())] for r in records]
    targets = [original_targets(r) for r in records]
    metrics, curves = metric_values(
        predictions, targets, split=cfg["eval"]["split"], collection_conf=cfg["eval"]["conf"]
    )
    actual = {
        "model_dtype": str(dtype),
        "input_dtype": str(dtype),
        "input_shapes": input_shapes,
        "input_shape_rule": "B19 ceil long-side resize, aspect sorted rect batches, pad0.5, stride32, centered LetterBox",
        "letterbox": {
            "auto": False,
            "scale_fill": False,
            "scaleup": False,
            "center": True,
            "value": 114,
            "interpolation": "INTER_LINEAR",
            "coordinate_restore": "actual per-axis resize gains and integer left/top padding",
        },
        "NMS": {
            "score": "objectness * class_probability (no second sigmoid)",
            "iou": cfg["eval"]["iou"],
            "conf": cfg["eval"]["conf"],
            "max_det": cfg["eval"]["max_det"],
            "complete_batch": True,
        },
        "fused": not any(isinstance(m, torch.nn.BatchNorm2d) for m in model.modules()),
        "TTA": False,
    }
    return metrics, curves, predictions, targets, (losses / max(batches, 1)).cpu().tolist(), actual


def validation_run(data, model, dataloader, compute_loss=None, **kwargs):
    evaluation = copy.deepcopy(model).eval()
    if HOOKS.cfg["eval"]["fuse"]:
        evaluation.float().fuse()
    records = HOOKS.manifest["splits"]["val"]["records"]
    values, curves, _, _, losses, actual = infer(evaluation, dataloader, HOOKS.cfg, records, compute_loss)
    if values["mAP50-95"] is None:
        raise ValueError("Training validation split has no GT; best selection is undefined")
    HOOKS.latest_values = values
    HOOKS.latest_curves = curves
    HOOKS.latest_geometry = actual
    peak = values["legacy_curve_peak"]
    return (
        (peak["P"], peak["R"], values["AP50"], values["mAP50-95"], *losses),
        np.array([values["mAP50-95"]]),
        (0.0, 0.0, 0.0),
    )


class TrainingHooks:
    """Own experiment auditing and selection while native code owns optimization."""

    def __init__(self, cfg, audit, data_manifest, directory, restored=None):
        self.cfg, self.audit, self.manifest, self.directory = cfg, audit, data_manifest, Path(directory)
        self.protocol = protocol(cfg, data_manifest)
        backend = read_json(HERE / "backend.lock.json")
        self.backend_identity = {"remote": backend["repository"], "branch": backend["tag"], "commit": backend["commit"]}
        write_json(self.directory / "protocol.json", self.protocol)
        write_json(self.directory / "b19_source.lock.json", read_json(HERE.parent / "shared/b19/source.lock.json"))
        write_json(self.directory / "shared.lock.json", read_json(HERE / "shared.lock.json"))
        self.restored = restored
        previous = restored.get("comparison", {}) if restored else {}
        self.best_map = previous.get("best_map", -1.0)
        self.best_epoch = previous.get("best_epoch")
        self.completed = previous.get("completed_epochs", 0)
        self.optimizer_updates = previous.get("optimizer_updates", 0)
        self.purpose = previous.get("purpose", "SMOKE" if cfg["name"].startswith("SMOKE") else "EXPERIMENT")

    def transfer(self, model, transferred, missing, weights):
        state = model.state_dict()
        write_json(
            self.directory / (f"resume_transfer_{os.getpid()}.json" if self.restored else "pretrained_transfer.json"),
            {
                "weights": str(weights),
                "weights_sha256": digest(weights),
                "nc": model.model[-1].nc,
                "head": type(model.model[-1]).__name__,
                "fused_before_training": not any(isinstance(m, torch.nn.BatchNorm2d) for m in model.modules()),
                "loaded_keys": sorted(transferred),
                "missing_keys": sorted(missing.missing_keys),
                "unexpected_keys": sorted(missing.unexpected_keys),
                "untransferred_shapes": {key: list(state[key].shape) for key in missing.missing_keys},
                "note": "80-class detection output convolution weights/biases differ for nc=1 and are not inherited",
            },
        )

    def amp(self, requested, actual, device):
        self.amp_result = {
            "requested": requested,
            "actual": bool(actual),
            "device": str(device),
            "mechanism": "native check_amp, verified local bus image and verified same-family fallback weights",
        }
        write_json(self.directory / "amp_check.json", self.amp_result)

    def established(self, model, optimizer, hyp, accumulate, nbs, amp, dataset, ema, scheduler, scaler, stopper, nb):
        self.ema = ema
        optimizer.register_step_post_hook(self.after_step)
        if self.restored:
            saved = self.restored["comparison"]["training_state"]
            scheduler.load_state_dict(saved["scheduler"])
            scaler.load_state_dict(saved["scaler"])
            for field in ("best_epoch", "best_fitness", "possible_stop"):
                setattr(stopper, field, saved["stopper"][field])
        requested = self.cfg["train"]
        effective = {
            **requested,
            **hyp,
            "optimizer": type(optimizer).__name__,
            "amp": bool(amp),
            "nominal_batch": nbs,
            "gradient_accumulation_initial": accumulate,
            "effective_batch_initial": requested["batch"] * accumulate,
            "warmup_min_iterations": 100,
            "warmup_iterations": max(round(hyp["warmup_epochs"] * nb), 100),
            "parameter_groups": self.groups(optimizer),
            "anchor_grid_pixels": (model.model[-1].anchors * model.model[-1].stride.view(-1, 1, 1))
            .detach()
            .cpu()
            .tolist(),
            "autoanchor": requested["autoanchor"],
            "loss_scaling": self.audit["official_defaults"]["scaling"],
            "unfused_parameters": sum(p.numel() for p in model.parameters()),
            "extra_native_augmentations": "none for b19/custom/none; native variant recorded separately",
        }
        differences = {
            key: {"requested": requested[key], "effective": value}
            for key, value in effective.items()
            if key in requested and value != requested[key]
        }
        write_json(
            self.directory / "effective_config.json",
            {"config": effective, "field_differences": differences, "AMP": self.amp_result},
        )
        write_json(self.directory / "optimizer_setup.json", effective["parameter_groups"])

    def after_step(self, optimizer, args, kwargs):
        self.optimizer_updates += 1

    @staticmethod
    def groups(optimizer):
        return [
            {
                k: group[k]
                for k in ("lr", "initial_lr", "weight_decay", "momentum", "betas", "nesterov", "eps")
                if k in group
            }
            | {"index": i, "parameters": sum(p.numel() for p in group["params"])}
            for i, group in enumerate(optimizer.param_groups)
        ]

    def epoch(self, epoch, dataset, dataloader):
        closed = dataset.set_epoch(epoch) if hasattr(dataset, "set_epoch") else False
        if closed and (
            epoch == max(0, self.cfg["train"]["epochs"] - self.cfg["augment"]["resolved"]["close_mosaic"])
            or self.restored
            and epoch == self.restored["epoch"] + 1
        ):
            write_json(
                self.directory / "close_mosaic_event.json",
                {
                    "epoch_zero_based": epoch,
                    "disabled": ["mosaic", "copy_paste", "mixup", "cutmix"],
                    "other_transforms_retained": True,
                    "workers": "fresh epoch iterators observe parent transforms",
                },
            )

    def fit(self, log_values, epoch, stop, optimizer, accumulate):
        header = [
            "epoch",
            "train_box",
            "train_obj",
            "train_cls",
            "legacy_P",
            "legacy_R",
            "AP50",
            "mAP50_95",
            "val_box",
            "val_obj",
            "val_cls",
            "lr0",
            "lr1",
            "lr2",
        ]
        path = self.directory / "results.csv"
        new = not path.exists()
        with path.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            if new:
                writer.writerow(header)
            writer.writerow([epoch, *[float(v) for v in log_values]])
        self.completed = epoch + 1
        self.stop_reason = (
            "native weighted-fitness early stopping"
            if stop
            else "requested epochs completed"
            if self.completed == self.cfg["train"]["epochs"]
            else "RUNNING"
        )
        with (self.directory / "optimizer_epochs.jsonl").open("a", encoding="utf-8") as stream:
            import json

            stream.write(
                json.dumps(
                    {
                        "epoch": epoch,
                        "accumulate_last_batch": int(accumulate),
                        "effective_batch_last_batch": int(accumulate) * self.cfg["train"]["batch"],
                        "groups": self.groups(optimizer),
                        "ema_updates": self.ema.updates,
                    }
                )
                + "\n"
            )
        write_json(self.directory / "validation_effective.json", self.latest_geometry)

    def checkpoint(self, ckpt, epoch, model, ema, optimizer, scheduler, scaler, stopper):
        is_best = self.latest_values["mAP50-95"] > self.best_map
        if is_best:
            self.best_map, self.best_epoch = self.latest_values["mAP50-95"], epoch
        ckpt["comparison"] = {
            "model": "yolov5n",
            "purpose": self.purpose,
            **self.protocol,
            "initial_weights": self.cfg["weights"],
            "config": self.cfg,
            "best_map": self.best_map,
            "best_epoch": self.best_epoch,
            "completed_epochs": self.completed,
            "optimizer_updates": self.optimizer_updates,
            "ema_updates": ema.updates,
            "training_state": {
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "stopper": {k: getattr(stopper, k) for k in ("best_epoch", "best_fitness", "possible_stop")},
            },
        }
        if is_best:
            path = self.directory / "weights/best.pt"
            atomic_torch_save(ckpt, path)
            provenance = self.checksum(path)
            write_json(
                self.directory / "best_selection.json",
                {
                    "split": "val",
                    "metric": "unified_mAP50_95",
                    "best_epoch_zero_based": epoch,
                    "value_fraction": self.best_map,
                    "weights_sha256": provenance["sha256"],
                    "native_best": "weights/native_best.pt (weighted native fitness)",
                    "test_used": False,
                },
            )

    def checksum(self, path):
        return checksum(
            path,
            {
                "kind": "comparison_checkpoint",
                "model": "yolov5n",
                "purpose": self.purpose,
                "protocol_sha256": self.protocol["protocol_sha256"],
                "run_config_sha256": self.audit["run_config_sha256"],
            },
        )

    def save(self, value, path):
        atomic_torch_save(value, path)
        self.checksum(path)

    def finish(self):
        training_figure(self.directory)
        write_json(
            self.directory / "training_summary.json",
            {
                "state": "COMPLETED",
                "purpose": self.purpose,
                "completed_epochs": self.completed,
                "best_epoch_zero_based": self.best_epoch,
                "best_metric": "unified_val_mAP50_95",
                "best_value_fraction": self.best_map,
                "optimizer_updates": self.optimizer_updates,
                "ema_updates": self.ema.updates,
                "update_count_source": "optimizer step post-hook; AMP-skipped steps do not count",
                "stop_reason": self.stop_reason,
                "native_early_stop_preserved": True,
            },
        )
        write_json(self.directory / "benchmark_status.json", {"state": "NOT_RUN"})


def train_native(cfg, audit, data_manifest, directory, restored=None):
    global HOOKS

    imports = bootstrap(cfg["cache"])
    os.environ["YOLOV5_LABEL_CACHE"] = str(Path(directory) / "cache/native-labels")
    write_json(Path(directory) / "environment.json", environment(imports))
    HOOKS = TrainingHooks(cfg, audit, data_manifest, directory, restored)
    native = backend_train()
    # Parse native defaults without allowing this outer program's argv into the old parser.
    previous = sys.argv
    try:
        sys.argv = [str(HERE / "vendor/yolov5/train.py")]
        opt = native.parse_opt()
    finally:
        sys.argv = previous
    t = cfg["train"]
    for key in (
        "epochs",
        "imgsz",
        "seed",
        "workers",
        "device",
        "optimizer",
        "cos_lr",
        "patience",
        "label_smoothing",
        "save_period",
    ):
        setattr(opt, key, t[key])
    opt.batch_size, opt.amp, opt.beta2 = t["batch"], t["amp"], t["beta2"]
    opt.save_dir, opt.data, opt.weights = str(directory), cfg["data"], cfg["weights"]
    opt.cfg = ""  # build from trusted official/experiment checkpoint YAML; original n structure retained
    opt.resume = bool(restored)
    if restored:
        opt.weights = str(Path(t["resume"]).resolve())
    opt.amp_probe = cfg["weights"]
    opt.noautoanchor, opt.noplots, opt.cache = not t["autoanchor"], True, False
    opt.comparison_data = {
        "nc": 1,
        "names": {0: "crack"},
        "train": data_manifest["splits"]["train"]["records"][0]["image"],
        "val": data_manifest["splits"]["val"]["records"][0]["image"],
    }
    hyp = {key: t[key] for key in audit["official_defaults"]["hyp"] if key not in cfg["augment"]["resolved"]}
    hyp.update({k: v for k, v in cfg["augment"]["resolved"].items() if k in audit["official_defaults"]["hyp"]})
    from utils.callbacks import Callbacks
    from utils.torch_utils import select_device

    device = select_device(t["device"], batch_size=t["batch"])
    native.train(hyp, opt, device, Callbacks())
    HOOKS.finish()


def validate_resume(cfg, data_manifest, directory):
    bootstrap(cfg["cache"])
    resume = cfg["train"]["resume"]
    path = Path(directory) / "weights/last.pt" if resume is True else Path(resume)
    cfg["train"]["resume"] = str(path.resolve())
    checkpoint = trusted_load(path)
    info = checkpoint.get("comparison", {})
    if info.get("model") != "yolov5n":
        raise ValueError("Resume requires this experiment's complete checkpoint")
    current = protocol(cfg, data_manifest)
    if info.get("protocol_sha256") != current["protocol_sha256"]:
        raise ValueError("Resume data/augmentation/evaluation protocol mismatch")
    previous = info["config"]
    changed = {
        k: {"before": flatten(previous).get(k), "requested": v}
        for k, v in flatten(cfg).items()
        if flatten(previous).get(k) != v
    }
    allowed = {
        "train.resume",
        "train.epochs",
        "train.workers",
        "train.device",
        "cache",
        "project",
        "reference.baseline_args",
        "data",
        "weights",
    }
    disallowed = set(changed) - allowed
    if disallowed:
        raise ValueError(f"Resume cannot change training invariants: {sorted(disallowed)}")
    if cfg["train"]["epochs"] <= checkpoint["epoch"] + 1 or checkpoint.get("optimizer") is None:
        raise ValueError("No optimizer state or no remaining requested epochs; implicit fine-tuning is disabled")
    if path.resolve().parent.parent != Path(directory).resolve():
        raise ValueError("Resume checkpoint must belong to the explicit project/name run directory")
    write_json(
        Path(directory) / f"resume_request_{os.getpid()}.json",
        {"checkpoint": str(path), "sha256": digest(path), "field_differences": changed},
    )
    return checkpoint
