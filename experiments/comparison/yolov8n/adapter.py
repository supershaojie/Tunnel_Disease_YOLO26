"""Production v8 Dataset/Trainer bridge; native detection model/loss/optimizer remain authoritative."""

import json
import os
import sys
from copy import copy, deepcopy
from types import SimpleNamespace

import numpy as np
import torch

from common import AUG_KEYS, environment, file_hash, write_json
from shared import evaluation, provider
from shared.b19.instance import Instances as B19Instances
from ultralytics.data.augment import Albumentations, Compose, Format, LetterBox
from ultralytics.data.dataset import YOLODataset
from ultralytics.models.yolo.detect import DetectionTrainer, DetectionValidator
from ultralytics.utils.instance import Instances
from ultralytics.utils.torch_utils import de_parallel


class NativeFormat:
    """Convert the shared Instances interface explicitly before native v8 formatting."""

    def __init__(self, bgr):
        self.format = Format(bbox_format="xywh", normalize=True, batch_idx=True, bgr=bgr)

    def __call__(self, labels):
        instance = labels["instances"]
        instance.convert_bbox("xyxy")
        labels["instances"] = Instances(
            instance.bboxes.copy(),
            instance.segments.copy(),
            instance.keypoints,
            bbox_format="xyxy",
            normalized=instance.normalized,
        )
        return self.format(labels)


class ComparisonDataset(YOLODataset):
    """Own augmentation once and leave collate/normalization native."""

    def __init__(self, *args, comparison, **kwargs):
        self.comparison = comparison
        self.augmentation_values = deepcopy(comparison["augment"]["effective"])
        super().__init__(*args, **kwargs)

    def update_labels_info(self, label):
        if self.augment and self.comparison["augment"]["profile"] in ["b19", "custom"]:
            label["instances"] = B19Instances(
                label.pop("bboxes"),
                np.zeros((0, 1000, 2), dtype=np.float32),
                None,
                bbox_format=label.pop("bbox_format"),
                normalized=label.pop("normalized"),
            )
            label.pop("segments", None)
            label.pop("keypoints", None)
            return label
        return super().update_labels_info(label)

    def build_transforms(self, hyp=None):
        profile = self.comparison["augment"]["profile"]
        if self.augment and profile in ["b19", "custom"]:
            transforms = provider.build(self, self.imgsz, self.augmentation_values)
            transforms.append(NativeFormat(self.augmentation_values["bgr"]))
            return transforms
        if self.augment and profile == "native":
            transforms = super().build_transforms(hyp)
            self.augmentation_values.update({key: getattr(hyp, key) for key in AUG_KEYS if hasattr(hyp, key)})
            return transforms
        # none and evaluation have no random augmentation, including native automatic extras.
        return Compose(
            [
                LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False),
                Format(bbox_format="xywh", normalize=True, batch_idx=True, bgr=0.0),
            ]
        )

    def close_mosaic(self, hyp):
        self.augmentation_values = provider.closed_values(self.augmentation_values)
        if self.comparison["augment"]["profile"] == "native":
            return super().close_mosaic(hyp)
        self.transforms = self.build_transforms(hyp)

    @classmethod
    def native_policy(cls, cfg):
        """Inspect the same native pipeline owner without loading images or creating caches."""
        probe = cls.__new__(cls)
        probe.comparison = cfg
        probe.augmentation_values = deepcopy(cfg["augment"]["effective"])
        probe.augment, probe.rect, probe.imgsz = True, cfg["train"]["rect"], cfg["train"]["imgsz"]
        probe.use_segments = probe.use_keypoints = probe.use_obb = False
        probe.data = {}
        transforms = probe.build_transforms(SimpleNamespace(**cfg["train"]))
        extra = next(t for t in transforms.transforms if isinstance(t, Albumentations))
        return probe.augmentation_values, {
            "albumentations": {
                "enabled": extra.transform is not None,
                "version": getattr(sys.modules.get("albumentations"), "__version__", None),
                "pipeline": extra.transform.to_dict() if extra.transform is not None else None,
            }
        }


def build_dataset(cfg, args, data, mode, batch, stride=32, pad=None):
    """Construct the production adapter for both train and independent evaluation."""
    return ComparisonDataset(
        img_path=data[mode],
        imgsz=args.imgsz,
        batch_size=batch,
        augment=mode == "train",
        hyp=args,
        rect=args.rect if mode == "train" else cfg["eval"]["rect"],
        cache=args.cache if mode == "train" else False,
        single_cls=False,
        stride=stride,
        pad=(0.0 if mode == "train" else 0.5) if pad is None else pad,
        prefix=f"{mode}: ",
        task="detect",
        data=data,
        fraction=1.0,
        comparison=cfg,
    )


class ComparisonValidator(DetectionValidator):
    """Reuse the native val forward/decoding and compute B19 metrics on those predictions."""

    def init_metrics(self, model):
        super().init_metrics(model)
        self.records = []
        self.observed_shapes = set()

    def update_metrics(self, preds, batch):
        self.observed_shapes.add(tuple(batch["img"].shape))
        for si, pred in enumerate(preds):
            pbatch = self._prepare_batch(si, batch)
            gt = torch.cat([pbatch["bbox"].reshape(-1, 4), pbatch["cls"].reshape(-1, 1)], dim=1)
            predn = self._prepare_pred(pred, pbatch)
            self.records.append(
                {
                    "image_id": batch["im_file"][si],
                    "predictions": predn.detach().float().cpu().numpy().reshape(-1, 6),
                    "gt": gt.detach().float().cpu().numpy().reshape(-1, 5),
                }
            )
        super().update_metrics(preds, batch)

    def get_stats(self):
        result = super().get_stats()
        self.unified, self.curves = evaluation.compute(self.records, "val", collection_conf=self.args.conf)
        result["fitness"] = self.unified["mAP50-95"] or 0.0
        result["metrics/unified_mAP50-95"] = result["fitness"]
        return result


class ComparisonTrainer(DetectionTrainer):
    """Production trainer with frozen shared augmentation and unified val-best selection."""

    context = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_callback("on_train_epoch_end", record_optimizer_epoch)

    def get_dataset(self):
        self.data = self.context["dataset"]
        return self.data["train"], self.data["val"]

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = super().get_model(cfg, weights, verbose)
        verify_model(model)
        source = weights.float().state_dict() if weights is not None else {}
        target = model.state_dict()
        inherited = [k for k, v in source.items() if k in target and v.shape == target[k].shape]
        reinitialized = [k for k in target if k not in inherited]
        mismatch = {
            k: {"source_shape": list(source[k].shape) if k in source else None, "target_shape": list(target[k].shape)}
            for k in reinitialized
        }
        if not inherited and not self.context["config"]["train"]["resume"]:
            raise RuntimeError("Official pretrained state was not inherited on the actual trainer construction path")
        equal = [k for k in inherited if torch.equal(source[k].cpu(), target[k].cpu())]
        if len(equal) != len(inherited):
            raise RuntimeError("Pretrained transfer mismatch")
        write_json(
            self.save_dir
            / (
                f"resume_{os.getpid()}_initialization.json"
                if self.context["config"]["train"]["resume"]
                else "initialization.json"
            ),
            {
                "pretrained": self.context["pretrained_provenance"],
                "inherited_keys": inherited,
                "reinitialized_keys": reinitialized,
                "reinitialized_shapes": mismatch,
                "verified_equal_keys": len(equal),
                "nc": model.model[-1].nc,
                "reg_max": model.model[-1].reg_max,
                "model_yaml": model.yaml,
                "modules": [m.__class__.__name__ for m in model.model],
                "native_classifier_branch": [m.__class__.__name__ for m in model.model[-1].cv3[0].modules()],
            },
        )
        return model

    def build_dataset(self, img_path, mode="train", batch=None):
        return build_dataset(
            self.context["config"],
            self.args,
            self.data,
            mode,
            batch,
            max(int(de_parallel(self.model).stride.max()), 32),
        )

    def get_validator(self):
        self.loss_names = ("box_loss", "cls_loss", "dfl_loss")
        args = copy(self.args)
        args.conf = self.context["config"]["eval"]["conf"]
        args.iou = self.context["config"]["eval"]["iou"]
        args.max_det = self.context["config"]["eval"]["max_det"]
        args.augment = False
        validator = ComparisonValidator(self.test_loader, save_dir=self.save_dir, args=args, _callbacks=self.callbacks)
        return validator

    def build_optimizer(self, model, name="auto", lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5):
        optimizer = super().build_optimizer(model, name, lr, momentum, decay, iterations)
        betas = self.context["config"]["train"].get("betas")
        if betas is not None:
            for group in optimizer.param_groups:
                group["betas"] = tuple(betas)
        self.optimizer_iterations = iterations
        return optimizer

    def validate(self):
        metrics = self.validator(self)
        metrics.pop("fitness", None)
        fitness = self.validator.unified["mAP50-95"] or 0.0
        if self.best_fitness is None or self.best_fitness < fitness:
            self.best_fitness = fitness
        return metrics, fitness

    def _setup_train(self, world_size):
        super()._setup_train(world_size)
        requested = self.context["config"]["train"]
        effective = vars(self.args).copy()
        groups = [
            {k: v for k, v in g.items() if k != "params"}
            | {"parameter_tensors": len(g["params"]), "parameter_elements": sum(p.numel() for p in g["params"])}
            for g in self.optimizer.param_groups
        ]
        effective.update(
            {
                "optimizer_class": type(self.optimizer).__name__,
                "optimizer_groups": groups,
                "accumulate": self.accumulate,
                "effective_batch": self.batch_size * self.accumulate,
                "optimizer_selection_iterations": self.optimizer_iterations,
                "amp_effective": self.amp,
                "loss_gains": {k: getattr(self.model.args, k) for k in ["box", "cls", "dfl"]},
                "weight_decay_scaled": groups[1]["weight_decay"],
            }
        )
        diffs = {
            k: {"requested": requested.get(k), "effective": v} for k, v in effective.items() if requested.get(k) != v
        }
        if requested["optimizer"] == "auto":
            diffs["lr0_optimizer"] = {
                "requested": requested["lr0"],
                "effective": groups[0]["lr"],
                "reason": "locked BaseTrainer.build_optimizer auto rule",
            }
            diffs["momentum_optimizer"] = {
                "requested": requested["momentum"],
                "effective": groups[0].get("momentum", groups[0].get("betas")),
            }
        prefix = f"resume_{os.getpid()}_" if self.context["config"]["train"]["resume"] else ""
        write_json(self.save_dir / (prefix + "effective_config.json"), effective)
        write_json(self.save_dir / (prefix + "requested_effective_diff.json"), diffs)
        write_json(self.save_dir / (prefix + "environment.json"), environment())

    def save_model(self):
        super().save_model()
        manifest_path = self.save_dir / "checkpoint_manifest.json"
        previous = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
        weights = previous.get("weights", {})
        augmentations = previous.get("training_augmentation", {})
        # Native save writes the same serialized bytes to each checkpoint touched by this event.
        saved_sha = file_hash(self.last)
        weights[self.last.name] = saved_sha
        augmentations[self.last.name] = deepcopy(self.context["protocol"]["config"]["augment"])
        if self.best_fitness == self.fitness:
            weights[self.best.name] = saved_sha
            augmentations[self.best.name] = deepcopy(augmentations[self.last.name])
        if self.save_period > 0 and self.epoch % self.save_period == 0:
            weights[f"epoch{self.epoch}.pt"] = saved_sha
            augmentations[f"epoch{self.epoch}.pt"] = deepcopy(augmentations[self.last.name])
        write_json(
            manifest_path,
            {
                "model": "yolov8n",
                "backend_sha": self.context["backend_sha"],
                "data_sha256": self.context["data_report"]["data_sha256"],
                "protocol_sha256": self.context["protocol"]["protocol_sha256"],
                "weights": weights,
                "original_pretrained_sha256": self.context["pretrained_provenance"]["sha256"],
                "execution_kind": self.context["config"]["_execution_kind"],
                "training_augmentation": augmentations,
            },
        )
        if self.best_fitness == self.fitness:
            selection = self.validator.unified["threshold_selection"]
            write_json(
                self.save_dir / "best_selection.json",
                {
                    "best_epoch": self.epoch + 1,
                    "metric": "unified val mAP50-95",
                    "value": self.fitness,
                    "weight_sha256": weights[self.best.name],
                    "val_confidence_selection": selection,
                    "protocol_sha256": self.context["protocol"]["protocol_sha256"],
                },
            )

    def final_eval(self):
        # Keep last.pt resumable; native final_eval strips optimizer and performs an extra val.
        # Best selection and val metrics already came from the same epoch's native forward.
        selection = json.loads((self.save_dir / "best_selection.json").read_text(encoding="utf-8"))
        write_json(
            self.save_dir / "training_summary.json",
            {
                "completed_epochs": self.epoch + 1,
                "best_epoch": selection["best_epoch"],
                "selection_metric": selection["metric"],
                "best_weight_sha256": file_hash(self.best),
                "optimizer_updates_ema": self.ema.updates,
                "stop_reason": "early_stopping"
                if self.stopper.possible_stop and self.epoch + 1 < self.epochs
                else "time_budget"
                if self.epoch + 1 < self.epochs
                else "epochs_completed",
                "server_training_status": "SMOKE" if self.context.get("smoke") else "COMPLETED",
            },
        )

    def _close_dataloader_mosaic(self):
        super()._close_dataloader_mosaic()
        write_json(
            self.save_dir / "augmentation_close.json",
            {
                "epoch_zero_based": getattr(self, "epoch", self.start_epoch),
                "closed_operations": list(provider.CLOSED_OPERATIONS),
                "values": self.train_loader.dataset.augmentation_values,
            },
        )


def verify_model(model, nc=1):
    """Reject other families/tasks and fused formal training initialization."""
    modules = [m.__class__.__name__ for m in model.model]
    head = model.model[-1]
    if "C2f" not in modules or "SPPF" not in modules or modules[-1] != "Detect":
        raise ValueError(f"Not an official YOLOv8 detection structure: {modules}")
    if head.nc != nc or head.reg_max != 16 or getattr(head, "end2end", False):
        raise ValueError("Expected native v8 Detect nc=1 reg_max=16 without end2end")
    nano = model.yaml.get("scale") == "n" or (
        model.yaml.get("depth_multiple") == 0.33 and model.yaml.get("width_multiple") == 0.25
    )
    if not nano:
        raise ValueError("Expected official YOLOv8n nano scale")
    return modules


def record_optimizer_epoch(trainer):
    """Capture actual warmup/scheduler parameter groups at each completed epoch."""
    groups = [{k: v for k, v in g.items() if k != "params"} for g in trainer.optimizer.param_groups]
    with (trainer.save_dir / "optimizer_epochs.jsonl").open("a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "epoch": trainer.epoch + 1,
                    "optimizer": type(trainer.optimizer).__name__,
                    "groups": groups,
                    "accumulate": trainer.accumulate,
                    "effective_batch": trainer.batch_size * trainer.accumulate,
                    "EMA_updates": trainer.ema.updates,
                    "loss_items": trainer.tloss.detach().cpu().tolist(),
                },
                default=float,
            )
            + "\n"
        )
