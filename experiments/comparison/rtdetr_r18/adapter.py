"""Native C2 RT-DETR training with the existing B19 data and metric owners."""

import json
from copy import copy, deepcopy

import torch
from common import HERE, MODEL, VENDOR, environment, file_hash, read_yaml, write_json
from shared import evaluation, provider
from shared.dataset import ComparisonDataset, build_dataset  # noqa: F401 - Shared production dataset API.

from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import ops
from ultralytics.utils.torch_utils import strip_optimizer


def decode(raw, input_shape, conf, max_det):
    """Decode native query probabilities using the actual input tensor height and width."""
    prediction = raw[0] if isinstance(raw, (tuple, list)) else raw
    height, width = input_shape[-2:]
    scale = prediction.new_tensor([width, height, width, height])
    outputs = []
    for image in prediction:
        boxes = ops.xywh2xyxy(image[:, :4] * scale)
        scores, classes = image[:, 4:].max(-1)
        keep = torch.where(scores >= conf)[0]
        keep = keep[scores[keep].argsort(descending=True)[:max_det]]
        outputs.append(torch.cat((boxes[keep], scores[keep, None], classes[keep, None]), dim=1))
    return outputs


def restore_boxes(boxes, ratio_pad, shape):
    """Undo the actual per-axis resize gains, including integer resize rounding."""
    gain_y, gain_x = ratio_pad[0]
    pad_x, pad_y = ratio_pad[1]
    boxes -= boxes.new_tensor([pad_x, pad_y, pad_x, pad_y])
    boxes /= boxes.new_tensor([gain_x, gain_y, gain_x, gain_y])
    return ops.clip_boxes(boxes, shape)


def records_from_batch(preds, batch):
    """Undo the shared resize/padding for both predictions and normalized GT boxes."""
    height, width = batch["img"].shape[-2:]
    records = []
    for index, prediction in enumerate(preds):
        pred = prediction.detach().float().clone()
        shape = batch["ori_shape"][index]
        ratio_pad = batch["ratio_pad"][index]

        pred[:, :4] = restore_boxes(pred[:, :4], ratio_pad, shape)
        mask = batch["batch_idx"] == index
        boxes = ops.xywh2xyxy(batch["bboxes"][mask].float())
        boxes *= boxes.new_tensor([width, height, width, height])
        boxes = restore_boxes(boxes, ratio_pad, shape)
        gt = torch.cat((boxes, batch["cls"][mask].reshape(-1, 1).float()), dim=1)
        records.append(
            {
                "image_id": batch["im_file"][index],
                "predictions": pred.cpu().numpy().reshape(-1, 6),
                "gt": gt.detach().cpu().numpy().reshape(-1, 5),
            }
        )
    return records


def verify_model(model, nc=1):
    """Require the pinned C2 R18 graph rather than another implementation with the same name."""
    from ultralytics.nn.tasks import RTDETRDetectionModel

    reference = read_yaml(VENDOR / "ultralytics/cfg/models/rt-detr/rtdetr-resnet18-lite.yaml")
    expected = [(row[0], row[2].split(".")[-1]) for row in reference["backbone"] + reference["head"]]
    actual = [(layer.f, type(layer).__name__) for layer in model.model]
    modules = {type(layer).__name__ for layer in model.modules()}
    if not isinstance(model, RTDETRDetectionModel) or actual != expected or model.model[-1].nc != nc:
        raise ValueError(f"Expected pinned LCBR C2 RT-DETR-R18 nc={nc}, got {actual}")
    if any(name.startswith(("LIF", "CBR", "DTR", "SCE")) for name in modules):
        raise ValueError("Innovation modules are not part of the C2 baseline")
    return [name for _, name in actual]


class ComparisonValidator(DetectionValidator):
    """Keep the native validation/loss lifecycle while selecting best with shared val AP."""

    def init_metrics(self, model):
        super().init_metrics(model)
        self.records = []
        self.observed_shapes = set()

    def preprocess(self, batch):
        batch = super().preprocess(batch)
        self.input_shape = tuple(batch["img"].shape)
        self.observed_shapes.add(self.input_shape)
        return batch

    def postprocess(self, raw):
        predictions = decode(raw, self.input_shape, self.args.conf, self.args.max_det)
        return [{"bboxes": p[:, :4], "conf": p[:, 4], "cls": p[:, 5]} for p in predictions]

    def update_metrics(self, preds, batch):
        predictions = [torch.cat((p["bboxes"], p["conf"][:, None], p["cls"][:, None]), dim=1) for p in preds]
        self.records.extend(records_from_batch(predictions, batch))
        super().update_metrics(preds, batch)

    def get_stats(self):
        result = super().get_stats()
        self.unified, self.curves = evaluation.compute(self.records, "val", collection_conf=self.args.conf)
        result["fitness"] = self.unified["mAP50-95"] or 0.0
        result["metrics/unified_mAP50-95"] = result["fitness"]
        return result


class ComparisonTrainer(RTDETRTrainer):
    """Replace only RT-DETR's stretched dataset and metric selection, preserving native training."""

    context = None

    def get_dataset(self):
        return self.context["dataset"]

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
        modules = verify_model(model)
        source = weights.float().state_dict() if weights is not None else {}
        target = model.state_dict()
        inherited = [name for name in target if name in source and source[name].shape == target[name].shape]
        changed = {name: list(value.shape) for name, value in target.items() if name not in inherited}
        record = {
            "source": self.context["pretrained_provenance"],
            "source_nc": weights.model[-1].nc if weights is not None else None,
            "target_nc": self.data["nc"],
            "same_name_shape_loaded": len(inherited),
            "target_state_tensors": len(target),
            "class_adapted_shapes": changed,
            "all_common_states_equal": all(torch.equal(source[name].cpu(), target[name].cpu()) for name in inherited),
            "scope": "full saved initialization, including original neck/encoder/decoder states; native nc adaptation",
            "modules": modules,
        }
        if not inherited or not record["all_common_states_equal"]:
            raise RuntimeError("Full original initialization was not inherited by the actual native trainer")
        write_json(self.save_dir / "initialization.json", record)
        return model

    def build_dataset(self, img_path, mode="train", batch=None):
        return build_dataset(self.context["config"], self.args, self.data, mode, batch)

    def get_validator(self):
        self.loss_names = "giou_loss", "cls_loss", "l1_loss"
        args = copy(self.args)
        settings = self.context["config"]["eval"]
        args.conf, args.max_det = settings["conf"], settings["max_det"]
        args.augment = False
        return ComparisonValidator(self.test_loader, save_dir=self.save_dir, args=args, _callbacks=self.callbacks)

    def validate(self):
        metrics = self.validator(self)
        metrics.pop("fitness", None)
        fitness = self.validator.unified["mAP50-95"] or 0.0
        if self.best_fitness is None or self.best_fitness < fitness:
            self.best_fitness = fitness
        return metrics, fitness

    def _setup_train(self):
        super()._setup_train()
        groups = [
            {key: value for key, value in group.items() if key != "params"}
            | {"parameter_tensors": len(group["params"]), "parameter_elements": sum(p.numel() for p in group["params"])}
            for group in self.optimizer.param_groups
        ]
        effective = {
            **vars(self.args),
            "optimizer_class": type(self.optimizer).__name__,
            "optimizer_groups": groups,
            "accumulate": self.accumulate,
            "effective_batch": self.batch_size * self.accumulate,
            "amp_effective": self.amp,
            "training_resize": "aspect-preserving BaseDataset resize + shared B19 exactly once",
            "validation_geometry": {"rect": self.context["config"]["eval"]["rect"], "padding": 0.5},
            "loss": "native RTDETRDetectionLoss, matcher, denoising and EMA unchanged",
        }
        write_json(self.save_dir / "effective_config.json", effective)
        write_json(self.save_dir / "environment.json", environment())

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

    def save_model(self):
        super().save_model()
        path = self.save_dir / "checkpoint_manifest.json"
        manifest = (
            json.loads(path.read_text(encoding="utf-8"))
            if path.is_file()
            else {
                "model": MODEL,
                "backend_sha": self.context["backend_sha"],
                "data_sha256": self.context["data_report"]["data_sha256"],
                "execution_kind": "SMOKE" if self.context["smoke"] else "FORMAL",
                "weights": {},
                "training_augmentation": {},
                "training_provenance": {},
            }
        )
        filenames = [self.last.name]
        if self.best_fitness == self.fitness:
            filenames.append(self.best.name)
            write_json(
                self.save_dir / "best_selection.json",
                {"best_epoch": self.epoch + 1, "metric": "shared val mAP50-95", "fitness": self.fitness},
            )
        if self.save_period > 0 and self.epoch % self.save_period == 0:
            filenames.append(f"epoch{self.epoch}.pt")
        record = {
            "native_checkpoint": {"epoch": self.epoch, "train_results": self.read_results_csv()},
            "comparison_config": self.context["config"],
            "protocol": self.context["protocol"],
            "initialization": json.loads((self.save_dir / "initialization.json").read_text(encoding="utf-8")),
            "backend_lock": json.loads((HERE / "backend.lock.json").read_text(encoding="utf-8")),
            "effective_config": json.loads((self.save_dir / "effective_config.json").read_text(encoding="utf-8")),
        }
        for filename in filenames:
            manifest["weights"][filename] = file_hash(self.wdir / filename)
            manifest["training_augmentation"][filename] = deepcopy(self.context["protocol"]["config"]["augment"])
            manifest["training_provenance"][filename] = record
        write_json(path, manifest)

    def final_eval(self):
        """Finalize native checkpoints; independent val/test is an explicit later process."""
        last = strip_optimizer(self.last)
        strip_optimizer(self.best, updates={"train_results": last.get("train_results")})
        path = self.save_dir / "checkpoint_manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        for filename in [self.last.name, self.best.name]:
            manifest["weights"][filename] = file_hash(self.wdir / filename)
            manifest["training_provenance"][filename]["native_checkpoint"]["train_results"] = last.get(
                "train_results", {}
            )
        write_json(path, manifest)
        selection = json.loads((self.save_dir / "best_selection.json").read_text(encoding="utf-8"))
        write_json(
            self.save_dir / "training_summary.json",
            {
                "completed_epochs": self.epoch + 1,
                "best_epoch": selection["best_epoch"],
                "selection_metric": selection["metric"],
                "best_weight_sha256": manifest["weights"][self.best.name],
                "last_weight_sha256": manifest["weights"][self.last.name],
                "optimizer_updates_ema": self.ema.updates,
                "stop_reason": "epochs_completed"
                if self.epoch + 1 >= self.epochs
                else "time_budget"
                if self.args.time
                else "early_stopping",
                "server_training_status": "SMOKE" if self.context["smoke"] else "COMPLETED",
            },
        )
