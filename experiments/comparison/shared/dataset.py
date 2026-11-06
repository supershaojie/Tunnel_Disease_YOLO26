"""Shared B19 dataset and native formatting, independent of model/trainer APIs."""

import sys
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
from common import AUG_KEYS

from shared import provider
from shared.b19.instance import Instances as B19Instances
from ultralytics.data.augment import Albumentations, Compose, Format, LetterBox
from ultralytics.data.dataset import YOLODataset
from ultralytics.utils.instance import Instances


class NativeFormat:
    """Convert the shared Instances interface explicitly before native backend formatting."""

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
