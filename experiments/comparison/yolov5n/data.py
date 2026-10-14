"""Read-only dataset manifests and the v5 sample/label boundary for frozen B19 augmentation."""

import copy
import math
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from common import load_yaml
from comparison_io import write_json
from shared.data import label_path


def image_read(path):
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unreadable image: {path}")
    return image


def read_labels(path):
    if not path.is_file() or not path.read_text(encoding="utf-8").strip():
        return np.zeros((0, 5), dtype=np.float32)
    rows = [line.split() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if any(len(row) != 5 for row in rows):
        raise ValueError(f"Detection labels must have five fields (class/xywh): {path}")
    labels = np.array(rows, dtype=np.float32)
    if (
        not np.isfinite(labels).all()
        or (labels[:, 0] != 0).any()
        or (labels[:, 1:] < 0).any()
        or (labels[:, 1:] > 1).any()
        or (labels[:, 3:] <= 0).any()
    ):
        raise ValueError(f"Invalid crack=0 normalized labels: {path}")
    if len(np.unique(labels, axis=0)) != len(labels):
        raise ValueError(f"Duplicate GT labels: {path}")
    return labels


def manifest(data_yaml):
    from shared.data import audit

    dataset, result = audit(data_yaml, load_yaml)
    for split, value in result["splits"].items():
        value["records"] = [
            {
                **entry,
                "image": str(Path(result["root"]) / entry["image_id"]),
                "label": str(label_path(Path(result["root"]) / entry["image_id"])),
                "shape_hw": [entry["height"], entry["width"]],
            }
            for entry in value["entries"]
        ]
    return {
        **result,
        "yaml_path": str(Path(data_yaml).resolve()),
        "names": {"0": "crack"},
        "historical_yaml_bytes_match": result["historical_yaml_matches"],
    }


class ComparisonDataset(Dataset):
    """Keep B19 sampling/resize geometry and expose v5 normalized labels once."""

    def __init__(self, records, cfg, split, stride=32):
        from augmentation import no_augmentation, transforms

        self.cfg, self.split, self.stride = cfg, split, stride
        self.imgsz = cfg["train"]["imgsz"]
        self.batch_size = cfg["train"]["batch"] if split == "train" else cfg["eval"]["batch"]
        self.augment = split == "train" and cfg["augment"]["profile"] != "none"
        self.rect = split != "train" and cfg["eval"]["rect"]
        self.records = list(records)
        if self.rect:
            ratios = np.array([r["shape_hw"][0] / r["shape_hw"][1] for r in self.records])
            self.records = [self.records[i] for i in ratios.argsort()]
        self.labels = [read_labels(Path(r["label"])) for r in self.records]
        self.shapes = np.array(
            [r["shape_hw"][::-1] for r in self.records], dtype=np.float64
        )  # native anchors expect WH
        self.im_files = [r["image"] for r in self.records]
        self.n = self.ni = len(self.records)
        self.indices = list(range(self.n))
        self.data = {"names": {0: "crack"}}
        self.cache = False
        self.buffer = []
        self.max_buffer_length = min(self.n, self.batch_size * 8, 1000)
        self.ims = [None] * self.n
        self.batch = np.floor(np.arange(self.n) / self.batch_size).astype(int)
        self.batch_shapes = []
        for batch in range(self.batch[-1] + 1):
            hw = np.array([r["shape_hw"] for r in self.records])[self.batch == batch]
            ar = hw[:, 0] / hw[:, 1]
            shape = [ar.max(), 1] if ar.max() < 1 else [1, 1 / ar.min()] if ar.min() > 1 else [1, 1]
            self.batch_shapes.append(np.ceil(np.array(shape) * self.imgsz / stride + 0.5).astype(int) * stride)
        self.current_values = cfg["augment"]["resolved"].copy()
        self.transforms = (
            transforms(self, self.current_values, cfg["augment"]["albumentations"])
            if self.augment
            else no_augmentation(self.imgsz)
        )

    def __len__(self):
        return self.n

    def set_epoch(self, epoch):
        from augmentation import schedule, transforms

        values, closed = schedule(self.cfg["augment"]["resolved"], epoch, self.cfg["train"]["epochs"])
        if self.augment and values != self.current_values:
            self.current_values = values
            self.transforms = transforms(self, values, self.cfg["augment"]["albumentations"])
        return closed

    def get_image_and_label(self, index):
        from shared.b19.instance import Instances

        record = self.records[index]
        image = self.ims[index]
        if image is None:
            image = image_read(record["image"])
            h0, w0 = image.shape[:2]
            ratio = self.imgsz / max(h0, w0)
            if ratio != 1:
                width, height = min(math.ceil(w0 * ratio), self.imgsz), min(math.ceil(h0 * ratio), self.imgsz)
                image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
            if self.augment:
                self.ims[index] = image
                self.buffer.append(index)
                if 1 < len(self.buffer) >= self.max_buffer_length:
                    self.ims[self.buffer.pop(0)] = None
        labels = self.labels[index].copy()
        result = {
            "im_file": record["image"],
            "ori_shape": tuple(record["shape_hw"]),
            "resized_shape": image.shape[:2],
            "img": image.copy(),
            "cls": labels[:, :1],
            "instances": Instances(
                labels[:, 1:], segments=np.zeros((0, 1000, 2), dtype=np.float32), bbox_format="xywh", normalized=True
            ),
        }
        if self.rect:
            result["rect_shape"] = self.batch_shapes[self.batch[index]]
        return result

    def __getitem__(self, index):
        index = self.indices[index]
        labels = self.get_image_and_label(index)
        original = labels["ori_shape"]
        resized = labels["resized_shape"]
        if not self.augment:
            letterbox = self.transforms.transforms[0]
            params = letterbox.get_params(copy.deepcopy(labels))
            shape = (
                (resized[0] / original[0] * params["ratio"][1], resized[1] / original[1] * params["ratio"][0]),
                (params["left"], params["top"]),
            )
        else:
            shape = None
        output = self.transforms(labels)
        image = output["img"]
        instances = output["instances"]
        instances.convert_bbox("xywh")
        instances.denormalize(image.shape[1], image.shape[0])
        instances.normalize(image.shape[1], image.shape[0])
        boxes = instances.bboxes.astype(np.float32)
        if (
            not np.isfinite(boxes).all()
            or (boxes < -1e-6).any()
            or (boxes > 1 + 1e-6).any()
            or (boxes[:, 2:] <= 0).any()
        ):
            raise ValueError(f"Augmentation produced invalid normalized boxes: {self.im_files[index]}")
        target = torch.zeros((len(boxes), 6), dtype=torch.float32)
        if len(boxes):
            target[:, 1] = torch.from_numpy(output["cls"].reshape(-1))
            target[:, 2:] = torch.from_numpy(boxes)
        # B19 Format's BGR probability is part of the protocol; normalize pixels in native train only.
        if image.shape[2] == 3 and (not self.augment or random.uniform(0, 1) > self.current_values["bgr"]):
            image = image[..., ::-1]
        return (
            torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))),
            target,
            self.im_files[index],
            (original, shape),
        )

    @staticmethod
    def collate_fn(batch):
        images, targets, paths, shapes = zip(*batch)
        for index, target in enumerate(targets):
            target[:, 0] = index
        return torch.stack(images), torch.cat(targets), paths, shapes


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def loader(records, cfg, split, stride=32):
    dataset = ComparisonDataset(records, cfg, split, stride)
    batch = cfg["train"]["batch"] if split == "train" else cfg["eval"]["batch"]
    workers = cfg["train"]["workers"] if split == "train" else cfg["eval"]["workers"]
    generator = torch.Generator().manual_seed(cfg["train"]["seed"])
    return DataLoader(
        dataset,
        batch_size=batch,
        shuffle=split == "train",
        num_workers=workers,
        persistent_workers=False,
        worker_init_fn=seed_worker,
        generator=generator,
        pin_memory=cfg["train"]["device"] != "cpu",
        collate_fn=dataset.collate_fn,
    ), dataset


def create_dataloader(path, imgsz, batch_size, stride, single_cls, augment=False, **kwargs):
    from runtime import HOOKS

    split = "train" if augment else "val"
    records = HOOKS.manifest["splits"][split]["records"]
    if augment and HOOKS.cfg["augment"]["profile"] == "native":
        from augmentation import optional_transform
        from utils.dataloaders import create_dataloader as native_loader

        root = Path(HOOKS.cfg["cache"]) / "native-index"
        root.mkdir(parents=True, exist_ok=True)
        index = root / f"{HOOKS.manifest['splits'][split]['sha256']}.txt"
        index.write_text("\n".join(r["image"] for r in records) + "\n", encoding="utf-8")
        result = native_loader(
            str(index),
            imgsz,
            batch_size,
            stride,
            single_cls,
            augment=True,
            albumentations=optional_transform(HOOKS.cfg["augment"]["albumentations"]),
            **kwargs,
        )
        if {str(Path(p).resolve()) for p in result[1].im_files} != {str(Path(r["image"]).resolve()) for r in records}:
            raise ValueError("Native dataset skipped images; full manifest identity is required")
        return result
    return loader(records, HOOKS.cfg, split, stride)


def save_manifest(directory, value):
    write_json(Path(directory) / "data_manifest.json", value)
    for split, entry in value["splits"].items():
        write_json(
            Path(directory) / f"{split}_image_index.json",
            [{k: v for k, v in r.items() if k not in ("image", "label")} for r in entry["records"]],
        )
