"""Backend boundary for the common committed provider and deterministic B19 geometry."""

from geometry import LetterBox
from shared.b19.augment import Compose
from shared.provider import build, closed_values


def optional_transform(policy):
    if policy == "disabled":
        return None
    from importlib.metadata import version
    from optional_albumentations import Albumentations

    if version("albumentations") != "2.0.8":
        raise RuntimeError("frozen_default requires albumentations==2.0.8")
    transform = Albumentations(p=1.0)
    if transform.transform is None:
        raise RuntimeError("Explicit optional augmentation unavailable")
    return transform


def transforms(dataset, values, albumentations="disabled"):
    sequence = build(dataset, dataset.imgsz, values)
    transform = optional_transform(albumentations)
    if transform is not None:
        sequence.insert(3, transform)
    return sequence


def no_augmentation(imgsz):
    return Compose([LetterBox((imgsz, imgsz), scaleup=False)])


def schedule(values, epoch, epochs):
    closed = values["close_mosaic"] > 0 and 0 <= epochs - values["close_mosaic"] <= epoch
    return closed_values(values) if closed else values.copy(), closed
