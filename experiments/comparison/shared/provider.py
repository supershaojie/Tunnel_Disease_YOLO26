"""Frozen B19 operator ordering and close-mosaic policy for detection backends."""

from .b19.augment import Compose, CopyPaste, CutMix, MixUp, Mosaic, RandomFlip, RandomHSV, RandomPerspective

CLOSED_OPERATIONS = ("mosaic", "copy_paste", "mixup", "cutmix")


def closed_values(values):
    """Apply the B19 dataset close-mosaic transition."""
    return {**values, **dict.fromkeys(CLOSED_OPERATIONS, 0.0)}


def build(dataset, imgsz, values):
    """Use B19 operators once, with backend formatting owned by its adapter."""
    mosaic = Mosaic(dataset, imgsz=imgsz, p=values["mosaic"])
    affine = RandomPerspective(
        **{k: values[k] for k in ["degrees", "translate", "scale", "shear", "perspective"]}, size=(imgsz, imgsz)
    )
    pre = Compose([mosaic, affine])
    if values["copy_paste_mode"] == "flip":
        pre.insert(1, CopyPaste(dataset, p=values["copy_paste"], mode="flip"))
    else:
        pre.append(CopyPaste(dataset, pre_transform=Compose([mosaic, affine]), p=values["copy_paste"], mode="mixup"))
    return Compose(
        [
            pre,
            MixUp(dataset, pre_transform=pre, p=values["mixup"]),
            CutMix(dataset, pre_transform=pre, p=values["cutmix"]),
            RandomHSV(hgain=values["hsv_h"], sgain=values["hsv_s"], vgain=values["hsv_v"]),
            RandomFlip(direction="vertical", p=values["flipud"]),
            RandomFlip(direction="horizontal", p=values["fliplr"]),
        ]
    )
