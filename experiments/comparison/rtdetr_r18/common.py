"""Bind the C2 RT-DETR backend to the existing comparison protocol."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared import common as _shared

_shared.configure(Path(__file__).resolve().parent)


def __getattr__(name):
    """Expose shared utilities without duplicating their implementation."""
    return getattr(_shared, name)


def resolve(args):
    """Reuse strict config resolution and reject unsupported native resume changes."""
    cfg = _shared.resolve(args)
    if cfg["train"]["resume"]:
        raise ValueError("This experiment starts from the fixed initialization; resume is not implemented")
    if cfg["train"].get("betas") is not None:
        raise ValueError("Use the native AdamW momentum/beta1 recipe; betas overrides are not implemented")
    return cfg


def protocol(cfg, data_hash, training_augmentation=None):
    """Record query decoding and actual geometry without an inapplicable NMS IoU."""
    result = _shared.protocol(cfg, data_hash, training_augmentation)
    result["config"]["eval"].pop("iou", None)
    result["config"]["eval"]["postprocess"] = "native RT-DETR query scores; no NMS, sigmoid or objectness"
    result["protocol_sha256"] = _shared.digest(result["config"])
    result["adapter_sha256"] = _shared.digest(
        {
            **{
                name: _shared.file_hash(_shared.HERE / name)
                for name in ["adapter.py", "runtime.py", "evaluation_runtime.py"]
            },
            "shared/dataset.py": _shared.file_hash(_shared.SHARED / "dataset.py"),
        }
    )
    return result
