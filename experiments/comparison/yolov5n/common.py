"""Typed configuration, source isolation, and public comparison protocol."""

import argparse
import copy
import importlib.util
import json
import math
import os
import sys
from pathlib import Path

import yaml

from comparison_io import canonical_hash, digest, read_json

HERE = Path(__file__).resolve().parent
VENDOR = HERE / "vendor/yolov5"
SHARED = HERE.parent / "shared"
sys.path.insert(0, str(HERE.parent))  # pure shared package is needed before native checkpoint bootstrap
DEFAULT = HERE / "configs/default.yaml"
AUG = dict(
    hsv_h=0.024,
    hsv_s=0.84,
    hsv_v=0.535,
    degrees=11.0,
    translate=0.17,
    scale=0.735,
    shear=3.5,
    perspective=0.00055,
    flipud=0.0,
    fliplr=0.5,
    bgr=0.0,
    mosaic=1.0,
    mixup=0.135,
    cutmix=0.03,
    copy_paste=0.0,
    copy_paste_mode="flip",
    close_mosaic=10,
)
PROBABILITIES = {
    "hsv_h",
    "hsv_s",
    "hsv_v",
    "translate",
    "flipud",
    "fliplr",
    "bgr",
    "mosaic",
    "mixup",
    "cutmix",
    "copy_paste",
}


class UniqueLoader(yaml.SafeLoader):
    """Reject duplicate YAML keys before applying configuration precedence."""

    def construct_mapping(self, node, deep=False):
        keys = [self.construct_object(key, deep=deep) for key, _ in node.value]
        if len(keys) != len(set(keys)):
            raise ValueError(f"Duplicate YAML keys at line {node.start_mark.line + 1}")
        return super().construct_mapping(node, deep=deep)


def load_yaml(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        value = yaml.load(stream, Loader=UniqueLoader)
    if not isinstance(value, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return value


def flatten(value, prefix=""):
    result = {}
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(item, dict) and item:
            result.update(flatten(item, name))
        else:
            result[name] = item
    return result


def merge(target, update, schema, sources, origin, prefix=""):
    for key, value in update.items():
        name = f"{prefix}.{key}" if prefix else key
        if key not in schema:
            raise ValueError(f"Unknown configuration key: {name}")
        if isinstance(schema[key], dict):
            if not isinstance(value, dict):
                raise ValueError(f"{name} must be a mapping")
            merge(target[key], value, schema[key], sources, origin, name)
        else:
            target[key] = value
            sources[name] = origin


class UniqueStore(argparse.Action):
    """Keep contradictory repeated explicit options from being silently discarded."""

    def __call__(self, parser, namespace, values, option_string=None):
        assigned = vars(namespace).setdefault("_assigned_options", {})
        if self.dest in assigned and assigned[self.dest] != values:
            raise argparse.ArgumentError(self, f"Contradictory repeated option: {option_string}")
        assigned[self.dest] = values
        setattr(namespace, self.dest, values)


def parser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", type=Path, action=UniqueStore, default=DEFAULT)
    for name in ("data", "weights", "baseline-args", "project", "name", "device"):
        p.add_argument("--" + name, action=UniqueStore)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--resume",
        action=UniqueStore,
        nargs="?",
        const=True,
        default=None,
        help="Explicit checkpoint, or this run's last.pt",
    )
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    return p


def configuration(args):
    official = read_json(HERE / "official_defaults.json")
    public = load_yaml(DEFAULT)
    native = official["parser"]
    hyp = {k: v for k, v in official["hyp"].items() if k not in AUG}
    recipe = {
        **hyp,
        "cos_lr": native["cos_lr"],
        "patience": native["patience"],
        "label_smoothing": native["label_smoothing"],
        "save_period": native["save_period"],
        "beta2": official["adam_beta2"],
    }
    cfg = copy.deepcopy(public)
    cfg["train"] = {**recipe, **cfg["train"]}
    schema = copy.deepcopy(cfg)
    schema["augment"]["overrides"] = AUG.copy()
    schema["train"]["resume"] = False
    sources = {k: "public_preset" for k in flatten(public)}
    sources.update({"train." + k: "official_v7.0" for k in recipe})
    raw = load_yaml(args.config)
    merge(cfg, raw, schema, sources, "config_file")
    cli = {}
    mapping = {
        "data": "data",
        "weights": "weights",
        "baseline_args": "reference.baseline_args",
        "project": "project",
        "name": "name",
        "device": "train.device",
        "resume": "train.resume",
        "split": "eval.split",
    }
    for field, key in mapping.items():
        value = getattr(args, field, None)
        if value is not None:
            cli[key] = value
    for setting in args.set:
        if "=" not in setting:
            raise ValueError(f"Expected --set key=value, got {setting}")
        key, text = setting.split("=", 1)
        value = yaml.safe_load(text)
        if key in cli and cli[key] != value:
            raise ValueError(f"Conflicting explicit settings for {key}: {cli[key]!r} and {value!r}")
        if any(k.startswith(key + ".") or key.startswith(k + ".") for k in cli):
            raise ValueError(f"Overlapping explicit settings for {key}")
        cli[key] = value
    for key, value in cli.items():
        update = value
        for part in reversed(key.split(".")):
            update = {part: update}
        merge(cfg, update, schema, sources, "explicit_cli")
    cfg["train"]["optimizer"] = (
        native["optimizer"] if cfg["train"]["optimizer"] == "official_default" else cfg["train"]["optimizer"]
    )
    if cfg["train"]["optimizer"] == "SGD" and sources["train.beta2"] != "official_v7.0":
        raise ValueError("Explicit beta2 requires Adam/AdamW; SGD has no beta2 parameter")
    validate_config(cfg)
    issues = []
    profile = cfg["augment"]["profile"]
    baseline = Path(cfg["reference"]["baseline_args"])
    if profile == "b19":
        if baseline.is_file():
            reference = load_yaml(baseline)
            missing = set(AUG) - set(reference)
            if missing:
                raise ValueError(f"B19 args missing augmentation fields: {sorted(missing)}")
            augmentation = {k: reference[k] for k in AUG}
            reference_info = {"state": "VERIFIED_FILE", "sha256": digest(baseline), "path": str(baseline)}
        else:
            if not args.dry_run:
                raise FileNotFoundError(
                    f"B19 reference args missing: {baseline}; provide --baseline-args or an explicit variant profile"
                )
            augmentation = AUG.copy()
            reference_info = {
                "state": "UNVERIFIED",
                "path": str(baseline),
                "source": "historical checklist only (dry-run)",
            }
            issues.append("B19 server args UNVERIFIED; dry-run uses historical checklist")
    else:
        defaults = load_yaml(HERE.parents[2] / "ultralytics/cfg/default.yaml")
        augmentation = {k: defaults[k] for k in AUG}
        reference_info = {"state": "EXPLICIT_VARIANT", "profile": profile}
        if profile == "native":
            augmentation.update({k: v for k, v in official["hyp"].items() if k in AUG})
            augmentation.update(cutmix=0.0, close_mosaic=0, bgr=0.0)
        if profile == "none":
            augmentation = {k: ("flip" if k == "copy_paste_mode" else 0) for k in AUG}
    augmentation.update(cfg["augment"]["overrides"])
    validate_augmentation(augmentation)
    if augmentation["copy_paste"]:
        raise ValueError("Detection-only labels have no polygons for CopyPaste; nonzero request is unsupported")
    if profile == "none" and (
        cfg["augment"]["albumentations"] != "disabled"
        or any(v for k, v in augmentation.items() if k != "copy_paste_mode")
    ):
        raise ValueError("none profile requires disabled Albumentations and zero random augmentation values")
    if profile == "native" and (augmentation["cutmix"] or augmentation["bgr"] or augmentation["close_mosaic"]):
        raise ValueError("v7.0 native profile has no CutMix, BGR randomization or close_mosaic; use b19/custom")
    cfg["augment"]["resolved"] = augmentation
    if cfg["name"] is None:
        t = cfg["train"]
        cfg["name"] = (
            f"compare_yolov5n_b19_e{t['epochs']}_i{t['imgsz']}_b{t['batch']}_{t['optimizer'].lower()}_s{t['seed']}"
        )
        sources["name"] = "generated_from_final_parameters"
    if not cfg["name"] or Path(cfg["name"]).name != cfg["name"]:
        raise ValueError("name must be one directory component")
    return cfg, {
        "raw_user_config": raw,
        "explicit_cli": cli,
        "sources": sources,
        "official_defaults": official,
        "baseline_reference": reference_info,
        "issues": issues,
        "run_config_sha256": canonical_hash(cfg),
    }


def validate_augmentation(aug):
    for key, value in aug.items():
        if key == "copy_paste_mode":
            if value not in ("flip", "mixup"):
                raise ValueError("copy_paste_mode must be flip or mixup")
        elif isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"Invalid augmentation value {key}={value!r}")
        elif key in PROBABILITIES and value > 1:
            raise ValueError(f"Probability {key} must be in [0,1]")
        elif key == "close_mosaic" and not isinstance(value, int):
            raise ValueError("close_mosaic must be an integer")
    if aug["scale"] > 1:
        raise ValueError("scale must be in [0,1] for the scalar B19 recipe")


def validate_config(cfg):
    if cfg["protocol"] != "comparison-b19-v1" or cfg["backend"] != {"name": "yolov5n", "upstream_tag": "v7.0"}:
        raise ValueError("This entry implements comparison-b19-v1 / original yolov5n / v7.0 only")
    if cfg["train"]["optimizer"] not in ("SGD", "Adam", "AdamW"):
        raise ValueError("Locked v7.0 supports SGD/Adam/AdamW; it has no optimizer=auto mode")
    if cfg["augment"]["profile"] not in ("b19", "custom", "none", "native"):
        raise ValueError("Unknown augmentation profile")
    if cfg["augment"]["albumentations"] not in ("disabled", "frozen_default"):
        raise ValueError("albumentations must be disabled or frozen_default")
    for field in ("data", "weights", "cache", "project"):
        if not isinstance(cfg[field], str) or not cfg[field]:
            raise ValueError(f"{field} must be a nonempty path string")
    for section, keys in {
        "train": ["epochs", "imgsz", "batch", "workers", "seed", "patience", "save_period"],
        "eval": ["batch", "workers", "max_det"],
        "benchmark": ["imgsz", "batch", "warmup", "iterations"],
        "package": ["max_bytes"],
    }.items():
        for key in keys:
            value = cfg[section][key]
            minimum = (
                -1 if key == "save_period" else 0 if key in ("workers", "seed", "patience", "warmup", "examples") else 1
            )
            if type(value) is not int or value < minimum:
                raise ValueError(f"{section}.{key} must be an integer >= {minimum}")
    for section, keys in {
        "train": ["amp", "autoanchor", "cos_lr"],
        "eval": ["half", "rect", "augment", "fuse"],
        "benchmark": ["half"],
    }.items():
        for key in keys:
            if type(cfg[section][key]) is not bool:
                raise ValueError(f"{section}.{key} must be a YAML boolean")
    for key, value in cfg["train"].items():
        if key in (
            "lr0",
            "lrf",
            "momentum",
            "weight_decay",
            "warmup_epochs",
            "warmup_momentum",
            "warmup_bias_lr",
            "box",
            "cls",
            "cls_pw",
            "obj",
            "obj_pw",
            "iou_t",
            "anchor_t",
            "fl_gamma",
            "beta2",
            "label_smoothing",
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid train.{key}={value}")
    for key in ("conf", "iou", "display_conf"):
        value = cfg["eval"][key]
        if type(value) not in (float, int) or not 0 <= value <= 1:
            raise ValueError(f"eval.{key} must be a probability")
    if cfg["eval"]["display_conf"] < cfg["eval"]["conf"]:
        raise ValueError("display_conf cannot be below the collected prediction confidence")
    for key in ("momentum", "beta2", "warmup_momentum", "label_smoothing"):
        if not 0 <= cfg["train"][key] < 1:
            raise ValueError(f"train.{key} must be in [0,1)")
    if cfg["eval"]["split"] not in ("val", "test") or cfg["eval"]["augment"]:
        raise ValueError("Evaluation requires split=val/test and augment=false")
    if cfg["benchmark"]["batch"] != 1:
        raise ValueError("Comparison latency benchmark requires batch=1")
    if cfg["train"]["imgsz"] < 64 or cfg["train"]["imgsz"] % 32 or cfg["benchmark"]["imgsz"] % 32:
        raise ValueError("Input size must be a stride-32 multiple (training minimum 64)")
    if not isinstance(cfg["train"]["device"], str) or (
        cfg["train"]["device"] != "cpu" and not cfg["train"]["device"].isdigit()
    ):
        raise ValueError("Use one explicit CUDA device index or cpu; concurrent experiments use separate processes")
    if type(cfg["train"]["resume"]) not in (bool, str):
        raise ValueError("train.resume must be false/true or a checkpoint path")


def verify_sources():
    lock = read_json(HERE / "backend.lock.json")
    for name, expected in lock["vendor_files"].items():
        if digest(VENDOR / name) != expected:
            raise ValueError(f"Vendored source integrity mismatch: {name}")
    shared = read_json(HERE / "shared.lock.json")
    for name, expected in shared["files"].items():
        if digest(SHARED / name) != expected:
            raise ValueError(f"Shared provider integrity mismatch: {name}")
    return {"backend": lock, "frozen": read_json(SHARED / "b19/source.lock.json"), "shared": shared}


def bootstrap(cache):
    verify_sources()
    Path(cache).mkdir(parents=True, exist_ok=True)
    os.environ["YOLOv5_AUTOINSTALL"] = "false"
    os.environ["YOLO_AUTOINSTALL"] = "false"
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["COMET_MODE"] = "DISABLED"
    os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
    os.environ["YOLOV5_CONFIG_DIR"] = str(Path(cache) / "v5-config")
    os.environ["YOLOV5_LABEL_CACHE"] = str(Path(cache) / "native-labels")
    for namespace in ("models", "utils"):
        for name, module in list(sys.modules.items()):
            if name == namespace or name.startswith(namespace + "."):
                path = getattr(module, "__file__", None)
                if path and not Path(path).resolve().is_relative_to(VENDOR):
                    raise RuntimeError(f"Conflicting {name} loaded from {path}; start a new independent process")
    paths = (str(VENDOR), str(HERE.parent))
    sys.path[:] = [*paths, *(path for path in sys.path if path not in paths)]
    import models.yolo
    import utils.general

    imports = {"models": models.yolo.__file__, "utils": utils.general.__file__, "vendor": str(VENDOR)}
    if any(not Path(imports[key]).resolve().is_relative_to(VENDOR) for key in ("models", "utils")):
        raise RuntimeError(f"Native imports must resolve inside the locked vendor: {imports}")
    return imports


def backend_train():
    spec = importlib.util.spec_from_file_location("comparison_native_v5_train", VENDOR / "train.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def protocol(cfg, manifest):
    core = canonical_hash(
        {
            "b19": {
                name: digest(SHARED / "b19" / name) for name in ("augment.py", "instance.py", "ops.py", "metrics.py")
            },
            "provider": digest(SHARED / "provider.py"),
        }
    )
    public = {
        "id": cfg["protocol"],
        "data": manifest["data_sha256"],
        "augment": {
            "profile": cfg["augment"]["profile"],
            "values": cfg["augment"]["resolved"],
            "implementation": canonical_hash(
                {name: digest(VENDOR / name) for name in ("utils/augmentations.py", "utils/dataloaders.py")}
            )
            if cfg["augment"]["profile"] == "native"
            else core,
            "extras": {"albumentations": {"enabled": False}}
            if cfg["augment"]["albumentations"] == "disabled"
            else {"albumentations": {"enabled": True, "version": "2.0.8", "policy": "frozen_default"}},
        },
        "eval": {
            k: v for k, v in cfg["eval"].items() if k not in ("split", "fixed_images", "display_conf", "examples")
        },
        "imgsz": cfg["train"]["imgsz"],
        "metrics_implementation": digest(SHARED / "b19/metrics.py"),
        "matching_implementation": canonical_hash(
            {name: digest(SHARED / name) for name in ("b19/matching.py", "evaluation.py")}
        ),
    }
    return {
        "public": public,
        "config": public,
        "protocol_sha256": canonical_hash(public),
        "shared_augmentation_sha256": core,
        "augment_core_sha256": core,
        "adapter_sha256": canonical_hash(
            {name: digest(HERE / name) for name in ("data.py", "geometry.py", "augmentation.py")}
        ),
        "run_config_sha256": canonical_hash(cfg),
    }


def print_plan(cfg, audit):
    print(
        json.dumps(
            {
                "mode": "DRY_RUN",
                "config": cfg,
                "audit": audit,
                "server_training": "NOT_STARTED",
                "output_directory_created": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
