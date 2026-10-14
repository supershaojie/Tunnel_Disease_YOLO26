"""Configuration, immutable source locks, provenance and isolated backend bootstrap."""

import argparse
import hashlib
import json
import os
import platform
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
VENDOR = HERE / "vendor/ultralytics_source"
SHARED = HERE.parent / "shared"
BASE_SHA = "4a1b167dee2b0c7353d9d7ead2911383d1ebf1d6"
AUG_KEYS = (
    "hsv_h",
    "hsv_s",
    "hsv_v",
    "degrees",
    "translate",
    "scale",
    "shear",
    "perspective",
    "flipud",
    "fliplr",
    "bgr",
    "mosaic",
    "mixup",
    "cutmix",
    "copy_paste",
    "copy_paste_mode",
    "close_mosaic",
)
PROB_KEYS = (
    "hsv_h",
    "hsv_s",
    "hsv_v",
    "translate",
    "scale",
    "perspective",
    "flipud",
    "fliplr",
    "bgr",
    "mosaic",
    "mixup",
    "cutmix",
    "copy_paste",
)
OPTIMIZERS = ("auto", "SGD", "Adam", "Adamax", "AdamW", "NAdam", "RAdam", "RMSProp")


class UniqueLoader(yaml.SafeLoader):
    """Reject duplicate YAML fields instead of losing user intent."""


def _mapping(loader, node, deep=False):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise ValueError(f"Duplicate YAML key: {key}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


def read_yaml(path):
    """Load a mapping with safe typed YAML and unique keys."""
    result = yaml.load(Path(path).read_text(encoding="utf-8"), Loader=UniqueLoader)
    if not isinstance(result, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return result


def digest(value):
    """Hash canonical JSON without paths or presentation-dependent whitespace."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def file_hash(path):
    """Compute a streaming SHA256 for source, data and checkpoints."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    """Atomically save JSON provenance."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    temp.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, default=str) + "\n", encoding="utf-8"
    )
    os.replace(temp, path)


class UniqueStore(argparse.Action):
    """Assign a scalar/list option once without discarding contradictory repeated requests."""

    def __call__(self, parser, namespace, values, option_string=None):
        assigned = vars(namespace).setdefault("_assigned_options", {})
        if self.dest in assigned and assigned[self.dest] != values:
            raise argparse.ArgumentError(self, f"Contradictory repeated option: {option_string}")
        assigned[self.dest] = values
        setattr(namespace, self.dest, values)


def verify_backend():
    """Reject a modified vendor tree before trusted checkpoint loading."""
    lock = json.loads((HERE / "backend.lock.json").read_text(encoding="utf-8"))
    bad = [
        name for name, sha in lock["files"].items() if not (VENDOR / name).is_file() or file_hash(VENDOR / name) != sha
    ]
    if bad:
        raise ValueError(f"Backend lock mismatch: {bad[:8]}")
    return lock


def parser(mode):
    """Build the common supported command interface without importing Torch."""
    p = argparse.ArgumentParser(description=f"Locked YOLOv8n comparison: {mode}")
    p.add_argument("--config", action=UniqueStore, default=str(HERE / "configs/default.yaml"))
    for option in ("data", "weights", "baseline-args", "project", "name", "device", "cache-dir", "pretrained-cache"):
        p.add_argument("--" + option, action=UniqueStore)
    p.add_argument("--set", action="append", default=[], metavar="KEY=YAML_VALUE")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--resume", action=UniqueStore, metavar="LAST_PT", help="Explicit own last.pt; no automatic latest-run lookup"
    )
    p.add_argument("--split", action=UniqueStore, choices=["val", "test"])
    p.add_argument("--result-dir", action=UniqueStore)
    p.add_argument("--output", action=UniqueStore)
    p.add_argument(
        "--val-metrics", action=UniqueStore, help="Trusted val metrics.json containing the frozen confidence threshold"
    )
    p.add_argument(
        "--results", action=UniqueStore, nargs="+", help="Result directories for protocol-checked aggregate packaging"
    )
    return p


def resolve(args):
    """Resolve defaults, presets, config and explicit CLI without silently changing values."""
    official = read_yaml(VENDOR / "ultralytics/cfg/default.yaml")
    official = {k: None if isinstance(v, str) and v.lower() == "none" else v for k, v in official.items()}
    preset = read_yaml(HERE / "configs/default.yaml")
    raw = read_yaml(args.config)
    cfg = deepcopy(preset)
    sources = {}
    schemas = {k: set(v) if isinstance(v, dict) else None for k, v in preset.items()}
    schemas["train"] = (set(official) - set(AUG_KEYS) - {"data", "model", "project", "name", "task", "mode", "cfg"}) | {
        "betas"
    }
    schemas["augment"] = {"profile", "overrides"}

    def assign(key, value, source):
        if isinstance(value, dict):
            raise ValueError(f"Expected leaf config key: {key}; use individual --set key=value fields")
        parts = key.split(".")
        if parts[0] not in schemas:
            raise ValueError(f"Unknown config key: {key}")
        if len(parts) == 2:
            if schemas[parts[0]] is None or parts[1] not in schemas[parts[0]]:
                raise ValueError(f"Unknown config key: {key}")
        elif len(parts) == 3:
            if parts[:2] != ["augment", "overrides"] or parts[2] not in AUG_KEYS:
                raise ValueError(f"Unknown config key: {key}")
        elif len(parts) != 1 or isinstance(cfg[parts[0]], dict):
            raise ValueError(f"Expected leaf config key: {key}")
        target = cfg
        for part in parts[:-1]:
            target = target[part]
        target[parts[-1]] = value
        sources[key] = source

    def walk(mapping, prefix=""):
        for key, value in mapping.items():
            name = prefix + str(key)
            if isinstance(value, dict):
                if name == "augment.overrides" and not value:
                    continue
                if name.split(".")[0] not in schemas:
                    raise ValueError(f"Unknown config key: {name}")
                if name.count(".") and name != "augment.overrides":
                    raise ValueError(f"Unknown mapping: {name}")
                walk(value, name + ".")
            else:
                assign(name, value, "config")

    walk(raw)
    explicit = {}
    for item in args.set:
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"Expected key=value: {item}")
        value = yaml.load(value, Loader=UniqueLoader)
        if key in explicit and explicit[key] != value:
            raise ValueError(f"Contradictory duplicate override: {key}")
        explicit[key] = value
    for name, key in [
        ("data", "data"),
        ("weights", "weights"),
        ("baseline_args", "reference.baseline_args"),
        ("project", "project"),
        ("name", "name"),
        ("device", "train.device"),
        ("cache_dir", "cache_dir"),
        ("pretrained_cache", "pretrained_cache"),
        ("split", "eval.split"),
        ("resume", "train.resume"),
    ]:
        value = getattr(args, name, None)
        if value is not None:
            if key in explicit and explicit[key] != value:
                raise ValueError(f"Contradictory CLI and --set: {key}")
            explicit[key] = value
    for key, value in explicit.items():
        assign(key, value, "CLI")
    if cfg["protocol"] != "comparison-b19-v1" or cfg["backend"] != preset["backend"]:
        raise ValueError("This entry requires comparison-b19-v1 and locked YOLOv8n v8.3.0")
    requested = deepcopy(official)
    for key in official:
        sources.setdefault("train." + key, "public_preset" if key in preset["train"] else "locked_official_default")
    for key, value in cfg["train"].items():
        requested[key] = official.get(key) if value == "official_default" else value
    if requested["optimizer"] not in OPTIMIZERS:
        raise ValueError(f"Unsupported locked-version optimizer: {requested['optimizer']}")
    for key in ["epochs", "batch", "imgsz", "nbs"]:
        if type(requested[key]) is not int or requested[key] < 1:
            raise ValueError(f"train.{key} must be a positive integer; implicit autobatch is disabled")
    if requested["imgsz"] % 32:
        raise ValueError("imgsz must be a multiple of 32; implicit rounding is disabled")
    for key, value in requested.items():
        default = official.get(key)
        if isinstance(default, bool) and type(value) is not bool and key not in ["cache", "resume"]:
            raise ValueError(f"train.{key} must be boolean")
        if isinstance(value, float) and not __import__("math").isfinite(value):
            raise ValueError(f"Nonfinite train.{key}")
    if type(requested["workers"]) is not int or requested["workers"] < 0:
        raise ValueError("workers must be a nonnegative integer")
    if requested["resume"] is not False and not isinstance(requested["resume"], str):
        raise ValueError("resume must be false or an explicit checkpoint path")
    if (
        not requested["val"]
        or requested["fraction"] != 1.0
        or requested["single_cls"]
        or requested["classes"] is not None
    ):
        raise ValueError("Main comparison requires val=true, fraction=1, original class mapping and all images")
    if not requested["save"] or requested["pretrained"] is not True or requested["split"] != "val":
        raise ValueError("Main training requires save=true, official pretrained=true and val-only epoch selection")
    if requested["save_hybrid"]:
        raise ValueError("save_hybrid injects GT boxes into validation predictions and invalidates comparison metrics")
    for key in ["lr0", "lrf", "weight_decay", "warmup_epochs", "warmup_bias_lr", "box", "cls", "dfl"]:
        if type(requested[key]) not in [int, float] or requested[key] < 0:
            raise ValueError(f"train.{key} must be nonnegative numeric")
    for key in ["momentum", "warmup_momentum"]:
        if type(requested[key]) not in [int, float] or not 0 <= requested[key] < 1:
            raise ValueError(f"train.{key} must be in [0,1)")
    if type(requested["patience"]) is not int or requested["patience"] < 0:
        raise ValueError("patience must be a nonnegative integer")
    if requested["time"] is not None and (type(requested["time"]) not in [int, float] or requested["time"] < 0):
        raise ValueError("time must be null or nonnegative hours")
    if "," in str(requested["device"]):
        raise ValueError("This isolated comparison entry uses one CPU/GPU per process; select one device")
    if requested.get("betas") is not None:
        betas = requested["betas"]
        if requested["optimizer"] not in ["Adam", "Adamax", "AdamW", "NAdam", "RAdam"]:
            raise ValueError("Explicit betas require an explicit Adam-family optimizer")
        if (
            not isinstance(betas, list)
            or len(betas) != 2
            or any(type(b) not in [int, float] or not 0 <= b < 1 for b in betas)
        ):
            raise ValueError("betas must be a two-element sequence in [0,1)")
    profile = cfg["augment"]["profile"]
    if profile not in ["b19", "custom", "none", "native"]:
        raise ValueError(f"Unknown augment profile: {profile}")
    baseline = Path(cfg["reference"]["baseline_args"])
    baseline_verified = baseline.is_file()
    if profile == "b19":
        if baseline_verified:
            baseline_values = read_yaml(baseline)
        elif args.dry_run:
            baseline_values = read_yaml(HERE / "reference/b19_archived_args.yaml")
        else:
            raise FileNotFoundError(f"B19 args missing: {baseline}; pass --baseline-args")
        missing = set(AUG_KEYS) - set(baseline_values)
        if missing:
            raise ValueError(f"Missing B19 augment fields: {sorted(missing)}")
        augment = {key: baseline_values[key] for key in AUG_KEYS}
    else:
        augment = {key: official.get(key, 0.0 if key == "cutmix" else "flip") for key in AUG_KEYS}
    if profile == "none":
        augment.update({key: 0.0 for key in AUG_KEYS if key not in ["copy_paste_mode", "close_mosaic"]})
        augment["close_mosaic"] = 0
    augment.update(cfg["augment"]["overrides"])
    if profile == "none" and (
        requested["multi_scale"] or any(augment[key] != 0 for key in AUG_KEYS if key not in ["copy_paste_mode"])
    ):
        raise ValueError("none profile cannot enable random augmentations")
    for key in PROB_KEYS:
        if type(augment[key]) not in [int, float] or not 0 <= augment[key] <= 1:
            raise ValueError(f"augment.{key} must be in [0,1]")
    for key in ["degrees", "shear"]:
        if type(augment[key]) not in [int, float] or not 0 <= augment[key] <= 180:
            raise ValueError(f"Invalid augment.{key}")
    if type(augment["close_mosaic"]) is not int or not 0 <= augment["close_mosaic"] <= requested["epochs"]:
        raise ValueError("close_mosaic must be an integer between 0 and epochs")
    if augment["copy_paste_mode"] not in ["flip", "mixup"]:
        raise ValueError("copy_paste_mode must be flip or mixup")
    if augment["copy_paste"]:
        raise ValueError("Detection-only crack boxes have no segments for CopyPaste; nonzero request unsupported")
    if profile == "native" and augment["cutmix"]:
        raise ValueError("Locked native v8 has no CutMix; use b19/custom")
    if profile in ["b19", "custom"] and requested["rect"] and any(augment[k] for k in ["mosaic", "mixup", "cutmix"]):
        raise ValueError("rect training conflicts with shared mix augmentations")
    requested["close_mosaic"] = augment["close_mosaic"]
    for key in AUG_KEYS:
        if key in official:
            requested[key] = augment[key]
        sources["augment.effective." + key] = (
            "CLI_or_config_override"
            if key in cfg["augment"]["overrides"]
            else "baseline_args"
            if profile == "b19"
            else profile + "_profile"
        )
    for key in ["half", "rect", "augment", "fuse"]:
        if type(cfg["eval"][key]) is not bool:
            raise ValueError(f"eval.{key} must be boolean")
    if cfg["eval"]["augment"]:
        raise ValueError("Comparison evaluation uses no TTA")
    for key in ["conf", "iou", "display_conf"]:
        if type(cfg["eval"][key]) not in [int, float] or not 0 <= cfg["eval"][key] <= 1:
            raise ValueError(f"eval.{key} must be in [0,1]")
    if cfg["eval"]["display_conf"] < cfg["eval"]["conf"]:
        raise ValueError(
            "display_conf must be >= prediction collection conf; omitted detections cannot be reconstructed"
        )
    for key in ["batch", "max_det"]:
        if type(cfg["eval"][key]) is not int or cfg["eval"][key] < 1:
            raise ValueError(f"eval.{key} must be positive integer")
    if cfg["eval"]["split"] not in ["val", "test"]:
        raise ValueError("eval.split must be val or test")
    if cfg["benchmark"]["batch"] != 1:
        raise ValueError("Independent single-image benchmark requires batch=1")
    for key in ["half", "fuse"]:
        if type(cfg["benchmark"][key]) is not bool:
            raise ValueError(f"benchmark.{key} must be boolean")
    for key in ["imgsz", "warmup", "iterations"]:
        v = cfg["benchmark"][key]
        if type(v) is not int or v < (0 if key == "warmup" else 1):
            raise ValueError(f"Invalid benchmark.{key}")
    if type(cfg["package"]["max_bytes"]) is not int or cfg["package"]["max_bytes"] < 1:
        raise ValueError("package.max_bytes must be positive integer")
    cfg["name"] = (
        cfg["name"]
        or f"compare_yolov8n_b19_e{requested['epochs']}_i{requested['imgsz']}_b{requested['batch']}_s{requested['seed']}"
    )
    cfg["train"] = requested
    cfg["augment"]["effective"] = augment
    cfg["_provenance"] = {
        "user_config": raw,
        "official_defaults": official,
        "sources": sources,
        "baseline_args_status": "VERIFIED" if baseline_verified else "UNVERIFIED_PREVIEW_ONLY",
        "baseline_args_sha256": file_hash(baseline) if baseline_verified else None,
    }
    return cfg


def protocol(cfg, data_hash, training_augmentation=None):
    """Build model-independent protocol plus separate backend adapter and run hashes."""
    if training_augmentation is not None:
        augmentation = deepcopy(training_augmentation)
    else:
        augmentation = training_policy(cfg)
    public = {
        "id": cfg["protocol"],
        "data": data_hash,
        "augment": augmentation,
        "eval": {k: v for k, v in cfg["eval"].items() if k not in ["split", "fixed_images", "display_conf"]},
        "imgsz": cfg["train"]["imgsz"],
        "metrics_implementation": file_hash(SHARED / "b19/metrics.py"),
        "matching_implementation": digest(
            {name: file_hash(SHARED / name) for name in ["b19/matching.py", "evaluation.py"]}
        ),
    }
    return {
        "config": public,
        "protocol_sha256": digest(public),
        "run_config_sha256": digest(cfg),
        "augment_core_sha256": augmentation["implementation"],
        "adapter_sha256": digest({name: file_hash(HERE / name) for name in ["adapter.py", "runtime.py"]}),
    }


def training_policy(cfg):
    """Freeze the initial pipeline and its full close schedule, independent of later inference."""
    profile = cfg["augment"]["profile"]
    values = deepcopy(cfg["augment"]["effective"])
    extras = {"albumentations": {"enabled": False}}
    if profile in ["b19", "custom"]:
        core = digest(
            {
                "b19": {
                    name: file_hash(SHARED / "b19" / name)
                    for name in ["augment.py", "instance.py", "ops.py", "metrics.py"]
                },
                "provider": file_hash(SHARED / "provider.py"),
            }
        )
    else:
        core = digest(
            {
                name: file_hash(VENDOR / "ultralytics" / name)
                for name in ["data/augment.py", "data/dataset.py", "data/base.py", "utils/instance.py", "utils/ops.py"]
            }
        )
        if profile == "native":
            from adapter import ComparisonDataset

            values, extras = ComparisonDataset.native_policy(cfg)
    return {
        "profile": profile,
        "values": values,
        "requested_values": deepcopy(cfg["augment"]["effective"]),
        "implementation": core,
        "extras": extras,
        "training": {
            key: cfg["train"][key]
            for key in ["imgsz", "multi_scale", "batch", "rect", "cache", "epochs", "time", "seed"]
        },
        "close_schedule": "native_dynamic_epoch_budget" if cfg["train"]["time"] else "fixed_epoch_budget",
        "close_epoch_zero_based": cfg["train"]["epochs"] - values["close_mosaic"]
        if values["close_mosaic"] and not cfg["train"]["time"]
        else None,
    }


def result_directory(run, args, mode, split=None):
    """Share the checkpoint-owned result location between runtime, measurement and launcher."""
    return Path(args.result_dir or Path(run) / ("benchmark" if mode == "benchmark" else "evaluate_" + split)).resolve()


def bootstrap(cfg, pretrained=None, data_identity=None):
    """Import only the locked private package in each independent Python process."""
    lock = verify_backend()
    if "ultralytics" in sys.modules:
        raise RuntimeError("Use a fresh Python process; do not mix named ultralytics packages")
    cache = Path(cfg["cache_dir"]).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_AUTOINSTALL"] = "false"
    os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
    os.environ["YOLO_CONFIG_DIR"] = str(cache / "settings")
    os.environ["YOLO_COMPARISON_CACHE_DIR"] = str(cache / "yolov8n")
    os.environ["YOLO_COMPARISON_DATA_IDENTITY"] = data_identity or "no_dataset"
    if pretrained is not None:
        os.environ["YOLO_COMPARISON_PRETRAINED"] = str(Path(pretrained).resolve())
    sys.path.insert(0, str(VENDOR))
    sys.path.insert(0, str(HERE.parent))
    import ultralytics

    if (
        ultralytics.__version__ != lock["version"]
        or Path(ultralytics.__file__).resolve() != VENDOR / "ultralytics/__init__.py"
    ):
        raise RuntimeError("Unexpected backend import")
    print(
        json.dumps(
            {
                "ultralytics_version": ultralytics.__version__,
                "ultralytics_file": ultralytics.__file__,
                "upstream_sha": lock["commit"],
            },
            ensure_ascii=False,
        )
    )
    return ultralytics


def environment():
    """Measure local environment without describing it as server verification."""
    import torch
    import torchvision
    import ultralytics

    return {
        "sys_executable": sys.executable,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "backend_version": ultralytics.__version__,
        "backend_file": ultralytics.__file__,
        "host": platform.node(),
    }


def status(directory, phase, state, **details):
    """Record independent lifecycle states with a PID and UTC timestamps."""
    path = Path(directory) / f"{phase}_status.json"
    now = datetime.now(timezone.utc).isoformat()
    previous = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    record = {
        "phase": phase,
        "status": state,
        "pid": os.getpid(),
        "log": os.environ.get("YOLO_COMPARISON_LOG"),
        "timestamp_utc": now,
        "started_utc": now if state == "RUNNING" else previous.get("started_utc", now),
        "ended_utc": now if state in ["COMPLETED", "FAILED"] else None,
        **details,
    }
    write_json(path, record)
    with path.with_suffix(".jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
