"""Build and train b19 + DTR-C2PSA from audited original weights and the complete b19 recipe."""

# ruff: noqa: E402 -- Direct script execution must select this worktree and disable auto-install before imports.

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import random
import shutil
import sys
import tempfile
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ["YOLO_AUTOINSTALL"] = "false"

import numpy as np
import torch

import ultralytics
from ultralytics.cfg import get_cfg
from ultralytics.data.utils import IMG_FORMATS
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.modules.dtr import C2PSA_DTR
from ultralytics.nn.tasks import DetectionModel, load_checkpoint
from ultralytics.utils import DEFAULT_CFG_DICT, YAML
from ultralytics.utils.git import GitRepo

BASE_SHA = "4a1b167dee2b0c7353d9d7ead2911383d1ebf1d6"
WEIGHTS_SHA = "9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef"
DATA_SHA = "1f18760508e9dbf2332cd7102ee9c15e11e08f3d15fc4202c8ee0ed1bb785b12"
MODEL_YAML = Path(__file__).with_name("yolo26n-dtr.yaml")
BASELINE_ARGS = Path(__file__).with_name("baseline_args.yaml")
RUN_NAME = "dtr_c2psa_b19_e200_i640_b32_s42"
RUNTIME_FIELDS = {"model", "data", "pretrained", "project", "name", "save_dir", "device"}
REMOVED = "model.10.m."
ADDED = "model.10.dtr_blocks."


def sha256(path):
    """Hash a local artifact without loading weights or modifying it."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_weights(path):
    """Require the original, immutable yolo26n checkpoint for a new training run."""
    path = Path(path).resolve(strict=True)
    if sha256(path) != WEIGHTS_SHA:
        raise ValueError(f"Original yolo26n.pt SHA256 mismatch: {path}")
    return path


def rng_state():
    """Snapshot only initialized random sources, without initializing a CUDA context."""
    return {
        "cpu": torch.get_rng_state().clone(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
        "python": random.getstate(),
        "numpy": np.random.get_state(),
    }


def rng_equal(before, after):
    """Report exact equality for each random source independently."""
    return {
        "cpu": torch.equal(before["cpu"], after["cpu"]),
        "cuda": len(before["cuda"]) == len(after["cuda"])
        and all(torch.equal(a, b) for a, b in zip(before["cuda"], after["cuda"])),
        "python": before["python"] == after["python"],
        "numpy": all(np.array_equal(a, b) for a, b in zip(before["numpy"], after["numpy"])),
    }


def key_summary(model, keys):
    """Separate parameter elements, buffers, and tensor counts, retaining the full key inventory."""
    state, parameters = model.state_dict(), dict(model.named_parameters())
    keys = sorted(keys)
    return {
        "tensor_count": len(keys),
        "parameter_tensor_count": sum(k in parameters for k in keys),
        "buffer_tensor_count": sum(k not in parameters for k in keys),
        "parameter_elements": sum(state[k].numel() for k in keys if k in parameters),
        "buffer_elements": sum(state[k].numel() for k in keys if k not in parameters),
        "keys": keys,
    }


class DTRTrainer(DetectionTrainer):
    """Own the new-run initialization boundary; reuse native loss, optimizer, and training lifecycle."""

    max_oom_retries = 0  # Abort at the native retry owner before it mutates batch=32.

    def setup_model(self):
        """Give an explicit checkpoint ownership of its learned state, regardless of saved pretrained arguments."""
        if not isinstance(self.model, torch.nn.Module) and str(self.model).endswith(".pt"):
            weights, checkpoint = load_checkpoint(self.model)
            self.model = self.get_model(cfg=weights.yaml, weights=weights)
            return checkpoint
        return super().setup_model()

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Construct the nc=1 native reference, then replace only PSA state inside an RNG fork."""
        if weights is not None and isinstance(weights.model[10], C2PSA_DTR):
            return super().get_model(deepcopy(weights.yaml), weights, verbose)
        if weights is None:
            raise ValueError("A new DTR run requires the verified original yolo26n.pt")
        verify_weights(weights.pt_path)
        if self.data["nc"] != 1 or self.data["names"] != {0: "crack"} or self.data["channels"] != 3:
            raise ValueError("This experiment requires the original one-class, three-channel crack dataset")
        reference = super().get_model(deepcopy(weights.yaml), weights, verbose=verbose)
        before = rng_state()
        try:
            with torch.random.fork_rng(devices=list(range(len(before["cuda"])))):
                model = self.set_model_names_for_load(
                    DetectionModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=False)
                )
        finally:
            random.setstate(before["python"])
            np.random.set_state(before["numpy"])
        old, new, source = reference.state_dict(), model.state_dict(), weights.state_dict()
        removed = {k for k in old if k.startswith(REMOVED)}
        added = set(new) - set(old)
        retained = set(old) - removed
        if set(old) - set(new) != removed or added != {k for k in new if k.startswith(ADDED)}:
            raise RuntimeError("Replacement changed state outside model.10.m / model.10.dtr_blocks")
        old_layers = reference.yaml["backbone"] + reference.yaml["head"]
        new_layers = model.yaml["backbone"] + model.yaml["head"]
        if len(old_layers) != len(new_layers) or any(
            a != b for i, (a, b) in enumerate(zip(old_layers, new_layers)) if i != 10
        ):
            raise RuntimeError("Unexpected architecture change outside layer 10")
        old_modules, new_modules = dict(reference.named_modules()), dict(model.named_modules())
        for key in retained:
            parent = key.rsplit(".", 1)[0]
            if old[key].shape != new[key].shape or type(old_modules[parent]) is not type(new_modules[parent]):
                raise RuntimeError(f"Retained key changed shape or module semantics: {key}")
        model.load_state_dict({**new, **{k: old[k] for k in retained}}, strict=True)
        inconsistent = [
            k for k in retained if not torch.equal(old[k], new[k]) or old[k].data_ptr() == new[k].data_ptr()
        ]
        if inconsistent:
            raise RuntimeError(f"Retained state mismatch or shared storage: {inconsistent}")
        loaded = {k for k in retained if k in source and source[k].shape == old[k].shape}
        if any(not torch.equal(old[k], source[k]) for k in loaded):
            raise RuntimeError("Reference state differs from compatible original pretrained tensors")
        gaps = retained - loaded
        self.migration_report = {
            "pretrained_retained": key_summary(reference, loaded),
            "native_nc80_to_nc1_gaps": key_summary(reference, gaps),
            "intentionally_removed": key_summary(reference, removed),
            "newly_initialized": key_summary(model, added),
            "retained_total": key_summary(reference, retained),
            "unexpected_missing_or_inconsistent": inconsistent,
            "rng_after_reference_unchanged": rng_equal(before, rng_state()),
            "reference_cpu_rng_sha256": hashlib.sha256(before["cpu"].numpy().tobytes()).hexdigest(),
            "reference_yaml": reference.yaml,
        }
        if not all(self.migration_report["rng_after_reference_unchanged"].values()):
            raise RuntimeError("DTR construction advanced the training RNG")
        return model


def environment():
    """Record the actual local environment and differences from the documented server baseline."""
    actual = {
        "python": platform.python_version(),
        "executable": sys.executable,
        "torch": torch.__version__,
        "torchvision": importlib.metadata.version("torchvision"),
        "cuda_runtime": torch.version.cuda,
        "ultralytics": ultralytics.__version__,
        "ultralytics_path": str(Path(ultralytics.__file__).resolve()),
        "albumentations_installed": importlib.util.find_spec("albumentations") is not None,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    expected = {
        "python": "3.12.3",
        "executable": "/root/miniconda3/bin/python",
        "torch": "2.8.0+cu128",
        "torchvision": "0.23.0+cu128",
        "cuda_runtime": "12.8",
        "ultralytics": "8.4.98",
        "albumentations_installed": False,
        "gpu": "NVIDIA GeForce RTX 4090",
    }
    if Path(ultralytics.__file__).resolve() != ROOT / "ultralytics" / "__init__.py":
        raise RuntimeError("Ultralytics must be imported from this experiment worktree")
    git = GitRepo(ROOT)
    return {
        "actual": actual,
        "source": {"commit": git.commit, "branch": git.branch, "worktree": str(git.root), "baseline_sha": BASE_SHA},
        "server_differences": {k: {"expected": v, "actual": actual[k]} for k, v in expected.items() if actual[k] != v},
        "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions() if d.metadata["Name"]},
        "dependencies_fully_locked": False,
    }


def audit_data(path):
    """Check the existing split inventory without inference, rewriting YAML, or creating dataset caches."""
    path = Path(path).resolve(strict=True)
    data = YAML.load(path)
    expected = {"train": "images/train", "val": "images/val", "test": "images/test", "nc": 1, "names": {0: "crack"}}
    if {k: v for k, v in data.items() if k != "path"} != expected:
        raise ValueError("Dataset YAML differs from the locked crack split (only root path relocation is allowed)")
    root = Path(data.get("path", path.parent))
    if not root.is_absolute():
        root = (ROOT / root).resolve()
    counts, boxes = {}, {}
    for split, expected_count in {"train": 8414, "val": 2404, "test": 1202}.items():
        images = [p for p in (root / data[split]).rglob("*") if p.is_file() and p.suffix[1:].lower() in IMG_FORMATS]
        counts[split] = len(images)
        if counts[split] != expected_count:
            raise ValueError(f"{split} image count {counts[split]} != {expected_count}")
        if split != "train":
            boxes[split] = sum(
                sum(bool(line.strip()) for line in p.read_text().splitlines())
                for p in (root / "labels" / split).rglob("*.txt")
            )
    if boxes != {"val": 2985, "test": 1477}:
        raise ValueError(f"Unexpected label counts: {boxes}")
    digest = sha256(path)
    return {
        "path": str(path),
        "sha256": digest,
        "historical_sha256_match": digest == DATA_SHA,
        "images": counts,
        "boxes": boxes,
        "content_identity": "YAML and counts checked; no full dataset content hash",
    }


def resolve_recipe(args):
    """Validate every mechanism field against the archived full recipe and alter only runtime paths."""
    baseline = YAML.load(args.baseline_args)
    locked = YAML.load(BASELINE_ARGS)
    differences = {
        k: (locked.get(k), baseline.get(k))
        for k in locked.keys() | baseline.keys()
        if k not in RUNTIME_FIELDS and (k not in baseline or k not in locked or baseline[k] != locked[k])
    }
    if differences or set(DEFAULT_CFG_DICT) - set(baseline):
        raise ValueError(f"Incomplete or changed b19 training recipe: {differences}")
    weights = verify_weights(args.weights)
    resolved = {
        **baseline,
        "model": str(MODEL_YAML),
        "pretrained": str(weights),
        "data": str(Path(args.data).resolve(strict=True)),
        "project": str(Path(args.project).resolve()),
        "name": args.name,
        "device": args.device,
    }
    resolved["save_dir"] = str(Path(resolved["project"]) / args.name)
    resolved = vars(get_cfg(overrides=resolved)) | {"save_dir": resolved["save_dir"]}
    diff = {
        k: {"baseline": baseline.get(k), "resolved": resolved.get(k)}
        for k in baseline.keys() | resolved.keys()
        if baseline.get(k) != resolved.get(k)
    }
    if set(diff) - RUNTIME_FIELDS:
        raise ValueError(f"Unexpected training mechanism changes: {diff}")
    return resolved, {
        "baseline_file": str(Path(args.baseline_args).resolve()),
        "baseline_sha256": sha256(args.baseline_args),
        "baseline": baseline,
        "resolved": resolved,
        "diff": diff,
    }


def preflight(args):
    """Audit configuration, existing data, weights, and environment before any output directory is reserved."""
    resolved, recipe = resolve_recipe(args)
    if Path(resolved["save_dir"]).exists():
        raise FileExistsError(f"Output already exists; resume and overwrite are disabled: {resolved['save_dir']}")
    if args.report:
        report_path, run_path = Path(args.report).resolve(), Path(resolved["save_dir"]).resolve()
        if report_path == run_path or run_path in report_path.parents:
            raise ValueError("The audit report must be outside the formal training output directory")
    report = {
        "base_sha": BASE_SHA,
        "formal_training": "NOT_STARTED",
        "recipe": recipe,
        "environment": environment(),
        "data": audit_data(args.data),
        "weights_sha256": WEIGHTS_SHA,
    }
    return resolved, report


def setup_dry_run(resolved, scratch):
    """Exercise the real Trainer constructor and setup_model/get_model chain in a temporary output tree."""
    overrides = {**resolved, "project": str(scratch), "name": "dry_run", "save_dir": str(Path(scratch) / "dry_run")}
    trainer = DTRTrainer(overrides=overrides)
    trainer.setup_model()
    trainer.set_model_attributes()
    return trainer


def check_training_contract(trainer):
    """Fail after native AMP/setup checks if actual training would violate the fixed experimental recipe."""
    if (
        trainer.batch_size != 32
        or trainer.args.imgsz != 640
        or not trainer.amp
        or type(trainer.optimizer).__name__ != "MuSGD"
    ):
        raise RuntimeError("Required actual training state is batch=32, imgsz=640, AMP enabled, and MuSGD")


def write_report(path, report):
    """Write an explicit report outside Git; callers choose its location."""
    if path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def main():
    """Run a read-only dry build or the explicitly invoked, fixed 200-epoch training entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--baseline-args", required=True)
    parser.add_argument("--project", default=str(ROOT / "runs" / "detect"))
    parser.add_argument("--name", default=RUN_NAME)
    parser.add_argument("--device", default="0")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report")
    args = parser.parse_args()
    resolved, report = preflight(args)
    if args.dry_run:
        with tempfile.TemporaryDirectory(prefix="dtr-dry-") as scratch:
            trainer = setup_dry_run(resolved, scratch)
            report["migration"] = trainer.migration_report
            report["setup_chain"] = (
                "DTRTrainer.__init__ -> DTRTrainer.setup_model -> BaseTrainer.setup_model -> DTRTrainer.get_model"
            )
            report["parameters_unfused"] = sum(p.numel() for p in trainer.model.parameters())
        write_report(args.report, report)
        print(
            json.dumps(
                {
                    "formal_training": "NOT_STARTED",
                    "parameters": report["parameters_unfused"],
                    "server_environment_differences": report["environment"]["server_differences"],
                    "report": args.report,
                },
                indent=2,
            )
        )
        return
    if report["environment"]["server_differences"]:
        write_report(args.report, report)
        raise RuntimeError(
            f"Server environment differs; no training started: {report['environment']['server_differences']}"
        )
    write_report(args.report, report)
    local_weights = ROOT / "yolo26n.pt"
    if local_weights.exists():
        verify_weights(local_weights)
    else:
        shutil.copyfile(resolved["pretrained"], local_weights)
        verify_weights(local_weights)
    os.chdir(ROOT)  # The native AMP check reuses the verified, local yolo26n.pt.
    Path(resolved["save_dir"]).mkdir(parents=True, exist_ok=False)
    write_report(Path(resolved["save_dir"]) / "dtr_preflight.json", report)
    trainer = DTRTrainer(overrides=resolved)
    trainer.add_callback("on_pretrain_routine_end", check_training_contract)
    trainer.add_callback(
        "on_pretrain_routine_end", lambda t: write_report(t.save_dir / "dtr_migration.json", t.migration_report)
    )
    trainer.train()


if __name__ == "__main__":
    main()
