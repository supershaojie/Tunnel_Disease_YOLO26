"""Build the fixed b19 SWR experiment and audit native-state/RNG inheritance before training."""

from __future__ import annotations

# ruff: noqa: E402 - bootstrap this worktree and disable auto-install before Ultralytics imports.

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import pickle
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
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.modules.swr import SWRFusion
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import YAML

BASE_SHA = "4a1b167dee2b0c7353d9d7ead2911383d1ebf1d6"
WEIGHT_SHA = "9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef"
DATA_SHA = "1f18760508e9dbf2332cd7102ee9c15e11e08f3d15fc4202c8ee0ed1bb785b12"
MODEL_YAML = ROOT / "experiments/swr_fusion/yolo26n-swr.yaml"
RECIPE = ROOT / "experiments/swr_fusion/b19_recipe.yaml"
RUN_NAME = "swr_fusion_b19_e200_i640_b32_s42"
RUNTIME_KEYS = {"model", "pretrained", "data", "project", "name", "save_dir", "device"}


def sha256(path: Path) -> str:
    """Hash a local file without downloading or modifying it."""
    with Path(path).open("rb") as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rng_digest() -> dict:
    """Snapshot CPU, Python, NumPy and already-initialized CUDA RNG states without consuming randomness."""
    return {
        "cpu": hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
        "python": hashlib.sha256(repr(random.getstate()).encode()).hexdigest(),
        "numpy": hashlib.sha256(pickle.dumps(np.random.get_state(), protocol=4)).hexdigest(),
        "cuda": [hashlib.sha256(s.cpu().numpy().tobytes()).hexdigest() for s in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_initialized()
        else [],
    }


def state_inventory(model: torch.nn.Module, keys: set) -> dict:
    """Separate state tensors, parameter elements and buffers, retaining the complete key lists."""
    params = dict(model.named_parameters())
    state = model.state_dict()
    return {
        "tensors": len(keys),
        "parameter_tensors": len(keys & params.keys()),
        "parameter_elements": sum(params[k].numel() for k in keys & params.keys()),
        "buffer_tensors": len(keys - params.keys()),
        "buffer_elements": sum(state[k].numel() for k in keys - params.keys()),
        "keys": sorted(keys),
    }


class SWRTrainer(DetectionTrainer):
    """Own fresh-training initialization while reusing the native detection training machinery."""

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Inherit native nc=1 state and leave extra module construction outside training RNG."""
        if weights is None:
            raise ValueError("Fresh SWR training requires the verified original yolo26n.pt")
        if isinstance(weights.model[16], SWRFusion):
            # A learned SWR checkpoint follows native loading, never original-weight initialization.
            return super().get_model(cfg=deepcopy(weights.yaml), weights=weights, verbose=verbose)
        if sha256(Path(weights.pt_path)) != WEIGHT_SHA:
            raise ValueError("SWR initialization only accepts the original yolo26n.pt SHA256")
        if self.data["nc"] != 1 or self.data["names"] != {0: "crack"}:
            raise ValueError("This experiment requires the single class {0: 'crack'}")
        reference = super().get_model(cfg=deepcopy(weights.yaml), weights=weights, verbose=False)
        after_reference = rng_digest()
        # All constructors in this fixed graph use CPU torch RNG only; assertions also audit other sources.
        with torch.random.fork_rng(devices=[]):
            model = self.set_model_names_for_load(
                DetectionModel(str(MODEL_YAML), nc=1, ch=self.data["channels"], verbose=verbose)
            )
        if rng_digest() != after_reference:
            raise RuntimeError("SWR construction changed the native post-construction RNG state")
        ref_graph = reference.yaml["backbone"] + reference.yaml["head"]
        new_graph = model.yaml["backbone"] + model.yaml["head"]
        for index, (old, new) in enumerate(zip(reference.model, model.model)):
            if index in {14, 15, 16}:
                continue
            if ref_graph[index] != new_graph[index] or old.f != new.f:
                raise RuntimeError(f"Unexpected graph change at layer {index}")
            if [(k, type(v)) for k, v in old.named_modules()] != [(k, type(v)) for k, v in new.named_modules()]:
                raise RuntimeError(f"Unexpected module semantics at layer {index}")
        src, dst, pretrained = reference.state_dict(), model.state_dict(), weights.state_dict()
        retained = {k for k in src if not k.startswith("model.16.")}
        added = {k for k in dst if k.startswith("model.16.")}
        if dst.keys() - added != retained or any(src[k].shape != dst[k].shape for k in retained):
            raise RuntimeError("Unexpected state change outside model.16")
        with torch.no_grad():
            for key in retained:
                dst[key].copy_(src[key])
        inconsistent = {
            k for k in retained if not torch.equal(src[k], dst[k]) or src[k].data_ptr() == dst[k].data_ptr()
        }
        if inconsistent:
            raise RuntimeError(f"Retained state mismatch or shared storage: {sorted(inconsistent)}")
        transferred = {k for k in retained if k in pretrained and pretrained[k].shape == src[k].shape}
        if any(not torch.equal(src[k], pretrained[k]) for k in transferred):
            raise RuntimeError(
                "Unexpected native pretrained inheritance (class remapping must not match crack to COCO)"
            )
        self.initialization_report = {
            "baseline_sha": BASE_SHA,
            "pretrained_retained": state_inventory(reference, transferred),
            "native_nc_adaptation_retained": state_inventory(reference, retained - transferred),
            "removed_native": state_inventory(reference, src.keys() - retained),
            "initialized_swr": state_inventory(model, added),
            "retained_total": state_inventory(reference, retained),
            "unexpected": state_inventory(model, inconsistent),
            "retained_bitwise_equal": True,
            "independent_storage": True,
            "rng_after_native": after_reference,
            "rng_after_swr": rng_digest(),
            "rng_equal": after_reference == rng_digest(),
        }
        return model


def disable_oom_retry(trainer: SWRTrainer):
    """Use the pinned trainer's exhausted retry budget before each epoch; OOM propagates without batch reduction."""
    trainer._oom_retries = 3


def check_runtime(trainer: SWRTrainer):
    """Verify the native AMP decision and fixed recipe before the first training batch."""
    if trainer.batch_size != 32 or not bool(trainer.amp) or type(trainer.optimizer).__name__ != "MuSGD":
        raise RuntimeError("Formal SWR training requires actual batch=32, AMP enabled and native MuSGD")
    if trainer.args.imgsz != 640 or trainer.args.epochs != 200:
        raise RuntimeError("Formal SWR training requires imgsz=640 and epochs=200")


def environment_report() -> dict:
    """Record actual dependencies and explicit differences from the known server environment."""
    expected = {
        "python": "3.12.3",
        "torch": "2.8.0+cu128",
        "torchvision": "0.23.0+cu128",
        "cuda_runtime": "12.8",
        "ultralytics": "8.4.98",
        "albumentations": None,
        "interpreter": "/root/miniconda3/bin/python",
        "gpu": "NVIDIA GeForce RTX 4090",
    }
    actual = {
        "python": platform.python_version(),
        "interpreter": sys.executable,
        "torch": torch.__version__,
        "torchvision": importlib.metadata.version("torchvision"),
        "cuda_runtime": torch.version.cuda,
        "ultralytics": ultralytics.__version__,
        "albumentations": importlib.metadata.version("albumentations")
        if importlib.util.find_spec("albumentations")
        else None,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    if Path(ultralytics.__file__).resolve() != ROOT / "ultralytics/__init__.py":
        raise RuntimeError("Ultralytics must be imported from this experiment worktree")
    return {
        "actual": actual,
        "import_path": ultralytics.__file__,
        "differences": {k: {"expected": v, "actual": actual[k]} for k, v in expected.items() if actual[k] != v},
        "dependencies": {
            d.metadata["Name"]: d.version for d in importlib.metadata.distributions() if d.metadata["Name"]
        },
    }


def resolve_config(args: argparse.Namespace) -> tuple[dict, dict]:
    """Read the complete archived recipe, reject mechanism drift and change only explicit runtime paths."""
    baseline = YAML.load(args.baseline_args)
    archived = YAML.load(RECIPE)
    if baseline.keys() != archived.keys():
        raise ValueError("baseline-args must contain the same complete key set as the archived b19 args.yaml")
    mismatches = {
        k: [archived[k], baseline[k]] for k in archived if k not in RUNTIME_KEYS and archived[k] != baseline[k]
    }
    if mismatches:
        raise ValueError(f"Baseline training-mechanism differences: {mismatches}")
    weights, data = Path(args.weights).resolve(), Path(args.data).resolve()
    if sha256(weights) != WEIGHT_SHA or sha256(data) != DATA_SHA:
        raise ValueError("Original weights or dataset YAML SHA256 differs from the locked b19 input")
    output = Path(args.project).resolve() / args.name
    if output.exists():
        raise FileExistsError(f"Output already exists; no overwrite or resume: {output}")
    resolved = dict(
        baseline,
        model=str(MODEL_YAML),
        pretrained="yolo26n.pt",
        data=str(data),
        project=str(output.parent),
        name=output.name,
        save_dir=str(output),
        device=str(args.device),
    )
    # Validate against this pinned source, never synthesize a recipe from newer defaults.
    parsed = vars(get_cfg(overrides=resolved))
    differences = {k: {"baseline": baseline[k], "experiment": parsed[k]} for k in baseline if baseline[k] != parsed[k]}
    if differences.keys() - RUNTIME_KEYS:
        raise ValueError(f"Unexpected resolved differences: {differences}")
    local_weight = ROOT / "yolo26n.pt"
    if local_weight.exists():
        if sha256(local_weight) != WEIGHT_SHA:
            raise ValueError(f"Existing AMP-check weight is not the original: {local_weight}")
    else:
        shutil.copyfile(weights, local_weight)
    return parsed, {
        "baseline_args_path": str(Path(args.baseline_args).resolve()),
        "baseline_args_sha256": sha256(Path(args.baseline_args)),
        "baseline": baseline,
        "resolved": parsed,
        "differences": differences,
        "source_weights": str(weights),
        "weights_sha256": sha256(weights),
        "data_yaml_sha256": sha256(data),
        "formal_training": "NOT_STARTED",
    }


def dataset_inventory(data: dict) -> dict:
    """Check the locked split counts using file metadata and label text, without image inference."""
    from ultralytics.data.utils import IMG_FORMATS, img2label_paths

    result = {}
    for split in ("train", "val", "test"):
        images = sorted(p for p in Path(data[split]).rglob("*") if p.suffix[1:].lower() in IMG_FORMATS)
        result[split] = {"path": data[split], "images": len(images)}
        if split != "train":
            labels = (Path(p) for p in img2label_paths([str(p) for p in images]))
            result[split]["boxes"] = sum(
                sum(bool(line.strip()) for line in p.read_text(encoding="utf-8").splitlines())
                for p in labels
                if p.is_file()
            )
    expected = {"train": (8414, None), "val": (2404, 2985), "test": (1202, 1477)}
    if any((result[s]["images"], result[s].get("boxes")) != counts for s, counts in expected.items()):
        raise ValueError(f"Dataset counts differ from the locked b19 splits: {result}")
    return result


def main():
    """Run a temporary setup-only audit or explicitly requested formal training."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--project", default=str(ROOT / "runs/detect"))
    parser.add_argument("--name", default=RUN_NAME)
    parser.add_argument("--device", default="0")
    parser.add_argument("--baseline-args", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)  # Native AMP checks resolve the verified local yolo26n.pt here.
    config, report = resolve_config(args)
    report["environment"] = environment_report()
    if args.dry_run:
        with tempfile.TemporaryDirectory(prefix="swr-preflight-") as temp:
            audit_config = dict(config, project=temp, name="setup-only", save_dir=str(Path(temp) / "setup-only"))
            trainer = SWRTrainer(overrides=audit_config)
            trainer.setup_model()  # Actual production setup_model -> overridden get_model.
            report["dataset"] = dataset_inventory(trainer.data)
            report["initialization"] = trainer.initialization_report
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    if report["environment"]["differences"]:
        raise RuntimeError(f"Formal server environment differs: {report['environment']['differences']}")
    if args.device != "0":
        raise ValueError("The fixed formal run uses device=0; other devices are available for dry-run only")
    # Reserve the exact output once, before native get_save_dir can silently choose an incremented name.
    output = Path(config["save_dir"])
    output.mkdir(parents=True, exist_ok=False)
    trainer = SWRTrainer(overrides=config)
    report["dataset"] = dataset_inventory(trainer.data)
    trainer.setup_model()
    report["initialization"] = trainer.initialization_report
    (output / "preflight.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    trainer.add_callback("on_train_epoch_start", disable_oom_retry)
    trainer.add_callback("on_train_start", check_runtime)
    trainer.train()


if __name__ == "__main__":
    main()
