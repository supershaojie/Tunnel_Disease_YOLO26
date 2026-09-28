"""Build and train the fixed b19 SCE experiment using an audited native initialization path."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import pickle
import random
import shutil
import subprocess
import sys
import tempfile
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ["YOLO_AUTOINSTALL"] = "false"

import numpy as np
import torch

import ultralytics
from ultralytics.cfg import get_cfg
from ultralytics.data.utils import IMG_FORMATS
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.modules import Detect, SCEFusion
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import DEFAULT_CFG_DICT, YAML
from ultralytics.utils.torch_utils import unwrap_model

ROOT = Path(__file__).resolve().parents[2]
BASE_SHA = "4a1b167dee2b0c7353d9d7ead2911383d1ebf1d6"
WEIGHTS_SHA = "9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef"
DATA_SHA = "1f18760508e9dbf2332cd7102ee9c15e11e08f3d15fc4202c8ee0ed1bb785b12"
MODEL_YAML = ROOT / "experiments/sce_fusion/yolo26n-sce.yaml"
RUN_NAME = "sce_fusion_b19_e200_i640_b32_s42"
RUNTIME_FIELDS = {"model", "pretrained", "data", "project", "name", "save_dir", "device"}


def sha256(path):
    """Hash a local artifact without loading it as executable content."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_report(path, report):
    """Write a JSON report to an explicit file, creating only its parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str, allow_nan=False) + "\n", encoding="utf-8"
    )


def environment():
    """Record the actual import, source revision, dependencies and available device."""
    if Path(ultralytics.__file__).resolve() != ROOT / "ultralytics/__init__.py":
        raise RuntimeError(f"Wrong Ultralytics import: {ultralytics.__file__}")
    versions = {}
    for name in ("torch", "torchvision", "numpy", "opencv-python", "Pillow", "PyYAML", "albumentations"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "NOT_INSTALLED"
    return {
        "python": sys.version,
        "executable": sys.executable,
        "ultralytics": ultralytics.__version__,
        "import": ultralytics.__file__,
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "base_commit": BASE_SHA,
        "dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)),
        "dependencies": versions,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "server_environment": "UNVERIFIED: no server connection is made by these scripts",
    }


def verified_weights(path):
    """Require the original local YOLO26n checkpoint; never download a replacement."""
    path = Path(path).expanduser().resolve(strict=True)
    if sha256(path) != WEIGHTS_SHA:
        raise ValueError(f"Original yolo26n.pt SHA256 mismatch: {path}")
    return path


def prepare_amp_weights(path):
    """Provide the verified original under the name used by the unmodified native AMP check."""
    path = verified_weights(path)
    local = ROOT / "yolo26n.pt"
    if local.exists():
        verified_weights(local)
    else:
        with path.open("rb") as source, local.open("xb") as target:
            shutil.copyfileobj(source, target)
        verified_weights(local)
    return path


def audit_data(path):
    """Check the local single-class split and record provenance without running validation."""
    path = Path(path).resolve(strict=True)
    data = YAML.load(path)
    names = data.get("names")
    if names not in ({0: "crack"}, ["crack"]) or data.get("nc", 1) != 1:
        raise ValueError("SCE requires the original single-class crack dataset")
    expected = {"train": "images/train", "val": "images/val", "test": "images/test", "nc": 1, "names": {0: "crack"}}
    differences = {k: {"historical": expected.get(k), "actual": v} for k, v in data.items() if expected.get(k) != v}
    if set(differences) - {"path", "train", "val", "test"}:
        raise ValueError(f"Non-path dataset YAML changes: {differences}")
    root = Path(data.get("path", path.parent))
    if not root.is_absolute():
        root = (path.parent / root).resolve()
    splits = {}
    for split, count in (("train", 8414), ("val", 2404), ("test", 1202)):
        directory = root / data[split]
        images = sorted(p for p in directory.rglob("*") if p.suffix[1:].lower() in IMG_FORMATS)
        if len(images) != count:
            raise ValueError(f"{split} has {len(images)} images, expected {count}: {directory}")
        boxes, digest = 0, hashlib.sha256()
        for image in images:
            label = Path(str(image).replace(f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}")).with_suffix(".txt")
            content = label.read_bytes()
            for line in content.decode("utf-8").splitlines():
                fields = line.split()
                if fields:
                    if len(fields) != 5 or float(fields[0]) != 0:
                        raise ValueError(f"Invalid crack detection label: {label}")
                    boxes += 1
            digest.update(image.relative_to(directory).as_posix().encode() + b"\0" + content + b"\n")
        if split in {"val", "test"} and boxes != {"val": 2985, "test": 1477}[split]:
            raise ValueError(f"Unexpected {split} box count: {boxes}")
        splits[split] = {
            "images": len(images),
            "boxes": boxes,
            "labels_sha256": digest.hexdigest(),
            "path": str(directory),
        }
    return {
        "path": str(path),
        "sha256": sha256(path),
        "historical_sha256": DATA_SHA,
        "yaml_byte_match": sha256(path) == DATA_SHA,
        "path_differences": differences,
        "splits": splits,
        "server_image_byte_identity": "UNVERIFIED: split counts and local label digests are not remote image hashes",
    }


def configuration(options):
    """Read the entire baseline recipe and change only declared runtime paths/device."""
    original = YAML.load(Path(options.baseline_args).resolve(strict=True))
    missing = set(DEFAULT_CFG_DICT) - set(original)
    if missing:
        raise ValueError(f"Incomplete baseline args.yaml, missing: {sorted(missing)}")
    required = {
        "epochs": 200,
        "imgsz": 640,
        "batch": 32,
        "seed": 42,
        "optimizer": "MuSGD",
        "amp": True,
        "deterministic": True,
        "resume": False,
        "exist_ok": False,
        "cls_remap": True,
        "quantize": None,
    }
    for key, value in required.items():
        if original.get(key) != value:
            raise ValueError(f"b19 recipe mismatch: {key}={original.get(key)!r}, expected {value!r}")
    weights = verified_weights(options.weights)
    final = original.copy()
    final.update(
        model=str(MODEL_YAML),
        pretrained="yolo26n.pt",
        data=str(Path(options.data).resolve(strict=True)),
        project=str(Path(options.project).resolve()),
        name=options.name,
    )
    final.pop("save_dir", None)
    if options.device is not None:
        final["device"] = options.device
    if "," in str(final["device"]):
        raise ValueError("This fixed b19 experiment uses one device; multi-GPU/DDP is outside its scope")
    final = vars(get_cfg(overrides=final))
    changes = {
        k: {"baseline": original.get(k), "resolved": final.get(k)}
        for k in original.keys() | final.keys()
        if original.get(k) != final.get(k)
    }
    illegal = set(changes) - RUNTIME_FIELDS - (set(final) - set(original))
    if illegal:
        raise ValueError(f"Unexpected training mechanism changes: {sorted(illegal)}")
    return final, {
        "baseline_path": str(Path(options.baseline_args).resolve()),
        "baseline_sha256": sha256(options.baseline_args),
        "original": original,
        "resolved": final.copy(),
        "differences": changes,
        "native_defaults_absent_from_archive": {k: final[k] for k in set(final) - set(original)},
        "weights": str(weights),
        "weights_sha256": WEIGHTS_SHA,
    }


def rng_fingerprint():
    """Fingerprint all RNGs already in use without initializing a new CUDA RNG."""
    states = {
        "cpu": torch.get_rng_state().numpy().tobytes(),
        "python": pickle.dumps(random.getstate()),
        "numpy": pickle.dumps(np.random.get_state()),
    }
    if torch.cuda.is_initialized():
        states.update({f"cuda:{i}": v.cpu().numpy().tobytes() for i, v in enumerate(torch.cuda.get_rng_state_all())})
    return {k: hashlib.sha256(v).hexdigest() for k, v in states.items()}


def remap_key(key):
    """Rename only the complete native Detect prefix."""
    return "model.27." + key[len("model.23.") :] if key.startswith("model.23.") else key


def state_inventory(model, keys):
    """Separate state tensor count from parameter elements, excluding buffers from the latter."""
    parameters = dict(model.named_parameters())
    return {
        "tensor_count": len(keys),
        "parameter_elements": sum(parameters[k].numel() for k in keys if k in parameters),
        "keys": sorted(keys),
    }


class SCETrainer(DetectionTrainer):
    """Own reference construction, graph-state migration and the fixed-batch experiment policy."""

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Use native nc adaptation first; isolate all extra construction RNG consumption."""
        if weights is None:
            raise ValueError("SCE training requires the SHA-verified original checkpoint")
        if any(isinstance(m, SCEFusion) for m in weights.modules()):
            # Native checkpoint reconstruction already uses new graph keys. Never remap a trained SCE model.
            return super().get_model(cfg=deepcopy(weights.yaml), weights=weights, verbose=verbose)
        verified_weights(weights.pt_path)
        if self.data["nc"] != 1 or self.data["channels"] != 3 or type(weights.model[-1]) is not Detect:
            raise ValueError("Expected RGB single-class adaptation of the native YOLO26n Detect checkpoint")
        # Capture the common native setup boundary, after dataset/callback initialization and checkpoint loading.
        self.reference_rng = (
            random.getstate(),
            np.random.get_state(),
            torch.get_rng_state(),
            torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
        )
        self.reference_verbose = verbose
        reference = super().get_model(cfg=deepcopy(weights.yaml), weights=weights, verbose=verbose)
        before = rng_fingerprint()
        python_rng, numpy_rng = random.getstate(), np.random.get_state()
        try:
            with torch.random.fork_rng(
                devices=list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
            ):
                model = self.set_model_names_for_load(DetectionModel(str(MODEL_YAML), nc=1, ch=3, verbose=verbose))
        finally:
            random.setstate(python_rng)
            np.random.set_state(numpy_rng)
        source, destination = reference.state_dict(), model.state_dict()
        mapping = {key: remap_key(key) for key in source}
        if len(set(mapping.values())) != len(source):
            raise RuntimeError("Native state mapping is not one-to-one")
        for key, target in mapping.items():
            if target not in destination or source[key].shape != destination[target].shape:
                raise RuntimeError(f"Native state mismatch: {key} -> {target}")
        model.load_state_dict({target: source[key] for key, target in mapping.items()}, strict=False)
        if any(
            not torch.equal(source[k], destination[v]) or source[k].data_ptr() == destination[v].data_ptr()
            for k, v in mapping.items()
        ):
            raise RuntimeError("Native state must be equal without shared mutable storage")
        added = set(destination) - set(mapping.values())
        if any(not key.startswith("model.23.") for key in added):
            raise RuntimeError("Unexpected non-SCE state in the new graph")
        pretrained = weights.state_dict()
        retained = {
            k
            for k in source
            if k in pretrained and source[k].shape == pretrained[k].shape and torch.equal(source[k], pretrained[k])
        }
        adapted = set(source) - retained
        if any(not k.startswith("model.23.") or "cv3" not in k for k in adapted):
            raise RuntimeError("Unexpected pretrained gap outside class-adapted detection branches")
        after = rng_fingerprint()
        if before != after:
            raise RuntimeError("SCE construction changed the native training RNG")
        self.transfer_report = {
            "status": "PASS",
            "original_pretrained_retained": state_inventory(reference, retained),
            "native_class_adaptation": state_inventory(reference, adapted),
            "detect_renamed": state_inventory(reference, {k for k in source if k.startswith("model.23.")}),
            "native_total": state_inventory(reference, source),
            "new_sce": state_inventory(model, added),
            "intentional_removals": [],
            "unexpected_omissions": [],
            "mapping": mapping,
            "all_native_values_equal": True,
            "shared_native_storage": False,
            "rng_after_reference": before,
            "rng_after_sce": after,
        }
        return model

    def build_optimizer(self, model, *args, **kwargs):
        """Preserve native b19 MuSGD head-name grouping despite Detect's graph relocation."""
        graph = unwrap_model(model).model
        # This temporary name-only view is never executed or stored on the training model.
        view = torch.nn.Module()
        view.model = torch.nn.ModuleList([*graph[:23], graph[27], graph[23]])
        return super().build_optimizer(view, *args, **kwargs)

    @property
    def _oom_retries(self):
        """No retries have occurred because this experiment never retries a smaller batch."""
        return 0

    @_oom_retries.setter
    def _oom_retries(self, value):
        """Reject the native retry request before it changes batch size or rebuilds the pipeline."""
        if value:
            raise RuntimeError("SCE fixed batch=32: CUDA memory failure; automatic batch reduction is disabled")


def parser(verify=False):
    """Expose the documented train/verify arguments."""
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--data", required=True)
    result.add_argument("--weights", required=True)
    result.add_argument("--baseline-args", required=True)
    result.add_argument("--device", default=None)
    result.add_argument("--report", required=verify, help="Explicit JSON report file; parent directories are created")
    if not verify:
        result.add_argument("--project", default=str(ROOT / "runs/detect/sce_fusion"))
        result.add_argument("--name", default=RUN_NAME)
        result.add_argument("--dry-run", action="store_true", help="Build/audit in a temporary directory; do not train")
    else:
        result.set_defaults(project=str(ROOT / "runs/detect/sce_fusion"), name=RUN_NAME)
    return result


def construct(options, temporary_project=None):
    """Exercise the real native setup_model -> SCETrainer.get_model production path."""
    final, config_report = configuration(options)
    data_report = audit_data(options.data)
    prepare_amp_weights(options.weights)
    os.chdir(ROOT)
    if temporary_project:
        final.update(project=str(temporary_project), name="construction")
    target = Path(final["project"]) / final["name"]
    if target.exists():
        raise FileExistsError(f"Experiment output already exists; refusing overwrite, rename or resume: {target}")
    trainer = SCETrainer(overrides=final)
    trainer.setup_model()
    trainer.set_model_attributes()
    config_report["trainer_effective"] = vars(trainer.args).copy()
    config_report["effective_differences"] = {
        k: {"resolved": config_report["resolved"].get(k), "effective": v}
        for k, v in vars(trainer.args).items()
        if config_report["resolved"].get(k) != v
    }
    return trainer, {
        "environment": environment(),
        "configuration": config_report,
        "dataset": data_report,
        "transfer": trainer.transfer_report,
        "formal_training": "NOT_STARTED",
    }


def main():
    """Run a temporary construction audit or start the explicitly requested fixed training recipe."""
    options = parser().parse_args()
    report_path = options.report or ROOT / "artifacts/sce_fusion/dry_run.json" if options.dry_run else options.report
    report = {"formal_training": "NOT_STARTED"}
    try:
        if options.dry_run:
            with tempfile.TemporaryDirectory(prefix="sce-dry-") as directory:
                _, report = construct(options, directory)
                report["status"] = "PASS"
        else:
            trainer, report = construct(options)
            report_path = report_path or trainer.save_dir / "sce_report.json"
            write_report(report_path, report)

            def record_start(active):
                if not active.amp:
                    raise RuntimeError("Native AMP check did not enable AMP; fixed b19 training cannot continue")
                report.update(
                    formal_training="STARTED",
                    actual_amp=active.amp,
                    actual_batch=active.batch_size,
                    actual_configuration=vars(active.args).copy(),
                )
                write_report(report_path, report)

            trainer.add_callback("on_train_start", record_start)
            trainer.train()
            report.update(status="PASS", formal_training="COMPLETED")
    except Exception as error:
        report.update(status="FAIL", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if report_path:
            write_report(report_path, report)


if __name__ == "__main__":
    main()
