"""Shared provenance, configuration and bounded diagnostics for CSA-C3k2."""

import hashlib
import importlib.metadata
import json
import os
import random
import shutil
import subprocess
import sys
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

import ultralytics
from ultralytics.data.utils import IMG_FORMATS, check_det_dataset
from ultralytics.nn.modules import CSAUnit, CurveSampler
from ultralytics.utils import DEFAULT_CFG_DICT, YAML

ROOT = Path(__file__).resolve().parents[2]
MODEL_YAML = ROOT / "experiments/csa_c3k2/yolo26n-csa.yaml"
BASE_SHA = "4a1b167dee2b0c7353d9d7ead2911383d1ebf1d6"
WEIGHTS_SHA = "9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef"
DATA_SHA = "1f18760508e9dbf2332cd7102ee9c15e11e08f3d15fc4202c8ee0ed1bb785b12"
RECIPE_SHA = "81c506eb50bcb29a50d37c080ba35b212eb4faff42dff65ca5e535f02dd0ee7f"
RUN_NAME = "csa_c3k2_b19_e200_i640_b32_s42"
RUNTIME_KEYS = {"model", "pretrained", "data", "project", "name", "save_dir", "device"}


def sha256(path):
    """Hash a required local file without invoking any download mechanism."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_report(path, report):
    """Write an explicit JSON report, creating its parent directory."""
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str, allow_nan=False) + "\n", encoding="utf-8"
    )


def environment():
    """Record the actual local import, source revision, device and dependency versions."""
    if Path(ultralytics.__file__).resolve() != ROOT / "ultralytics/__init__.py":
        raise RuntimeError(f"Wrong Ultralytics import: {ultralytics.__file__}")
    versions = {}
    for package in ("torch", "torchvision", "numpy", "opencv-python", "PyYAML", "albumentations", "ultralytics-thop"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "NOT_INSTALLED"
    return {
        "python": sys.version,
        "executable": sys.executable,
        "ultralytics": ultralytics.__version__,
        "import_path": ultralytics.__file__,
        "worktree": str(ROOT),
        "base_commit": BASE_SHA,
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "working_tree": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True).strip(),
        "versions": versions,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "YOLO_AUTOINSTALL": os.environ.get("YOLO_AUTOINSTALL"),
    }


def checked_weights(path):
    """Require the specified original checkpoint and supply the native AMP check's local filename."""
    path = Path(path).expanduser().resolve(strict=True)
    actual = sha256(path)
    if actual != WEIGHTS_SHA:
        raise ValueError(f"Original yolo26n.pt SHA256 mismatch: {path}: {actual}")
    target = ROOT / "yolo26n.pt"
    if target.exists():
        if sha256(target) != WEIGHTS_SHA:
            raise FileExistsError(f"Refusing to replace mismatched AMP checkpoint: {target}")
    else:
        with path.open("rb") as source, target.open("xb") as dest:
            shutil.copyfileobj(source, dest)
    return path


def recipe(args):
    """Read every b19 argument and allow only documented runtime path/device replacements."""
    baseline_path = Path(args.baseline_args).expanduser().resolve(strict=True)
    original = YAML.load(baseline_path)
    missing = DEFAULT_CFG_DICT.keys() - original.keys()
    extra = original.keys() - DEFAULT_CFG_DICT.keys() - {"save_dir"}
    if missing or extra:
        raise ValueError(f"Incomplete/incompatible b19 args: missing={sorted(missing)}, extra={sorted(extra)}")
    mechanism = {k: v for k, v in original.items() if k not in RUNTIME_KEYS}
    fingerprint = hashlib.sha256(json.dumps(mechanism, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if fingerprint != RECIPE_SHA:
        raise ValueError("The complete non-runtime recipe differs from the audited b19 args.yaml")
    weights = checked_weights(args.weights)
    project = Path(getattr(args, "project", None) or ROOT / "runs/csa_c3k2").expanduser().resolve()
    name = getattr(args, "name", None) or RUN_NAME
    if Path(name).name != name or name in {".", ".."}:
        raise ValueError("--name must be a single output directory name")
    final = dict(original)
    final.update(
        model=str(MODEL_YAML),
        pretrained=str(weights),
        data=str(Path(args.data).expanduser().resolve(strict=True)),
        project=str(project),
        name=name,
        save_dir=str(project / name),
        device=args.device if args.device is not None else original["device"],
    )
    return final, {
        "baseline_args_path": str(baseline_path),
        "baseline_args_sha256": sha256(baseline_path),
        "recipe_sha256": fingerprint,
        "original": original,
        "resolved": final.copy(),
        "differences": differences(original, final),
        "checkpoint": {"path": str(weights), "sha256": WEIGHTS_SHA},
        "model_yaml_sha256": sha256(MODEL_YAML),
    }


def differences(original, final):
    """Describe each changed field without hiding default additions or native normalization."""
    return {
        k: {"before": original.get(k), "after": final.get(k)}
        for k in original.keys() | final.keys()
        if original.get(k) != final.get(k)
    }


def audit_data(path):
    """Check dataset metadata and split counts without evaluating images or writing label caches."""
    path = Path(path).resolve(strict=True)
    raw = YAML.load(path)
    historical = {"train": "images/train", "val": "images/val", "test": "images/test", "nc": 1, "names": {0: "crack"}}
    changes = differences(historical, raw)
    if any(k not in {"path", "train", "val", "test"} for k in changes):
        raise ValueError(f"Dataset semantics differ from b19: {changes}")
    data = check_det_dataset(str(path), autodownload=False)
    counts = {}
    for split, expected_images, expected_boxes in (("train", 8414, None), ("val", 2404, 2985), ("test", 1202, 1477)):
        folder = Path(data[split])
        images = sorted(p for p in folder.rglob("*") if p.suffix[1:].lower() in IMG_FORMATS)
        labels = [
            Path(str(p).replace(f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}")).with_suffix(".txt")
            for p in images
        ]
        boxes = 0
        for label in labels:
            for line in label.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    values = [float(v) for v in line.split()]
                    if (
                        len(values) != 5
                        or values[0] != 0
                        or not all(np.isfinite(values))
                        or not all(0 <= v <= 1 for v in values[1:])
                    ):
                        raise ValueError(f"Invalid crack detection label: {label}")
                    boxes += 1
        counts[split] = {"images": len(images), "boxes": boxes, "path": str(folder)}
        if len(images) != expected_images or expected_boxes is not None and boxes != expected_boxes:
            raise ValueError(f"Unexpected {split} counts: {counts[split]}")
    return {
        "status": "PASS",
        "path": str(path),
        "sha256": sha256(path),
        "historical_sha256": DATA_SHA,
        "byte_identical": sha256(path) == DATA_SHA,
        "yaml_differences": changes,
        "splits": counts,
        "scope": "Metadata, image/label correspondence and counts; no server image-content comparison or evaluation",
    }


def rng_state():
    """Snapshot Python, NumPy, CPU and already initialized CUDA random generators."""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "cpu": torch.get_rng_state().clone(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
    }


def rng_digest(state=None):
    """Produce stable audit digests without consuming random numbers."""
    state = rng_state() if state is None else state
    return {
        "python": hashlib.sha256(repr(state["python"]).encode()).hexdigest(),
        "numpy": hashlib.sha256(
            state["numpy"][1].tobytes() + repr((state["numpy"][0], *state["numpy"][2:])).encode()
        ).hexdigest(),
        "cpu": hashlib.sha256(state["cpu"].numpy().tobytes()).hexdigest(),
        "cuda": [hashlib.sha256(s.cpu().numpy().tobytes()).hexdigest() for s in state["cuda"]],
    }


@contextmanager
def isolated_rng():
    """Let the native reference consume RNG, but isolate innovation construction and diagnostics."""
    state = rng_state()
    with torch.random.fork_rng(devices=list(range(len(state["cuda"])))):
        try:
            yield
        finally:
            random.setstate(state["python"])
            np.random.set_state(state["numpy"])


def parameter_count(model):
    """Count parameters only, excluding buffers."""
    return sum(p.numel() for p in model.parameters())


def fusion_audit(model):
    """Measure a native fused copy and distinguish unavailable unfused state in already-fused checkpoints."""
    was_fused = model.is_fused()
    fused = deepcopy(model).eval().fuse(verbose=False)
    counts = {
        "checkpoint": parameter_count(model),
        "checkpoint_already_fused": was_fused,
        "unfused": None if was_fused else parameter_count(model),
        "layer4_unfused": None if was_fused else parameter_count(model.model[4]),
        "unfused_status": "UNVERIFIED" if was_fused else "PASS",
        "unfused_note": "Use the original unfused checkpoint to measure BN/O2M state" if was_fused else None,
        "fused": parameter_count(fused),
        "layer4_fused": parameter_count(fused.model[4]),
        "O2M_removed_by_native_fuse": fused.model[-1].cv2 is None and fused.model[-1].cv3 is None,
    }
    return counts, fused


class Diagnostics:
    """Attach removable hooks for at most four samples; store summaries, never full activations."""

    def __init__(self, model, samples=0):
        """Register opt-in hooks with per-module sample limits."""
        if not 0 <= samples <= 4:
            raise ValueError("diagnostic samples must be in 0..4")
        self.records, self.handles, self.seen = {}, [], {}
        self.samples = samples
        if not samples:
            return
        for name, module in model.named_modules():
            if isinstance(module, CurveSampler):
                self.handles.append(module.offset[-1].register_forward_hook(self.offset_hook(name, module)))
            if isinstance(module, CSAUnit):
                self.handles.append(module.selector[-1].register_forward_hook(self.selector_hook(name)))
                self.handles.append(module.register_forward_hook(self.residual_hook(name)))

    def take(self, key, output):
        """Select only the remaining requested samples, including within a larger validation batch."""
        count = min(self.samples - self.seen.get(key, 0), output.shape[0])
        self.seen[key] = self.seen.get(key, 0) + max(count, 0)
        return output[:count].detach().float() if count > 0 else None

    @staticmethod
    def quantiles(tensor):
        """Summarize bounded diagnostic tensors."""
        q = torch.tensor([0, 0.05, 0.5, 0.95, 1], device=tensor.device)
        return tensor.flatten().quantile(q).cpu().tolist()

    def offset_hook(self, name, sampler):
        """Measure the actual predictor outputs at each sampled forward."""

        def hook(module, inputs, output):
            value = self.take(name, output)
            if value is not None:
                steps, cumulative, _ = sampler.curve_offsets(value)
                self.records.setdefault(name, []).append(
                    {
                        "samples": len(value),
                        "step_quantiles": self.quantiles(steps),
                        "cumulative_quantiles": self.quantiles(cumulative),
                        "tanh_saturation_abs_ge_0.99": float((steps.abs() >= 0.99).float().mean()),
                    }
                )

        return hook

    def selector_hook(self, name):
        """Report group/branch order [local,horizontal,vertical]."""

        def hook(module, inputs, output):
            key = name + ".selection"
            value = self.take(key, output)
            if value is not None:
                b, _, h, w = value.shape
                weights = value.reshape(b, 4, 3, h, w).softmax(2)
                self.records.setdefault(key, []).append(
                    {"samples": b, "group_branch_mean": weights.mean((0, 3, 4)).cpu().tolist()}
                )

        return hook

    def residual_hook(self, name):
        """Measure residual/input norms without altering the model's return type."""

        def hook(module, inputs, output):
            key = name + ".residual"
            value = self.take(key, output)
            if value is not None:
                original = inputs[0][: len(value)].detach().float()
                ratios = (value - original).flatten(1).norm(dim=1) / original.flatten(1).norm(dim=1).clamp_min(1e-12)
                self.records.setdefault(key, []).append({"ratios": ratios.cpu().tolist()})

        return hook

    def close(self):
        """Remove every hook before deepcopy, serialization or ordinary inference."""
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
