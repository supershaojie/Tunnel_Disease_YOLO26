"""Launch the locked b19 BDF experiment; --dry-run builds through the native Trainer without training."""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

import ultralytics
from ultralytics.cfg import get_cfg
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.autobackend import check_class_names
from ultralytics.utils import DEFAULT_CFG_DICT, YAML
from ultralytics.utils.torch_utils import init_seeds

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "ultralytics/cfg/models/26/yolo26n-bdf-fusion.yaml"
WEIGHTS_SHA256 = "9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef"
RUN_FIELDS = {"model", "data", "project", "name", "device", "resume", "save_dir", "pretrained", "cfg"}


def locked_config(baseline_args=None):
    """Recover b19 defaults and reject incompatible optional historical args, excluding run-specific fields."""
    config = {**DEFAULT_CFG_DICT, **YAML.load(Path(__file__).with_name("b19.yaml"))}
    if baseline_args:
        historical = YAML.load(baseline_args)
        differences = {
            k: (v, config.get(k)) for k, v in historical.items() if k not in RUN_FIELDS and config.get(k) != v
        }
        if differences:
            raise ValueError(f"Baseline args differ from locked b19: {differences}")
    return {k: v for k, v in config.items() if k not in RUN_FIELDS}


def check_weights(path):
    """Require the original checkpoint, including for dry-run; never download or accept trained smoke weights."""
    path = Path(path).resolve(strict=True)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != WEIGHTS_SHA256:
        raise ValueError(f"Original yolo26n.pt SHA256 mismatch: {digest}")
    return path


def environment():
    """Report actual versions and the differences from the verified server environment."""
    actual = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "ultralytics": ultralytics.__version__,
        "albumentations_installed": importlib.util.find_spec("albumentations") is not None,
    }
    expected = {
        "python": "3.12.3",
        "torch": "2.8.0+cu128",
        "cuda_runtime": "12.8",
        "ultralytics": "8.4.98",
        "albumentations_installed": False,
    }
    differences = {k: {"actual": actual[k], "expected": v} for k, v in expected.items() if actual[k] != v}
    actual["executable"] = sys.executable
    actual["source"] = str(Path(ultralytics.__file__).resolve())
    actual["dependencies"] = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}
    return actual, differences


def main():
    """Validate explicit paths and build or launch a fresh native DetectionTrainer."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument("--name", default="bdf_fusion_b19_e200_i640_b32_s42")
    parser.add_argument("--device", default="0")
    parser.add_argument("--baseline-args", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    cli = parser.parse_args()
    if Path(ultralytics.__file__).resolve() != ROOT / "ultralytics/__init__.py":
        raise RuntimeError("Import must resolve to this experiment worktree")
    weights = check_weights(cli.weights)
    data_path = cli.data.resolve(strict=True)
    data = YAML.load(data_path)
    names = check_class_names(data.get("names", {0: "crack"}))
    if len(names) != 1 or data.get("nc", 1) != 1 or data.get("channels", 3) != 3:
        raise ValueError("The locked b19 experiment requires nc=1 and three image channels")
    for split in ("train", "val", "test"):
        if not data.get(split):
            raise ValueError(f"Dataset YAML must preserve the b19 {split} split")
    if Path(cli.name).name != cli.name or cli.name in {".", ".."}:
        raise ValueError("--name must be a single output directory name")
    config = locked_config(cli.baseline_args)
    config.update(
        model=str(MODEL),
        data=str(data_path),
        pretrained=str(weights),
        resume=False,
        project=str(cli.project.resolve()),
        name=cli.name,
        device=cli.device,
    )
    args = get_cfg(overrides=config)
    actual, differences = environment()
    print(
        json.dumps(
            {
                "training_status": "NOT_STARTED",
                "environment": actual,
                "server_environment_differences": differences,
                "config": vars(args),
            },
            indent=2,
        )
    )
    if cli.dry_run:
        # Use precisely the production rebuild/load methods, without Trainer.__init__ directory/data side effects.
        trainer = DetectionTrainer.__new__(DetectionTrainer)
        trainer.args, trainer.data, trainer.model = args, {"nc": 1, "channels": 3, "names": names}, str(MODEL)
        trainer.resume = False
        init_seeds(args.seed, deterministic=args.deterministic)
        trainer.setup_model()
        trainer.set_model_attributes()
        print(f"DRY_RUN_OK: {sum(p.numel() for p in trainer.model.parameters())} parameters; training NOT_STARTED")
        return
    if differences or Path(sys.executable).resolve() != Path("/root/miniconda3/bin/python").resolve():
        raise RuntimeError(f"Server environment mismatch; packages were not modified: {differences}")
    if not torch.cuda.is_available():
        raise RuntimeError("The locked experiment requires CUDA; batch/AMP/optimizer will not be changed")
    if (cli.project / cli.name).exists():
        raise FileExistsError("Experiment output already exists; refusing to overwrite or silently rename it")
    check_weights(Path.cwd() / "yolo26n.pt")  # Native AMP check resolves this name in the working directory.
    DetectionTrainer(overrides=config).train()


if __name__ == "__main__":
    main()
