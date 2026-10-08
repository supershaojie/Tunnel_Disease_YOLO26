"""Launch the fixed b19 + DGA experiment, or audit its construction without starting training."""

# ruff: noqa: E402 - select the experiment source and disable package installation before importing Ultralytics.

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ["YOLO_AUTOINSTALL"] = "false"

import torch

import ultralytics
from ultralytics.cfg import get_cfg
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.modules import C2PSA_DGA
from ultralytics.nn.tasks import load_checkpoint
from ultralytics.utils import LOGGER, YAML
from ultralytics.utils.torch_utils import init_seeds

MODEL = ROOT / "experiments/dga_c2psa/yolo26n-dga.yaml"
RECIPE = MODEL.with_name("b19_recipe.yaml")
WEIGHTS_SHA256 = "9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef"
RUN_FIELDS = {"model", "data", "project", "name", "pretrained", "resume", "save_dir", "device"}
PREDICTOR = "model.10.m.0.attn.geometry_predictor."


class DGATrainer(DetectionTrainer):
    """Use the native Trainer and verify all common state during its actual YAML rebuild."""

    def _build_train_pipeline(self):
        """Enforce the fixed recipe before native loader/optimizer creation, including an OOM retry."""
        if self.batch_size != 32 or not self.amp:
            raise RuntimeError("DGA requires batch=32 and working AMP; stop instead of training an altered recipe.")
        return super()._build_train_pipeline()

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Audit against native nc=1 construction from the same starting CPU RNG, then return DGA."""
        if self.data["nc"] != 1 or self.data["channels"] != 3:
            raise ValueError("This fixed b19 experiment requires nc=1 and three input channels.")
        # This reference is an audit, not a module replacement; it cannot advance the training RNG.
        with torch.random.fork_rng(devices=[]):
            native = super().get_model(ROOT / "ultralytics/cfg/models/26/yolo26n.yaml", weights, verbose=False)
            native_rng = torch.get_rng_state()
        model = super().get_model(cfg, weights, verbose)
        native_state, state = native.state_dict(), model.state_dict()
        missing = sorted(native_state.keys() - state.keys())
        mismatched = [k for k, v in native_state.items() if k in state and not torch.equal(v, state[k])]
        added = sorted(state.keys() - native_state.keys())
        if missing or mismatched or not torch.equal(native_rng, torch.get_rng_state()):
            raise RuntimeError(f"Common native state/RNG mismatch: missing={missing}, mismatched={mismatched}")
        if not added or any(not k.startswith(PREDICTOR) for k in added):
            raise RuntimeError(f"Unexpected added state: {added}")
        psa = model.model[10]
        if not isinstance(psa, C2PSA_DGA) or len(psa.m) != 1:
            raise RuntimeError("Expected exactly one DGA PSABlock at layer 10.")
        attn = psa.m[0].attn
        if (psa.c, attn.num_heads, attn.key_dim, attn.head_dim) != (128, 2, 32, 64):
            raise RuntimeError("Unexpected YOLO26n attention dimensions.")
        source = weights.state_dict() if weights is not None else {}
        gaps = [k for k, v in native_state.items() if k not in source or source[k].shape != v.shape]
        unexpected = [k for k in gaps if not k.startswith(("model.23.cv3.", "model.23.one2one_cv3."))]
        if weights is None or unexpected:
            raise RuntimeError(f"Original pretrained weights required; unexpected native transfer gaps: {unexpected}")
        self.dga_audit = {
            "common_state_tensors_equal": len(native_state),
            "rng_equal": True,
            "native_nc_adaptation_gaps": gaps,
            "unexpected_native_gaps": unexpected,
            "new_state_keys": added,
            "new_parameters": sum(p.numel() for p in attn.geometry_predictor.parameters()),
            "parameters_unfused": sum(p.numel() for p in model.parameters()),
        }
        LOGGER.info("DGA construction audit: " + json.dumps(self.dga_audit))
        return model


def training_config(options):
    """Recover the historical recipe, checking optional server args before replacing run-specific fields."""
    recipe = YAML.load(RECIPE)
    if options.baseline_args:
        historical = {k: v for k, v in YAML.load(options.baseline_args).items() if k not in RUN_FIELDS}
        differences = {
            k: (recipe.get(k), historical.get(k))
            for k in recipe.keys() | historical.keys()
            if recipe.get(k) != historical.get(k)
        }
        if differences:
            raise ValueError(f"Baseline args differ from the recovered b19 recipe (bundled, supplied): {differences}")
    weights = Path(options.weights).resolve(strict=True)
    digest = hashlib.sha256(weights.read_bytes()).hexdigest()
    if digest != WEIGHTS_SHA256:
        raise ValueError(f"Expected original yolo26n.pt SHA256 {WEIGHTS_SHA256}, got {digest}")
    recipe.update(
        model=str(MODEL),
        data=str(Path(options.data).resolve(strict=True)),
        pretrained=str(weights),
        resume=False,
        project=str(Path(options.project).resolve()),
        name=options.name,
        device=options.device,
    )
    return vars(get_cfg(overrides=recipe))


def environment_report():
    """Report the source import and actual installed dependency versions without changing the environment."""
    imported = Path(ultralytics.__file__).resolve()
    if imported != ROOT / "ultralytics/__init__.py":
        raise RuntimeError(f"Wrong ultralytics import: {imported}")
    packages = {}
    for name in ("torchvision", "numpy", "opencv-python", "pillow", "scipy", "ultralytics-thop", "PyYAML"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    report = {
        "python": platform.python_version(),
        "executable": sys.executable,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "ultralytics": ultralytics.__version__,
        "ultralytics_path": str(imported),
        "albumentations_installed": importlib.util.find_spec("albumentations") is not None,
        "dependencies": packages,
    }
    LOGGER.info("Environment: " + json.dumps(report))
    return report


def main():
    """Audit only on --dry-run; otherwise start the isolated, fixed-recipe native training loop."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--weights", default="yolo26n.pt")
    parser.add_argument("--project", default=str(ROOT / "runs/detect"))
    parser.add_argument("--name", default="dga_c2psa_b19_e200_i640_b32_s42")
    parser.add_argument("--device", default="0")
    parser.add_argument("--baseline-args", help="Optional complete historical b19 args.yaml to cross-check.")
    parser.add_argument(
        "--dry-run", action="store_true", help="Audit on CPU without a Trainer run directory or training."
    )
    options = parser.parse_args()
    environment = environment_report()
    config = training_config(options)
    LOGGER.info("Resolved b19 configuration: " + json.dumps(config))
    if options.dry_run:
        # Reuse the real get_model method without BaseTrainer.__init__ creating run directories.
        trainer = object.__new__(DGATrainer)
        trainer.args = get_cfg(overrides=config)
        data = YAML.load(config["data"])
        names = data["names"]
        trainer.data = {"nc": len(names), "names": names, "channels": data.get("channels", 3)}
        weights, _ = load_checkpoint(config["pretrained"])
        init_seeds(config["seed"], deterministic=config["deterministic"])
        trainer.get_model(config["model"], weights, verbose=False)
        LOGGER.info("Formal training status: NOT_STARTED (dry-run completed; no output directory created).")
        return
    expected = {
        "python": "3.12.3",
        "torch": "2.8.0+cu128",
        "cuda_runtime": "12.8",
        "ultralytics": "8.4.98",
        "albumentations_installed": False,
    }
    drift = {k: (expected[k], environment[k]) for k in expected if expected[k] != environment[k]}
    if drift:
        raise RuntimeError(f"Core environment differs from confirmed b19 server (expected, actual): {drift}")
    if (Path(config["project"]) / config["name"]).exists():
        raise FileExistsError("The experiment output already exists; use a new independent --name.")
    # Native AMP check resolves yolo26n.pt in cwd. Require the verified local file to avoid a download.
    amp_weights = Path.cwd() / "yolo26n.pt"
    if not amp_weights.is_file() or hashlib.sha256(amp_weights.read_bytes()).hexdigest() != WEIGHTS_SHA256:
        raise ValueError(
            "Place a link/copy of the original yolo26n.pt in the working directory for the native AMP check."
        )
    DGATrainer(overrides=config).train()


if __name__ == "__main__":
    main()
