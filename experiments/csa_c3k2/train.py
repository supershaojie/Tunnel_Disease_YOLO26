"""Audited b19 initialization and the fixed CSA-C3k2 training entry point."""

import argparse
import os
import sys
import tempfile
from copy import deepcopy
from pathlib import Path

os.environ["YOLO_AUTOINSTALL"] = "false"
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from experiments.csa_c3k2.common import (
    MODEL_YAML,
    ROOT,
    RUNTIME_KEYS,
    RUN_NAME,
    WEIGHTS_SHA,
    audit_data,
    differences,
    environment,
    isolated_rng,
    recipe,
    rng_digest,
    sha256,
    write_report,
)
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.nn.modules import CSAC3k2
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import YAML


class CSATrainer(DetectionTrainer):
    """Own only the experimental construction and fail-fast batch policy; reuse native training/loss/optimizer."""

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Build the exact native nc=1 reference first, then copy retained state into an RNG-isolated CSA graph."""
        if weights is None or not hasattr(weights, "pt_path"):
            raise ValueError("CSA new-run construction requires setup_model() with the verified original checkpoint")
        if isinstance(weights.model[4], CSAC3k2):
            raise ValueError(
                "Training resumes are not supported by this new-experiment entry point; use native checkpoint inference"
            )
        if (
            sha256(weights.pt_path) != WEIGHTS_SHA
            or Path(weights.pt_path).resolve() != Path(self.args.pretrained).resolve()
        ):
            raise ValueError("Trainer weights are not the verified original pretrained source")
        if Path(cfg).resolve() != MODEL_YAML or self.data["nc"] != 1 or self.data["channels"] != 3:
            raise ValueError("CSA v1 requires its fixed YAML and the single-class, three-channel b19 dataset")
        native_yaml = YAML.load(ROOT / "ultralytics/cfg/models/26/yolo26.yaml")
        for key in ("backbone", "head", "end2end", "reg_max", "scales"):
            if weights.yaml[key] != native_yaml[key]:
                raise ValueError(f"Pretrained architecture differs from locked b19: {key}")

        reference = super().get_model(cfg=deepcopy(weights.yaml), weights=weights, verbose=verbose)
        after_reference = rng_digest()
        with isolated_rng():
            model = self.set_model_names_for_load(DetectionModel(str(MODEL_YAML), nc=1, ch=3, verbose=verbose))
        if rng_digest() != after_reference:
            raise RuntimeError("Innovation construction advanced external RNG")

        old, new, pretrained = reference.state_dict(), model.state_dict(), weights.state_dict()
        kept = {k: v for k, v in old.items() if not k.startswith("model.4.")}
        removed = [k for k in old if k.startswith("model.4.")]
        added = [k for k in new if k.startswith("model.4.")]
        for index, (source, target) in enumerate(zip(reference.model, model.model)):
            if index != 4 and (type(source) is not type(target) or source.f != target.f):
                raise RuntimeError(f"Retained layer semantics changed at {index}")
        if len(reference.model) != len(model.model) or set(new) - set(added) != set(kept):
            raise RuntimeError("Unexpected retained state keys")
        if any(old[k].shape != new[k].shape for k in kept) or not torch.equal(reference.stride, model.stride):
            raise RuntimeError("Retained tensor shapes or stride changed")
        model.load_state_dict({**new, **kept}, strict=True)
        copied = model.state_dict()
        if any(not torch.equal(copied[k], old[k]) or copied[k].data_ptr() == old[k].data_ptr() for k in kept):
            raise RuntimeError("Retained state is unequal or shares storage with the reference")

        source_keys = [k for k in kept if k in pretrained and pretrained[k].shape == old[k].shape]
        gaps = [k for k in kept if k not in source_keys]
        if any(not torch.equal(old[k], pretrained[k]) for k in source_keys):
            raise RuntimeError("Native pretrained transfer did not preserve expected source values")
        if any(not k.startswith(("model.23.cv3.", "model.23.one2one_cv3.")) for k in gaps):
            raise RuntimeError("An unanticipated pretrained mismatch is outside native class adaptation")

        def category(keys, owner):
            params = dict(owner.named_parameters())
            return {
                "tensor_count": len(keys),
                "parameter_elements": sum(params[k].numel() for k in keys if k in params),
                "keys": sorted(keys),
            }

        self.transfer_report = {
            "status": "PASS",
            "production_path": "BaseTrainer.setup_model -> CSATrainer.get_model -> DetectionTrainer.get_model(reference)",
            "weights_received": weights.pt_path,
            "cfg_received": str(cfg),
            "retained_from_original_pretrained": category(source_keys, reference),
            "native_class_adaptation_gaps": category(gaps, reference),
            "intentionally_removed": category(removed, reference),
            "newly_initialized": category(added, model),
            "unexpected_missing": {"tensor_count": 0, "parameter_elements": 0, "keys": []},
            "retained_total": category(list(kept), reference),
            "all_retained_equal": True,
            "independent_storage": True,
            "rng_after_reference": after_reference,
            "rng_after_production": rng_digest(),
        }
        return model

    def _handle_train_batch_failure(self, error, epoch):
        """End this experiment on memory failure before native recovery can halve its fixed batch."""
        if isinstance(error, torch.cuda.OutOfMemoryError) or any(
            s in str(error) for s in ("CUDNN_STATUS_INTERNAL_ERROR", "unable to find an engine")
        ):
            raise RuntimeError(
                "CSA fixed batch=32 experiment stopped on memory failure; batch/AMP/optimizer were not changed"
            ) from error
        raise error


def make_trainer(config, temporary=None):
    """Use the real constructor and inherited setup_model; temporary builds never reserve formal output."""
    effective = config.copy()
    if temporary is not None:
        effective.update(project=str(temporary), name="construction", save_dir=str(Path(temporary) / "construction"))
    if "," in str(effective["device"]):
        raise ValueError("The fixed first experiment supports a single explicitly selected device")
    Path(effective["save_dir"]).mkdir(parents=True, exist_ok=False)
    return CSATrainer(cfg=effective, overrides={})


def parser():
    """Expose the documented new-run interface."""
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--data", required=True)
    result.add_argument("--weights", required=True)
    result.add_argument("--baseline-args", required=True)
    result.add_argument("--project", default=None)
    result.add_argument("--name", default=RUN_NAME)
    result.add_argument("--device", default=None)
    result.add_argument("--dry-run", action="store_true")
    result.add_argument("--report", type=Path)
    return result


def main():
    """Build in a temporary directory for dry-run, or launch a fresh fixed-recipe native training run."""
    args = parser().parse_args()
    os.chdir(ROOT)
    report = {"formal_training": "NOT_STARTED", "status": "FAIL", "environment": environment()}
    config, report["configuration"] = recipe(args)
    formal_dir = Path(config["save_dir"])
    if args.dry_run and args.report and args.report.resolve().is_relative_to(formal_dir):
        raise ValueError("Dry-run report must be outside the formal experiment output directory")
    if not args.dry_run and formal_dir.exists():
        raise FileExistsError(f"Formal experiment output already exists: {formal_dir}")
    destination = args.report
    try:
        report["dataset"] = audit_data(config["data"])
        if args.dry_run:
            with tempfile.TemporaryDirectory(prefix="csa-dry-") as temp, isolated_rng():
                trainer = make_trainer(config, temp)
                trainer.setup_model()
                trainer.set_model_attributes()
                report["transfer"] = trainer.transfer_report
                report["actual_trainer_args"] = vars(trainer.args).copy()
                report["actual_differences"] = differences(config, vars(trainer.args))
                report["status"] = "PASS"
        else:
            trainer = make_trainer(config)
            destination = args.report or trainer.save_dir / "csa_run.json"

            def record_ready(current):
                if not current.amp or current.device.type != "cuda":
                    raise RuntimeError("Fixed CSA training requires native AMP to pass on CUDA; no FP32 fallback")
                actual = vars(current.args).copy()
                drift = {k: v for k, v in differences(config, actual).items() if k not in RUNTIME_KEYS}
                if drift:
                    raise RuntimeError(f"Effective training mechanism changed: {drift}")
                report.update(
                    transfer=current.transfer_report,
                    actual_trainer_args=actual,
                    formal_training="STARTED",
                    status="RUNNING",
                )
                write_report(destination, report)

            trainer.add_callback("on_pretrain_routine_end", record_ready)
            trainer.train()
            report.update(formal_training="COMPLETED", status="PASS")
    except Exception as error:
        report.update(status="FAIL", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if destination:
            write_report(destination, report)
        elif args.dry_run:
            print(f"CSA dry-run: {report['status']}; formal_training=NOT_STARTED (use --report for the full audit)")


if __name__ == "__main__":
    main()
