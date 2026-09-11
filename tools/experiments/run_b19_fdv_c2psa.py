"""Train the fixed FDV v1 from the archived b19 recipe and original COCO checkpoint."""

# Direct entry must resolve this worktree before package imports.
# ruff: noqa: E402

import argparse
import copy
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

import ultralytics
from tools.experiments import b19_common as common
from ultralytics.cfg import DEFAULT_CFG_DICT
from ultralytics.data.utils import check_det_dataset
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.modules import FDV_C2PSA

NAME = "yolo26n_b19_fdv_c2psa_v1"
BASE = Path("/root/autodl-tmp/projects/Tunnel_Disease_YOLO26")


def options_parser():
    """Expose locations and audit stages, with no training hyperparameter override interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, default=BASE)
    parser.add_argument("--baseline-args", type=Path)
    parser.add_argument("--pretrained", type=Path)
    parser.add_argument("--pretrained-sha256", default=common.PRETRAINED_SHA256)
    parser.add_argument("--baseline-launcher", type=Path, default=Path(__file__).with_name("b19_launcher_expanded.txt"))
    parser.add_argument("--project", type=Path, default=ROOT / "runs/detect")
    parser.add_argument("--name", choices=[NAME], default=NAME)
    return parser


def audit_arguments(raw, effective):
    """Compare every resolved field; only verified identity and equivalent file locations may differ."""
    allowed = {"model", "pretrained", "data", "project", "name", "save_dir"}
    differences = {
        k: [raw.get(k), effective.get(k)] for k in raw.keys() | effective.keys() if raw.get(k) != effective.get(k)
    }
    illegal = {k: v for k, v in differences.items() if k not in allowed}
    if illegal:
        raise ValueError(f"Non-identity b19 training differences: {illegal}")
    return differences


def require_runtime():
    """Require the recorded server environment instead of changing recipe for local hardware."""
    expected = common.REFERENCE["environment"]
    actual = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    if actual != expected:
        raise RuntimeError(f"Formal b19 environment mismatch: actual={actual}, expected={expected}")
    return actual


class AuditedTrainer(DetectionTrainer):
    """Use the native trainer, auditing reconstruction and enforcing the fixed batch at the retry boundary."""

    def __init__(self, overrides, _callbacks=None):
        """Atomically claim the canonical output and prevent native automatic name incrementing."""
        self.expected_args = copy.deepcopy(overrides)
        output = Path(overrides["project"]) / overrides["name"]
        output.mkdir(parents=True, exist_ok=False)
        super().__init__(
            cfg={**DEFAULT_CFG_DICT, "save_dir": str(output)}, overrides=overrides.copy(), _callbacks=_callbacks
        )

    @property
    def _oom_retries(self):
        """Fixed-batch training never requests memory recovery."""
        return 0

    @_oom_retries.setter
    def _oom_retries(self, value):
        """Re-raise the native OOM at the retry request, before it changes batch or args."""
        if value:
            error = sys.exc_info()[1]
            if error is None:
                raise RuntimeError("FDV v1 forbids automatic batch reduction")
            raise error

    def get_dataset(self):
        """Read the existing fixed dataset without replacement downloads."""
        return check_det_dataset(self.args.data, autodownload=False)

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Audit the actual native pretrained loading into the final training model."""
        with torch.random.fork_rng(devices=[]):
            baseline = super().get_model(common.baseline_architecture(), weights, verbose=False)
        candidate = super().get_model(cfg, weights, verbose)
        if weights is None:
            raise RuntimeError("Original pretrained checkpoint is required")
        self.weight_audit = common.audit_weights(baseline, candidate, weights)
        if self.weight_audit["matched_tensors"] != common.REFERENCE["transferred_items"]:
            raise RuntimeError("Pretrained coverage differs from native b19")
        assert type(candidate.model[10]) is FDV_C2PSA and candidate.model[10].m[0].attn.theta_c.count_nonzero() == 0
        return candidate


def audit_training_setup(trainer):
    """Check effective recipe and optimizer after native initialization and before the first training batch."""
    audit_arguments(trainer.expected_args, vars(trainer.args))
    assert trainer.batch_size == 32 and trainer.amp is True
    model = trainer.model
    parameter_ids = [id(p) for group in trainer.optimizer.param_groups for p in group["params"]]
    assert len(parameter_ids) == len(set(parameter_ids))
    new = {k: p for k, p in model.named_parameters() if k == "model.10.m.0.attn.theta_c"}
    assert len(new) == 1 and all(id(p) in parameter_ids and p.requires_grad for p in new.values())
    signature = [
        {k: v for k, v in group.items() if k not in {"params", "initial_lr"}}
        for group in trainer.optimizer.param_groups
    ]
    with torch.random.fork_rng(devices=[]):
        baseline = DetectionTrainer.get_model(trainer, common.baseline_architecture(), verbose=False)
    optimizer = trainer.build_optimizer(
        baseline, "MuSGD", trainer.args.lr0, trainer.args.momentum, trainer.args.weight_decay
    )
    expected = [{k: v for k, v in group.items() if k != "params"} for group in optimizer.param_groups]
    assert signature == expected and type(trainer.optimizer).__name__ == "MuSGD"
    common.write_json(
        trainer.save_dir / "provenance/optimizer.json", {"groups": signature, "new_parameters": list(new)}
    )
    common.write_json(trainer.save_dir / "provenance/weights.json", trainer.weight_audit)


def record_completion(trainer):
    """Publish completion only after the native trainer finishes and produces both checkpoints."""
    required = ["weights/best.pt", "weights/last.pt", "args.yaml", "results.csv"]
    hashes = {name: common.sha256(trainer.save_dir / name) for name in required}
    common.write_json(
        trainer.save_dir / "completed.json",
        {
            "commit": common.git("rev-parse", "HEAD"),
            "files": hashes,
            "epochs_completed": trainer.epoch + 1,
            "early_stop": trainer.stopper.possible_stop,
        },
    )


def main():
    """Run a fresh-process preflight, then train from the original seed and checkpoint."""
    parser = options_parser()
    parser.add_argument("--stage", choices=["train", "preflight"], default="train")
    args = parser.parse_args()
    common.require_clean_source()
    require_runtime()
    raw, config, evidence = common.resolve_recipe(args)
    audit_arguments(raw, config)
    evidence["launcher"] = common.launcher_evidence(args, raw)
    evidence["source_sha256"] = common.source_hashes()
    args.project.resolve().mkdir(parents=True, exist_ok=True)
    audit_dir = Path(tempfile.mkdtemp(prefix=f"{NAME}_preflight_", dir=args.project.resolve()))
    command = [
        sys.executable,
        str(Path(__file__).with_name("verify_b19_fdv_c2psa.py")),
        "--baseline-root",
        str(args.baseline_root),
        "--baseline-args",
        evidence["args_path"],
        "--pretrained",
        evidence["initial_path"],
        "--baseline-launcher",
        str(args.baseline_launcher),
        "--project",
        str(args.project),
        "--output",
        str(audit_dir),
    ]
    subprocess.run(command, check=True, cwd=ROOT)
    checks = json.loads((audit_dir / "checks.json").read_text(encoding="utf-8"))
    assert checks["passed"] and not checks["local_only"]
    assert (
        checks["commit"] == evidence["commit"]
        and checks["recipe"]["config_differences"] == evidence["config_differences"]
    )
    common.require_clean_source()
    assert evidence["source_sha256"] == common.source_hashes()
    receipt = common.sha256(audit_dir / "checks.json")
    if args.stage == "preflight":
        return
    evidence["preflight_checks_sha256"] = receipt
    trainer = AuditedTrainer(config)
    provenance = trainer.save_dir / "provenance"
    common.write_json(provenance / "resolved.json", {"config": config, "evidence": evidence})
    shutil.copytree(audit_dir, provenance / "preflight")
    shutil.copyfile(evidence["args_path"], provenance / "b19_original_args.yaml")
    shutil.copyfile(args.baseline_launcher, provenance / "b19_launcher_expanded.txt")
    trainer.add_callback("on_pretrain_routine_end", audit_training_setup)
    trainer.add_callback("on_train_end", record_completion)
    trainer.train()


if __name__ == "__main__":
    main()
