"""Train the frozen b19 + DTR + SCE graph, or audit its production construction without training."""

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ["YOLO_AUTOINSTALL"] = "false"
os.environ["ULTRALYTICS_SAFE_LOAD"] = "true"

import torch

from experiments.sce_fusion.train import (
    BASE_SHA,
    RUNTIME_FIELDS,
    SCETrainer,
    prepare_amp_weights,
    sha256,
    verified_weights,
    write_report,
)
from experiments.sce_fusion.train import (
    audit_data as sce_audit_data,
)
from experiments.sce_fusion.train import (
    environment as sce_environment,
)
from ultralytics.cfg import get_cfg
from ultralytics.nn.modules import C2PSA, C2PSA_DTR, Detect, Index, SCEFusion
from ultralytics.nn.tasks import load_checkpoint
from ultralytics.utils import DEFAULT_CFG_DICT, YAML
from ultralytics.utils.files import WorkingDirectory

MODEL_YAML = Path(__file__).with_name("yolo26n-dtr-sce.yaml")
BASELINE_ARGS = Path(__file__).with_name("baseline_args.yaml")
RUN_NAME = "dtr_sce_b19_e200_i640_b32_s42"
SOURCES = {
    "b19": BASE_SHA,
    "dtr": "8cf1b0d09f2488d320525f131f7966cf20792345",
    "sce": "57b59daee717b5cb048f226adb8ad23909a6e95a",
}
FROZEN = {
    "dtr": "d429ea673adf36ca24361df31cf4eaa5c2fb326990abcfd15e8c38277604f53f",
    "sce": "cb9194f0563f9ac5f22d0ad23a38f7e3644bcbe33c30321b6ef0ebb3c750c5ad",
}
REMOVED = "model.10.m."
NEW = ("model.10.dtr_blocks.", "model.23.")


def frozen_sources():
    """Verify the two immutable module files, normalizing CRLF only in memory."""
    actual = {
        name: hashlib.sha256(
            (ROOT / f"ultralytics/nn/modules/{name}.py").read_bytes().replace(b"\r\n", b"\n")
        ).hexdigest()
        for name in FROZEN
    }
    if actual != FROZEN:
        raise RuntimeError(f"Frozen module source mismatch: {actual}")
    return {"commits": SOURCES, "LF_sha256": actual}


def identity(model, variant="dtr_sce"):
    """Check the actual graph and native head before accepting a checkpoint's requested experiment identity."""
    has_dtr, has_sce = variant in {"dtr", "dtr_sce"}, variant in {"sce", "dtr_sce"}
    graph = model.model
    detect = 27 if has_sce else 23
    if len(graph) != detect + 1 or type(graph[10]) is not (C2PSA_DTR if has_dtr else C2PSA):
        raise ValueError(f"Checkpoint graph does not match variant={variant}")
    if (
        sum(isinstance(m, C2PSA_DTR) for m in model.modules()) != int(has_dtr)
        or sum(isinstance(m, SCEFusion) for m in model.modules()) != int(has_sce)
        or type(graph[detect]) is not Detect
        or graph[detect].nc != 1
        or graph[detect].reg_max != 1
        or not model.end2end
        or graph[0].conv.in_channels != 3
        or model.names != {0: "crack"}
        or model.stride.tolist() != [8, 16, 32]
    ):
        raise ValueError(f"Expected RGB/nc1/crack native end-to-end YOLO26n for {variant}")
    expected = YAML.load(ROOT / "experiments/sce_fusion/yolo26n-sce.yaml")
    layers = model.yaml["backbone"] + model.yaml["head"]
    native = expected["backbone"] + expected["head"]
    if has_dtr:
        native[10] = [-1, 1, "C2PSA_DTR", [1024, 0.5, 2, 2, 5]]
    if not has_sce:
        native = native[:23] + [[[16, 19, 22], 1, "Detect", ["nc"]]]
    if layers != native or model.yaml.get("scale") != "n":
        raise ValueError("Checkpoint YAML differs from the locked graph")
    if has_sce:
        if type(graph[23]) is not SCEFusion or graph[23].f != [16, 19, 22]:
            raise ValueError("SCE must consume the cached P3/P4/P5 neck outputs")
        for i, c in enumerate((64, 128, 256)):
            node = graph[24 + i]
            if type(node) is not Index or node.f != 23 or node.index != i or graph[23].channels[i] != c:
                raise ValueError("Incorrect SCE Index wiring")
    if graph[detect].f != ([24, 25, 26] if has_sce else [16, 19, 22]):
        raise ValueError("Unexpected Detect inputs")
    return {
        "variant": variant,
        "nodes": len(graph),
        "dtr": 10 if has_dtr else None,
        "sce": 23 if has_sce else None,
        "detect": detect,
        "stride": model.stride.tolist(),
    }


class DTRSCETrainer(SCETrainer):
    """Reuse SCE's migration owner, optimizer name view and fixed-batch OOM policy."""

    model_yaml = MODEL_YAML
    removed_prefixes = (REMOVED,)
    new_prefixes = NEW

    def setup_model(self):
        """Give an explicit learned checkpoint ownership before consulting any saved pretrained path."""
        if not isinstance(self.model, torch.nn.Module) and str(self.model).endswith(".pt"):
            path = Path(self.model).resolve(strict=True)
            with WorkingDirectory(path.parent):
                weights, checkpoint = load_checkpoint(path.name)
            identity(weights)
            self.model = self.get_model(cfg=weights.yaml, weights=weights)
            return checkpoint
        return super().setup_model()

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Distinguish learned combination checkpoints from SHA-verified native initialization."""
        if weights is not None and any(isinstance(m, SCEFusion) for m in weights.modules()):
            identity(weights)
        model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
        identity(model)
        return model


def environment(device="0"):
    """Compare the actual local runtime to the formal server contract without changing packages."""
    report = sce_environment()
    actual = {
        "python": platform.python_version(),
        "executable": sys.executable,
        "torch": torch.__version__,
        "torchvision": report["dependencies"]["torchvision"],
        "cuda_runtime": torch.version.cuda,
        "ultralytics": report["ultralytics"],
        "gpu": torch.cuda.get_device_name(int(device))
        if str(device).isdecimal() and torch.cuda.is_available()
        else None,
        "albumentations": report["dependencies"]["albumentations"],
    }
    expected = {
        "python": "3.12.3",
        "executable": "/root/miniconda3/bin/python",
        "torch": "2.8.0+cu128",
        "torchvision": "0.23.0+cu128",
        "cuda_runtime": "12.8",
        "ultralytics": "8.4.98",
        "gpu": "NVIDIA GeForce RTX 4090",
        "albumentations": "NOT_INSTALLED",
    }
    report["server_differences"] = {
        k: {"expected": v, "actual": actual[k]} for k, v in expected.items() if actual[k] != v
    }
    report["selected_device"] = str(device)
    report["gpu"] = actual["gpu"]
    report["server_environment"] = "PASS" if not report["server_differences"] else "UNVERIFIED"
    return report


def audit_data(path):
    """Allow only a root path relocation of the existing, fixed crack dataset."""
    data = YAML.load(Path(path).resolve(strict=True))
    expected = {"train": "images/train", "val": "images/val", "test": "images/test", "nc": 1, "names": {0: "crack"}}
    if {k: v for k, v in data.items() if k != "path"} != expected:
        raise ValueError("Dataset YAML differs from the fixed split; only root path relocation is allowed")
    return sce_audit_data(path)


def configuration(options):
    """Compare every recipe field with the verified archive, then apply only runtime overrides."""
    baseline = YAML.load(Path(options.baseline_args).resolve(strict=True))
    locked = YAML.load(BASELINE_ARGS)
    differences = {
        k: (locked.get(k), baseline.get(k))
        for k in locked.keys() | baseline.keys()
        if k not in RUNTIME_FIELDS and (k not in baseline or k not in locked or baseline[k] != locked[k])
    }
    if differences or set(DEFAULT_CFG_DICT) - set(baseline):
        raise ValueError(
            f"Incomplete or changed b19 recipe: {differences}; missing={set(DEFAULT_CFG_DICT) - set(baseline)}"
        )
    if not options.name or Path(options.name).name != options.name or options.name in {".", ".."}:
        raise ValueError("--name must be one output directory name")
    if str(options.device) != "cpu" and not str(options.device).isdecimal():
        raise ValueError(
            "Select a single explicit CUDA index or cpu; automatic selection/DDP is outside this experiment"
        )
    weights = verified_weights(options.weights)
    target = Path(options.project).expanduser().resolve() / options.name
    resolved = {
        **baseline,
        "model": str(MODEL_YAML),
        "pretrained": str(weights),
        "data": str(Path(options.data).expanduser().resolve(strict=True)),
        "project": str(target.parent),
        "name": target.name,
        "save_dir": str(target),
        "device": options.device,
    }
    resolved = vars(get_cfg(overrides=resolved)) | {"save_dir": str(target)}
    changes = {
        k: {"baseline": baseline.get(k), "resolved": resolved.get(k)}
        for k in baseline.keys() | resolved.keys()
        if baseline.get(k) != resolved.get(k)
    }
    if set(changes) - RUNTIME_FIELDS:
        raise ValueError(f"Unexpected recipe changes: {changes}")
    if options.report:
        report = Path(options.report).resolve()
        if report == target or target in report.parents:
            raise ValueError("--report must be outside the formal output directory")
    return resolved, {
        "baseline_file": str(Path(options.baseline_args).resolve()),
        "baseline_sha256": sha256(options.baseline_args),
        "baseline": baseline,
        "resolved": resolved.copy(),
        "differences": changes,
        "weights_sha256": sha256(weights),
    }


def parser(verify=False):
    """Expose exactly the documented train/verify options."""
    result = argparse.ArgumentParser(description=__doc__)
    for flag in ("data", "weights", "baseline-args"):
        result.add_argument(f"--{flag}", required=True)
    result.add_argument("--device", default="0")
    result.add_argument("--report", required=verify, help="Explicit JSON file outside the formal run directory")
    if verify:
        result.set_defaults(project=str(ROOT / "runs/detect"), name=RUN_NAME)
    else:
        result.add_argument("--project", default=str(ROOT / "runs/detect"))
        result.add_argument("--name", default=RUN_NAME)
        result.add_argument("--dry-run", action="store_true")
    return result


def construct(options, temporary_project=None):
    """Enter the real production Trainer setup chain, reserving only the intended output directory."""
    resolved, recipe = configuration(options)
    report = {
        "formal_training": "NOT_STARTED",
        "sources": frozen_sources(),
        "environment": environment(options.device),
        "configuration": recipe,
        "dataset": audit_data(options.data),
    }
    if temporary_project is None and report["environment"]["server_differences"]:
        raise RuntimeError(f"Formal server contract mismatch: {report['environment']['server_differences']}")
    if temporary_project is not None:
        resolved.update(
            project=str(temporary_project), name="construction", save_dir=str(Path(temporary_project) / "construction")
        )
    target = Path(resolved["save_dir"])
    target.mkdir(parents=True, exist_ok=False)  # Atomic reservation; explicit save_dir disables automatic suffixes.
    trainer = DTRSCETrainer(overrides=resolved)
    trainer.setup_model()
    trainer.set_model_attributes()
    report["configuration"]["trainer_effective"] = vars(trainer.args).copy()
    report["transfer"] = trainer.transfer_report
    report["identity"] = identity(trainer.model)
    report["setup_chain"] = (
        "DTRSCETrainer.setup_model -> BaseTrainer.setup_model -> DTRSCETrainer.get_model -> SCETrainer.get_model -> DetectionTrainer.get_model"
    )
    return trainer, report


def main():
    """Audit in a temporary directory or run the explicitly invoked formal training recipe."""
    options = parser().parse_args()
    # Validate the report destination before writing even an error report.
    resolved, _ = configuration(options)
    report = {"formal_training": "NOT_STARTED"}
    report_files = [options.report] if options.report else []
    results_csv = None
    try:
        if options.dry_run:
            torch.set_num_threads(min(4, torch.get_num_threads()))
            with tempfile.TemporaryDirectory(prefix="dtr-sce-dry-") as scratch:
                trainer, report = construct(options, scratch)
                report["parameters_unfused"] = sum(p.numel() for p in trainer.model.parameters())
        else:
            runtime = environment(options.device)
            if runtime["server_differences"]:
                report["environment"] = runtime
                raise RuntimeError(f"Formal server contract mismatch: {runtime['server_differences']}")
            prepare_amp_weights(resolved["pretrained"])
            trainer, report = construct(options)
            report_files.append(trainer.save_dir / "dtr_sce_audit.json")
            results_csv = trainer.csv
            report.update(result_directory=str(trainer.save_dir), summary=str(report_files[-1]))

            def record_start(active):
                if (
                    not active.amp
                    or active.batch_size != 32
                    or active.args.imgsz != 640
                    or type(active.optimizer).__name__ != "MuSGD"
                ):
                    raise RuntimeError("Formal b19 requires AMP, batch32, imgsz640 and MuSGD; native setup failed")
                report.update(
                    formal_training="STARTED", actual_amp=active.amp, actual_configuration=vars(active.args).copy()
                )
                write_report(active.save_dir / "dtr_sce_audit.json", report)

            trainer.add_callback("on_pretrain_routine_end", record_start)
            with WorkingDirectory(ROOT):
                trainer.train()
            report.update(formal_training="COMPLETED", final_metrics=trainer.metrics)
        report["status"] = "PASS"
    except Exception as error:
        report.update(
            status="FAIL",
            formal_training="FAILED" if report["formal_training"] == "STARTED" else report["formal_training"],
            error=f"{type(error).__name__}: {error}",
        )
        raise
    finally:
        if results_csv is not None and results_csv.is_file():
            with results_csv.open(encoding="utf-8", newline="") as stream:
                report["last_epoch"] = next(reversed(list(csv.DictReader(stream))), None)
        for path in report_files:
            write_report(path, report)
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k
                in {
                    "status",
                    "formal_training",
                    "parameters_unfused",
                    "last_epoch",
                    "final_metrics",
                    "result_directory",
                    "summary",
                }
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
