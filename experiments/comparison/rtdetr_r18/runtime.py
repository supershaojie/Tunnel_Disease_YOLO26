"""Prepare the exact initialization and launch native C2 training."""

import json
import shutil
from pathlib import Path

from common import (
    MODEL,
    bootstrap,
    file_hash,
    parser,
    protocol,
    read_yaml,
    resolve,
    status,
    verify_backend,
    write_json,
)

INIT_NAME = "rtdetr_r18_lite_imagenet_backbone_init.pt"
INIT_SHA = "fe8501bbcc1d1d5b366bdb10fd47d0fbb91edff1f9558f39e31683a550bcad8e"


def weight_path(cfg):
    """Use only the original complete initialization, never another detector's weights."""
    return Path(cfg["weights"] or Path(cfg["pretrained_cache"]) / INIT_NAME).resolve()


def verified_pretrained(cfg):
    """Verify the user-specified original checkpoint before trusted native loading."""
    path = weight_path(cfg)
    if not path.is_file():
        raise FileNotFoundError(
            f"Original initialization missing: {path}. Copy only {INIT_NAME} from the LCBR weights directory "
            f"(FileZilla target: /root/autodl-tmp/projects/Tunnel_Disease_YOLO26/weights/pretrained/{INIT_NAME})."
        )
    if file_hash(path) != INIT_SHA:
        raise ValueError(f"Original initialization SHA256 mismatch: {path}; expected {INIT_SHA}")
    return path, {
        "filename": INIT_NAME,
        "sha256": INIT_SHA,
        "source": verify_backend()["upstream_repository"],
        "source_commit": verify_backend()["commit"],
        "source_path": "weights/" + INIT_NAME,
        "initialization": "ImageNet backbone + saved original random neck/encoder/decoder; nc=80 epoch=-1",
        "comparison_note": "YOLO comparisons use their own COCO detection pretraining; initialization differs",
    }


def prepare(cfg, args):
    """Copy an existing LCBR initialization into the shared cache or reuse it."""
    destination = weight_path(cfg)
    if not destination.exists() and args.source_weights and not args.dry_run:
        source = Path(args.source_weights).resolve()
        if file_hash(source) != INIT_SHA:
            raise ValueError(f"Source initialization SHA256 mismatch: {source}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    if args.dry_run:
        print(json.dumps({"weight": str(destination), "exists": destination.is_file(), "expected_sha256": INIT_SHA}))
        return
    path, provenance = verified_pretrained(cfg)
    write_json(path.with_suffix(".source.json"), provenance)
    print(json.dumps({"prepared": str(path), **provenance}, ensure_ascii=False))


def checkpoint_info(path, snapshot=None):
    """Verify an owned native checkpoint against its compact experiment manifest."""
    path = Path(path).resolve()
    run = path.parent.parent
    manifest = json.loads((run / "checkpoint_manifest.json").read_text(encoding="utf-8"))
    if manifest["model"] != MODEL or manifest["backend_sha"] != verify_backend()["commit"]:
        raise ValueError("Checkpoint model/backend differs from this experiment")
    if file_hash(path) != manifest["weights"].get(path.name):
        raise ValueError(f"Checkpoint SHA256 mismatch: {path}")
    if snapshot is not None:
        shutil.copyfile(path, snapshot)
        if file_hash(snapshot) != manifest["weights"][path.name]:
            raise ValueError("Checkpoint changed while capturing evaluation snapshot")
    return run, manifest


def train(cfg, args):
    """Use native RTDETRTrainer construction, matching, loss, EMA and parameter groups."""
    if args.dry_run:
        print(json.dumps(cfg, ensure_ascii=False, indent=2))
        return
    from shared.data import audit, backend_paths

    path, provenance = verified_pretrained(cfg)
    dataset, report = audit(cfg["data"], read_yaml)
    directory = (Path(cfg["project"]) / cfg["name"]).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    status(directory, "train", "RUNNING")
    try:
        bootstrap(cfg, path, data_identity=report["data_sha256"])
        from adapter import ComparisonTrainer

        dataset["channels"] = 3
        paths = backend_paths(dataset, Path(cfg["cache_dir"]) / ("splits-" + report["data_sha256"]))
        public = protocol(cfg, report["data_sha256"])
        ComparisonTrainer.context = {
            "config": cfg,
            "dataset": paths,
            "data_report": report,
            "pretrained_provenance": provenance,
            "protocol": public,
            "backend_sha": verify_backend()["commit"],
            "smoke": False,
        }
        for name, value in {
            "run_config": cfg,
            "data_manifest": report,
            "protocol": public,
            "initialization_source": provenance,
            "backend.lock": verify_backend(),
        }.items():
            write_json(directory / (name + ".json"), value)
        native = {k: v for k, v in cfg["train"].items() if k != "betas"}
        native.update(
            model=str(path),
            data=str(Path(cfg["data"]).resolve()),
            task="detect",
            mode="train",
            project=str(directory.parent),
            name=directory.name,
            exist_ok=True,
        )
        trainer = ComparisonTrainer(overrides=native)
        trainer.train()
        status(directory, "train", "COMPLETED", completed_epochs=trainer.epoch + 1)
    except BaseException as error:
        status(directory, "train", "FAILED", reason=f"{type(error).__name__}: {error}")
        raise


def main(mode):
    """Expose the existing comparison CLI with only a source-file option for prepare."""
    command = parser(mode)
    command.add_argument("--source-weights", help="Existing original LCBR initialization to copy, prepare only")
    args = command.parse_args()
    cfg = resolve(args)
    if mode in ["prepare", "train"]:
        return globals()[mode](cfg, args)
    if mode == "evaluate":
        from evaluation_runtime import evaluate

        return evaluate(cfg, args)
    if mode in ["benchmark", "complexity"]:
        from measure import benchmark, complexity

        return {"benchmark": benchmark, "complexity": complexity}[mode](cfg, args)
    if mode == "package":
        from artifacts import package

        return package(cfg, args)
    raise ValueError(mode)
