"""Build an actually size-checked core result archive without modifying original artifacts."""

import argparse
import gzip
import hashlib
import json
import os
import zipfile
from pathlib import Path

from comparison_io import digest, now, read_json, status, write_json

CORE = (
    "shared.lock.json",
    "initial_requested_config.json",
    "initial_config_audit.json",
    "requested_config.json",
    "raw_user_config.json",
    "config_audit.json",
    "official_defaults.json",
    "effective_config.json",
    "initialization.json",
    "pretrained_transfer.json",
    "backend.lock.json",
    "b19_source.lock.json",
    "environment.json",
    "protocol.json",
    "optimizer_setup.json",
    "optimizer_epochs.jsonl",
    "amp_check.json",
    "close_mosaic_event.json",
    "data_manifest.json",
    "train_image_index.json",
    "val_image_index.json",
    "test_image_index.json",
    "best_selection.json",
    "training_summary.json",
    "train_status.json",
    "benchmark_status.json",
    "results.csv",
    "opt.yaml",
    "hyp.yaml",
    "training_curves.svg",
    "weights/best.pt.source.json",
    "weights/last.pt.source.json",
)
EVAL = (
    "metrics.json",
    "summary.csv",
    "requested_config.json",
    "config_audit.json",
    "environment.json",
    "image_index.json",
    "predictions.npz",
    "ground_truth.npz",
    "curves.csv",
    "curves.npz",
    "PR_curve.svg",
    "P_curve.svg",
    "R_curve.svg",
    "F1_curve.svg",
    "confusion_matrix.json",
    "confusion_matrix.svg",
    "val_threshold.json",
    "fixed_images.json",
    "evaluate_status.json",
)


def check_exports(directory):
    import numpy as np

    values = read_json(directory / "metrics.json")
    index = read_json(directory / "image_index.json")
    for name, total in (("predictions.npz", values["predictions"]), ("ground_truth.npz", values["boxes"])):
        with np.load(directory / name, allow_pickle=False) as arrays:
            offsets = arrays["image_offsets"]
            if arrays["boxes"].dtype != np.float32 or arrays["classes"].dtype != np.int32:
                raise ValueError(f"Non-lossless export dtype: {name}")
            if (
                len(offsets) != len(index) + 1
                or offsets[0] != 0
                or offsets[-1] != total
                or (np.diff(offsets) < 0).any()
                or len(arrays["boxes"]) != total
            ):
                raise ValueError(f"Incomplete image/prediction offsets: {name}")
            if name == "predictions.npz" and (
                arrays["scores"].dtype != np.float32
                or np.diff(offsets).max(initial=0) > values["actual"]["NMS"]["max_det"]
            ):
                raise ValueError("Invalid prediction scores/counts")
    if len(index) != values["images"] or read_json(directory / "evaluate_status.json")["state"] != "COMPLETED":
        raise ValueError("Evaluation incomplete")
    return values


def members(run):
    run = Path(run).resolve()
    required = (
        "requested_config.json",
        "config_audit.json",
        "official_defaults.json",
        "effective_config.json",
        "initialization.json",
        "pretrained_transfer.json",
        "protocol.json",
        "data_manifest.json",
        "best_selection.json",
        "training_summary.json",
        "results.csv",
        "environment.json",
        "backend.lock.json",
        "b19_source.lock.json",
        "train.log",
    )
    for name in required:
        if not (run / name).is_file():
            raise FileNotFoundError(f"Required core artifact missing: {run / name}")
    values = {}
    for name in CORE:
        path = run / name
        if path.is_file():
            if path.is_symlink() or not path.resolve().is_relative_to(run):
                raise ValueError(f"Artifact outside run directory: {path}")
            values[name] = path.read_bytes()
    values["train.log.gz"] = gzip.compress((run / "train.log").read_bytes(), mtime=0)
    for split in ("val", "test"):
        directory = run / "evaluation" / split
        check_exports(directory)
        for name in EVAL:
            path = directory / name
            if path.is_file():
                if path.is_symlink() or not path.resolve().is_relative_to(run):
                    raise ValueError(f"Artifact outside run directory: {path}")
                values[f"evaluation/{split}/{name}"] = path.read_bytes()
        for path in sorted((directory / "examples").glob("*")):
            if path.suffix not in (".jpg", ".json") or path.is_symlink() or not path.resolve().is_relative_to(run):
                raise ValueError(f"Unexpected example artifact: {path}")
            values[path.relative_to(run).as_posix()] = path.read_bytes()
    benchmark = run / "benchmark/benchmark.json"
    if benchmark.exists():
        values["benchmark/benchmark.json"] = benchmark.read_bytes()
    else:
        values["benchmark/benchmark.json"] = json.dumps(
            {"state": "NOT_RUN", "parameters_flops_latency": "UNVERIFIED; no zero/FPS substitution"}
        ).encode()
    return values


def archive(values, output, max_bytes, selection):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Archive already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    inventory = [
        {"name": name, "bytes": len(value), "sha256": hashlib.sha256(value).hexdigest()}
        for name, value in sorted(values.items())
    ]
    values = {
        **values,
        "package_manifest.json": json.dumps(
            {"members": inventory, "selection": selection, "strict_max_bytes": max_bytes}, indent=2
        ).encode(),
    }
    temporary = output.with_name(output.name + f".{os.getpid()}.partial")
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as bundle:
            for name, value in sorted(values.items()):
                bundle.writestr(name, value)
        size = temporary.stat().st_size
        if size >= max_bytes:
            largest = sorted(inventory, key=lambda r: r["bytes"], reverse=True)[:8]
            raise ValueError(
                f"Archive exceeds strict limit: {size} >= {max_bytes}; required values preserved; largest members={largest}"
            )
        with zipfile.ZipFile(temporary) as bundle:
            if bundle.testzip() is not None:
                raise ValueError("Archive CRC verification failed")
            for item in inventory:
                if hashlib.sha256(bundle.read(item["name"])).hexdigest() != item["sha256"]:
                    raise ValueError(f"Archive member integrity mismatch: {item['name']}")
        os.replace(temporary, output)
        write_json(
            str(output) + ".manifest.json",
            {"bytes": size, "sha256": digest(output), "members": inventory, "selection": selection},
        )
        return {"path": str(output), "bytes": size, "sha256": digest(output), "max_bytes": max_bytes}
    finally:
        temporary.unlink(missing_ok=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--result-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, help="Default: run-name.core.zip beside the result directory")
    p.add_argument("--max-bytes", type=int, help="Override package.max_bytes from the saved requested config")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    run = args.result_dir.resolve()
    maximum = (
        args.max_bytes
        if args.max_bytes is not None
        else read_json(run / "requested_config.json")["package"]["max_bytes"]
    )
    if maximum <= 0:
        raise ValueError("max-bytes must be positive")
    output = args.output or run.with_name(run.name + ".core.zip")
    values = members(run)
    selection = {
        "included": "complete metrics/config/provenance/numeric curves/GT/float32 predictions/fixed examples/SVG; one compressed log",
        "excluded": "all weights, dataset images, caches, environments, event files; duplicate plot formats and duplicate full manifests",
        "original_artifacts": "preserved",
    }
    if args.dry_run:
        print(
            {
                "state": "DRY_RUN",
                "members": len(values),
                "uncompressed_bytes": sum(map(len, values.values())),
                "max_bytes": maximum,
                "archive_written": False,
            }
        )
        return
    started = now()
    status(run, "package", "RUNNING", started=started)
    try:
        result = archive(values, output, maximum, selection)
        status(run, "package", "COMPLETED", started=started, ended=now(), **result)
        print(result)
    except BaseException as error:
        status(run, "package", "FAILED", started=started, ended=now(), reason=f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    main()
