"""Shared FP32 evaluation and verified archive writer reused from b19 experiment infrastructure."""

# ruff: noqa: E402 -- Resolve the experiment worktree before importing its package.

import gzip
import hashlib
import json
import logging
import os
import subprocess
import sys
import tarfile
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.experiments import b19_common as shared
from tools.experiments.b19_common import MODEL

import torch

import ultralytics
from ultralytics import YOLO
from ultralytics.nn.modules import DSDDetect
from ultralytics.utils import LOGGER, YAML


def evaluation_conditions():
    """Bind FP32 evaluation to the mapped CUDA device and actual reusable backend policy."""
    return dict(
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        cuda_device=dict(
            logical_device=0,
            name=torch.cuda.get_device_name(0),
            uuid=str(getattr(torch.cuda.get_device_properties(0), "uuid", "unavailable")),
        )
        if torch.cuda.is_available()
        else None,
        backend=shared.computation_conditions(),
    )


def provenance(run):
    """Identify the selected checkpoint and the exact source/environment used for postprocessing."""
    path = run / "weights/best.pt"
    return dict(
        weight=str(path),
        weight_sha256=shared.sha256(path),
        commit=shared.git("rev-parse", "HEAD"),
        status=shared.git("status", "--short", "--untracked-files=no"),
        python=sys.version,
        torch=torch.__version__,
        cuda=torch.version.cuda,
        ultralytics=ultralytics.__version__,
        import_path=ultralytics.__file__,
        execution_conditions=evaluation_conditions(),
        source_sha256={
            str(p.relative_to(ROOT)): shared.sha256(p)
            for p in (MODEL, Path(__file__), ROOT / "ultralytics/nn/modules/dsd_head.py")
        },
    )


def completed_run(run):
    """Require actual successful training and the matching selected best checkpoint."""
    status = run.parent / f"{run.name}_train.exit_status"
    if not status.is_file() or status.read_text().strip() != "0":
        raise RuntimeError(f"Training has no successful process exit status: {status}")
    record = json.loads((run / "completed.json").read_text(encoding="utf-8"))
    if not record.get("completed") or Path(record["run"]).resolve() != run:
        raise ValueError("Completion record does not belong to this run")
    if record["best_sha256"] != shared.sha256(run / "weights/best.pt"):
        raise ValueError("best.pt differs from the completed training checkpoint")
    return record


def test_best(run, data, *, split="test", block_type=DSDDetect, evidence=None, output=None, capture=None, layer=23):
    """Evaluate once in FP32; store exact metrics and JSON from that same evaluation."""
    evidence = provenance(run) if evidence is None else evidence
    evidence["data_sha256"] = shared.sha256(data)
    output = run / "test" if output is None else output
    settings = dict(
        data=str(data),
        split=split,
        imgsz=640,
        batch=32,
        workers=8,
        device=0,
        conf=0.001,
        iou=0.7,
        max_det=300,
        rect=True,
        augment=False,
        quantize=None,
        plots=True,
        save_json=True,
        save_txt=True,
        save_conf=True,
        project=str(output.parent),
        name=output.name,
        exist_ok=False,
        save_dir=str(output),
    )
    receipt = output / "metrics.json"
    if receipt.is_file():
        previous = json.loads(receipt.read_text(encoding="utf-8"))
        if previous["settings"] == settings and previous["evidence"] == evidence:
            print(f"Reusing completed {split}: {receipt}")
            return previous
        raise FileExistsError(f"Conflicting test evidence, preserving: {output}")
    output.mkdir(parents=True, exist_ok=False)
    model = YOLO(evidence["weight"])
    assert type(model.model.model[layer]) is block_type
    observation = {}
    execution_conditions = evaluation_conditions()

    def capture_validation(validator):
        shared.write_json(output / "predictions.json", validator.jdict)
        tensors = list(model.model.parameters())
        assert tensors and all(p.dtype == torch.float32 for p in tensors)
        assert validator.args.split == split
        actual_conditions = evaluation_conditions()
        shared.assert_close_tree(execution_conditions, actual_conditions, path="evaluation_conditions")
        observation.update(
            execution_conditions=actual_conditions,
            actual_parameter_devices=sorted({str(p.device) for p in tensors}),
            actual_parameter_dtype="torch.float32",
            actual_split=validator.args.split,
            args=vars(validator.args),
            save_dir=str(validator.save_dir),
            speed=validator.speed,
            images=validator.seen,
            targets=int(validator.metrics.nt_per_class.sum()),
        )
        if capture is not None:
            observation.update(capture(validator))

    model.add_callback("on_val_end", capture_validation)
    handler = logging.FileHandler(output / f"{split}.log", encoding="utf-8")
    LOGGER.addHandler(handler)
    try:
        with redirect_stdout(shared.TeeStream(sys.stdout, handler.stream)), redirect_stderr(
            shared.TeeStream(sys.stderr, handler.stream)
        ):
            metrics = model.val(**settings)
            assert Path(observation["save_dir"]) == output
            # BaseValidator writes the actual AutoBackend precision back into args.quantize.
            assert observation["args"]["quantize"] is None
            result = dict(
                evidence=evidence,
                settings=settings,
                results_dict={k: float(v) for k, v in metrics.results_dict.items()},
                ap75=float(metrics.box.all_ap[:, 5].mean()),
                **observation,
            )
            YAML.save(output / "args.yaml", observation["args"])
            shared.write_json(receipt, result)
            return result
    finally:
        LOGGER.removeHandler(handler)
        handler.close()


def package(run, output, *, source_files=(), required_files=None, evidence=None, exclude_dirs=(), include_last=False):
    """Include only current-run evidence and source dependencies, then verify every archived file hash."""
    if output.exists():
        raise FileExistsError(output)
    required = (
        required_files
        if required_files is not None
        else [
            "args.yaml",
            "results.csv",
            "train.log",
            "weights/best.pt",
            "test/metrics.json",
            "test/predictions.json",
            "diagnostics.json",
            "val_metrics.json",
            "completed.json",
        ]
    )
    for name in required:
        if not (run / name).is_file():
            raise FileNotFoundError(run / name)
    shared.write_json(run / "environment.json", provenance(run) if evidence is None else evidence)
    (run / "pip-freeze.txt").write_text(
        subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True), encoding="utf-8"
    )
    files = {}
    for path in run.rglob("*"):
        if any(folder in path.parents for folder in exclude_dirs):
            continue
        if path.is_file() and path != run / "package_manifest.json":
            if path.suffix == ".pt" and path not in {
                run / "weights/best.pt",
                *([run / "weights/last.pt"] if include_last else []),
            }:
                continue
            if path.suffix in {".yaml", ".json", ".jsonl", ".csv", ".log", ".txt", ".png", ".jpg", ".pt"}:
                files["run/" + path.relative_to(run).as_posix()] = path
    source = [
        "ultralytics/nn/modules/dsd_head.py",
        "ultralytics/nn/modules/__init__.py",
        "ultralytics/nn/tasks.py",
        "ultralytics/cfg/models/26/yolo26n-dsd-head-v1.yaml",
        "tools/experiments/run_b19_dsd_head.py",
        "tools/experiments/finish_b19_dsd_head.py",
        "tools/experiments/server_b19_dsd_head_v1.sh",
        "tools/experiments/b19_launcher_expanded.txt",
        "tools/experiments/b19_reference.json",
        "tools/experiments/b19_common.py",
        "tools/experiments/b19_finish.py",
        "docs/experiments/b19_dsd_head_v1.md",
        "tests/test_dsd_head.py",
    ] + list(source_files)
    for suffix in (
        f"{stage}.{extension}"
        for stage in ("preflight", "train", "test", "diagnose")
        for extension in ("exit_status", "process_status.json", "console.log")
    ):
        path = run.parent / f"{run.name}_{suffix}"
        if path.is_file():
            files["run/" + path.name] = path
    patch = run / "source.patch"
    patch.write_bytes(
        subprocess.check_output(
            [
                "git",
                "-c",
                f"safe.directory={ROOT.as_posix()}",
                "diff",
                "--binary",
                shared.REFERENCE["source_commit"],
                "HEAD",
            ],
            cwd=ROOT,
        )
    )
    files["source.patch"] = patch
    files.update({"source/" + Path(name).as_posix(): ROOT / name for name in source})
    # Bundle the exact Git source tree for an independently reconstructable checkout, without datasets/runs.
    source_archive = run / "source.tar"
    subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={ROOT.as_posix()}",
            "archive",
            "--format=tar",
            "-o",
            str(source_archive),
            "HEAD",
        ],
        cwd=ROOT,
        check=True,
    )
    files["source.tar"] = source_archive
    if output in [p.resolve() for p in files.values()]:
        raise ValueError("Archive must not contain itself")
    manifest = {name: shared.sha256(path) for name, path in files.items()}
    shared.write_json(run / "package_manifest.json", manifest)
    files["package_manifest.json"] = run / "package_manifest.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as target, tarfile.open(fileobj=target, mode="w:gz") as archive:
        for name, path in files.items():
            archive.add(path, arcname=name, recursive=False)
    with tarfile.open(output, "r:gz") as archive:
        for name, expected in manifest.items():
            digest = hashlib.sha256()
            with archive.extractfile(name) as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != expected:
                raise ValueError(f"Archive integrity mismatch: {name}")
    with gzip.open(output, "rb") as stream:
        while stream.read(1024 * 1024):
            pass  # Read through the gzip footer and CRC.
    digest = shared.sha256(output)
    output.with_name(output.name + ".sha256").write_text(f"{digest}  {output.name}\n", encoding="utf-8")
    print(
        json.dumps(
            dict(archive=str(output), size_bytes=output.stat().st_size, sha256=digest, verified_files=len(manifest))
        )
    )
