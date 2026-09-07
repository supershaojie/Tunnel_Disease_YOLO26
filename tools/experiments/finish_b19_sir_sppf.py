"""Evaluate best.pt once, diagnose fixed validation images, or package this SIR experiment."""

# ruff: noqa: E402 -- Resolve the experiment worktree before importing its package.

import argparse
import gzip
import hashlib
import json
import logging
import subprocess
import sys
import tarfile
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.experiments import run_b19_sir_sppf as shared
from tools.experiments.run_b19_sir_sppf import MODEL

import cv2
import torch

import ultralytics
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from ultralytics.data.utils import IMG_FORMATS, check_det_dataset
from ultralytics.nn.modules import SPPF_SIR
from ultralytics.nn.tasks import load_checkpoint
from ultralytics.utils import LOGGER, YAML


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
        source_sha256={
            str(p.relative_to(ROOT)): shared.sha256(p)
            for p in (MODEL, Path(__file__), ROOT / "ultralytics/nn/modules/sir_sppf.py")
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


def test_best(run, data, *, split="test", block_type=SPPF_SIR, evidence=None, output=None, capture=None):
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
    assert type(model.model.model[9]) is block_type
    observation = {}

    def capture_validation(validator):
        shared.write_json(output / "predictions.json", validator.jdict)
        observation.update(
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


@torch.no_grad()
def diagnose(run, data):
    """Inspect 16 fixed real val images using a temporary hook, without adding state to normal forward."""
    evidence = provenance(run)
    output = run / "diagnostics.json"
    if output.exists():
        raise FileExistsError(f"Preserving existing diagnostic: {output}")
    model, _ = load_checkpoint(evidence["weight"])
    assert type(model.model[9]) is SPPF_SIR
    model = model.float().eval().to("cuda:0")
    cfg = check_det_dataset(str(data), autodownload=False)
    images = sorted(p for p in Path(cfg["val"]).rglob("*") if p.suffix[1:].lower() in IMG_FORMATS)[:16]
    if len(images) != 16:
        raise ValueError("Expected 16 real validation images")
    rows = []

    def inspect(module, inputs, routed):
        x = inputs[0]
        z = [module.cv1(x)]
        z.extend(module.m(z[-1]) for _ in range(module.n))
        increments = [b - a for a, b in zip(z, z[1:])]
        gates = module.router(torch.cat([z[0], *increments], 1)).tanh().chunk(module.n, 1)
        correction = 0
        scales = []
        for raw, delta, gate in zip(z[1:], increments, gates):
            correction = correction + 0.5 * gate * delta
            coefficient = 1 + 0.5 * gate
            scales.append(
                dict(
                    tanh_mean=gate.mean().item(),
                    tanh_std=gate.std(unbiased=False).item(),
                    coefficient_mean=coefficient.mean().item(),
                    coefficient_std=coefficient.std(unbiased=False).item(),
                    saturation_ratio=(gate.abs() >= 0.99).float().mean().item(),
                    correction_raw_norm_ratio=(correction.norm() / (raw.norm() + 1e-6)).item(),
                )
            )
        bypass = module.cv2(torch.cat(z, 1))
        if module.add:
            bypass = bypass + x
        rows.append(
            dict(scales=scales, module_change_norm_ratio=((routed - bypass).norm() / (bypass.norm() + 1e-6)).item())
        )

    hook = model.model[9].register_forward_hook(inspect)
    try:
        for path in images:
            img = cv2.imread(str(path))
            if img is None:
                raise ValueError(f"Unreadable validation image: {path}")
            img = LetterBox((640, 640), auto=False, stride=32)(image=img)
            tensor = torch.from_numpy(img[..., ::-1].transpose(2, 0, 1).copy()).unsqueeze(0).to("cuda:0").float() / 255
            model(tensor)
            rows[-1].update(image=str(path), sha256=shared.sha256(path))
    finally:
        hook.remove()
    shared.write_json(
        output,
        dict(
            evidence=evidence,
            mode="best.pt FP32 eval/no_grad; first 16 sorted val images; 640 letterbox; no augmentation",
            saturation_definition="abs(tanh(logit)) >= 0.99",
            interpretation="Feature diagnostic using the same trained SPPF weights; not a retrained ablation or crack-edge proof",
            images=rows,
        ),
    )


def package(run, output, *, source_files=(), required_files=None, evidence=None, exclude_dirs=()):
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
            if path.suffix == ".pt" and path != run / "weights/best.pt":
                continue
            if path.suffix in {".yaml", ".json", ".jsonl", ".csv", ".log", ".txt", ".png", ".jpg", ".pt"}:
                files["run/" + path.relative_to(run).as_posix()] = path
    source = [
        "ultralytics/nn/modules/sir_sppf.py",
        "ultralytics/nn/modules/__init__.py",
        "ultralytics/nn/tasks.py",
        "ultralytics/cfg/models/26/yolo26n-sir-sppf-v1.yaml",
        "tools/experiments/run_b19_sir_sppf.py",
        "tools/experiments/finish_b19_sir_sppf.py",
        "tools/experiments/server_b19_sir_sppf_v1.sh",
        "tools/experiments/b19_launcher_expanded.txt",
        "tools/experiments/b19_reference.json",
        "docs/experiments/b19_sir_sppf_v1.md",
        "tests/test_sir_sppf.py",
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
    # Retain all core evidence and best.pt; bound optional plots before writing a light package.
    optional = [n for n, p in files.items() if p.suffix.lower() in {".png", ".jpg"} and "batch" in p.name]
    for name in optional[8:]:
        del files[name]
    if sum(p.stat().st_size for p in files.values()) > 150 * 1024**2:
        print("Largest package inputs:", sorted(((p.stat().st_size, n) for n, p in files.items()), reverse=True)[:12])
        raise ValueError("Core package inputs still exceed 150 MiB; inspect listed files before packaging")
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


def main(argv=None):
    """Expose independent completion steps; package never triggers another test evaluation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("test", "diagnose", "package"), required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--data", type=Path, help="Defaults to the data path saved in this run's args.yaml")
    parser.add_argument("--output", type=Path, help="Package destination; defaults to a sibling run-name.tar.gz")
    args = parser.parse_args(argv)
    run = args.run.resolve()
    completed_run(run)
    if args.stage == "package":
        package(run, (args.output or run.with_suffix(".tar.gz")).resolve())
    else:
        data = (args.data or Path(YAML.load(run / "args.yaml")["data"])).resolve()
        {"test": test_best, "diagnose": diagnose}[args.stage](run, data)


if __name__ == "__main__":
    main()
