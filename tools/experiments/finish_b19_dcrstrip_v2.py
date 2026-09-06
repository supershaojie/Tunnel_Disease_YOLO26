"""Evaluate best.pt once, diagnose fixed validation images, or package this v2 experiment."""

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

from tools.experiments import run_b19_dcrstrip as shared
from tools.experiments.run_b19_dcrstrip_v2 import MODEL

import cv2
import torch

import ultralytics
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from ultralytics.data.utils import IMG_FORMATS, check_det_dataset
from ultralytics.nn.modules import C3k2_DCRStripV2
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
            for p in (MODEL, Path(__file__), ROOT / "ultralytics/nn/modules/dcr_strip_v2.py")
        },
    )


def test_best(run, data):
    """Run one FP32 split=test val call and save its unrounded metrics, effective args, plots and JSON."""
    evidence = provenance(run)
    model = YOLO(evidence["weight"])
    assert type(model.model.model[4]) is C3k2_DCRStripV2
    output = run / "test"
    output.mkdir(exist_ok=False)

    observation = {}

    def capture_validation(validator):
        observation.update(
            args=vars(validator.args),
            speed=validator.speed,
            images=validator.seen,
            targets=int(validator.metrics.nt_per_class.sum()),
        )

    model.add_callback("on_val_end", capture_validation)
    handler = logging.FileHandler(output / "test.log", encoding="utf-8")
    LOGGER.addHandler(handler)
    try:
        with redirect_stdout(shared.TeeStream(sys.stdout, handler.stream)), redirect_stderr(
            shared.TeeStream(sys.stderr, handler.stream)
        ):
            metrics = model.val(
                data=str(data),
                split="test",
                imgsz=640,
                batch=32,
                workers=8,
                device=0,
                conf=0.001,
                iou=0.7,
                max_det=300,
                rect=True,
                augment=False,
                quantize=None,  # 8.4.98: half=False maps to None; None selects conventional FP32.
                plots=True,
                save_json=True,
                project=str(run),
                name="test",
                exist_ok=True,  # Directory was atomically reserved above, so there is no silent second test.
            )
            shared.write_json(
                output / "metrics.json",
                dict(
                    evidence=evidence,
                    results_dict={k: float(v) for k, v in metrics.results_dict.items()},
                    **observation,
                ),
            )
            YAML.save(output / "args.yaml", observation["args"])
    except Exception:
        logging.getLogger("ultralytics").exception("v2 test failed")
        raise
    finally:
        LOGGER.removeHandler(handler)
        handler.close()


@torch.no_grad()
def diagnose(run, data):
    """Read four fixed validation images with no augmentation and record compact branch statistics."""
    evidence = provenance(run)
    model, _ = load_checkpoint(evidence["weight"])
    assert type(model.model[4]) is C3k2_DCRStripV2
    model = model.float().eval().to("cuda:0")
    cfg = check_det_dataset(str(data), autodownload=False)
    val = Path(cfg["val"])
    # The recorded b19 split is images/val, validated by the training runner.
    images = sorted(p for p in val.rglob("*") if p.suffix[1:].lower() in IMG_FORMATS)[:4]
    if len(images) != 4:
        raise ValueError(f"Expected four real validation images under {val}")
    block = model.model[4].dcr
    rows = []

    def inspect(module, inputs):
        feature = inputs[0]
        content, compensation, weights, q = module.components(feature)
        delta = module.expand(content + compensation)
        rows.append(
            dict(
                gate_mean=weights.mean((0, 2, 3)).cpu().tolist(),
                gate_std=weights.std((0, 2, 3), unbiased=False).cpu().tolist(),
                q_min=q.amin((0, 2, 3)).cpu().tolist(),
                q_max=q.amax((0, 2, 3)).cpu().tolist(),
                content_norm=content.norm().item(),
                beta_contrast_norm=compensation.norm().item(),
                beta_contrast_content_ratio=(compensation.norm() / (content.norm() + 1e-6)).item(),
                residual_main_ratio=(module.alpha * delta).norm().item() / (feature.norm().item() + 1e-6),
            )
        )

    hook = block.register_forward_pre_hook(inspect)
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
    ratios = torch.tensor([row["residual_main_ratio"] for row in rows])
    shared.write_json(
        run / "diagnostics.json",
        dict(
            evidence=evidence,
            alpha=block.alpha.item(),
            beta=block.beta.item(),
            directions=["H", "V", "D", "AD"],
            mode="best.pt FP32 eval/no_grad; first four sorted val images; 640 square letterbox; no augmentation",
            q_interpretation="relative contrast statistic, not calibrated crack probability",
            residual_main_ratio=dict(
                mean=ratios.mean().item(),
                std=ratios.std(unbiased=False).item(),
                min=ratios.min().item(),
                max=ratios.max().item(),
            ),
            images=rows,
        ),
    )


def package(run, output):
    """Include only current-run evidence and source dependencies, then verify every archived file hash."""
    required = [
        "args.yaml",
        "results.csv",
        "train.log",
        "weights/best.pt",
        "test/metrics.json",
        "test/predictions.json",
        "diagnostics.json",
    ]
    for name in required:
        if not (run / name).is_file():
            raise FileNotFoundError(run / name)
    shared.write_json(run / "environment.json", provenance(run))
    (run / "pip-freeze.txt").write_text(
        subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True), encoding="utf-8"
    )
    files = {}
    for path in run.rglob("*"):
        if path.is_file() and path != run / "package_manifest.json":
            if path.suffix == ".pt" and path != run / "weights/best.pt":
                continue
            if path.suffix in {".yaml", ".json", ".jsonl", ".csv", ".log", ".txt", ".png", ".jpg", ".pt"}:
                files["run/" + path.relative_to(run).as_posix()] = path
    source = [
        "ultralytics/nn/modules/dcr_strip.py",
        "ultralytics/nn/modules/dcr_strip_v2.py",
        "ultralytics/nn/modules/__init__.py",
        "ultralytics/nn/tasks.py",
        "ultralytics/engine/trainer.py",
        "ultralytics/cfg/models/26/yolo26n-dcrstrip-v2.yaml",
        "tools/experiments/run_b19_dcrstrip.py",
        "tools/experiments/run_b19_dcrstrip_v2.py",
        "tools/experiments/finish_b19_dcrstrip_v2.py",
        "tools/experiments/server_b19_dcrstrip_v2.sh",
        "tools/experiments/b19_launcher_expanded.txt",
        "tools/experiments/b19_reference.json",
        "docs/experiments/b19_dcrstrip_v2.md",
        "tests/test_dcr_strip_v2.py",
    ]
    files.update({"source/" + name: ROOT / name for name in source})
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
    print(json.dumps(dict(archive=str(output), sha256=shared.sha256(output), verified_files=len(manifest))))


def main(argv=None):
    """Expose independent completion steps; package never triggers another test evaluation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("test", "diagnose", "package"), required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--data", type=Path, help="Defaults to the data path saved in this run's args.yaml")
    parser.add_argument("--output", type=Path, help="Package destination; defaults to a sibling run-name.tar.gz")
    args = parser.parse_args(argv)
    run = args.run.resolve()
    if args.stage == "package":
        package(run, (args.output or run.with_suffix(".tar.gz")).resolve())
    else:
        data = (args.data or Path(YAML.load(run / "args.yaml")["data"])).resolve()
        {"test": test_best, "diagnose": diagnose}[args.stage](run, data)


if __name__ == "__main__":
    main()
