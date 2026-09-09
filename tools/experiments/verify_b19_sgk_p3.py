"""Reproducible local SGK contracts and real-image Validator smoke test, never a formal preflight."""

# ruff: noqa: E402 -- Import this worktree before loading Ultralytics.

import argparse
import os
import subprocess
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.experiments import run_b19_sgk_p3 as run
from tools.experiments.finish_b19_sgk_p3 import CurveValidator, diagnose

import torch
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel, load_checkpoint


def main():
    """Check original source evidence and explicitly label bounded local development results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--baseline-args", type=Path, required=True)
    parser.add_argument("--dataset-zip", type=Path, required=True)
    parser.add_argument("--reference-zip", type=Path, required=True)
    parser.add_argument("--baseline-launcher", type=Path)
    options = parser.parse_args()
    os.chdir(ROOT)
    torch.set_num_threads(2)
    output = ROOT / "artifacts/sgk_local"
    output.mkdir(parents=True, exist_ok=True)
    evidence = dict(
        scope="Local development only; no server preflight or accuracy claim",
        commit=run.git("rev-parse", "HEAD"),
        python=sys.version,
        torch=torch.__version__,
        cuda=torch.version.cuda,
    )
    assert run.sha256(options.pretrained) == run.PRETRAINED_SHA256
    assert run.sha256(options.baseline_args) == run.REFERENCE["args_sha256"]
    assert (
        run.sha256(options.baseline_args.parent / "results.csv")
        == "44795068d0ebc687c29ff7b7b8cbf337469546b2d4dd515cb0616c270bdf9507"
    )
    actual = run.YAML.load(options.baseline_args)
    assert actual == run.REFERENCE["args"]
    launcher = options.baseline_launcher or ROOT / "tools/experiments/b19_launcher_expanded.txt"
    evidence["launcher"] = dict(
        run.launcher_evidence(SimpleNamespace(baseline_launcher=launcher), actual),
        original_server_file_provided=options.baseline_launcher is not None,
    )
    for name in ("pretrained", "baseline_args", "dataset_zip", "reference_zip"):
        path = getattr(options, name).resolve()
        evidence[name] = dict(path=str(path), sha256=run.sha256(path))
    assert evidence["reference_zip"]["sha256"] == "a0c197f38e1510291a9a762fc7dc0a8c13694742fff9e3132c50a3ee32ddadcf"
    with zipfile.ZipFile(options.reference_zip) as archive:
        evidence["reference_members"] = {
            n: __import__("hashlib").sha256(archive.read(n)).hexdigest()
            for n in archive.namelist()
            if n.endswith(("CGhalfConv_2025ESWA.py", "MRFAConv_2025ICCV.py"))
        }
    data = output / "real_val_subset"
    with zipfile.ZipFile(options.dataset_zip) as archive:
        names = archive.namelist()
        counts = {}
        for split, expected in run.REFERENCE["dataset_counts"].items():
            images = sorted(
                n for n in names if f"/images/{split}/" in n and n.lower().endswith((".jpg", ".png", ".jpeg"))
            )
            labels = [n for n in names if f"/labels/{split}/" in n and n.endswith(".txt")]
            targets = sum(len(archive.read(n).decode().splitlines()) for n in labels)
            assert len(images) == expected
            if split in ("val", "test"):
                assert targets == {"val": 2985, "test": 1477}[split]
            counts[split] = dict(images=len(images), targets=targets)
            if split == "val":
                for image in images[:16]:
                    label = str(Path(image.replace("/images/", "/labels/")).with_suffix(".txt")).replace("\\", "/")
                    for member in (image, label):
                        destination = data / Path(member).relative_to(Path(member).parts[0])
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        destination.write_bytes(archive.read(member))
    evidence["dataset_counts"] = counts
    run.write_json(output / "source_evidence.json", evidence)
    run.structural_checks(output)
    env = {
        **os.environ,
        "SGK_TEST_PRETRAINED": str(options.pretrained.resolve()),
        "YOLO_AUTOINSTALL": "false",
        "YOLO_OFFLINE": "true",
    }
    with (output / "unittest.log").open("w", encoding="utf-8") as stream:
        result = subprocess.run(
            [sys.executable, "-m", "unittest", "tests.test_sgk_p3", "-v"],
            cwd=ROOT,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
    result.check_returncode()
    run.YAML.save(
        data / "data.yaml",
        dict(path=str(data), train="images/val", val="images/val", test="images/val", names={0: "crack"}, nc=1),
    )
    # Nonzero projection is confined to a disposable functional probe, not a trained experiment.
    source, _ = load_checkpoint(options.pretrained)
    model = DetectionModel(str(run.MODEL), nc=1, verbose=False).eval()
    model.load(source)
    model.names = {0: "crack"}
    with torch.no_grad():
        model.model[16].sgk.Po.weight.normal_(std=0.01)
    import tempfile

    probe = Path(tempfile.mkdtemp(prefix="functional-probe-", dir=output))
    (probe / "weights").mkdir()
    torch.save(
        dict(model=model, train_args={k: v for k, v in actual.items() if k != "save_dir"}), probe / "weights/best.pt"
    )
    facade = YOLO(probe / "weights/best.pt")
    calls = []
    handle = facade.model.model[16].sgk.register_forward_hook(lambda *args: calls.append(True))
    try:
        metrics = facade.val(
            validator=CurveValidator,
            data=str(data / "data.yaml"),
            imgsz=640,
            batch=1,
            device=0 if torch.cuda.is_available() else "cpu",
            workers=0,
            quantize=None,
            conf=0.001,
            rect=True,
            plots=True,
            save_json=True,
            project=str(probe),
            name="validator",
            exist_ok=False,
        )
        assert calls
    finally:
        handle.remove()
    run.write_json(
        output / "validator.json",
        dict(
            scope="16 real validation images, untrained nonzero-branch functional probe; not evaluation",
            sgk_calls=len(calls),
            metrics=metrics.results_dict,
            output=str(probe),
        ),
    )
    diagnose(probe, data / "data.yaml", device="cuda:0" if torch.cuda.is_available() else "cpu")
    run.write_json(
        output / "complete.json",
        dict(
            passed=True,
            scope=evidence["scope"],
            checks=[
                "original hashes",
                "dataset counts",
                "structure",
                "formula indices",
                "native task gradients",
                "MuSGD",
                "EMA",
                "fresh reload",
                "fuse",
                "CUDA batch1",
                "real Validator",
                "real-image diagnose",
            ],
            server_preflight_passed=False,
            functional_probe=str(probe),
        ),
    )
    print(f"Local verification completed: {output}; server batch32 preflight remains required")


if __name__ == "__main__":
    main()
