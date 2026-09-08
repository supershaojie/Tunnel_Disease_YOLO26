"""Evaluate best.pt once, diagnose fixed validation images, or package this PKC experiment."""

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

from tools.experiments import run_b19_pkc_sppf as shared
from tools.experiments.run_b19_pkc_sppf import MODEL

import torch

import ultralytics
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils.metrics import ConfusionMatrix
import numpy as np
import copy
from ultralytics.data.utils import IMG_FORMATS, check_det_dataset, img2label_paths
from ultralytics.nn.modules import SPPF_PKC
from ultralytics.nn.tasks import load_checkpoint
from ultralytics.utils import LOGGER, YAML


def checkpoint_provenance(run):
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
            for p in (MODEL, Path(__file__), ROOT / "ultralytics/nn/modules/pkc_sppf.py")
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


def test_best(run, data, *, split="test", block_type=SPPF_PKC, evidence=None, output=None, capture=None):
    """Evaluate once in FP32; store exact metrics and JSON from that same evaluation."""
    evidence = checkpoint_provenance(run) if evidence is None else evidence
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
            metrics = model.val(validator=CurveValidator, **settings)
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


def archive_package(run, output, *, source_files=(), required_files=None, evidence=None, exclude_dirs=()):
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
    shared.write_json(run / "environment.json", checkpoint_provenance(run) if evidence is None else evidence)
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
        "ultralytics/nn/modules/pkc_sppf.py",
        "ultralytics/nn/modules/__init__.py",
        "ultralytics/nn/tasks.py",
        "ultralytics/cfg/models/26/yolo26n-pkc-sppf-v1.yaml",
        "tools/experiments/run_b19_pkc_sppf.py",
        "tools/experiments/finish_b19_pkc_sppf.py",
        "tools/experiments/server_b19_pkc_sppf_v1.sh",
        "tools/experiments/b19_launcher_expanded.txt",
        "tools/experiments/b19_reference.json",
        "docs/experiments/b19_pkc_sppf_v1.md",
        "tests/test_pkc_sppf.py",
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


class CurveValidator(DetectionValidator):
    """Export native matched TP flags before metrics consumes them; never run a second evaluation."""

    def get_stats(self):
        stats = {k: np.concatenate(v, 0) for k, v in self.metrics.stats.items() if k in {"tp", "conf", "target_cls"}}
        order = np.argsort(-stats["conf"], kind="stable")
        scores, tp = stats["conf"][order], stats["tp"][order]
        # Evaluate all equal-confidence detections together, for reproducible Precision/FPPI comparisons.
        ends = np.flatnonzero(np.r_[scores[1:] != scores[:-1], True]) if len(scores) else np.array([], dtype=int)
        curves = {}
        for label, column in (("IoU50", 0), ("IoU75", 5)):
            true = tp[:, column].cumsum()[ends]
            false = ends + 1 - true
            curves[label] = dict(
                confidence=scores[ends].tolist(),
                precision=(true / (ends + 1)).tolist(),
                recall=(true / max(len(stats["target_cls"]), 1)).tolist(),
                fppi=(false / self.seen).tolist(),
            )
        shared.write_json(
            self.save_dir / "operating_curves.json",
            dict(
                images=self.seen,
                targets=len(stats["target_cls"]),
                curves=curves,
                interpretation="Native matching; select any operating threshold on val only, then freeze it for test",
            ),
        )
        return super().get_stats()


COUNTS = {"val": (2404, 2985), "test": (1202, 1477)}
EVAL_ARTIFACTS = (
    "operating_curves.json",
    "args.yaml",
    "predictions.json",
    "BoxPR_curve.png",
    "confusion_matrix.png",
    "confusion_matrix_normalized.png",
)


def provenance(run, data):
    """Bind reports to PKC v1 code, best weight, effective data file, and current image/label manifests."""
    evidence = checkpoint_provenance(run)
    evidence["source_sha256"] = shared.source_hashes(
        (Path(__file__), ROOT / "tools/experiments/server_b19_pkc_sppf_v1.sh")
    )
    evidence.update(version=1, data=str(data), data_sha256=shared.sha256(data))
    cfg = check_det_dataset(str(data), autodownload=False)
    manifest = {}
    for split in COUNTS:
        images = sorted(p for p in Path(cfg[split]).rglob("*") if p.suffix[1:].lower() in IMG_FORMATS)
        entries = [
            [str(p), p.stat().st_size, p.stat().st_mtime_ns, shared.sha256(label)]
            for p, label in zip(images, img2label_paths(list(map(str, images))))
        ]
        manifest[split] = hashlib.sha256(json.dumps(entries).encode()).hexdigest()
    evidence["image_stat_label_content_sha256"] = manifest
    return evidence


def report_directory(run, stage, evidence):
    """Reuse matching completed reports; preserve failed or different evidence in separate directories."""
    digest = hashlib.sha256(json.dumps([stage, evidence], sort_keys=True).encode()).hexdigest()
    parent = run / "evaluation"
    primary = parent / f"{stage}-{digest[:20]}"
    for candidate in (primary, *sorted(parent.glob(primary.name + "-retry-*"))):
        if (candidate / "metrics.json").is_file():
            record = json.loads((candidate / "metrics.json").read_text(encoding="utf-8"))
            artifacts = record.get("artifacts", {})
            required = EVAL_ARTIFACTS if stage in COUNTS else ()
            if record.get("evidence") == evidence and all(
                (candidate / name).is_file() and shared.sha256(candidate / name) == artifacts.get(name)
                for name in required
            ):
                return candidate
    candidate = primary
    attempt = 1
    while candidate.exists():
        candidate = parent / f"{primary.name}-retry-{attempt}"
        attempt += 1
    return candidate


def evaluate_best(run, data):
    """Run one independent FP32 val and test each, preserving full precision and native matrix thresholds."""
    evidence = provenance(run, data)
    records = {}
    for split, counts in COUNTS.items():

        def capture(validator):
            assert (validator.seen, int(validator.metrics.nt_per_class.sum())) == counts
            assert validator.args.conf == 0.001 and validator.args.quantize is None
            assert validator.args.rect and not validator.args.augment
            ap = validator.metrics.box.all_ap
            assert ap.shape == (1, 10)
            return dict(
                metrics={
                    "P": float(validator.metrics.box.mp),
                    "R": float(validator.metrics.box.mr),
                    "AP50": float(ap[:, 0].mean()),
                    "AP75": float(ap[:, 5].mean()),
                    "mAP50-95": float(ap.mean()),
                },
                metric_units="fraction (multiply by 100 for percent); no rounding",
                ap_by_iou={f"{0.50 + 0.05 * i:.2f}": float(ap[:, i].mean()) for i in range(10)},
                confusion_matrix=dict(
                    matrix=validator.confusion_matrix.matrix.tolist(),
                    confidence=0.25,
                    iou=0.45,
                    threshold_source="DetectionValidator.update_metrics -> ConfusionMatrix.process_batch: "
                    "default detection conf=0.001 maps to 0.25; default matching IoU=0.45",
                ),
            )

        output = report_directory(run, split, evidence)
        record = test_best(
            run, data, split=split, block_type=SPPF_PKC, evidence=evidence.copy(), output=output, capture=capture
        )
        record["artifacts"] = {name: shared.sha256(output / name) for name in EVAL_ARTIFACTS}
        shared.write_json(output / "metrics.json", record)
        records[split] = dict(
            path=str(output.relative_to(run) / "metrics.json"), sha256=shared.sha256(output / "metrics.json")
        )
    shared.write_json(run / "evaluation.json", dict(evidence=evidence, reports=records))


def diagnose(run, data, device="cuda:0"):
    """Record feature/task-gradient statistics and branch-on/off TP/FP/FN on 16 fixed validation images."""
    evidence = provenance(run, data)
    cfg = check_det_dataset(str(data), autodownload=False)
    images = sorted(p for p in Path(cfg["val"]).rglob("*") if p.suffix[1:].lower() in IMG_FORMATS)[:16]
    assert len(images) == 16
    evidence["samples"] = [[str(p), shared.sha256(p)] for p in images]
    output = report_directory(run, "diagnostics", evidence)
    if (output / "metrics.json").is_file():
        shared.write_json(
            run / "diagnostics.json",
            dict(path=str(output.relative_to(run) / "metrics.json"), sha256=shared.sha256(output / "metrics.json")),
        )
        return
    output.mkdir(parents=True, exist_ok=False)
    model, _ = load_checkpoint(evidence["weight"])
    assert type(model.model[9]) is SPPF_PKC
    model = model.float().eval().to(device)
    model.requires_grad_(True)
    model.args = shared.get_cfg(overrides=model.args)
    model.criterion = None
    # This is an inference bypass diagnostic, not an independently trained baseline or ablation.
    bypass = copy.deepcopy(model).requires_grad_(False)
    with torch.no_grad():
        bypass.model[9].pkc["project"].weight.zero_()
    validator = DetectionValidator(
        args=dict(
            data=str(data), imgsz=640, batch=32, rect=False, device=0, conf=0.001, iou=0.7, max_det=300, quantize=None
        ),
        save_dir=output,
    )
    validator.data, validator.stride, validator.end2end = cfg, 32, model.end2end
    dataset = validator.build_dataset(cfg["val"], batch=32, mode="val")
    lookup = {str(Path(path).resolve()): i for i, path in enumerate(dataset.im_files)}
    rows, tensors, handles = [], {}, []
    for name in ("l5", "l9", "l13", "project"):

        def capture(module, args, value, name=name):
            tensors[name] = value

        handles.append(model.model[9].pkc[name].register_forward_hook(capture))
    handles.append(model.model[9].cv2.register_forward_hook(lambda m, a, value: tensors.update(base=value.detach())))
    handles.append(
        model.model[9].register_forward_hook(
            lambda m, a, value: tensors.update(y0=tensors["base"] + a[0].detach() if m.add else tensors["base"])
        )
    )

    def stats(value):
        value = value.detach().float()
        assert torch.isfinite(value).all()
        return dict(mean=value.mean().item(), std=value.std(unbiased=False).item(), norm=value.norm().item())

    try:
        for path in images:
            batch = dataset.collate_fn([dataset[lookup[str(path.resolve())]]])
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            batch["img"] = batch["img"].float() / 255
            predictions = model(batch["img"])
            loss, _ = model.loss(batch, predictions)
            grads = torch.autograd.grad(loss.sum(), [tensors[k] for k in ("l5", "l9", "l13")])
            delta = tensors["project"].detach()
            y0 = tensors["y0"]
            row = dict(
                image=str(path),
                sha256=shared.sha256(path),
                label_sha256=shared.sha256(img2label_paths([str(path)])[0]),
                delta_y0_norm_ratio=(delta.norm() / (y0.norm() + 1e-6)).item(),
                delta=stats(delta),
                y0=stats(y0),
                loss=loss.detach().cpu().tolist(),
                scales={
                    k: dict(feature=stats(tensors[k]), task_gradient=stats(g))
                    for k, g in zip(("l5", "l9", "l13"), grads)
                },
            )
            with torch.no_grad():
                outcomes = {}
                for name, pred in (("branch_on", predictions), ("branch_off", bypass(batch["img"]))):
                    detected = validator.postprocess(pred)[0]
                    from ultralytics.utils.ops import xywh2xyxy

                    targets = dict(cls=batch["cls"].view(-1), bboxes=xywh2xyxy(batch["bboxes"]) * 640)
                    matrix = ConfusionMatrix(names=model.names)
                    matrix.process_batch(detected, targets, conf=0.25, iou_thres=0.5)
                    outcomes[name] = dict(
                        TP=int(matrix.matrix[0, 0]), FP=int(matrix.matrix[0, 1]), FN=int(matrix.matrix[1, 0])
                    )
                row["detections"] = outcomes
                row["on_minus_off"] = {
                    k: outcomes["branch_on"][k] - outcomes["branch_off"][k] for k in ("TP", "FP", "FN")
                }
            rows.append(row)
            tensors.clear()
            del predictions, loss, grads
    finally:
        for handle in handles:
            handle.remove()
    shared.write_json(
        output / "metrics.json",
        dict(
            evidence=evidence,
            images=rows,
            mode="FP32 eval; native no-augmentation square640 val transform; per-image task gradients; no optimizer update",
            thresholds=dict(confidence=0.25, matching_iou=0.5, max_det=300),
            interpretation="Trained branch on/off diagnostic only; no claim of b19 improvement or training ablation",
        ),
    )
    shared.write_json(
        run / "diagnostics.json",
        dict(path=str(output.relative_to(run) / "metrics.json"), sha256=shared.sha256(output / "metrics.json")),
    )


def package(run, data, output):
    """Package existing matching evaluation and diagnosis, never invoking training or evaluation."""
    evidence = provenance(run, data)
    if evidence["status"]:
        raise RuntimeError("Commit tracked source changes before packaging: source.tar must match the evaluated code")
    index = json.loads((run / "evaluation.json").read_text(encoding="utf-8"))
    assert index["evidence"] == evidence
    diagnostic = json.loads((run / "diagnostics.json").read_text(encoding="utf-8"))
    selected = []
    for report in (*index["reports"].values(), diagnostic):
        path = (run / report["path"]).resolve()
        path.relative_to(run)
        assert shared.sha256(path) == report["sha256"]
        selected.append(path.parent)
    detail = json.loads((run / diagnostic["path"]).read_text(encoding="utf-8"))
    assert {k: v for k, v in detail["evidence"].items() if k != "samples"} == evidence
    assert all(shared.sha256(path) == digest for path, digest in detail["evidence"]["samples"])
    required = [
        "args.yaml",
        "results.csv",
        "train.log",
        "weights/best.pt",
        "completed.json",
        "val_metrics.json",
        "evaluation.json",
        "diagnostics.json",
    ]
    required += [str(path.relative_to(run) / "metrics.json") for path in selected]
    for split in COUNTS:
        folder = (run / index["reports"][split]["path"]).parent
        report = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
        assert report["evidence"] == evidence
        assert all(shared.sha256(folder / name) == report["artifacts"][name] for name in EVAL_ARTIFACTS)
        required += [str((folder / name).relative_to(run)) for name in EVAL_ARTIFACTS]
    archive_package(
        run,
        output,
        required_files=required,
        evidence=evidence,
        exclude_dirs=tuple(p for p in (run / "evaluation").iterdir() if p not in selected),
    )


def main(argv=None):
    """Expose the PKC v1 completion stages; test includes both independent val and held-out test."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("test", "diagnose", "package"), required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    run = args.run.resolve()
    assert run == ROOT / "runs/detect" / shared.NAME
    completed_run(run)
    data = Path(YAML.load(run / "args.yaml")["data"]).resolve()
    if args.stage == "package":
        package(run, data, (args.output or run.with_suffix(".tar.gz")).resolve())
    else:
        {"test": evaluate_best, "diagnose": diagnose}[args.stage](run, data)


if __name__ == "__main__":
    main()
