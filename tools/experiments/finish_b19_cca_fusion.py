"""Evaluate best.pt once, diagnose fixed validation images, or package this CCA experiment."""

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

from tools.experiments import run_b19_cca_fusion as shared

import torch

import ultralytics
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionValidator
import numpy as np
from ultralytics.data.utils import IMG_FORMATS, check_det_dataset, img2label_paths
from ultralytics.nn.modules import Concat_CCA_Fusion
from ultralytics.nn.tasks import load_checkpoint
from ultralytics.utils import LOGGER, YAML


def checkpoint_provenance(run, experiment=shared.EXPERIMENT):
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
        backend=dict(
            tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
            tf32_cudnn=torch.backends.cudnn.allow_tf32,
            cudnn_benchmark=torch.backends.cudnn.benchmark,
            cudnn_deterministic=torch.backends.cudnn.deterministic,
        ),
        source_sha256={
            str(p.relative_to(ROOT)): shared.sha256(p)
            for p in (
                experiment.model,
                experiment.finish_entry,
                Path(sys.modules[experiment.block_type.__module__].__file__),
            )
        },
    )


def completed_run(run):
    """Require actual successful training and the matching selected best checkpoint."""
    pointer = run.parent / f"{run.name}_train.current_attempt"
    status = Path(pointer.read_text(encoding="utf-8").strip()) / "exit_status"
    if not status.is_file() or status.read_text(encoding="utf-8").strip() != "0":
        raise RuntimeError(f"Training has no successful process exit status: {status}")
    record = json.loads((run / "completed.json").read_text(encoding="utf-8"))
    if not record.get("completed") or Path(record["run"]).resolve() != run:
        raise ValueError("Completion record does not belong to this run")
    if record["best_sha256"] != shared.sha256(run / "weights/best.pt"):
        raise ValueError("best.pt differs from the completed training checkpoint")
    if record["commit"] != shared.git("rev-parse", "HEAD"):
        raise ValueError("Training commit differs from current source")
    if shared.git("status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("Commit source changes before evaluating the fixed training checkpoint")
    return record


def test_best(run, data, *, split="test", block_type=Concat_CCA_Fusion, evidence=None, output=None, capture=None):
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
    if block_type is not None:
        assert type(model.model.model[12]) is block_type
    observation = {}
    forwards = []

    def observe_forward(module, args):
        forwards.append(
            dict(
                parameter_dtype=str(next(module.parameters()).dtype),
                input_dtype=str(args[0].dtype),
                autocast=torch.is_autocast_enabled(),
                end2end=module.end2end,
            )
        )

    observation["unfused_parameters"] = sum(p.numel() for p in model.model.parameters())
    model.model.register_forward_pre_hook(observe_forward)
    model.add_callback("on_val_start", lambda v: forwards.clear())

    def capture_validation(validator):
        assert forwards and all(
            x["parameter_dtype"] == "torch.float32" and not x["autocast"] and x["end2end"] for x in forwards
        )
        observation["actual_forwards"] = forwards
        shared.write_json(output / "predictions.json", validator.jdict)
        observation.update(
            args=vars(validator.args),
            save_dir=str(validator.save_dir),
            speed=validator.speed,
            backend=dict(
                amp_enabled=torch.is_autocast_enabled(),
                fp16=validator.args.quantize == 16,
                tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
                tf32_cudnn=torch.backends.cudnn.allow_tf32,
                cudnn_benchmark=torch.backends.cudnn.benchmark,
            ),
            peak_cuda_memory_bytes=torch.cuda.max_memory_allocated(),
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
    required = [Path(name).as_posix() for name in required]
    for name in required:
        if not (run / name).is_file():
            raise FileNotFoundError(run / name)
    shared.write_json(run / "environment.json", checkpoint_provenance(run) if evidence is None else evidence)
    (run / "pip-freeze.txt").write_text(
        subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True), encoding="utf-8"
    )
    files = {}
    for path in run.rglob("*"):
        if any(folder in path.parents for folder in exclude_dirs) or any(
            ".attempt." in part for part in path.relative_to(run).parts
        ):
            continue
        if path.is_file() and path.name not in {"package_manifest.json", "package_result.json"}:
            if path.suffix == ".pt" and path != run / "weights/best.pt":
                continue
            if path.suffix in {".yaml", ".json", ".jsonl", ".csv", ".log", ".txt", ".png", ".jpg", ".pt"}:
                files["run/" + path.relative_to(run).as_posix()] = path
    for name in required:
        assert "run/" + name in files, f"Required result was not selected for archival: {name}"
    source = [
        "tools/experiments/verify_b19_cca_fusion.py",
        "docs/experiments/b19_cca_fusion_v1_server.md",
        "ultralytics/nn/modules/cca_fusion.py",
        "ultralytics/nn/modules/__init__.py",
        "ultralytics/nn/tasks.py",
        "ultralytics/cfg/models/26/yolo26n-cca-fusion-v1.yaml",
        "tools/experiments/run_b19_cca_fusion.py",
        "tools/experiments/finish_b19_cca_fusion.py",
        "tools/experiments/server_b19_cca_fusion_v1.sh",
        "tools/experiments/b19_launcher_expanded.txt",
        "tools/experiments/b19_reference.json",
        "tools/experiments/b19_archived_args.yaml",
        "tools/experiments/b19_dataset_manifest.json",
        "tools/experiments/deploy_b19_cca_fusion_v1.sh",
        "docs/experiments/b19_cca_fusion_v1.md",
        "tests/test_cca_fusion.py",
    ] + list(source_files)
    # Transient attempts and their console aliases are not archival inputs. Durable train/eval logs
    # already live in the final run. Successful stage receipts are verified by package() before this call.
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
    manifest = {name: dict(sha256=shared.sha256(path), size_bytes=path.stat().st_size) for name, path in files.items()}
    shared.write_json(run / "package_manifest.json", manifest)
    files["package_manifest.json"] = run / "package_manifest.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as target, tarfile.open(fileobj=target, mode="w:gz", dereference=True) as archive:
        for name, path in files.items():
            archive.add(path, arcname=name, recursive=False)
    with tarfile.open(output, "r:gz") as archive:
        for name, expected in manifest.items():
            member = archive.getmember(name)
            if not member.isfile():
                raise ValueError(f"Expected materialized regular archive member: {name}")
            digest = hashlib.sha256()
            with archive.extractfile(member) as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != expected["sha256"] or member.size != expected["size_bytes"]:
                raise ValueError(f"Archive integrity mismatch: {name}")
    with gzip.open(output, "rb") as stream:
        while stream.read(1024 * 1024):
            pass  # Read through the gzip footer and CRC.
    digest = shared.sha256(output)
    output.with_name(output.name + ".sha256").write_text(f"{digest}  {output.name}\n", encoding="utf-8")
    result = dict(
        archive=str(output.resolve()),
        size_bytes=output.stat().st_size,
        sha256=digest,
        verified_files=len(manifest),
        gzip_crc_verified=True,
        key_members=["run/" + name for name in required] + ["source.tar", "package_manifest.json"],
    )
    shared.write_json(output.with_name(output.name + ".json"), result)
    shared.write_json(run / "package_result.json", result)
    print(json.dumps(result))


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
    "BoxP_curve.png",
    "BoxR_curve.png",
    "BoxF1_curve.png",
    "confusion_matrix.png",
    "confusion_matrix_normalized.png",
)


def provenance(run, data, experiment=shared.EXPERIMENT):
    """Bind reports to the selected CCA code, best weight, effective data file, and current image/label manifests."""
    evidence = checkpoint_provenance(run, experiment)
    evidence["source_sha256"] = shared.source_hashes(
        (
            Path(__file__),
            ROOT / "tools/experiments/server_b19_cca_fusion_v1.sh",
            *(ROOT / p for p in experiment.source_files),
        )
    )
    evidence.update(version=experiment.version, data=str(data), data_sha256=shared.sha256(data))
    evidence["dataset_manifest"] = shared.dataset_manifest(data)
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


def evaluate_best(run, data, only_split=None, baseline_best=None, experiment=shared.EXPERIMENT):
    """Run one independent FP32 val and test each, preserving full precision and native matrix thresholds."""
    evidence = provenance(run, data, experiment)
    report_run = run if baseline_best is None else run / "baseline_comparison"
    if baseline_best is not None:
        evidence.update(
            weight=str(baseline_best.resolve()),
            weight_sha256=shared.sha256(baseline_best),
            model_role="native_b19_comparison",
        )
        assert evidence["weight_sha256"] == "d0b2ca5a5d30de9ed002c64c9238b182ddeccce644c055ae5f2dba566878bd5e"
        baseline = YOLO(baseline_best).model
        assert shared.architecture_signature(baseline.yaml) == shared.architecture_signature(
            shared.baseline_architecture()
        )
        assert baseline.model[-1].nc == 1
        del baseline
    if only_split is None:
        records, operating = {}, None
        for split in COUNTS:
            subprocess.run(
                [sys.executable, str(experiment.finish_entry), "--stage", "test", "--run", str(run), "--split", split]
                + (["--baseline-best", str(baseline_best)] if baseline_best is not None else []),
                check=True,
                cwd=ROOT,
            )
            output = report_directory(report_run, split, evidence)
            records[split] = dict(
                path=str(output.relative_to(report_run) / "metrics.json"), sha256=shared.sha256(output / "metrics.json")
            )
            curve = json.loads((output / "operating_curves.json").read_text(encoding="utf-8"))["curves"]["IoU50"]
            if split == "val":
                valid = [i for i, p in enumerate(curve["precision"]) if p >= 0.85]
                operating = dict(
                    matching_iou=0.5,
                    target_precision=0.85,
                    threshold_source="val only, frozen before test",
                    attainable=bool(valid),
                )
                if valid:
                    i = max(valid, key=lambda i: curve["recall"][i])
                    operating.update(threshold=curve["confidence"][i], val={k: v[i] for k, v in curve.items()})
                shared.write_json(output / "frozen_operating_point.json", dict(evidence=evidence, operating=operating))
            elif operating["attainable"]:
                selected = [j for j, c in enumerate(curve["confidence"]) if c >= operating["threshold"]]
                operating["test"] = {
                    k: v[selected[-1]] if selected else (None if k == "precision" else 0.0)
                    for k, v in curve.items()
                    if k != "confidence"
                }
        shared.write_json(
            report_run / "evaluation.json", dict(evidence=evidence, reports=records, recall_at_precision=operating)
        )
        return
    for split, counts in COUNTS.items():
        if split != only_split:
            continue

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

        output = report_directory(report_run, split, evidence)
        record = test_best(
            run,
            data,
            split=split,
            block_type=experiment.block_type if baseline_best is None else None,
            evidence=evidence.copy(),
            output=output,
            capture=capture,
        )
        record["artifacts"] = {name: shared.sha256(output / name) for name in EVAL_ARTIFACTS}
        shared.write_json(output / "metrics.json", record)


def match_boxes(detected, targets, threshold):
    """Use native greedy IoU matching order and return matched target identities and their IoUs."""
    from ultralytics.utils.metrics import box_iou

    keep = detected["conf"] >= 0.25
    boxes = detected["bboxes"][keep]
    iou = box_iou(targets, boxes).cpu().numpy()
    pairs = np.array(np.where(iou > threshold)).T
    if len(pairs):
        matches = np.c_[pairs, iou[pairs[:, 0], pairs[:, 1]]]
        matches = matches[matches[:, 2].argsort()[::-1]]
        matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
        matches = matches[matches[:, 2].argsort()[::-1]]
        matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
    else:
        matches = np.empty((0, 3))
    return dict(
        TP=len(matches),
        FP=len(boxes) - len(matches),
        FN=len(targets) - len(matches),
        matched={str(int(t)): float(v) for t, _, v in matches},
    )


@torch.no_grad()
def diagnose(run, data, device="cuda:0", experiment=shared.EXPERIMENT):
    """Inspect fixed validation images with unchanged weights and a temporary residual-output bypass."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from ultralytics.utils.ops import xywh2xyxy

    evidence = provenance(run, data, experiment)
    cfg = check_det_dataset(str(data), autodownload=False)
    images = sorted(p for p in Path(cfg["val"]).rglob("*") if p.suffix[1:].lower() in IMG_FORMATS)[:16]
    assert len(images) == 16
    evidence["samples"] = [[str(p), shared.sha256(p)] for p in images]
    output = report_directory(run, "diagnostics", evidence)
    output.mkdir(parents=True, exist_ok=False)
    shared.write_json(output / "samples.json", evidence["samples"])
    model, _ = load_checkpoint(evidence["weight"])
    model = model.float().eval().to(device)
    block = model.model[12]
    assert type(block) is experiment.block_type
    assert torch.count_nonzero(block.Wo.weight) > 0
    validator = DetectionValidator(
        args=dict(
            data=str(data), imgsz=640, batch=32, rect=False, device=0, conf=0.001, iou=0.7, max_det=300, quantize=None
        ),
        save_dir=output,
    )
    validator.data, validator.stride, validator.end2end = cfg, 32, model.end2end
    dataset = validator.build_dataset(cfg["val"], batch=32, mode="val")
    lookup = {str(Path(p).resolve()): i for i, p in enumerate(dataset.im_files)}
    tensors, rows = {}, []

    def capture(module, args, result):
        tensors.update(up=args[0][0], low=args[0][1], high=args[0][2], residual=result[:, :256] - args[0][0])

    hook = block.register_forward_hook(capture)

    def stats(x):
        assert torch.isfinite(x).all()
        return dict(mean=x.mean().item(), min=x.min().item(), max=x.max().item())

    try:
        for index, path in enumerate(images):
            batch = dataset.collate_fn([dataset[lookup[str(path.resolve())]]])
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            batch["img"] = batch["img"].float() / 255
            normal = model(batch["img"])
            up, residual = tensors["up"], tensors["residual"]
            delta, weights, q, k = block.correspondence(tensors["low"], tensors["high"])
            h, w = tensors["high"].shape[-2:]
            valid_count = torch.nn.functional.unfold(torch.ones(1, 1, h, w, device=device), 3, padding=1)
            valid_count = valid_count.sum(1).reshape(1, 1, h, w).repeat_interleave(2, 2).repeat_interleave(2, 3)[0, 0]
            entropy = -(weights * weights.clamp_min(1e-30).log()).sum(1)[0]
            offsets = weights.new_tensor([(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)])
            displacement = torch.einsum("bnhw,nd->bdhw", weights, offsets)[0]
            phase = weights.reshape(1, 9, h, 2, w, 2).permute(0, 1, 2, 4, 3, 5).reshape(1, 9, h, w, 4)
            ratio = residual.float().norm(dim=1) / up.float().norm(dim=1).clamp_min(1e-6)
            row = dict(
                image=str(path),
                sha256=shared.sha256(path),
                label_sha256=shared.sha256(img2label_paths([str(path)])[0]),
                q_norm=stats(q.float().norm(dim=1)),
                k_norm=stats(k.float().norm(dim=1)),
                residual_over_original_U=stats(ratio),
                denominator_floor=1e-6,
                center_weight=stats(weights[:, 4]),
                noncenter_weight=stats(1 - weights[:, 4]),
                phase_distribution_deviation=stats((phase - phase.mean(-1, keepdim=True)).abs().sum(1)),
                expected_correspondence_position_displacement=stats(displacement.norm(dim=0)),
                support={
                    str(int(n)): dict(
                        positions=int((valid_count == n).sum()),
                        entropy=stats(entropy[valid_count == n]),
                        normalized_entropy=stats(entropy[valid_count == n] / max(float(np.log(n)), 1e-6)),
                        center=stats(weights[0, 4][valid_count == n]),
                    )
                    for n in valid_count.unique().tolist()
                },
            )
            row.update(experiment.trainer_type.mechanism_diagnostics(block, up, tensors["low"], tensors["high"]))
            # This hook changes only the returned R, with no parameter/checkpoint mutation.
            bypass = block.Wo.register_forward_hook(lambda m, a, v: torch.zeros_like(v))
            try:
                off = model(batch["img"])
            finally:
                bypass.remove()
            row["raw_one2one_max_abs_delta"] = {
                name: (v - off[1]["one2one"][name]).abs().max().item()
                for name, v in normal[1]["one2one"].items()
                if isinstance(v, torch.Tensor)
            }
            targets = xywh2xyxy(batch["bboxes"]) * batch["img"].new_tensor([640, 640, 640, 640])
            detected = {name: validator.postprocess(pred)[0] for name, pred in (("on", normal), ("off", off))}
            row["matching"] = {}
            for threshold in (0.5, 0.75):
                outcomes = {name: match_boxes(pred, targets, threshold) for name, pred in detected.items()}
                on, disabled = outcomes["on"]["matched"], outcomes["off"]["matched"]
                row["matching"][str(threshold)] = dict(
                    outcomes,
                    recovered=sorted(set(on) - set(disabled)),
                    lost=sorted(set(disabled) - set(on)),
                    common_target_iou_delta={t: on[t] - disabled[t] for t in on.keys() & disabled.keys()},
                )
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            display = batch["img"][0].permute(1, 2, 0).cpu().numpy()
            for axis, name in zip(axes[:2], ("on", "off")):
                axis.imshow(display)
                axis.set_title("CCA residual " + name)
                for box in detected[name]["bboxes"][detected[name]["conf"] >= 0.25].cpu().tolist():
                    x1, y1, x2, y2 = box
                    axis.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color="red", lw=1))
                axis.axis("off")
            axes[2].imshow(display, extent=(0, 40, 40, 0))
            yy, xx = np.mgrid[:40, :40]
            axes[2].quiver(
                xx[::2, ::2],
                yy[::2, ::2],
                displacement[1, ::2, ::2].cpu(),
                -displacement[0, ::2, ::2].cpu(),
                color="yellow",
            )
            axes[2].set_title("Expected correspondence position (coarse units)")
            fig.tight_layout()
            fig.savefig(output / f"{index:02d}_diagnostic.png", dpi=120)
            plt.close(fig)
            rows.append(row)
    finally:
        hook.remove()
    assert max(row["residual_over_original_U"]["max"] for row in rows) > 0
    detail = dict(
        evidence=evidence,
        images=rows,
        weight_norms={n: p.float().norm().item() for n, p in block.named_parameters()},
        preflight_update_evidence=json.loads((run / "provenance/checks.json").read_text(encoding="utf-8")),
        mode="FP32 eval, fixed first 16 sorted validation images; same preprocessing and weights; output-hook R bypass",
        support_labels={"4": "corner", "6": "edge", "9": "interior"},
        interpretation="Expected position of correspondence weights, not explicit offsets or measured alignment error. Heatmaps are not crack probabilities. This is a same-weight functional diagnostic, not a trained ablation or full evaluation.",
    )
    detail["artifacts"] = {p.name: shared.sha256(p) for p in output.iterdir() if p.is_file()}
    shared.write_json(output / "metrics.json", detail)
    shared.write_json(
        run / "diagnostics.json",
        dict(path=str(output.relative_to(run) / "metrics.json"), sha256=shared.sha256(output / "metrics.json")),
    )


def package(run, data, output, experiment=shared.EXPERIMENT):
    """Package existing matching evaluation and diagnosis, never invoking training or evaluation."""
    evidence = provenance(run, data, experiment)
    for stage in ("train", "test", "diagnose"):
        pointer = run.parent / f"{run.name}_{stage}.current_attempt"
        attempt = Path(pointer.read_text(encoding="utf-8").strip())
        if (attempt / "exit_status").read_text(encoding="utf-8").strip() != "0":
            raise RuntimeError(f"Current {stage} attempt did not succeed: {attempt}")
    checks = json.loads((run / "provenance/checks.json").read_text(encoding="utf-8"))
    origin = json.loads((run / "provenance/resolved.json").read_text(encoding="utf-8"))
    assert checks["passed"] and not checks["missing_effective_parameters"]
    process = json.loads((run / "provenance/preflight_process_status.json").read_text(encoding="utf-8"))
    assert process["python"] == 0 and process["commit"] == evidence["commit"]
    assert origin["commit"] == evidence["commit"] and origin["data_sha256"] == evidence["data_sha256"]
    assert origin["dataset_manifest"] == evidence["dataset_manifest"]
    assert origin["source_sha256"] == evidence["source_sha256"]
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
    assert all(
        shared.sha256((run / diagnostic["path"]).parent / name) == digest
        for name, digest in detail["artifacts"].items()
    )
    required = [
        "args.yaml",
        "results.csv",
        "train.log",
        "weights/best.pt",
        "completed.json",
        "val_metrics.json",
        "evaluation.json",
        "diagnostics.json",
        "provenance/checks.json",
        "provenance/preflight_process_status.json",
        "provenance/resolved.json",
        "provenance/final_weight_audit.json",
        "provenance/final_optimizer.json",
        "provenance/b19_original_args.yaml",
        "provenance/b19_launcher_expanded.txt",
    ]
    required += [str(path.relative_to(run) / "metrics.json") for path in selected]
    for split in COUNTS:
        folder = (run / index["reports"][split]["path"]).parent
        report = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
        assert report["evidence"] == evidence
        assert all(shared.sha256(folder / name) == report["artifacts"][name] for name in EVAL_ARTIFACTS)
        required += [str((folder / name).relative_to(run)) for name in EVAL_ARTIFACTS]
    frozen = (run / index["reports"]["val"]["path"]).with_name("frozen_operating_point.json")
    frozen_record = json.loads(frozen.read_text(encoding="utf-8"))
    assert frozen_record["evidence"] == evidence
    assert frozen_record["operating"] == {k: v for k, v in index["recall_at_precision"].items() if k != "test"}
    required.append(str(frozen.relative_to(run)))
    comparison = run / "baseline_comparison/evaluation.json"
    if experiment.version == 2 and not comparison.is_file():
        raise FileNotFoundError(f"CCA v2 requires the same-condition b19 comparison: {comparison}")
    if comparison.exists():
        baseline_index = json.loads(comparison.read_text(encoding="utf-8"))
        assert (
            baseline_index["evidence"]["weight_sha256"]
            == "d0b2ca5a5d30de9ed002c64c9238b182ddeccce644c055ae5f2dba566878bd5e"
        )
        assert baseline_index["evidence"]["commit"] == evidence["commit"]
        assert baseline_index["evidence"]["data_sha256"] == evidence["data_sha256"]
        for item in baseline_index["reports"].values():
            path = comparison.parent / item["path"]
            assert shared.sha256(path) == item["sha256"]
            detail = json.loads(path.read_text(encoding="utf-8"))
            assert detail["evidence"] == baseline_index["evidence"]
            assert all(shared.sha256(path.parent / name) == detail["artifacts"][name] for name in EVAL_ARTIFACTS)
    archive_package(
        run,
        output,
        source_files=experiment.source_files,
        required_files=required,
        evidence=evidence,
        exclude_dirs=tuple(p for p in (run / "evaluation").iterdir() if p not in selected),
    )


def main(argv=None, experiment=shared.EXPERIMENT):
    """Expose the selected CCA completion stages; test includes both independent val and held-out test."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("test", "diagnose", "package"), required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--split", choices=tuple(COUNTS))
    parser.add_argument(
        "--baseline-best",
        type=Path,
        help="Optional original trained b19 checkpoint; evaluate through identical FP32 entry",
    )
    args = parser.parse_args(argv)
    run = args.run.resolve()
    assert run == ROOT / "runs/detect" / experiment.name
    completed_run(run)
    data = Path(YAML.load(run / "args.yaml")["data"]).resolve()
    if args.stage == "package":
        package(
            run,
            data,
            (
                args.output
                or ROOT
                / "artifacts/experiments"
                / f"{experiment.name}_{shared.git('rev-parse', '--short=12', 'HEAD')}.tar.gz"
            ).resolve(),
            experiment=experiment,
        )
    else:
        if args.stage == "test":
            evaluate_best(run, data, args.split, args.baseline_best, experiment)
        else:
            diagnose(run, data, experiment=experiment)


if __name__ == "__main__":
    main()
