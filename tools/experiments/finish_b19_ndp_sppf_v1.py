"""Independent FP32 val/test, fixed-val NDP diagnosis and verifiable result packaging."""

# ruff: noqa: E402 -- Bind this worktree and offline environment before importing its package.

import argparse
import json
import os
import subprocess
import sys
import tarfile
import tempfile
from importlib.metadata import distributions
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("YOLO_AUTOINSTALL", "false")
os.environ.setdefault("YOLO_OFFLINE", "true")

import cv2
import numpy as np
import torch

from tools.experiments import b19_common as common
from tools.experiments.ndp_experiment import V1
from tools.experiments.verify_b19_ndp_sppf_v1 import bypass
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from ultralytics.data.utils import IMG_FORMATS, check_det_dataset
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.nn.tasks import load_checkpoint
from ultralytics.utils import YAML
from ultralytics.utils.metrics import box_iou
from ultralytics.utils.ops import scale_boxes, xywh2xyxy


class RecallValidator(DetectionValidator):
    """Retain native IoU=0.5 matching before the native metrics clear their raw stats."""

    def get_stats(self):
        stats = self.metrics.stats
        self.audit_raw = {k: np.concatenate(v) for k, v in stats.items() if v}
        self.recall_raw = {
            "conf": np.concatenate(stats["conf"]),
            "tp": np.concatenate(stats["tp"])[:, 0],
            "targets": sum(len(x) for x in stats["target_cls"]),
        }
        return super().get_stats()


def recall_curve(raw, images, threshold=None, select=True):
    """Use val-only tied confidence boundaries; freeze that threshold on test."""
    conf, tp = raw["conf"], raw["tp"]
    order = np.argsort(-conf, kind="stable")
    conf, tp = conf[order], tp[order]
    ends = np.r_[np.flatnonzero(np.diff(conf)) + 1, len(conf)] if len(conf) else np.array([], dtype=int)
    rows = []
    cumulative = np.cumsum(tp)
    for n in ends:
        true = int(cumulative[n - 1])
        false = int(n - true)
        rows.append(
            {
                "threshold": float(conf[n - 1]),
                "tp": true,
                "fp": false,
                "fn": raw["targets"] - true,
                "precision": true / int(n),
                "recall": true / max(raw["targets"], 1),
                "fppi": false / images,
            }
        )
    feasible = [r for r in rows if r["precision"] >= 0.85] if select else []
    selected = max(feasible, key=lambda r: (r["recall"], -r["fppi"])) if feasible else None
    fixed = None
    if threshold is not None:
        mask = conf >= threshold
        true = int(tp[mask].sum())
        false = int(mask.sum() - true)
        fixed = {
            "threshold": threshold,
            "tp": true,
            "fp": false,
            "fn": raw["targets"] - true,
            "precision": true / max(int(mask.sum()), 1),
            "recall": true / max(raw["targets"], 1),
            "fppi": false / images,
        }
    return {
        "iou": 0.5,
        "selection_rule": "val only: maximum recall among tied confidence boundaries with precision >= 0.85",
        "selection_status": "selected" if selected else "unavailable or selection disabled on test",
        "matching": "native DetectionValidator; confidence ties retained together",
        "curve": rows,
        "val_selected": selected,
        "frozen_threshold_result": fixed,
    }


def evaluate(weight, data, output, split, device=0, batch=32, workers=8, experiment=V1, threshold=None, baseline=False):
    """Run the repository's unified evaluation settings and capture the real backend precision and predictions."""
    output.mkdir(parents=True, exist_ok=False)
    model = YOLO(weight)
    if not baseline:
        assert type(model.model.model[9]) is experiment.block_type
    settings = {
        "data": str(data),
        "split": split,
        "imgsz": 640,
        "batch": batch,
        "workers": workers,
        "device": device,
        "conf": 0.001,
        "iou": 0.7,
        "max_det": 300,
        "rect": True,
        "augment": False,
        "quantize": None,
        "plots": True,
        "save_json": True,
        "save_txt": True,
        "save_conf": True,
        "project": str(output.parent),
        "name": output.name,
        "exist_ok": False,
        "save_dir": str(output),
    }
    record = {
        "split": split,
        "precision": "FP32",
        "weight": str(weight),
        "weight_sha256": common.sha256(weight),
        "commit": common.git("rev-parse", "HEAD"),
        "data_sha256": common.sha256(data),
        "source_sha256": common.source_hashes(),
        "settings": settings,
        "precision_mapping": "half=False is quantize=None in this pinned source",
    }

    def capture(validator):
        assert validator.args.quantize is None
        params = list(model.model.parameters())
        assert all(p.dtype == torch.float32 for p in params)
        assert validator.args.split == split and Path(validator.save_dir) == output
        assert validator.args.rect and not validator.args.augment
        ap = validator.metrics.box.all_ap
        record.update(
            images=validator.seen,
            targets=int(validator.metrics.nt_per_class.sum()),
            speed=validator.speed,
            args=vars(validator.args),
            results_dict=validator.metrics.results_dict,
            ap_by_iou={f"{0.5 + i * 0.05:.2f}": float(ap[:, i].mean()) for i in range(10)},
            confusion_matrix=validator.confusion_matrix.matrix.tolist(),
            confusion_thresholds={"conf": validator.args.conf, "iou": 0.45},
            actual_parameter_dtype="torch.float32",
            fused=model.model.is_fused(),
            precision_recall_rule="native best-F1 operating point; see recall.json for fixed thresholds",
            backend=common.computation_conditions(),
        )
        np.savez_compressed(output / "matching_stats.npz", **validator.audit_raw)
        recall = recall_curve(validator.recall_raw, validator.seen, threshold, select=split == "val")
        if split == "test":
            recall["val_selected"] = None
            recall["threshold_source"] = "frozen validation threshold; no test selection"
        record["recall"] = {
            "val_selected": recall["val_selected"],
            "frozen_threshold_result": recall["frozen_threshold_result"],
        }
        common.write_json(output / "recall.json", recall)
        record["memory"] = {
            "peak_allocated_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
            "peak_reserved_bytes": torch.cuda.max_memory_reserved() if torch.cuda.is_available() else None,
        }
        common.write_json(output / "predictions.json", validator.jdict)
        YAML.save(output / "args.yaml", vars(validator.args))

    calls = {"ndp": 0, "one2one": 0}

    def observe_ndp(module, inputs, output):
        assert output.dtype == torch.float32
        calls["ndp"] += 1

    def observe_head(module, inputs, output):
        assert module.end2end and output[1]["one2one"]["scores"].numel() > 0
        assert baseline or calls["ndp"] > calls["one2one"]
        calls["one2one"] += 1

    handles = [] if baseline else [model.model.model[9].register_forward_hook(observe_ndp)]
    handles.append(model.model.model[-1].register_forward_hook(observe_head))
    model.add_callback("on_val_end", capture)
    try:
        model.val(validator=RecallValidator, **settings)
    finally:
        for handle in handles:
            handle.remove()
    assert calls["one2one"] > 0
    record["inference_path"] = dict(version=experiment.version, **calls)
    record["artifacts"] = {p.relative_to(output).as_posix(): common.sha256(p) for p in output.rglob("*") if p.is_file()}
    common.write_json(output / "metrics.json", record)
    print(
        json.dumps(
            {
                "split": split,
                "P_R_AP50_mAP50_95": record["results_dict"],
                "AP75": record["ap_by_iou"]["0.75"],
                "output": str(output),
            },
            indent=2,
        )
    )
    return record


def fixed_subset(data, output, count=32):
    """Write a deterministic val-only list without moving or copying the original images."""
    dataset = check_det_dataset(str(data), autodownload=False)
    images = sorted(p.resolve() for p in Path(dataset["val"]).rglob("*") if p.suffix[1:].lower() in IMG_FORMATS)[:count]
    assert len(images) == count
    output.mkdir(parents=True, exist_ok=False)
    subset = output / "data.yaml"
    subset.with_suffix(".txt").write_text("\n".join(map(str, images)) + "\n", encoding="utf-8")
    YAML.save(
        subset,
        {
            "path": str(output.resolve()),
            "train": str(subset.with_suffix(".txt")),
            "val": str(subset.with_suffix(".txt")),
            "names": dataset["names"],
        },
    )
    common.write_json(
        output / "images.json", [{"path": str(p), "id": p.stem, "sha256": common.sha256(p)} for p in images]
    )
    return subset, images


def completed_run(run, experiment=V1):
    """Require this run's successful process exit and both unchanged final weights."""
    common.require_clean_source()
    assert run.name == experiment.name
    status = run.parent / f"{run.name}_train.exit_status"
    assert status.read_text(encoding="utf-8").strip() == "0", f"Training did not exit successfully: {status}"
    record = json.loads((run / "completed.json").read_text(encoding="utf-8"))
    assert record["completed"] and Path(record["run"]).resolve() == run
    assert record["commit"] == common.git("rev-parse", "HEAD"), "Deploy the recorded training commit"
    for name in ("best", "last"):
        assert record[f"{name}_sha256"] == common.sha256(run / f"weights/{name}.pt")
    original = json.loads((run / "provenance/resolved.json").read_text(encoding="utf-8"))
    assert original["evidence"]["source_sha256"] == common.source_hashes(), "Experiment source changed since training"
    data = Path(YAML.load(run / "args.yaml")["data"]).resolve()
    assert common.sha256(data) == original["evidence"]["data_sha256"]
    assert common.dataset_manifest(data) == original["evidence"]["dataset_manifest"], "Dataset changed since training"
    return data


def bind_report(run, name, folder):
    """Record a completed immutable report; retries get new folders and retain all prior evidence."""
    common.write_json(
        run / f"{name}.json",
        {
            "path": (folder / "metrics.json").relative_to(run).as_posix(),
            "sha256": common.sha256(folder / "metrics.json"),
            "stage_attempt": os.environ.get("B19_STAGE_ATTEMPT"),
        },
    )


def distribution(tensor):
    """Summarize a finite tensor without retaining its full values."""
    values = tensor.float().flatten()
    assert values.numel() and torch.isfinite(values).all()
    quantiles = values.quantile(values.new_tensor([0.05, 0.5, 0.95])).tolist()
    return {
        "norm": values.norm().item(),
        "finite": True,
        "min": values.min().item(),
        "max": values.max().item(),
        "mean": values.mean().item(),
        "std": values.std(unbiased=False).item(),
        "max_abs": values.abs().max().item(),
        "p05": quantiles[0],
        "p50": quantiles[1],
        "p95": quantiles[2],
    }


def match_objects(pred, gt, threshold):
    """Retain ground-truth identities using the pinned native Validator greedy IoU matching order."""
    iou = box_iou(gt[:, 1:], pred[:, :4]).cpu().numpy()
    iou *= gt[:, 0:1].cpu().numpy() == pred[:, 5].cpu().numpy()[None, :]
    pairs = np.array(np.nonzero(iou >= threshold)).T
    if len(pairs) > 1:
        pairs = pairs[iou[pairs[:, 0], pairs[:, 1]].argsort()[::-1]]
        pairs = pairs[np.unique(pairs[:, 1], return_index=True)[1]]
        pairs = pairs[np.unique(pairs[:, 0], return_index=True)[1]]
    found = {int(g): {"prediction": int(d), "iou": float(iou[g, d])} for g, d in pairs}
    fp = sorted(set(range(len(pred))) - {v["prediction"] for v in found.values()})
    return {"tp": len(found), "fp": len(fp), "fn": len(gt) - len(found), "matches": found, "fp_indices": fp}


@torch.no_grad()
def diagnose(run, data, output, experiment=V1, device="cuda:0"):
    """Measure E5/E9/E13 and Delta/Y0 and matched detection changes on exactly 16 validation images."""
    model, _ = load_checkpoint(run / "weights/best.pt")
    assert type(model.model[9]) is experiment.block_type
    model.float().eval().to(device)
    _, images = fixed_subset(data, output / "fixed_val", 16)
    branch = model.model[9]
    rows = []
    eps = 1e-12
    validator = DetectionValidator(
        args={"conf": 0.25, "iou": 0.7, "max_det": 300, "quantize": None, "save_dir": str(output)}
    )
    validator.end2end = True
    for path in images:
        source = cv2.imread(str(path))
        if source is None:
            raise ValueError(f"Unreadable image: {path}")
        image = LetterBox((640, 640), auto=False, stride=32)(image=source)
        tensor = torch.from_numpy(image[..., ::-1].transpose(2, 0, 1).copy()).unsqueeze(0).to(device).float() / 255
        stats = []

        def observe(module, inputs, result, stats=stats):
            from ultralytics.nn.modules.ndp_sppf import ndp_window

            z = module.cv1(inputs[0])  # diagnostic eval-only pass: no BN update
            pyramid = [z]
            pyramid.extend(module.m(pyramid[-1]) for _ in range(module.n))
            base = module.cv2(torch.cat(pyramid, 1))
            base = base + inputs[0] if module.add else base
            u = module.ndp_in(z)
            responses, scales = [], {}
            for k in (5, 9, 13):
                e, weight_stats = ndp_window(u, k, diagnostics=True)
                responses.append(e)
                scales[str(k)] = {key: distribution(value) for key, value in weight_stats.items()}
                scales[str(k)]["E"] = distribution(e)
                # Keep counts and entropy per location so borders can be interpreted correctly.
                np.savez_compressed(
                    output / f"{path.stem}_k{k}_aggregation.npz",
                    E=e.cpu().numpy(),
                    **{key: v.cpu().numpy() for key, v in weight_stats.items()},
                )
            delta = module.ndp_out(torch.cat(responses, 1))
            correlations = {}
            for i, j in ((0, 1), (0, 2), (1, 2)):
                a, b = responses[i].flatten().float(), responses[j].flatten().float()
                a, b = a - a.mean(), b - b.mean()
                denom = a.norm() * b.norm()
                correlations[f"E{(5, 9, 13)[i]}_E{(5, 9, 13)[j]}"] = float((a * b).sum() / denom) if denom > 0 else None
            values = {
                "scales": scales,
                "correlations": correlations,
                "Delta": distribution(delta),
                "Y0": distribution(base),
                "Delta_over_Y0": float(delta.norm() / (base.norm() + eps)),
                "epsilon": eps,
            }
            torch.testing.assert_close(result, base + delta, atol=1e-6, rtol=1e-5)
            stats.append(values)

        handle = branch.register_forward_hook(observe)
        try:
            enabled = model(tensor)
        finally:
            handle.remove()
        with bypass(model):
            disabled = model(tensor)
        assert len(stats) == 1
        predictions = {}
        for key, result in (("on", enabled), ("off", disabled)):
            processed = validator.postprocess(result)[0]
            boxes = scale_boxes(tensor.shape[2:], processed["bboxes"].clone(), source.shape[:2])
            predictions[key] = torch.cat((boxes, processed["conf"][:, None], processed["cls"][:, None]), 1).cpu()
        label = Path(str(path).replace(f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}")).with_suffix(".txt")
        gt = torch.tensor(
            [[float(v) for v in line.split()] for line in label.read_text().splitlines()], dtype=torch.float32
        ).reshape(-1, 5)
        gt[:, 1:] = xywh2xyxy(gt[:, 1:]) * torch.tensor(
            [source.shape[1], source.shape[0], source.shape[1], source.shape[0]]
        )
        changes = {}
        for iou in (0.5, 0.75):
            on, off = (match_objects(predictions[k], gt, iou) for k in ("on", "off"))
            common_ids = set(on["matches"]) & set(off["matches"])
            # Track newly appearing false positives by spatially matching FP boxes with the same rule.
            on_fp = predictions["on"][on["fp_indices"]]
            off_fp = predictions["off"][off["fp_indices"]]
            fp_gt = torch.cat((off_fp[:, 5:6], off_fp[:, :4]), 1)
            fp_match = match_objects(on_fp, fp_gt, 0.5)
            changes[str(iou)] = {
                "on": on,
                "off": off,
                "recovered_targets": sorted(set(on["matches"]) - set(off["matches"])),
                "lost_targets": sorted(set(off["matches"]) - set(on["matches"])),
                "new_fp_count": len(on_fp) - fp_match["tp"],
                "lost_fp_count": len(off_fp) - fp_match["tp"],
                "new_fp_rule": "one-to-one IoU>=0.5 between same-class on/off unmatched predictions",
                "localization": [
                    {
                        "target": g,
                        "on_iou": on["matches"][g]["iou"],
                        "off_iou": off["matches"][g]["iou"],
                        "delta_iou": on["matches"][g]["iou"] - off["matches"][g]["iou"],
                    }
                    for g in sorted(common_ids)
                ],
            }
        panels = []
        for key in ("off", "on"):
            panel = source.copy()
            for idx, row in enumerate(gt):
                x1, y1, x2, y2 = map(int, row[1:].tolist())
                cv2.rectangle(panel, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(panel, f"GT {idx}", (x1, max(y1, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
            for idx, row in enumerate(predictions[key]):
                x1, y1, x2, y2 = map(int, row[:4].tolist())
                cv2.rectangle(panel, (x1, y1), (x2, y2), (0, 128, 255), 1)
                cv2.putText(
                    panel,
                    f"{idx} {float(row[4]):.2f}",
                    (x1, max(y1 + 14, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 128, 255),
                    1,
                )
            cv2.putText(panel, f"NDP {key}; conf=0.25", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            panels.append(panel)
        assert cv2.imwrite(str(output / f"{path.stem}_off_on.jpg"), np.concatenate(panels, axis=1))
        rows.append(
            {
                "image": str(path),
                "sha256": common.sha256(path),
                "distribution_pooling": stats[0],
                "changes": changes,
                "ground_truth": gt.tolist(),
                "predictions": {k: v.tolist() for k, v in predictions.items()},
            }
        )
    result = {
        "split": "val",
        "precision": "FP32",
        "device": str(device),
        "samples": rows,
        "settings": {
            "imgsz": 640,
            "conf": 0.25,
            "iou": 0.7,
            "max_det": 300,
            "end2end": True,
            "augment": False,
            "rect": False,
        },
        "selection": "first 16 lexicographically sorted val paths, identical preprocessing and native end2end postprocessing",
        "weights": {k: float(p.norm()) for k, p in branch.named_parameters()},
        "weight_sha256": common.sha256(run / "weights/best.pt"),
        "commit": common.git("rev-parse", "HEAD"),
        "source_sha256": common.source_hashes(),
        "data_sha256": common.sha256(data),
        "backend": common.computation_conditions(),
        "interpretation": "Same trained weights, residual on/off only; not a retrained ablation. Weights are aggregation coefficients, not foreground confidence; E is a centered distribution summary. No evaluation gradients are claimed.",
    }
    result["artifacts"] = {p.relative_to(output).as_posix(): common.sha256(p) for p in output.rglob("*") if p.is_file()}
    common.write_json(output / "metrics.json", result)
    bind_report(run, "diagnostics", output)


def package(run, data, experiment=V1):
    """Verify outputs and export source, weights, curves, predictions, statistics, logs and checksums."""
    common.require_clean_source()
    assert run.name == experiment.name
    required = ["args.yaml", "results.csv", "results.png", "completed.json", "weights/best.pt", "weights/last.pt"]
    required += [
        "provenance/weights.json",
        "provenance/optimizer.json",
        "provenance/effective_config.json",
        "provenance/resolved.json",
        "provenance/preflight/structural.json",
        "val_fp32.json",
        "test_fp32.json",
        "diagnostics.json",
    ]
    missing = [name for name in required if not (run / name).is_file()]
    missing += [
        f"successful {stage} stage"
        for stage in ("train", "test", "diagnose")
        if not (run.parent / f"{experiment.name}_{stage}.exit_status").is_file()
        or (run.parent / f"{experiment.name}_{stage}.exit_status").read_text(encoding="utf-8").strip() != "0"
    ]
    if missing:
        raise FileNotFoundError(f"Incomplete result; missing stages/artifacts: {missing}")
    common.verify_preflight(run / "provenance/preflight")
    data = completed_run(run, experiment)
    for name in ("val_fp32", "test_fp32", "diagnostics"):
        pointer = json.loads((run / f"{name}.json").read_text(encoding="utf-8"))
        path = run / pointer["path"]
        assert common.sha256(path) == pointer["sha256"]
        report = json.loads(path.read_text(encoding="utf-8"))
        assert report["weight_sha256"] == common.sha256(run / "weights/best.pt")
        assert report["commit"] == common.git("rev-parse", "HEAD") and report["precision"] == "FP32"
        assert report["source_sha256"] == common.source_hashes()
        assert report["data_sha256"] == common.sha256(data)
        for artifact, digest in report["artifacts"].items():
            assert common.sha256(path.parent / artifact) == digest
        if name != "diagnostics":
            required_evaluation = {
                "predictions.json",
                "matching_stats.npz",
                "recall.json",
                "confusion_matrix.png",
                "args.yaml",
                "BoxPR_curve.png",
                "BoxF1_curve.png",
                "BoxP_curve.png",
                "BoxR_curve.png",
            }
            absent = required_evaluation - set(report["artifacts"])
            if absent:
                raise FileNotFoundError(f"Incomplete {name} evaluation artifacts: {sorted(absent)}")
            expected = (2404, 2985) if name == "val_fp32" else (1202, 1477)
            assert (report["images"], report["targets"]) == expected
    bundles = ROOT / "artifacts/experiments"
    bundles.mkdir(parents=True, exist_ok=True)
    output = bundles / f"{experiment.name}_{common.git('rev-parse', '--short=12', 'HEAD')}.tar.gz"
    if output.exists():
        raise FileExistsError(f"Preserving existing package: {output}")
    staging = Path(tempfile.mkdtemp(prefix=f"{experiment.name}_package_", dir=bundles))
    staged_archive = staging / output.name
    with (staging / "source.tar").open("wb") as stream:
        subprocess.run(
            ["git", "-c", f"safe.directory={ROOT.as_posix()}", "archive", "HEAD"], cwd=ROOT, stdout=stream, check=True
        )
    common.write_json(
        staging / "sources.json",
        {
            "commit": common.git("rev-parse", "HEAD"),
            "origin": common.git("remote", "get-url", "origin"),
            "source_sha256": common.source_hashes(),
            "initial_weight_sha256": common.PRETRAINED_SHA256,
            "data": str(data),
            "data_sha256": common.sha256(data),
            "reference_files": json.loads((ROOT / "tools/experiments/ndp_sources.json").read_text(encoding="utf-8")),
            "references": ["https://arxiv.org/abs/2101.00440", "https://arxiv.org/abs/1908.04156"],
            "experiment": {"version": experiment.version, "name": experiment.name, "model": str(experiment.model)},
        },
    )
    selected = {
        f"run/{p.relative_to(run).as_posix()}": p
        for p in run.rglob("*")
        if p.is_file() and (p.suffix != ".pt" or p in (run / "weights/best.pt", run / "weights/last.pt"))
    }
    completed = json.loads((run / "completed.json").read_text())
    attempts = {"train": completed["train_attempt"], "preflight": completed["preflight_attempt"]}
    for name in ("val_fp32", "test_fp32", "diagnostics"):
        pointer = json.loads((run / f"{name}.json").read_text())
        stage = "diagnose" if name == "diagnostics" else "test"
        if stage in attempts:
            assert attempts[stage] == pointer["stage_attempt"]
        attempts[stage] = pointer["stage_attempt"]
    for stage, folder in attempts.items():
        attempt = Path(folder).resolve()
        assert attempt.parent == run.parent and attempt.name.startswith(f"{experiment.name}_{stage}.attempt.")
        assert (attempt / "exit_status").read_text().strip() == "0"
        state = json.loads((attempt / "process_status.json").read_text())
        assert state["state"] == "exited" and state["exit_status"] == 0
        if stage == "train":
            assert (attempt / "commit.txt").read_text().strip() == completed["commit"]
        for item in attempt.rglob("*"):
            if item.is_file():
                selected[f"execution/{attempt.name}/{item.relative_to(attempt).as_posix()}"] = item
    common.write_json(
        staging / "dependencies.json",
        {
            "python": sys.version,
            "torch": torch.__version__,
            "packages": {d.metadata["Name"]: d.version for d in distributions() if d.metadata["Name"]},
        },
    )
    selected.update({p.name: p for p in staging.iterdir() if p.is_file()})
    common.write_json(
        staging / "checksums.json",
        {name: {"bytes": p.stat().st_size, "sha256": common.sha256(p)} for name, p in selected.items()},
    )
    selected["checksums.json"] = staging / "checksums.json"
    with tarfile.open(staged_archive, "w:gz") as archive:
        for name, path in sorted(selected.items()):
            archive.add(path, arcname=f"{experiment.name}/{name}", recursive=False)
    # Verify archive payload, not just the source files that were added.
    import hashlib

    manifest = json.loads((staging / "checksums.json").read_text(encoding="utf-8"))
    with tarfile.open(staged_archive) as archive:
        for name, row in manifest.items():
            stream = archive.extractfile(f"{experiment.name}/{name}")
            digest = hashlib.sha256()
            size = 0
            for chunk in iter(lambda stream=stream: stream.read(1024 * 1024), b""):
                digest.update(chunk)
                size += len(chunk)
            assert digest.hexdigest() == row["sha256"] and size == row["bytes"]
    # Same-filesystem hard link publishes only a complete verified archive and refuses an existing destination.
    output.hardlink_to(staged_archive)
    receipt = {
        "path": str(output),
        "bytes": output.stat().st_size,
        "sha256": common.sha256(output),
        "verified_files": len(manifest),
    }
    common.write_json(output.with_suffix(output.suffix + ".json"), receipt)
    import gzip

    with gzip.open(output, "rb") as stream:
        while stream.read(1024 * 1024):
            pass
    output.with_suffix(output.suffix + ".sha256").write_text(f"{receipt['sha256']}  {output.name}\n", encoding="utf-8")
    subprocess.run(["sha256sum", "-c", output.name + ".sha256"], cwd=bundles, check=True)
    common.write_json(output.with_suffix(output.suffix + ".manifest.json"), manifest)
    print(json.dumps(receipt, indent=2))


def main(argv=None):
    """Expose independent stage processes; internal checkpoint checks cannot create formal stage receipts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("test", "diagnose", "package"))
    parser.add_argument("--checkpoint-check", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expected-images", type=int)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    args = parser.parse_args(argv)
    common.record_process()
    experiment = V1
    if args.checkpoint_check:
        result = evaluate(
            args.checkpoint_check,
            args.data,
            args.output,
            args.split,
            experiment=experiment,
            threshold=args.threshold,
            baseline=args.baseline,
        )
        if args.expected_images is not None:
            assert result["images"] == args.expected_images
        else:
            assert (result["images"], result["targets"]) == ((2404, 2985) if args.split == "val" else (1202, 1477))
        return
    run = ROOT / "runs/detect" / experiment.name
    if args.stage == "package":
        package(run, None, experiment)
        return
    data = completed_run(run, experiment)
    output = Path(tempfile.mkdtemp(prefix=f"{args.stage}_", dir=run))
    if args.stage == "test":
        threshold_args = []
        for split in ("val", "test"):
            folder = output / f"{split}_fp32"
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--checkpoint-check",
                    str(run / "weights/best.pt"),
                    "--data",
                    str(data),
                    "--output",
                    str(folder),
                    "--split",
                    split,
                    *threshold_args,
                ],
                cwd=ROOT,
                check=True,
            )
            bind_report(run, f"{split}_fp32", folder)
            if split == "val":
                selected = json.loads((folder / "metrics.json").read_text())["recall"]["val_selected"]
                threshold_args = ["--threshold", str(selected["threshold"])] if selected else []
        baseline_weight = (
            Path("/root/autodl-tmp/projects/Tunnel_Disease_YOLO26/runs/detect")
            / common.REFERENCE["args"]["name"]
            / "weights/best.pt"
        )
        baseline_report = {"weight": str(baseline_weight), "available": baseline_weight.is_file(), "retrained": False}
        if baseline_weight.is_file():
            baseline_report["sha256"] = common.sha256(baseline_weight)
            baseline_report["reports"] = {}
            threshold_args = []
            for split in ("val", "test"):
                folder = output / f"baseline_{split}_fp32"
                subprocess.run(
                    [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--checkpoint-check",
                        str(baseline_weight),
                        "--data",
                        str(data),
                        "--output",
                        str(folder),
                        "--split",
                        split,
                        "--baseline",
                        *threshold_args,
                    ],
                    cwd=ROOT,
                    check=True,
                )
                baseline_report["reports"][split] = {
                    "path": str(folder.relative_to(run)),
                    "sha256": common.sha256(folder / "metrics.json"),
                    "stage_attempt": os.environ.get("B19_STAGE_ATTEMPT"),
                }
                if split == "val":
                    selected = json.loads((folder / "metrics.json").read_text())["recall"]["val_selected"]
                    threshold_args = ["--threshold", str(selected["threshold"])] if selected else []
        common.write_json(run / "baseline_comparison.json", baseline_report)
    elif args.stage == "diagnose":
        diagnose(run, data, output, experiment)
    else:
        parser.error("--stage is required")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        raise SystemExit(error.returncode if error.returncode > 0 else 128 - error.returncode) from error
