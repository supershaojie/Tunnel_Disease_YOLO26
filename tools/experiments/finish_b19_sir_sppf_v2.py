"""Evaluate validation-selected v2 best.pt in FP32, diagnose independent corrections, and package evidence."""

# ruff: noqa: E402 -- Resolve the experiment worktree before importing its package.

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.experiments import finish_b19_sir_sppf as common
from tools.experiments import run_b19_sir_sppf as shared
from tools.experiments import run_b19_sir_sppf_v2 as run_v2

import cv2
import torch

from ultralytics.data.augment import LetterBox
from ultralytics.data.utils import IMG_FORMATS, check_det_dataset, img2label_paths
from ultralytics.nn.modules import SPPF_SIR_V2
from ultralytics.nn.tasks import load_checkpoint
from ultralytics.utils import YAML

COUNTS = {"val": (2404, 2985), "test": (1202, 1477)}
EVAL_ARTIFACTS = (
    "args.yaml",
    "predictions.json",
    "BoxPR_curve.png",
    "confusion_matrix.png",
    "confusion_matrix_normalized.png",
)


def provenance(run, data):
    """Bind reports to v2 code, best weight, effective data file, and current image/label manifests."""
    evidence = common.provenance(run)
    evidence["source_sha256"] = shared.source_hashes(
        (*run_v2.SOURCE_FILES, Path(common.__file__), ROOT / "tools/experiments/server_b19_sir_sppf_v1.sh")
    )
    evidence.update(version=2, data=str(data), data_sha256=shared.sha256(data))
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
        record = common.test_best(
            run, data, split=split, block_type=SPPF_SIR_V2, evidence=evidence.copy(), output=output, capture=capture
        )
        record["artifacts"] = {name: shared.sha256(output / name) for name in EVAL_ARTIFACTS}
        shared.write_json(output / "metrics.json", record)
        records[split] = dict(
            path=str(output.relative_to(run) / "metrics.json"), sha256=shared.sha256(output / "metrics.json")
        )
    shared.write_json(run / "evaluation.json", dict(evidence=evidence, reports=records))


@torch.no_grad()
def correction_statistics(module, x, routed):
    """Measure r_i independently; 1+0.5*tanh(L_i) weights d_i relative to the raw previous scale."""
    z = [module.cv1(x)]
    z.extend(module.m(z[-1]) for _ in range(module.n))
    increments = [b - a for a, b in zip(z, z[1:])]
    gates = module.router(torch.cat([z[0], *increments], 1)).tanh().chunk(module.n, 1)
    scales = []
    for raw, delta, gate in zip(z[1:], increments, gates):
        correction = 0.5 * gate * delta
        scales.append(
            dict(
                tanh_mean=gate.mean().item(),
                tanh_std=gate.std(unbiased=False).item(),
                coefficient_mean=(1 + 0.5 * gate).mean().item(),
                coefficient_std=(0.5 * gate).std(unbiased=False).item(),
                saturation_ratio=(gate.abs() >= 0.99).float().mean().item(),
                correction_norm=correction.norm().item(),
                correction_raw_norm_ratio=(correction.norm() / (raw.norm() + 1e-6)).item(),
            )
        )
    bypass = module.cv2(torch.cat(z, 1))
    if module.add:
        bypass = bypass + x
    return dict(scales=scales, module_change_norm_ratio=((routed - bypass).norm() / (bypass.norm() + 1e-6)).item())


@torch.no_grad()
def diagnose(run, data):
    """Inspect fixed validation samples on an independent eval model; reuse only matching evidence."""
    evidence = provenance(run, data)
    cfg = check_det_dataset(str(data), autodownload=False)
    images = sorted(p for p in Path(cfg["val"]).rglob("*") if p.suffix[1:].lower() in IMG_FORMATS)[:16]
    assert len(images) == 16
    evidence["samples"] = [[str(p), shared.sha256(p)] for p in images]
    output = report_directory(run, "diagnostics", evidence)
    if not (output / "metrics.json").is_file():
        output.mkdir(parents=True, exist_ok=False)
        model, _ = load_checkpoint(evidence["weight"])
        assert type(model.model[9]) is SPPF_SIR_V2
        model = model.float().eval().to("cuda:0")
        rows = []
        hook = model.model[9].register_forward_hook(
            lambda module, inputs, result: rows.append(correction_statistics(module, inputs[0], result))
        )
        try:
            for path in images:
                image = cv2.imread(str(path))
                if image is None:
                    raise ValueError(f"Unreadable validation image: {path}")
                image = LetterBox((640, 640), auto=False, stride=32)(image=image)
                tensor = torch.from_numpy(image[..., ::-1].transpose(2, 0, 1).copy()).unsqueeze(0)
                model(tensor.to("cuda:0").float() / 255)
                rows[-1].update(image=str(path), sha256=shared.sha256(path))
        finally:
            hook.remove()
        shared.write_json(
            output / "metrics.json",
            dict(
                evidence=evidence,
                images=rows,
                formula=run_v2.MODULE_CONFIG["formula"],
                mode="FP32 eval/no_grad; first 16 sorted val images; 640 letterbox; no augmentation",
                interpretation="Feature diagnosis of the same trained module; neither baseline gain nor evidence of test failure causality",
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
    common.package(
        run,
        output,
        required_files=required,
        evidence=evidence,
        exclude_dirs=tuple(p for p in (run / "evaluation").iterdir() if p not in selected),
        source_files=[str(p.relative_to(ROOT)) for p in run_v2.SOURCE_FILES]
        + [
            "ultralytics/nn/modules/sir_sppf_v2.py",
            "ultralytics/cfg/models/26/yolo26n-sir-sppf-v2.yaml",
            "tests/test_sir_sppf_v2.py",
            "docs/experiments/b19_sir_sppf_v2.md",
            "docs/experiments/b19_sir_sppf_reload_fix.md",
        ],
    )


def main(argv=None):
    """Expose the v2 completion stages; test includes both independent val and held-out test."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("test", "diagnose", "package"), required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    run = args.run.resolve()
    assert run == ROOT / "runs/detect" / run_v2.NAME
    common.completed_run(run)
    data = Path(YAML.load(run / "args.yaml")["data"]).resolve()
    if args.stage == "package":
        package(run, data, (args.output or run.with_suffix(".tar.gz")).resolve())
    else:
        {"test": evaluate_best, "diagnose": diagnose}[args.stage](run, data)


if __name__ == "__main__":
    main()
