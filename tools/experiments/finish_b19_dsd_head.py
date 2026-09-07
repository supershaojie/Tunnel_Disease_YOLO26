"""Independent FP32 val/test, fixed-validation DSD diagnostics, and verified experiment packaging."""

# ruff: noqa: E402 -- Resolve the experiment worktree before imports.

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.experiments import b19_common as shared
from tools.experiments import b19_finish as common
from tools.experiments import run_b19_dsd_head as experiment

import cv2
import torch

from ultralytics.data.augment import LetterBox
from ultralytics.data.utils import IMG_FORMATS, check_det_dataset, img2label_paths
from ultralytics.nn.modules import DSDDetect
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
    """Bind evidence to the actual code, best weight, data and both evaluation manifests."""
    evidence = common.provenance(run)
    evidence.update(experiment=experiment.NAME, precision="FP32", data=str(data), data_sha256=shared.sha256(data))
    evidence["source_sha256"] = shared.source_hashes(experiment.SOURCE_FILES)
    cfg = check_det_dataset(str(data), autodownload=False)
    evidence["manifests"] = {}
    for split in COUNTS:
        images = sorted(p for p in Path(cfg[split]).rglob("*") if p.suffix[1:].lower() in IMG_FORMATS)
        entries = [
            [str(p), p.stat().st_size, p.stat().st_mtime_ns, shared.sha256(label)]
            for p, label in zip(images, img2label_paths(list(map(str, images))))
        ]
        evidence["manifests"][split] = hashlib.sha256(json.dumps(entries).encode()).hexdigest()
    return evidence


def artifacts_match(folder, record, required=()):
    """Check required and declared artifact bytes before reuse or packaging."""
    artifacts = record.get("artifacts", {})
    return all(
        (folder / name).is_file() and shared.sha256(folder / name) == artifacts.get(name)
        for name in set(required) | set(artifacts)
    )


def report_directory(run, stage, evidence):
    """Preserve mismatched or failed attempts and reuse only completed matching evidence."""
    digest = hashlib.sha256(json.dumps([stage, evidence], sort_keys=True).encode()).hexdigest()
    parent = run / "evaluation"
    primary = parent / f"{stage}-{digest[:20]}"
    for candidate in (primary, *sorted(parent.glob(primary.name + "-retry-*"))):
        if (candidate / "metrics.json").is_file():
            record = json.loads((candidate / "metrics.json").read_text(encoding="utf-8"))
            if record.get("evidence") == evidence and artifacts_match(
                candidate, record, EVAL_ARTIFACTS if stage in COUNTS else ()
            ):
                return candidate
    candidate, attempt = primary, 1
    while candidate.exists():
        candidate = parent / f"{primary.name}-retry-{attempt}"
        attempt += 1
    return candidate


def evaluate_split(run, data, split):
    """Run a single split in its own process through YOLO.val and the real fused Validator."""
    evidence = provenance(run, data)

    def capture(validator):
        assert (validator.seen, int(validator.metrics.nt_per_class.sum())) == COUNTS[split]
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
            metric_units="fraction; unrounded",
            split=split,
            precision="FP32",
            ap_by_iou={f"{0.50 + 0.05 * i:.2f}": float(ap[:, i].mean()) for i in range(10)},
        )

    output = report_directory(run, split, evidence)
    record = common.test_best(run, data, split=split, evidence=evidence, output=output, capture=capture)
    record["artifacts"] = {
        p.relative_to(output).as_posix(): shared.sha256(p)
        for p in output.rglob("*")
        if p.is_file() and p.name != "metrics.json"
    }
    assert artifacts_match(output, record, EVAL_ARTIFACTS)
    shared.write_json(output / "metrics.json", record)
    shared.write_json(
        run / f"evaluation_{split}.json",
        dict(path=str(output.relative_to(run) / "metrics.json"), sha256=shared.sha256(output / "metrics.json")),
    )


def load_report(run, index):
    """Read a report whose relative location and checksum have been verified."""
    path = (run / index["path"]).resolve()
    path.relative_to(run)
    assert shared.sha256(path) == index["sha256"]
    record = json.loads(path.read_text(encoding="utf-8"))
    assert artifacts_match(path.parent, record)
    return path, record


def distribution(tensor):
    """Record signed coefficient tails, saturation and a fixed histogram."""
    t = tensor.float().flatten()
    return dict(
        mean=t.mean().item(),
        std=t.std(unbiased=False).item(),
        min=t.min().item(),
        max=t.max().item(),
        quantiles=t.quantile(torch.tensor([0.01, 0.05, 0.5, 0.95, 0.99], device=t.device)).tolist(),
        saturation_ratio=(t.abs() >= 0.99).float().mean().item(),
        histogram=torch.histc(t, bins=20, min=-1, max=1).long().tolist(),
        histogram_range=[-1, 1],
    )


@torch.no_grad()
def diagnose(run, data):
    """Inspect both trained adapters and paired fused inference on sixteen fixed validation samples."""
    evidence = provenance(run, data)
    _, validation = load_report(run, json.loads((run / "evaluation_val.json").read_text(encoding="utf-8")))
    assert validation["evidence"] == evidence
    cfg = check_det_dataset(str(data), autodownload=False)
    images = sorted(p for p in Path(cfg["val"]).rglob("*") if p.suffix[1:].lower() in IMG_FORMATS)[:16]
    assert len(images) == 16
    evidence["samples"] = [[str(p), shared.sha256(p)] for p in images]
    output = report_directory(run, "diagnostics", evidence)
    if not (output / "metrics.json").is_file():
        output.mkdir(parents=True, exist_ok=False)
        model, _ = load_checkpoint(evidence["weight"])
        model = model.float().eval().to("cuda:0")
        head = model.model[-1]
        assert type(head) is DSDDetect
        rows, coefficients, handles = [], {}, []

        def coeff_hook(branch):
            def capture(module, inputs, result):
                coefficients[branch] = result.float().reshape(result.shape[0], 8, 4, *result.shape[-2:]).tanh()

            return capture

        def adapter_hook(branch):
            def capture(module, inputs, result):
                x = inputs[0].float()
                coeff = coefficients.pop(branch)
                rows[-1][branch] = dict(
                    directions=[
                        dict(dy=dy, dx=dx, **distribution(coeff[:, :, d, 1:-1, 1:-1]))
                        for d, (dy, dx) in enumerate(module.directions)
                    ],
                    groups=[[distribution(coeff[:, g, d, 1:-1, 1:-1]) for d in range(4)] for g in range(8)],
                    delta_over_x=((result.float() - x).norm() / (x.norm() + 1e-6)).item(),
                )

            return capture

        for branch in ("reg_adapter", "one2one_reg_adapter"):
            adapter = getattr(head, branch)
            handles += [
                adapter.coeff.register_forward_hook(coeff_hook(branch)),
                adapter.register_forward_hook(adapter_hook(branch)),
            ]
        tensors = []
        try:
            for path in images:
                image = cv2.imread(str(path))
                if image is None:
                    raise ValueError(f"Unreadable validation image: {path}")
                image = LetterBox((640, 640), auto=False, stride=32)(image=image)
                x = torch.from_numpy(image[..., ::-1].transpose(2, 0, 1).copy()).unsqueeze(0)
                x = x.to("cuda:0").float() / 255
                tensors.append(x.cpu())
                rows.append(dict(image=str(path), sha256=shared.sha256(path)))
                model(x)
        finally:
            for handle in handles:
                handle.remove()
        model.fuse()
        assert head.reg_adapter is None and sum(p.numel() for p in head.one2one_reg_adapter.parameters()) == 1744
        inference = []
        for x in tensors:
            x = x.to("cuda:0")
            calls = []
            hook = head.one2one_reg_adapter.register_forward_hook(lambda m, i, o: calls.append(True))
            on = model(x)
            hook.remove()
            # Output substitution is confined to this diagnostic process and never persisted in a checkpoint.
            hook = head.one2one_reg_adapter.register_forward_hook(lambda m, i, o: i[0])
            try:
                off = model(x)
            finally:
                hook.remove()
            assert calls == [True] and on[1]["one2many"] == {}
            a, b = on[1]["one2one"], off[1]["one2one"]
            n = a["feats"][0].shape[-2] * a["feats"][0].shape[-1]
            assert torch.equal(a["scores"], b["scores"]) and torch.equal(a["boxes"][..., n:], b["boxes"][..., n:])
            inference.append(
                dict(
                    one2one_calls=len(calls),
                    p3_box_change_max=(a["boxes"][..., :n] - b["boxes"][..., :n]).abs().max().item(),
                    selected_prediction_change_max=(on[0] - off[0]).abs().max().item(),
                )
            )
        shared.write_json(
            output / "metrics.json",
            dict(
                evidence=evidence,
                images=rows,
                fused_inference=inference,
                validation_metrics=validation["metrics"],
                mode="FP32 eval/no_grad; first 16 sorted val images; 640 letterbox; no augmentation",
                coefficients="tanh(z); actual multiplier is 0.025*tanh(z); interior positions only",
                interpretation="Adapter-off is inference diagnosis of this trained checkpoint, not an independently trained ablation. "
                "An affine feature nullspace does not establish illumination or box-position invariance.",
            ),
        )
    shared.write_json(
        run / "diagnostics.json",
        dict(path=str(output.relative_to(run) / "metrics.json"), sha256=shared.sha256(output / "metrics.json")),
    )


def package(run, data, output):
    """Require matching complete metrics, diagnostics and training evidence before creating an archive."""
    evidence = provenance(run, data)
    if evidence["status"]:
        raise RuntimeError("Commit source changes before packaging")
    required = [
        "args.yaml",
        "results.csv",
        "results.png",
        "train.log",
        "weights/best.pt",
        "weights/last.pt",
        "completed.json",
        "val_metrics.json",
        "evaluation_val.json",
        "evaluation_test.json",
        "diagnostics.json",
    ]
    selected = []
    for stage, index_name in (
        ("val", "evaluation_val.json"),
        ("test", "evaluation_test.json"),
        ("diagnostics", "diagnostics.json"),
    ):
        path, record = load_report(run, json.loads((run / index_name).read_text(encoding="utf-8")))
        observed = dict(record["evidence"])
        samples = observed.pop("samples", [])
        assert observed == evidence
        assert all(shared.sha256(p) == digest for p, digest in samples)
        assert artifacts_match(path.parent, record, EVAL_ARTIFACTS if stage in COUNTS else ())
        selected.append(path.parent)
        required.append(path.relative_to(run).as_posix())
    common.package(
        run,
        output,
        required_files=required,
        evidence=evidence,
        include_last=True,
        exclude_dirs=tuple(p for p in (run / "evaluation").iterdir() if p not in selected),
        source_files=[str(p.relative_to(ROOT)) for p in experiment.SOURCE_FILES],
    )


def main(argv=None):
    """Expose independent completion stages; each evaluation split runs in a fresh Python process."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("test", "diagnose", "package"), required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--split", choices=tuple(COUNTS), help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    run = args.run.resolve()
    assert run == ROOT / "runs/detect" / experiment.NAME
    common.completed_run(run)
    data = Path(YAML.load(run / "args.yaml")["data"]).resolve()
    if args.stage == "test":
        if args.split:
            evaluate_split(run, data, args.split)
        else:
            for split in COUNTS:
                subprocess.run(
                    [
                        sys.executable,
                        "-u",
                        str(Path(__file__).resolve()),
                        "--stage",
                        "test",
                        "--run",
                        str(run),
                        "--split",
                        split,
                    ],
                    cwd=ROOT,
                    check=True,
                )
    elif args.stage == "diagnose":
        diagnose(run, data)
    else:
        package(run, data, (args.output or run.with_suffix(".tar.gz")).resolve())


if __name__ == "__main__":
    main()
