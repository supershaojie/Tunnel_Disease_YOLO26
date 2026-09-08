"""Independent FP32 val/test, fixed-val MPDF diagnosis and verifiable result packaging."""

# ruff: noqa: E402 -- Select the worktree before importing the package.

import argparse
import json
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("YOLO_AUTOINSTALL", "false")
os.environ.setdefault("YOLO_OFFLINE", "true")

import cv2
import torch

from tools.experiments import b19_common as common
from tools.experiments.mpdf_experiment import V1
from tools.experiments.verify_b19_mpdf_p3 import bypass
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from ultralytics.data.utils import IMG_FORMATS, check_det_dataset
from ultralytics.nn.tasks import load_checkpoint
from ultralytics.utils import YAML


def evaluate(weight, data, output, split, device=0, batch=32, workers=8, experiment=V1):
    """Run the repository's unified evaluation settings and capture the real backend precision and predictions."""
    output.mkdir(parents=True, exist_ok=False)
    model = YOLO(weight)
    assert type(model.model.model[15]) is experiment.block_type
    settings = dict(
        data=str(data),
        split=split,
        imgsz=640,
        batch=batch,
        workers=workers,
        device=device,
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
    record = dict(
        split=split,
        precision="FP32",
        weight=str(weight),
        weight_sha256=common.sha256(weight),
        commit=common.git("rev-parse", "HEAD"),
        data_sha256=common.sha256(data),
        source_sha256=common.source_hashes(),
        settings=settings,
    )

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
            confusion_thresholds=dict(conf=0.25, iou=0.45),
            actual_parameter_dtype="torch.float32",
            backend=common.computation_conditions(),
        )
        common.write_json(output / "predictions.json", validator.jdict)
        YAML.save(output / "args.yaml", vars(validator.args))

    calls = dict(mpdf=0, one2one=0)

    def observe_mpdf(module, inputs, output):
        assert output.dtype == torch.float32
        calls["mpdf"] += 1

    def observe_head(module, inputs, output):
        assert module.end2end and output[1]["one2one"]["scores"].numel() > 0
        assert calls["mpdf"] > calls["one2one"]
        calls["one2one"] += 1

    handles = [model.model.model[15].register_forward_hook(observe_mpdf)]
    handles.append(model.model.model[-1].register_forward_hook(observe_head))
    model.add_callback("on_val_end", capture)
    try:
        model.val(**settings)
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
        dict(
            path=str(output.resolve()),
            train=str(subset.with_suffix(".txt")),
            val=str(subset.with_suffix(".txt")),
            names=dataset["names"],
        ),
    )
    common.write_json(output / "images.json", [dict(path=str(p), id=p.stem, sha256=common.sha256(p)) for p in images])
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
        dict(path=(folder / "metrics.json").relative_to(run).as_posix(), sha256=common.sha256(folder / "metrics.json")),
    )


def distribution(tensor):
    """Summarize a finite tensor without retaining its full values."""
    values = tensor.float().flatten()
    assert values.numel() and torch.isfinite(values).all()
    quantiles = values.quantile(values.new_tensor([0.05, 0.5, 0.95])).tolist()
    return dict(
        min=values.min().item(),
        max=values.max().item(),
        mean=values.mean().item(),
        std=values.std(unbiased=False).item(),
        max_abs=values.abs().max().item(),
        p05=quantiles[0],
        p50=quantiles[1],
        p95=quantiles[2],
    )


@torch.no_grad()
def diagnose(run, data, output, experiment=V1, device="cuda:0"):
    """Measure trained P3 corrections and fused one-to-one changes on 16 fixed validation images."""
    model, _ = load_checkpoint(run / "weights/best.pt")
    assert type(model.model[15]) is experiment.block_type
    model.float().eval().to(device)
    _, images = fixed_subset(data, output / "fixed_val", 16)
    branch = model.model[15]
    initial = torch.load(run / "initial_mpdf.pt", map_location="cpu", weights_only=True)
    weight_stats = {
        k: dict(
            norm=float(v.float().norm()), change_from_initial_norm=float((v.float().cpu() - initial[k].float()).norm())
        )
        for k, v in branch.state_dict().items()
    }
    # Native fuse keeps MPDF convolutions and nonlinearities, and drops the auxiliary Detect branch.
    model.fuse(verbose=False)
    rows, examples = [], []
    eps = 1e-12
    for path in images:
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Unreadable validation image: {path}")
        image = LetterBox((640, 640), auto=False, stride=32)(image=image)
        tensor = torch.from_numpy(image[..., ::-1].transpose(2, 0, 1).copy()).unsqueeze(0).to(device).float() / 255
        precision_rows = {}
        for precision in ["FP32", "AMP_FP16"] if str(device).startswith("cuda") else ["FP32"]:
            state = {}

            def capture_coeff(module, inputs, result):
                state["coeff"] = result

            def capture_fusion(module, inputs, result):
                state.update(u=inputs[0][0], h=inputs[0][2], refined=result[:, : module.c_high])

            handles = [
                branch.project.register_forward_hook(capture_coeff),
                branch.register_forward_hook(capture_fusion),
            ]
            try:
                with torch.autocast(device_type="cuda", enabled=precision != "FP32"):
                    enabled = model(tensor)
            finally:
                for handle in handles:
                    handle.remove()
            bx, by, bd = state["coeff"].chunk(3, dim=1)
            correction = torch.empty_like(state["u"])
            for phase, value in enumerate(
                ((bx + by + bd) * 0.5, (-bx + by - bd) * 0.5, (bx - by - bd) * 0.5, (-bx - by + bd) * 0.5)
            ):
                correction[:, :, phase // 2 :: 2, phase % 2 :: 2] = value
            from tools.experiments.verify_b19_mpdf_p3 import error_stats
            from torch.nn import functional as F

            scale = max(float(correction.abs().max()), float(state["u"].abs().max()))
            precision_rows[precision] = dict(
                input_dtype=str(state["u"].dtype),
                coefficient_dtype=str(bx.dtype),
                ratio=float(correction.float().norm() / (state["u"].float().norm() + eps)),
                coefficients={
                    k: dict(**distribution(v), energy=float(v.double().square().sum()))
                    for k, v in zip(("Bx", "By", "Bd"), (bx, by, bd))
                },
                block_mean=[
                    error_stats(F.avg_pool2d(correction, 2), scale, "AvgPool(R)", bx.dtype),
                    error_stats(
                        F.avg_pool2d(state["refined"], 2) - state["h"], scale, "AvgPool(U_refined)-H", bx.dtype
                    ),
                ],
                phase_mean=[float(correction[:, :, i // 2 :: 2, i % 2 :: 2].float().mean()) for i in range(4)],
            )
            if precision == "FP32":
                with bypass(model):
                    disabled = model(tensor)
                old, new = disabled[1]["one2one"], enabled[1]["one2one"]
                differences = {k: distribution((new[k] - old[k]).abs()) for k in ("boxes", "scores")}
                assert model.model[-1].end2end and enabled[1]["one2many"] == {}
                if len(examples) < 4:
                    examples.append(
                        (
                            path.stem,
                            state["u"][0].float().mean(0).cpu(),
                            correction[0].float().mean(0).cpu(),
                            correction[0, 0].float().cpu(),
                        )
                    )
        rows.append(
            dict(
                image=str(path),
                image_id=path.stem,
                sha256=common.sha256(path),
                precision=precision_rows,
                raw_one2one_difference=differences,
            )
        )
    assert any(row["raw_one2one_difference"][k]["max_abs"] > 0 for row in rows for k in ("boxes", "scores"))
    import matplotlib.pyplot as plt

    vmax = max(float(t.abs().max()) for _, u, r, _ in examples for t in (u, r))
    fig, axes = plt.subplots(len(examples), 2, figsize=(10, 4 * len(examples)))
    for index, (name, u, r, _) in enumerate(examples):
        for column, (label, value) in enumerate((("U", u), ("R", r))):
            plot = axes[index, column].imshow(value.numpy(), cmap="coolwarm", vmin=-vmax, vmax=vmax)
            axes[index, column].set_title(f"{name}: channel mean {label}")
    fig.colorbar(plot, ax=axes.ravel().tolist(), label="shared signed feature scale")
    fig.savefig(output / "feature_corrections.png", dpi=150)
    plt.close(fig)
    # A fixed individual channel exposes signs that could cancel in a channel mean.
    rmax = max(float(r.abs().max()) for *_, r in examples)
    fig, axes = plt.subplots(1, len(examples), figsize=(16, 4))
    for ax, (name, _, _, correction0) in zip(axes, examples):
        plot = ax.imshow(correction0.numpy(), cmap="coolwarm", vmin=-rmax, vmax=rmax)
        ax.set_title(f"{name}: R channel 0")
    fig.colorbar(plot, ax=axes.tolist(), label="shared correction scale (all samples)")
    fig.savefig(output / "correction_channel0.png", dpi=150)
    plt.close(fig)
    ap_reports = {}
    for split in ("val", "test"):
        pointer = json.loads((run / f"{split}_fp32.json").read_text(encoding="utf-8"))
        report = json.loads((run / pointer["path"]).read_text(encoding="utf-8"))
        assert report["weight_sha256"] == common.sha256(run / "weights/best.pt")
        ap_reports[split] = dict(pointer=pointer, results_dict=report["results_dict"], ap_by_iou=report["ap_by_iou"])
    preflight = json.loads((run / "provenance/preflight/preflight/checks.json").read_text(encoding="utf-8"))
    result = dict(
        split="val",
        precision="FP32",
        version=1,
        device=str(device),
        preprocessing=dict(
            selection="first 16 sorted validation paths",
            letterbox=[640, 640],
            auto=False,
            stride=32,
            color="RGB",
            normalization="/255",
        ),
        norm_dimensions="L2 over C,H,W independently for each sample; epsilon=1e-12",
        epsilon=eps,
        ratio_quantiles=distribution(torch.tensor([row["precision"]["FP32"]["ratio"] for row in rows])),
        visualization="Signed channel means with one shared color scale; inspect phase means, checkerboards and background texture. No translation-invariance claim.",
        weights=weight_stats,
        samples=rows,
        reused_fp32_evaluations=ap_reports,
        real_training_preflight_gradients=dict(steps=preflight["gradient_steps"], batches=preflight["batches"]),
        weight_sha256=common.sha256(run / "weights/best.pt"),
        commit=common.git("rev-parse", "HEAD"),
        source_sha256=common.source_hashes(),
        data_sha256=common.sha256(data),
        backend=common.computation_conditions(),
        interpretation="Post-training inference diagnostic, not an independently trained ablation. "
        "Responses, gradients and norm ratios do not establish AP improvement.",
        baseline_comparison="Archived b19 metrics are not asserted to be same-condition FP32 reevaluations.",
    )
    result["artifacts"] = {p.relative_to(output).as_posix(): common.sha256(p) for p in output.rglob("*") if p.is_file()}
    common.write_json(output / "metrics.json", result)
    bind_report(run, "diagnostics", output)


def package(run, data, experiment=V1):
    """Verify outputs and export source, weights, curves, predictions, statistics, logs and checksums."""
    common.require_clean_source()
    assert run.name == experiment.name
    required = ["args.yaml", "results.csv", "results.png", "completed.json", "weights/best.pt", "weights/last.pt"]
    required += [
        "initial_mpdf.pt",
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
        for artifact, digest in report["artifacts"].items():
            assert common.sha256(path.parent / artifact) == digest
        if name != "diagnostics":
            required_evaluation = {
                "predictions.json",
                "args.yaml",
                "BoxPR_curve.png",
                "BoxF1_curve.png",
                "BoxP_curve.png",
                "BoxR_curve.png",
                "confusion_matrix.png",
                "confusion_matrix_normalized.png",
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
        dict(
            commit=common.git("rev-parse", "HEAD"),
            origin=common.git("remote", "get-url", "origin"),
            source_sha256=common.source_hashes(),
            initial_weight_sha256=common.PRETRAINED_SHA256,
            data=str(data),
            data_sha256=common.sha256(data),
            reference_files=json.loads((ROOT / "tools/experiments/mpdf_sources.json").read_text(encoding="utf-8")),
            references=["https://arxiv.org/abs/2408.12879", "https://doi.org/10.3390/app16167991"],
            experiment=dict(version=experiment.version, name=experiment.name, model=str(experiment.model)),
        ),
    )
    selected = {
        f"run/{p.relative_to(run).as_posix()}": p
        for p in run.rglob("*")
        if p.is_file() and p.name not in {"preflight.pt", "reload_reference.pt"}
    }
    for stage in ("preflight", "train", "test", "diagnose"):
        for p in run.parent.glob(f"{experiment.name}_{stage}*"):
            if p.is_file():
                selected[f"execution/{p.name}"] = p
            elif p.is_dir():
                for item in p.rglob("*"):
                    if item.is_file():
                        selected[f"execution/{item.relative_to(run.parent).as_posix()}"] = item
    selected.update({p.name: p for p in staging.iterdir() if p.is_file()})
    common.write_json(
        staging / "checksums.json",
        {name: dict(bytes=p.stat().st_size, sha256=common.sha256(p)) for name, p in selected.items()},
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
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
                size += len(chunk)
            assert digest.hexdigest() == row["sha256"] and size == row["bytes"]
    # Same-filesystem hard link publishes only a complete verified archive and refuses an existing destination.
    output.hardlink_to(staged_archive)
    receipt = dict(
        path=str(output), bytes=output.stat().st_size, sha256=common.sha256(output), verified_files=len(manifest)
    )
    common.write_json(output.with_suffix(output.suffix + ".json"), receipt)
    print(json.dumps(receipt, indent=2))


def main(argv=None):
    """Expose independent stage processes; internal checkpoint checks cannot create formal stage receipts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("test", "diagnose", "package"))
    parser.add_argument("--checkpoint-check", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expected-images", type=int)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    args = parser.parse_args(argv)
    experiment = V1
    if args.stage and os.environ.get("B19_STAGE_ATTEMPT"):
        (Path(os.environ["B19_STAGE_ATTEMPT"]) / "python.pid").write_text(str(os.getpid()) + "\n")
    if args.checkpoint_check:
        result = evaluate(args.checkpoint_check, args.data, args.output, args.split, experiment=experiment)
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
                ],
                cwd=ROOT,
                check=True,
            )
            bind_report(run, f"{split}_fp32", folder)
    elif args.stage == "diagnose":
        diagnose(run, data, output, experiment)
    else:
        parser.error("--stage is required")


if __name__ == "__main__":
    main()
