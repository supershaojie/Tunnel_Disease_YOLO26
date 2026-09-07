"""Independent FP32 val/test, fixed-val RSC diagnosis and verifiable result packaging."""

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
from tools.experiments.run_b19_rsc_c2psa import NAME
from tools.experiments.verify_b19_rsc_c2psa import bypass
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from ultralytics.data.utils import IMG_FORMATS, check_det_dataset
from ultralytics.nn.modules import C2PSA_RSC
from ultralytics.nn.modules.rsc_c2psa import reciprocal_probabilities
from ultralytics.nn.tasks import load_checkpoint
from ultralytics.utils import YAML


def evaluate(weight, data, output, split, device=0, batch=32, workers=8):
    """Run the repository's unified evaluation settings and capture the real backend precision and predictions."""
    output.mkdir(parents=True, exist_ok=False)
    model = YOLO(weight)
    assert type(model.model.model[10]) is C2PSA_RSC and model.model.model[10].m[0].attn.enabled
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

    model.add_callback("on_val_end", capture)
    model.val(**settings)
    record["artifacts"] = {p.relative_to(output).as_posix(): common.sha256(p) for p in output.rglob("*") if p.is_file()}
    common.write_json(output / "metrics.json", record)
    return record


def completed_run(run):
    """Require this run's successful process exit and both unchanged final weights."""
    status = run.parent / f"{NAME}_train.exit_status"
    assert status.read_text().strip() == "0", f"Training did not exit successfully: {status}"
    record = json.loads((run / "completed.json").read_text())
    assert record["completed"] and Path(record["run"]).resolve() == run
    assert record["commit"] == common.git("rev-parse", "HEAD"), "Deploy the recorded training commit"
    for name in ("best", "last"):
        assert record[f"{name}_sha256"] == common.sha256(run / f"weights/{name}.pt")
    original = json.loads((run / "provenance/resolved.json").read_text())
    assert original["evidence"]["source_sha256"] == common.source_hashes(), "Experiment source changed since training"
    data = Path(YAML.load(run / "args.yaml")["data"]).resolve()
    assert common.sha256(data) == original["evidence"]["data_sha256"]
    return data


def bind_report(run, name, folder):
    """Record a completed immutable report; retries get new folders and retain all prior evidence."""
    common.write_json(
        run / f"{name}.json",
        dict(path=(folder / "metrics.json").relative_to(run).as_posix(), sha256=common.sha256(folder / "metrics.json")),
    )


@torch.no_grad()
def diagnose(run, data, output):
    """Record head statistics on the first 16 sorted validation images, without storing attention matrices."""
    model, _ = load_checkpoint(run / "weights/best.pt")
    model.float().eval().to("cuda:0")
    dataset = check_det_dataset(str(data), autodownload=False)
    images = sorted(p for p in Path(dataset["val"]).rglob("*") if p.suffix[1:].lower() in IMG_FORMATS)[:16]
    assert len(images) == 16
    rows = []
    for path in images:
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Unreadable validation image: {path}")
        image = LetterBox((640, 640), auto=False, stride=32)(image=image)
        tensor = torch.from_numpy(image[..., ::-1].transpose(2, 0, 1).copy()).unsqueeze(0).cuda().float() / 255
        blocks = []

        def inspect(attn, inputs):
            x = inputs[0]
            B, _, H, W = x.shape
            q, k, _ = (
                attn.qkv(x)
                .view(B, attn.num_heads, 2 * attn.key_dim + attn.head_dim, H * W)
                .split([attn.key_dim, attn.key_dim, attn.head_dim], dim=2)
            )
            scores = (q * attn.scale).transpose(-2, -1) @ k
            a = scores.softmax(-1)
            r = reciprocal_probabilities(scores)
            beta = attn.beta.view(1, attn.num_heads, 1, 1)
            mixed = (1 - beta) * a.float() + beta * r
            heads = []
            for h in range(attn.num_heads):
                old, new = a[:, h], mixed[:, h]

                def entropy(p):
                    return (-(p * p.clamp_min(1e-30).log()).sum(-1)).mean().item()

                delta = (new - old).abs()
                heads.append(
                    dict(
                        head=h,
                        beta=attn.beta[h].item(),
                        theta=attn.theta[h].item(),
                        entropy_original=entropy(old),
                        entropy_new=entropy(new),
                        diagonal_mass_original=old.diagonal(dim1=-2, dim2=-1).mean().item(),
                        diagonal_mass_new=new.diagonal(dim1=-2, dim2=-1).mean().item(),
                        difference_mean_abs=delta.mean().item(),
                        row_l1_mean=delta.sum(-1).mean().item(),
                        row_l1_max=delta.sum(-1).max().item(),
                        row_sum_max_error=(new.sum(-1) - 1).abs().max().item(),
                    )
                )
            blocks.append(dict(tokens=H * W, shape=[H, W], heads=heads))

        handles = [block.attn.register_forward_pre_hook(inspect) for block in model.model[10].m]
        try:
            enabled = model(tensor)
        finally:
            for handle in handles:
                handle.remove()
        with bypass(model):
            native = model(tensor)
        # Dense boxes and scores have stable anchor order; postprocessed top-k boxes need not.
        old = native[1]["one2one"]
        new = enabled[1]["one2one"]
        changes = {k: (new[k] - old[k]).abs().mean().item() for k in ("boxes", "scores")}
        rows.append(
            dict(
                image=str(path),
                sha256=common.sha256(path),
                blocks=blocks,
                disabled_inference_mean_abs_difference=changes,
            )
        )
    common.write_json(
        output / "metrics.json",
        dict(
            split="val",
            precision="FP32",
            samples=rows,
            weight_sha256=common.sha256(run / "weights/best.pt"),
            commit=common.git("rev-parse", "HEAD"),
            interpretation="Post-training native bypass is an inference diagnostic, not an independently trained ablation.",
            artifacts={},
        ),
    )
    bind_report(run, "diagnostics", output)


def package(run, data):
    """Verify outputs and export source, weights, curves, predictions, statistics, logs and checksums."""
    required = ["args.yaml", "results.csv", "results.png", "completed.json", "weights/best.pt", "weights/last.pt"]
    for name in required:
        if not (run / name).is_file():
            raise FileNotFoundError(run / name)
    for stage in ("preflight", "train", "test", "diagnose"):
        assert (run.parent / f"{NAME}_{stage}.exit_status").read_text().strip() == "0"
    for name in ("val_fp32", "test_fp32", "diagnostics"):
        pointer = json.loads((run / f"{name}.json").read_text())
        path = run / pointer["path"]
        assert common.sha256(path) == pointer["sha256"]
        report = json.loads(path.read_text())
        assert report["weight_sha256"] == common.sha256(run / "weights/best.pt")
        assert report["commit"] == common.git("rev-parse", "HEAD") and report["precision"] == "FP32"
        for artifact, digest in report["artifacts"].items():
            assert common.sha256(path.parent / artifact) == digest
        if name != "diagnostics":
            expected = (2404, 2985) if name == "val_fp32" else (1202, 1477)
            assert (report["images"], report["targets"]) == expected
    bundles = ROOT / "artifacts/experiments"
    bundles.mkdir(parents=True, exist_ok=True)
    output = bundles / f"{NAME}_{common.git('rev-parse', '--short=12', 'HEAD')}.tar.gz"
    if output.exists():
        raise FileExistsError(f"Preserving existing package: {output}")
    staging = Path(tempfile.mkdtemp(prefix=f"{NAME}_package_", dir=bundles))
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
            reference_files=json.loads((ROOT / "tools/experiments/rsc_sources.json").read_text(encoding="utf-8")),
            references=["https://arxiv.org/html/2408.04357v1", "https://arxiv.org/html/2508.09983v1"],
        ),
    )
    selected = {
        f"run/{p.relative_to(run).as_posix()}": p
        for p in run.rglob("*")
        if p.is_file() and p.name not in {"preflight.pt", "reload_reference.pt"}
    }
    for stage in ("preflight", "train", "test", "diagnose"):
        for p in run.parent.glob(f"{NAME}_{stage}*"):
            if p.is_file():
                selected[f"execution/{p.name}"] = p
            elif ".attempt." in p.name:
                for item in p.rglob("*"):
                    if item.is_file():
                        selected[f"execution/{item.relative_to(run.parent).as_posix()}"] = item
    selected.update({p.name: p for p in staging.iterdir() if p.is_file()})
    common.write_json(
        staging / "checksums.json",
        {name: dict(bytes=p.stat().st_size, sha256=common.sha256(p)) for name, p in selected.items()},
    )
    selected["checksums.json"] = staging / "checksums.json"
    with tarfile.open(output, "w:gz") as archive:
        for name, path in sorted(selected.items()):
            archive.add(path, arcname=f"{NAME}/{name}", recursive=False)
    # Verify archive payload, not just the source files that were added.
    import hashlib

    manifest = json.loads((staging / "checksums.json").read_text())
    with tarfile.open(output) as archive:
        for name, row in manifest.items():
            stream = archive.extractfile(f"{NAME}/{name}")
            digest = hashlib.sha256()
            size = 0
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
                size += len(chunk)
            assert digest.hexdigest() == row["sha256"] and size == row["bytes"]
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
    parser.add_argument("--split", choices=("val", "test"), default="val")
    args = parser.parse_args(argv)
    if args.checkpoint_check:
        result = evaluate(args.checkpoint_check, args.data, args.output, args.split)
        assert (result["images"], result["targets"]) == ((2404, 2985) if args.split == "val" else (1202, 1477))
        return
    run = ROOT / "runs/detect" / NAME
    data = completed_run(run)
    if args.stage == "package":
        package(run, data)
        return
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
        diagnose(run, data, output)
    else:
        parser.error("--stage is required")


if __name__ == "__main__":
    main()
