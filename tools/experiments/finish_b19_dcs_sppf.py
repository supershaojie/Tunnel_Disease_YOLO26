"""Evaluate, diagnose and archive DCS v1 with distinct stage receipts and an immutable b19 comparison."""

# ruff: noqa: E402 - Direct script entry must prioritize this worktree before importing Ultralytics.

import argparse
import csv
import gzip
import hashlib
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch

from tools.experiments import b19_common as common
from tools.experiments.run_b19_dcs_sppf import NAME, audit_arguments, require_runtime
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from ultralytics.data.utils import IMG_FORMATS, check_det_dataset, img2label_paths
from ultralytics.nn.modules import DCS_SPPF, SPPF
from ultralytics.nn.tasks import load_checkpoint
from ultralytics.utils import YAML
from ultralytics.utils.torch_utils import get_flops

CURVES = [
    "BoxPR_curve.png",
    "BoxP_curve.png",
    "BoxR_curve.png",
    "BoxF1_curve.png",
    "confusion_matrix.png",
    "confusion_matrix_normalized.png",
    "predictions.json",
    "args.yaml",
]


def provenance(run, data, name=NAME, model=common.MODEL, block_type=DCS_SPPF):
    """Require completed canonical training, unchanged files/code and the exact fixed dataset."""
    common.require_clean_source()
    completion = json.loads((run / "completed.json").read_text(encoding="utf-8"))
    resolved = json.loads((run / "provenance/resolved.json").read_text(encoding="utf-8"))
    if completion["commit"] != common.git("rev-parse", "HEAD"):
        raise ValueError("Finish must run at the training commit")
    for artifact_name, digest in completion["files"].items():
        if common.sha256(run / artifact_name) != digest:
            raise ValueError(f"Changed completed training file: {artifact_name}")
    audit_arguments(resolved["config"], YAML.load(run / "args.yaml"))
    if resolved["config"]["name"] != name or Path(resolved["config"]["project"]) / name != run:
        raise ValueError("Run path differs from its canonical training identity")
    if resolved["evidence"]["source_sha256"] != common.source_hashes():
        raise ValueError("Source files differ from the recorded training source")
    if common.sha256(data) != resolved["evidence"]["data_sha256"]:
        raise ValueError("Evaluation data YAML differs from the verified training data")
    manifest = common.dataset_manifest(data)
    if manifest != resolved["evidence"]["dataset_manifest"]:
        raise ValueError("Evaluation dataset differs from the verified training dataset")
    preflight = json.loads((run / "provenance/preflight/checks.json").read_text(encoding="utf-8"))
    if not preflight["passed"] or preflight["local_only"] or preflight["commit"] != completion["commit"]:
        raise ValueError("Missing matching formal preflight")
    assert preflight["recipe"] == resolved["evidence"], "Preflight receipt differs from training provenance"
    assert resolved["evidence"]["model_binding"] == common.model_binding(model, block_type)
    return {
        "model_binding": resolved["evidence"]["model_binding"],
        "commit": completion["commit"],
        "weight_sha256": completion["files"]["weights/best.pt"],
        "data_manifest": manifest,
        "data_sha256": resolved["evidence"]["data_sha256"],
        "complexity": preflight["complexity"],
    }


def evaluate(weights, data, output, block_type, evidence, split="test"):
    """Reuse native FP32 b19 evaluation settings and extract AP75 from its all_ap array."""
    binding = {**evidence, "weight_sha256": common.sha256(weights), "split": split}
    receipt = output / "metrics.json"
    if receipt.is_file():
        result = json.loads(receipt.read_text(encoding="utf-8"))
        if result["evidence"] != binding:
            raise ValueError(f"Existing evaluation has different provenance: {output}")
        for name, digest in result["artifacts"].items():
            if common.sha256(output / name) != digest:
                raise ValueError(f"Changed evaluation artifact: {output / name}")
        return result
    output.mkdir(parents=True, exist_ok=False)
    model = YOLO(str(weights))
    assert type(model.model.model[9]) is block_type
    complexity = {"params": sum(p.numel() for p in model.model.parameters()), "gflops": get_flops(model.model, 640)}
    observed = {}

    def observe_precision(module, inputs):
        assert inputs[0].dtype == torch.float32
        assert {p.dtype for p in module.parameters()} == {torch.float32}
        observed["precision"] = {"input": str(inputs[0].dtype), "model": "torch.float32"}

    precision_hook = model.model.register_forward_pre_hook(observe_precision)

    def capture(validator):
        common.write_json(output / "predictions.json", validator.jdict)
        observed.update(
            images=validator.seen, targets=int(validator.metrics.nt_per_class.sum()), args=vars(validator.args)
        )

    model.add_callback("on_val_end", capture)
    try:
        metrics = model.val(
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
    finally:
        precision_hook.remove()
    assert observed["images"] == common.REFERENCE["dataset_counts"][split]
    assert observed["targets"] == {"val": 2985, "test": 1477}[split]
    assert observed["args"]["quantize"] is None
    YAML.save(output / "args.yaml", observed["args"])
    result = {
        "evidence": binding,
        "metrics": {
            "P": float(metrics.box.mp),
            "R": float(metrics.box.mr),
            "mAP50": float(metrics.box.map50),
            "mAP50-95": float(metrics.box.map),
            "AP75": float(metrics.box.all_ap[:, 5].mean()),
            **complexity,
        },
        "ap_by_iou": metrics.box.all_ap.mean(axis=0).tolist(),
        **observed,
        "artifacts": {name: common.sha256(output / name) for name in CURVES},
    }
    common.write_json(receipt, result)
    return result


def test_and_compare(run, data, baseline_best, evidence, block_type=DCS_SPPF, historical=None):
    """Evaluate candidate val/test and original trained b19 test with identical settings; never retrain b19."""
    reference = json.loads(Path(__file__).with_name("b19_test_reference.json").read_text(encoding="utf-8"))
    if common.sha256(baseline_best) != reference["weight_sha256"]:
        raise ValueError("Comparison weight is not the archived formal b19 best.pt")
    output = run / "baseline_comparison"
    evaluate(run / "weights/best.pt", data, output / "dcs_val", block_type, evidence, "val")
    candidate = evaluate(run / "weights/best.pt", data, output / "dcs_test", block_type, evidence)
    baseline = evaluate(baseline_best, data, output / "b19_test", SPPF, evidence)
    rows = {
        key: {"b19": value, "DCS": candidate["metrics"][key], "delta": candidate["metrics"][key] - value}
        for key, value in baseline["metrics"].items()
    }
    b19_historical = {
        "P": reference["results_dict"]["metrics/precision(B)"],
        "R": reference["results_dict"]["metrics/recall(B)"],
        "mAP50": reference["results_dict"]["metrics/mAP50(B)"],
        "mAP50-95": reference["results_dict"]["metrics/mAP50-95(B)"],
        "AP75": reference["ap75"],
    }
    report = {
        "evidence": evidence,
        "comparison": rows,
        "b19_rerun_minus_reference": {k: baseline["metrics"][k] - v for k, v in b19_historical.items()},
        "minimum_success": rows["mAP50-95"]["delta"] > 0,
        "ap75_improved": rows["AP75"]["delta"] > 0,
        "recall_delta": rows["R"]["delta"],
    }
    if historical is not None:
        report["v1_history"] = {"archive_id": historical["archive_id"], "sha256": historical["sha256"], "rerun": False}
        report["three_way_metrics"] = {
            k: {
                "b19": baseline["metrics"][k],
                "DCS_v1_history": historical["dcs_test"]["metrics"][k],
                "DCS_v2": candidate["metrics"][k],
                "v2_minus_b19_pp": 100 * rows[k]["delta"],
            }
            for k in b19_historical
        }
        report["ap_by_iou"] = [
            {
                "iou": round(0.5 + 0.05 * i, 2),
                "b19": b,
                "DCS_v1_history": h,
                "DCS_v2": c,
                "v2_minus_b19_pp": 100 * (c - b),
            }
            for i, (b, h, c) in enumerate(
                zip(baseline["ap_by_iou"], historical["dcs_test"]["ap_by_iou"], candidate["ap_by_iou"])
            )
        ]
    common.write_json(output / "comparison.json", report)
    lines = [
        f"# b19 / {block_type.__name__}",
        "",
        "Same-environment FP32 test, fixed best.pt selected on training val.",
        "",
        "| Metric | b19 | DCS | Delta |",
        "| --- | --- | --- | --- |",
    ]
    lines += [f"| {k} | {v['b19']:.9g} | {v['DCS']:.9g} | {v['delta']:+.9g} |" for k, v in rows.items()]
    lines += [
        "",
        "B19 rerun minus reference: " + json.dumps(report["b19_rerun_minus_reference"]),
        "No automatic design changes or subsequent training are performed.",
    ]
    if historical is not None:
        lines += [
            "",
            "DCS v1 is historical, not rerun: " + historical["sha256"],
            "",
            "| Metric | b19 | DCS v1 history | DCS v2 | v2-b19 / pp |",
            "| --- | --- | --- | --- | --- |",
        ]
        lines += [
            f"| {k} | {v['b19']:.12g} | {v['DCS_v1_history']:.12g} | {v['DCS_v2']:.12g} | {v['v2_minus_b19_pp']:+.9g} |"
            for k, v in report["three_way_metrics"].items()
        ]
        lines += [
            f"| AP{v['iou'] * 100:.0f} | {v['b19']:.12g} | {v['DCS_v1_history']:.12g} | {v['DCS_v2']:.12g} | {v['v2_minus_b19_pp']:+.9g} |"
            for v in report["ap_by_iou"]
        ]
        lines += [
            "",
            "Test has informed development observations; it is not an untouched blind test. Single-run differences do not establish significance or unique causality.",
        ]
    (output / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def diagnose(
    weights, data, directory, evidence, device="cuda:0", block_type=DCS_SPPF, residual_control=None, historical=None
):
    """Capture live contrasts/refinements on the first 16 sorted validation images, with fixed hashes and CSV statistics."""
    receipt = directory / "dcs_sppf_diagnostics.json"
    if receipt.is_file():
        report = json.loads(receipt.read_text(encoding="utf-8"))
        assert report["evidence"] == evidence
        for name, digest in report["artifacts"].items():
            assert common.sha256(directory / name) == digest
        return report
    directory.mkdir(parents=True, exist_ok=False)
    dataset = check_det_dataset(str(data), autodownload=False)
    files = sorted(p for p in Path(dataset["val"]).rglob("*") if p.suffix[1:].lower() in IMG_FORMATS)[:16]
    assert len(files) == 16
    model, _ = load_checkpoint(weights, device=device)
    model.fuse(verbose=False)
    block = model.model[9]
    assert type(block) is block_type
    captured, handles = {}, []

    def capture(name, inputs=False):
        def hook(module, args, output):
            captured[name] = (args[0] if inputs else output).detach()

        return hook

    handles.append(block.register_forward_hook(capture("input", inputs=True)))
    handles.append(block.register_forward_hook(capture("output")))
    handles.append(block.cv2.register_forward_hook(capture("cv2")))
    handles.append(block.fuse.register_forward_hook(capture("fused_R")))
    for scale, branch in zip((5, 9, 13), block.refine):
        handles.append(branch.register_forward_hook(capture(f"C{scale}", inputs=True)))
        handles.append(branch.register_forward_hook(capture(f"R{scale}")))

    def stats(tensor):
        tensor = tensor.float()
        assert torch.isfinite(tensor).all()
        return {
            "mean": tensor.mean().item(),
            "std": tensor.std(unbiased=False).item(),
            "max": tensor.max().item(),
            "max_abs": tensor.abs().max().item(),
        }

    rows = []
    try:
        for path in files:
            captured.clear()
            original = cv2.imread(str(path))
            assert original is not None, path
            im = LetterBox(new_shape=(640, 640), auto=False)(image=original)
            im = (
                torch.from_numpy(np.ascontiguousarray(im[:, :, ::-1].transpose(2, 0, 1))).to(device).float()[None] / 255
            )
            with torch.no_grad():
                model(im)
            native = captured.pop("cv2")
            x = captured.pop("input")
            if block.add:
                native = native + x
            output = captured.pop("output")
            alpha = 0.10 * block.theta.float().tanh()
            residual = alpha * captured["fused_R"].float()
            injected, q = (
                (residual, torch.ones_like(residual[:, :1, :1, :1]))
                if residual_control is None
                else residual_control(residual, native.float(), block)
            )
            expected = native + injected.to(native.dtype)
            torch.testing.assert_close(output, expected, atol=0, rtol=0)
            observed = output.float() - native.float()

            def rms(value):
                return value.double().square().mean().sqrt().item()

            reference_rms = rms(native)
            denominator = max(reference_rms, 1e-12)
            positional = injected.float().square().mean(1).sqrt().flatten() / denominator
            if residual_control is not None:
                assert rms(injected) <= block.residual_budget * reference_rms + 2e-8 * reference_rms
            ratios = {
                "raw_ratio": rms(residual) / denominator,
                "injected_ratio": rms(injected) / denominator,
                "observed_ratio": rms(observed) / denominator,
            }
            row = {
                "image": path.name,
                "image_sha256": common.sha256(path),
                "label_sha256": common.sha256(img2label_paths([str(path)])[0]),
                "theta": block.theta.item(),
                "alpha": alpha.item(),
                "rho": getattr(block, "residual_budget", None),
                "eps": getattr(block, "residual_eps", None),
                "RMS_y0": reference_rms,
                "RMS_raw": rms(residual),
                "RMS_injected": rms(injected),
                "zero_reference": reference_rms == 0,
                "q": q.item(),
                **ratios,
                "budget_fraction": ratios["injected_ratio"] / block.residual_budget if residual_control else None,
                "injected": stats(injected),
                "raw": stats(residual),
                "observed": stats(observed),
                "position_relative_rms": {
                    "p95": torch.quantile(positional, 0.95).item(),
                    "p99": torch.quantile(positional, 0.99).item(),
                    "max": positional.max().item(),
                },
                "crosscheck_max_abs": (output - expected).abs().max().item(),
                **{key: stats(value) for key, value in captured.items()},
                "alpha_times_R": stats(residual),
                "Y_native": stats(native),
                "residual_ratio": (residual.float().norm() / native.float().norm().clamp_min(1e-12)).item(),
            }
            rows.append(row)
    finally:
        for handle in handles:
            handle.remove()
    if historical is not None:
        assert [(r["image"], r["image_sha256"], r["label_sha256"]) for r in rows] == [
            (r["image"], r["image_sha256"], r["label_sha256"]) for r in historical["diagnostic"]["images"]
        ], "Diagnostic selection differs from v1"
    flat = []
    for row in rows:
        item = {}
        for key, value in row.items():
            if isinstance(value, dict):
                item.update({f"{key}_{stat}": number for stat, number in value.items()})
            else:
                item[key] = value
        flat.append(item)
    csv_path = directory / "dcs_sppf_diagnostics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    report = {
        "evidence": evidence,
        "commit": common.git("rev-parse", "HEAD"),
        "weight_sha256": common.sha256(weights),
        "selection": "first 16 lexically sorted validation images; no test selection",
        "images": rows,
        "preprocessing": "native LetterBox 640x640, auto=False, RGB/255",
        "precision": "FP32 fused",
        "ratio_denominator_floor": 1e-12,
        "position_units": "RMS across channels at each pixel / whole-image RMS(y0)",
        "summary": {
            key: {
                "mean": float(np.mean([r[key] for r in rows])),
                "min": min(r[key] for r in rows),
                "max": max(r[key] for r in rows),
            }
            for key in ("raw_ratio", "injected_ratio", "observed_ratio", "q", "RMS_y0", "RMS_raw", "RMS_injected")
        },
        "artifacts": {csv_path.name: common.sha256(csv_path)},
    }
    common.write_json(receipt, report)
    return report


def archive_package(run, output, required):
    """Materialize canonical files, prune transient attempts before walking, and read/hash every archived member.

    Adapted from the repaired CCA v2 archive routine; includes both best and last weights and Markdown diagnostics.
    """
    if output.exists() or run == output or run in output.parents:
        raise ValueError("Package must be a new file outside the canonical run directory")
    for name in required:
        if not (run / name).is_file() or (run / name).is_symlink():
            raise FileNotFoundError(f"Missing regular canonical artifact: {run / name}")
    source_archive = run / "source.tar"
    subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={ROOT.as_posix()}",
            "-c",
            "core.autocrlf=false",
            "archive",
            "--format=tar",
            "-o",
            str(source_archive),
            "HEAD",
        ],
        cwd=ROOT,
        check=True,
    )
    (run / "source.patch").write_bytes(
        subprocess.check_output(
            [
                "git",
                "-c",
                f"safe.directory={ROOT.as_posix()}",
                "diff",
                "--binary",
                common.REFERENCE["source_commit"],
                "HEAD",
            ],
            cwd=ROOT,
        )
    )
    common.write_json(
        run / "source_manifest.json",
        {
            "commit": common.git("rev-parse", "HEAD"),
            "base_commit": common.REFERENCE["source_commit"],
            "files": common.source_hashes(),
            "diff_summary": common.git("diff", "--stat", common.REFERENCE["source_commit"], "HEAD"),
        },
    )
    files, skipped = {}, []
    for folder, dirs, names in os.walk(run, followlinks=False):
        skipped += [
            str((Path(folder) / d).relative_to(run))
            for d in dirs
            if ".attempt." in d or (Path(folder) / d).is_symlink()
        ]
        dirs[:] = sorted(d for d in dirs if ".attempt." not in d and not (Path(folder) / d).is_symlink())
        for name in sorted(names):
            path = Path(folder) / name
            if (
                ".attempt." in name
                or path.is_symlink()
                or not path.is_file()
                or name in {"package_manifest.json", "package_result.json"}
            ):
                skipped.append(path.relative_to(run).as_posix())
                continue
            files["run/" + path.relative_to(run).as_posix()] = path
    common.write_json(
        run / "package_skipped.json",
        {"paths": skipped, "reason": "transient attempts, aliases, nonregular files, or prior package metadata"},
    )
    files["run/package_skipped.json"] = run / "package_skipped.json"
    assert all("run/" + name in files for name in required)
    manifest = {
        name: {"sha256": common.sha256(path), "size_bytes": path.stat().st_size} for name, path in files.items()
    }
    common.write_json(run / "package_manifest.json", manifest)
    files["package_manifest.json"] = run / "package_manifest.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream, tarfile.open(fileobj=stream, mode="w:gz", dereference=True) as archive:
        for name, path in files.items():
            archive.add(path, arcname=name, recursive=False)
    with tarfile.open(output, "r:gz") as archive:
        assert set(archive.getnames()) == set(files)
        for name, expected in manifest.items():
            member = archive.getmember(name)
            assert member.isfile() and member.size == expected["size_bytes"], name
            digest = hashlib.sha256()
            with archive.extractfile(member) as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            assert digest.hexdigest() == expected["sha256"], name
    with gzip.open(output, "rb") as stream:
        while stream.read(1024 * 1024):
            pass
    digest = common.sha256(output)
    output.with_name(output.name + ".sha256").write_text(f"{digest}  {output.name}\n", encoding="utf-8")
    result = {"path": str(output), "sha256": digest, "members_verified": len(files), "gzip_crc_verified": True}
    common.write_json(run / "package_result.json", result)
    print(json.dumps(result))
    return result


def package(run, output, evidence):
    """Require successful test and diagnosis independently of packaging before producing a final archive."""
    comparison = json.loads((run / "baseline_comparison/comparison.json").read_text(encoding="utf-8"))
    diagnostic = json.loads(
        (run / "baseline_comparison/diagnostics/dcs_sppf_diagnostics.json").read_text(encoding="utf-8")
    )
    assert comparison["evidence"] == evidence
    assert diagnostic["weight_sha256"] == evidence["weight_sha256"] and diagnostic["commit"] == evidence["commit"]
    assert diagnostic["evidence"] == evidence
    for name, digest in diagnostic["artifacts"].items():
        assert common.sha256(run / "baseline_comparison/diagnostics" / name) == digest
    required = [
        "weights/best.pt",
        "weights/last.pt",
        "results.csv",
        "args.yaml",
        "results.png",
        "completed.json",
        "provenance/resolved.json",
        "provenance/b19_original_args.yaml",
        "provenance/b19_launcher_expanded.txt",
        "provenance/weights.json",
        "provenance/optimizer.json",
        "provenance/preflight/checks.json",
        "provenance/preflight/native_binding.json",
        "provenance/preflight/native_steps.json",
        "provenance/preflight/native_gradient_summary.json",
        "provenance/preflight/console.log",
        "baseline_comparison/comparison.json",
        "baseline_comparison/comparison.md",
        "baseline_comparison/diagnostics/dcs_sppf_diagnostics.json",
        "baseline_comparison/diagnostics/dcs_sppf_diagnostics.csv",
    ]
    stage_results = {}
    reference = json.loads(Path(__file__).with_name("b19_test_reference.json").read_text(encoding="utf-8"))
    for stage in ("dcs_val", "dcs_test", "b19_test"):
        folder = run / "baseline_comparison" / stage
        receipt = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
        assert receipt["evidence"] == {
            **evidence,
            "split": "val" if stage == "dcs_val" else "test",
            "weight_sha256": reference["weight_sha256"] if stage == "b19_test" else evidence["weight_sha256"],
        }, f"Stale evaluation receipt: {stage}"
        stage_results[stage] = receipt["metrics"]
        for name, digest in receipt["artifacts"].items():
            assert common.sha256(folder / name) == digest
        required += [f"baseline_comparison/{stage}/{name}" for name in ["metrics.json", *CURVES]]
    for metric, value in stage_results["b19_test"].items():
        candidate = stage_results["dcs_test"][metric]
        assert comparison["comparison"][metric] == {"b19": value, "DCS": candidate, "delta": candidate - value}
    return archive_package(run, output, required)


def main(argv=None, name=NAME, model=common.MODEL, block_type=DCS_SPPF, residual_control=None, historical=None):
    """Dispatch each finish stage and report failures under its actual stage name."""
    parser = argparse.ArgumentParser(description=f"Evaluate, diagnose and archive {name} with fixed b19 provenance.")
    parser.add_argument("--stage", choices=["test", "diagnose", "package", "all"], required=True)
    parser.add_argument("--run", type=Path, default=ROOT / "runs/detect" / name)
    parser.add_argument("--data", type=Path, default=Path(common.REFERENCE["args"]["data"]))
    parser.add_argument(
        "--baseline-best", type=Path, default=Path(common.REFERENCE["args"]["save_dir"]) / "weights/best.pt"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    run, data = args.run.resolve(), args.data.resolve()
    evidence = provenance(run, data, name, model, block_type)
    for stage in ["test", "diagnose", "package"] if args.stage == "all" else [args.stage]:
        try:
            if stage == "test":
                require_runtime()
                test_and_compare(run, data, args.baseline_best, evidence, block_type, historical)
            elif stage == "diagnose":
                require_runtime()
                diagnose(
                    run / "weights/best.pt",
                    data,
                    run / "baseline_comparison/diagnostics",
                    evidence,
                    block_type=block_type,
                    residual_control=residual_control,
                    historical=historical,
                )
            else:
                output = args.output or ROOT / "artifacts/experiments" / f"{name}_{evidence['commit'][:12]}.tar.gz"
                package(run, output.resolve(), evidence)
            common.write_json(run / f"{stage}_status.json", {"stage": stage, "passed": True, "evidence": evidence})
        except Exception as error:
            common.write_json(run / f"{stage}_status.json", {"stage": stage, "passed": False, "error": str(error)})
            raise


if __name__ == "__main__":
    main()
