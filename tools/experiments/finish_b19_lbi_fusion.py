"""Evaluate, diagnose and archive LBI v1 with distinct stage receipts and an immutable b19 comparison."""

# Direct entry must resolve this worktree before package imports.
# ruff: noqa: E402

import argparse
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

from tools.experiments import b19_common as common
from tools.experiments.diagnose_b19_lbi_fusion import diagnose
from tools.experiments.run_b19_lbi_fusion import NAME, audit_arguments, require_runtime
from ultralytics import YOLO
from ultralytics.nn.modules import Concat, Concat_LBI_Fusion
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


def provenance(run, data):
    """Require completed canonical training, unchanged files/code and the exact fixed dataset."""
    common.require_clean_source()
    completion = json.loads((run / "completed.json").read_text())
    resolved = json.loads((run / "provenance/resolved.json").read_text())
    if completion["commit"] != common.git("rev-parse", "HEAD"):
        raise ValueError("Finish must run at the training commit")
    for name, digest in completion["files"].items():
        if common.sha256(run / name) != digest:
            raise ValueError(f"Changed completed training file: {name}")
    audit_arguments(resolved["config"], YAML.load(run / "args.yaml"))
    if resolved["config"]["name"] != NAME or Path(resolved["config"]["project"]) / NAME != run:
        raise ValueError("Run path differs from its canonical training identity")
    if resolved["evidence"]["source_sha256"] != common.source_hashes():
        raise ValueError("Source files differ from the recorded training source")
    manifest = common.dataset_manifest(data)
    if manifest != resolved["evidence"]["dataset_manifest"]:
        raise ValueError("Evaluation dataset differs from the verified training dataset")
    preflight = json.loads((run / "provenance/preflight/checks.json").read_text())
    assert common.sha256(run / "provenance/preflight/checks.json") == resolved["evidence"]["preflight_checks_sha256"]
    if not preflight["passed"] or preflight["local_only"] or preflight["commit"] != completion["commit"]:
        raise ValueError("Missing matching formal preflight")
    return {
        "commit": completion["commit"],
        "weight_sha256": completion["files"]["weights/best.pt"],
        "data_manifest": manifest,
        "complexity": preflight["complexity"],
    }


def evaluate(weights, data, output, block_type, evidence, split="test"):
    """Reuse native FP32 b19 evaluation settings and extract AP75 from its all_ap array."""
    binding = {**evidence, "weight_sha256": common.sha256(weights), "split": split}
    receipt = output / "metrics.json"
    if receipt.is_file():
        result = json.loads(receipt.read_text())
        if result["evidence"] != binding:
            raise ValueError(f"Existing evaluation has different provenance: {output}")
        for name, digest in result["artifacts"].items():
            if common.sha256(output / name) != digest:
                raise ValueError(f"Changed evaluation artifact: {output / name}")
        return result
    output.mkdir(parents=True, exist_ok=False)
    model = YOLO(str(weights))
    assert type(model.model.model[15]) is block_type
    complexity = {"params": sum(p.numel() for p in model.model.parameters()), "gflops": get_flops(model.model, 640)}
    observed = {}

    def capture(validator):
        common.write_json(output / "predictions.json", validator.jdict)
        observed.update(
            images=validator.seen, targets=int(validator.metrics.nt_per_class.sum()), args=vars(validator.args)
        )

    model.add_callback("on_val_end", capture)
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


def test_and_compare(run, data, baseline_best, evidence):
    """Evaluate candidate val/test and original trained b19 test with identical settings; never retrain b19."""
    reference = json.loads(Path(__file__).with_name("b19_test_reference.json").read_text())
    if common.sha256(baseline_best) != reference["weight_sha256"]:
        raise ValueError("Comparison weight is not the archived formal b19 best.pt")
    output = run / "baseline_comparison"
    candidate_val = evaluate(run / "weights/best.pt", data, run / "evaluation/val", Concat_LBI_Fusion, evidence, "val")
    baseline_val = evaluate(baseline_best, data, output / "b19_val", Concat, evidence, "val")
    candidate = evaluate(run / "weights/best.pt", data, run / "evaluation/test", Concat_LBI_Fusion, evidence)
    baseline = evaluate(baseline_best, data, output / "b19_test", Concat, evidence)
    rows = {
        key: {"b19": value, "LBI": candidate["metrics"][key], "delta": candidate["metrics"][key] - value}
        for key, value in baseline["metrics"].items()
    }
    historical = {
        "P": reference["results_dict"]["metrics/precision(B)"],
        "R": reference["results_dict"]["metrics/recall(B)"],
        "mAP50": reference["results_dict"]["metrics/mAP50(B)"],
        "mAP50-95": reference["results_dict"]["metrics/mAP50-95(B)"],
        "AP75": reference["ap75"],
    }
    report = {
        "evidence": evidence,
        "comparison": rows,
        "val_comparison": {
            k: {"b19": v, "LBI": candidate_val["metrics"][k], "delta": candidate_val["metrics"][k] - v}
            for k, v in baseline_val["metrics"].items()
        },
        "b19_rerun_minus_reference": {k: baseline["metrics"][k] - v for k, v in historical.items()},
        "recall_improved": rows["R"]["delta"] > 0,
        "ap75_not_decreased": rows["AP75"]["delta"] >= 0,
    }
    common.write_json(output / "comparison.json", report)
    lines = [
        "# b19 / LBI-Fusion v1",
        "",
        "Same-environment FP32 test, fixed best.pt selected on training val.",
        "",
        "| Metric | b19 | LBI | Delta |",
        "| --- | --- | --- | --- |",
    ]
    lines += [f"| {k} | {v['b19']:.9g} | {v['LBI']:.9g} | {v['delta']:+.9g} |" for k, v in rows.items()]
    lines += [
        "",
        "B19 rerun minus reference: " + json.dumps(report["b19_rerun_minus_reference"]),
        "No automatic design changes or subsequent training are performed.",
    ]
    (output / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


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
            "files_scope": "Working-tree bytes; source.tar is the exact portable Git tree (LF, independent of core.autocrlf)",
            "diff_summary": common.git("diff", "--stat", common.REFERENCE["source_commit"], "HEAD"),
        },
    )
    files, excluded = {}, []
    for folder, dirs, names in os.walk(run, followlinks=False):
        excluded.extend(
            {"path": (Path(folder) / d).relative_to(run).as_posix(), "reason": "attempt or directory symlink"}
            for d in dirs
            if ".attempt." in d or (Path(folder) / d).is_symlink()
        )
        dirs[:] = sorted(d for d in dirs if ".attempt." not in d and not (Path(folder) / d).is_symlink())
        for name in sorted(names):
            path = Path(folder) / name
            if (
                ".attempt." in name
                or path.is_symlink()
                or not path.is_file()
                or name in {"package_manifest.json", "package_result.json"}
            ):
                excluded.append(
                    {
                        "path": path.relative_to(run).as_posix(),
                        "reason": "attempt, symlink, nonregular or regenerated receipt",
                    }
                )
                continue
            files["run/" + path.relative_to(run).as_posix()] = path
    assert all("run/" + name in files for name in required)
    common.write_json(
        run / "package_exclusions.json",
        {
            "excluded": excluded,
            "optional_console_log": "present" if (run / "console.log").is_file() else "missing (optional)",
            "hardlinks": "dereference=True; materialized regular files, no external tar link targets",
        },
    )
    files["run/package_exclusions.json"] = run / "package_exclusions.json"
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
    comparison = json.loads((run / "baseline_comparison/comparison.json").read_text())
    diagnostic = json.loads((run / "diagnostics/lbi_fusion_diagnostics.json").read_text())
    assert comparison["evidence"] == evidence
    assert diagnostic["weight_sha256"] == evidence["weight_sha256"] and diagnostic["commit"] == evidence["commit"]
    assert common.sha256(run / "diagnostics/lbi_fusion_diagnostics.csv") == diagnostic["csv_sha256"]
    required = [
        "weights/best.pt",
        "weights/last.pt",
        "results.csv",
        "args.yaml",
        "results.png",
        "completed.json",
        "baseline_comparison/comparison.json",
        "baseline_comparison/comparison.md",
        "diagnostics/lbi_fusion_diagnostics.json",
        "diagnostics/lbi_fusion_diagnostics.csv",
        "diagnostics/validation_samples.json",
        "provenance/resolved.json",
        "provenance/weights.json",
        "provenance/optimizer.json",
        "provenance/b19_original_args.yaml",
        "provenance/preflight/checks.json",
        "provenance/preflight/native_steps.json",
        "provenance/preflight/native_batches.json",
        "provenance/preflight/native_gradient_summary.json",
    ]
    for stage in ("evaluation/val", "evaluation/test", "baseline_comparison/b19_val", "baseline_comparison/b19_test"):
        folder = run / stage
        receipt = json.loads((folder / "metrics.json").read_text())
        expected_weight = (
            evidence["weight_sha256"]
            if stage.startswith("evaluation/")
            else json.loads(Path(__file__).with_name("b19_test_reference.json").read_text())["weight_sha256"]
        )
        assert receipt["evidence"] == {
            **evidence,
            "weight_sha256": expected_weight,
            "split": "val" if stage.endswith("val") else "test",
        }
        for name, digest in receipt["artifacts"].items():
            assert common.sha256(folder / name) == digest
        required += [f"{stage}/{name}" for name in ["metrics.json", *CURVES]]
    return archive_package(run, output, required)


def main(argv=None):
    """Dispatch each finish stage and report failures under its actual stage name."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["test", "diagnose", "package", "all"], required=True)
    parser.add_argument("--run", type=Path, default=ROOT / "runs/detect" / NAME)
    parser.add_argument("--data", type=Path, default=Path(common.REFERENCE["args"]["data"]))
    parser.add_argument(
        "--baseline-best", type=Path, default=Path(common.REFERENCE["args"]["save_dir"]) / "weights/best.pt"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    run, data = args.run.resolve(), args.data.resolve()
    evidence = provenance(run, data)
    for stage in ["test", "diagnose", "package"] if args.stage == "all" else [args.stage]:
        try:
            if stage == "test":
                require_runtime()
                test_and_compare(run, data, args.baseline_best, evidence)
            elif stage == "diagnose":
                require_runtime()
                diagnose(run / "weights/best.pt", data, run / "diagnostics")
            else:
                output = args.output or ROOT / "artifacts/experiments" / f"{NAME}_{evidence['commit'][:12]}.tar.gz"
                package(run, output.resolve(), evidence)
            common.write_json(run / f"{stage}_status.json", {"stage": stage, "passed": True, "evidence": evidence})
        except Exception as error:
            common.write_json(run / f"{stage}_status.json", {"stage": stage, "passed": False, "error": str(error)})
            raise


if __name__ == "__main__":
    main()
