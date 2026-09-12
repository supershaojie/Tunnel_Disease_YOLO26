"""Replay authorized DCS server tensors using current trusted architectures; never train or load model pickles."""

# ruff: noqa: E402 -- Direct entry must import this worktree, never archived Python.
import argparse
import copy
import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from torch.nn import functional as F

from tools.experiments import b19_common as common
from tools.experiments.b19_detect_fuse_audit import (
    PROFILES,
    attributes,
    diagnostic,
    fuse_precision,
    fuse_precision_checks,
    snapshot,
)
from tools.experiments.run_b19_dcs_sppf_v2 import MODEL
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import YAML
from ultralytics.utils.torch_utils import init_seeds

VARIANTS = ("native", "v1", "v2_zero", "v2_updated", "v2_diagnostic")


def restore_fixture(directory, variant):
    """Strictly load each saved whole-model state into a local fixed graph, checking all inference attributes."""
    assert variant in VARIANTS
    source = torch.load(directory / "source.pt", map_location="cpu", weights_only=True)
    cfg = (
        common.baseline_architecture() if variant == "native" else YAML.load(common.MODEL if variant == "v1" else MODEL)
    )
    cfg.update(nc=1, scale="n")
    assert source["yaml"]["nc"] == 1
    assert common.architecture_signature(cfg) == common.architecture_signature(source["yaml"])
    # Never pass external YAML to parse_model (which accepts activation expressions) or arbitrary archived setattr.
    with torch.random.fork_rng():
        model = DetectionModel(copy.deepcopy(cfg), verbose=False).float().eval()
    model.load_state_dict(source["state"], strict=True)
    assert isinstance(source["names"], dict) and set(source["names"]) == {0}
    assert isinstance(source["names"][0], str)
    model.names = copy.deepcopy(source["names"])
    common.assert_close_tree(source["state"], snapshot(model.state_dict()), 0, 0, "fixture.state")
    common.assert_close_tree(source["attributes"], attributes(model), 0, 0, "fixture.attributes")
    x = torch.load(directory / "before.pt", map_location="cpu", weights_only=True)["input_after"].clone()
    assert x.dtype == torch.float32 and tuple(x.shape) == (1, 3, 128, 160) and torch.isfinite(x).all()
    return model, x


def first_block_replay(directory, device, output):
    """Feed exactly the same saved model.1 tensor to original Conv/BN/SiLU and saved folded Conv/SiLU."""
    device, output = torch.device(device), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    source, folded, layers = [
        torch.load(directory / f"{name}.pt", map_location="cpu", weights_only=True)
        for name in ("source", "fused_state", "layer_outputs")
    ]
    name = "model.2.cv1."
    state = {k: v.to(device) for k, v in source["state"].items() if k.startswith(name)}
    x = layers[0]["model.1"].to(device).clone()
    assert x.dtype == torch.float32 and tuple(x.shape) == (1, 32, 32, 40)
    assert tuple(state[name + "conv.weight"].shape) == (32, 32, 1, 1)
    assert source["attributes"]["bn"][name + "bn"]["eps"] == 0.001
    report = dict(
        strict_gate_passed=False,
        device=str(device),
        torch=torch.__version__,
        python=sys.version,
        cuda=torch.version.cuda,
        cudnn=torch.backends.cudnn.version(),
        gpu=torch.cuda.get_device_name(device) if x.is_cuda else None,
        capability=torch.cuda.get_device_capability(device) if x.is_cuda else None,
        input_shape=list(x.shape),
        profiles={},
        source_sha256={
            f"{name}.pt": common.sha256(directory / f"{name}.pt") for name in ("source", "fused_state", "layer_outputs")
        },
    )
    try:
        assert torch.backends.cudnn.allow_tf32, "Native incident diagnostic requires recorded TF32 permission"
        assert not torch.backends.cuda.matmul.allow_tf32 and torch.get_float32_matmul_precision() == "highest"
        for profile in PROFILES[:2]:
            result = report["profiles"][profile] = {}
            print(f"BEGIN first block {device}/{profile}", flush=True)
            with fuse_precision(device, profile) as conditions, torch.no_grad():
                result["precision"] = conditions
                before = F.silu(
                    F.batch_norm(
                        F.conv2d(x.clone(), state[name + "conv.weight"]),
                        state[name + "bn.running_mean"],
                        state[name + "bn.running_var"],
                        state[name + "bn.weight"],
                        state[name + "bn.bias"],
                        training=False,
                        eps=0.001,
                    )
                )
                after = F.silu(
                    F.conv2d(
                        x.clone(),
                        folded["state"][name + "conv.weight"].to(device),
                        folded["state"][name + "conv.bias"].to(device),
                    )
                )
                torch.save(
                    dict(input=snapshot(x), before=snapshot(before), after=snapshot(after)), output / f"{profile}.pt"
                )
                result["comparison"] = diagnostic(snapshot(before), snapshot(after), 1e-4, 1e-4, "model.2.cv1")
                result["status"] = "PASS" if result["comparison"]["passed"] else "FAIL"
            print(f"END first block {device}/{profile}: {result['status']}", flush=True)
        assert report["profiles"]["strict_fp32_equivalence"]["comparison"]["passed"], "Strict first block failed"
        report["strict_gate_passed"] = True
    except BaseException:
        report["traceback"] = traceback.format_exc()
        raise
    finally:
        common.write_json(output / "first_block.json", report)
    return report


def main():
    """Run both first-block fixtures and all five complete source states, retaining original failures separately."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evidence-root", type=Path, required=True, help="Validated extraction root containing preflight/"
    )
    parser.add_argument("--device", choices=("cpu", "cuda:0"), required=True)
    parser.add_argument("--threads", type=int, help="Optional local replay CPU thread count; default preserves runtime")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA NOT RUN: unavailable; no CPU fallback")
    if args.threads is not None:
        torch.set_num_threads(args.threads)
    init_seeds(42, deterministic=True)
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(
        strict_gate_passed=False,
        commit=common.git("rev-parse", "HEAD"),
        source_sha256=common.source_hashes(),
        backend=common.computation_conditions(),
        device=str(device),
        torch=torch.__version__,
        python=sys.version,
        cuda=torch.version.cuda,
        cudnn=torch.backends.cudnn.version(),
        gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        capability=torch.cuda.get_device_capability(device) if device.type == "cuda" else None,
        RTX4090_incident_replay="NOT RUN",
        server_B32_640="NOT RUN",
        formal_training="NOT STARTED",
        original_server={},
        first_block={},
        full_model={},
        saved_fold_comparisons={},
        matches_incident_runtime=False,
    )
    try:
        directories = {}
        for variant in VARIANTS:
            matches = list((args.evidence_root / "preflight" / variant).glob("fuse-evidence-*"))
            assert len(matches) == 1, f"Expected one unambiguous original attempt for {variant}"
            directories[variant] = matches[0]
            recorded = json.loads((matches[0] / "audit.json").read_text(encoding="utf-8"))
            report["original_server"][variant] = recorded
        recorded = report["original_server"]["native"]
        report["matches_incident_runtime"] = (
            device.type == "cuda"
            and all(report[key] == recorded[key] for key in ("gpu", "torch", "cuda", "cudnn", "backend"))
            and report["python"].split()[0] == recorded["python"].split()[0]
        )
        report["RTX4090_device_replay"] = "RUNNING" if report["gpu"] == recorded["gpu"] else "NOT RUN"
        if report["matches_incident_runtime"]:
            report["RTX4090_incident_replay"] = "RUNNING"
        for variant in ("native", "v2_updated"):
            report["first_block"][variant] = first_block_replay(
                directories[variant], device, args.output / "first_block" / variant
            )
        models, inputs = {}, []
        for variant in VARIANTS:
            models[variant], x = restore_fixture(directories[variant], variant)
            inputs.append(x)
        for x in inputs[1:]:
            common.assert_close_tree(inputs[0], x, 0, 0, "same_server_input")
        report["same_server_input"] = True
        fuse_precision_checks(models, inputs[0].to(device), args.output / "full_model", report["full_model"])
        for variant in VARIANTS:
            saved = torch.load(directories[variant] / "fused_state.pt", map_location="cpu", weights_only=True)
            target = Path(report["full_model"]["profiles"]["strict_fp32_equivalence"][variant]["directory"])
            refold = torch.load(target / "fused_state.pt", map_location="cpu", weights_only=True)
            result = report["saved_fold_comparisons"][variant] = diagnostic(saved, refold, 0, 0, "saved_fold_vs_refold")
            assert result["passed"], f"Server saved folded state differs: {variant}"
        report["strict_gate_passed"] = True
        if report["RTX4090_device_replay"] == "RUNNING":
            report["RTX4090_device_replay"] = "COMPLETED: consult all named profile verdicts"
        if report["matches_incident_runtime"]:
            report["RTX4090_incident_replay"] = "PASS"  # Strict gate; native/AMP diagnostic FAIL remain in full_model.
    except BaseException:
        report["traceback"] = traceback.format_exc()
        if report.get("RTX4090_device_replay") == "RUNNING":
            report["RTX4090_device_replay"] = "FAIL"
        if report["matches_incident_runtime"]:
            report["RTX4090_incident_replay"] = "FAIL"
        raise
    finally:
        common.write_json(args.output / "replay.json", report)


if __name__ == "__main__":
    main()
