"""Read-only replay of archived server tensors on an explicitly named device; never train or load model pickles."""

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
from tools.experiments.lbi_fuse_audit import diagnostic, fuse_precision, fuse_precision_checks, snapshot
from tools.experiments.verify_b19_lbi_fusion import FUSE_ATOL, FUSE_RTOL
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.torch_utils import init_seeds


def restore_fixture(directory):
    """Reconstruct a complete saved YAML/state/attribute fixture and verify every loaded tensor before inference."""
    source = torch.load(directory / "source.pt", weights_only=True, map_location="cpu")
    model = DetectionModel(copy.deepcopy(source["yaml"]), verbose=False).eval()
    model.load_state_dict(source["state"], strict=True)
    model.names = source["names"]
    modules = dict(model.named_modules())
    for name, attrs in source["attributes"].items():
        for key, value in attrs.items():
            if key not in ("type", "training"):
                setattr(modules[name], key, copy.deepcopy(value))
    common.assert_close_tree(source["state"], snapshot(model.state_dict()), 0, 0, "fixture.state")
    common.assert_close_tree(source["attributes"], snapshot(common.inference_attributes(model)), 0, 0, "fixture.attrs")
    return model, source["input"].clone()


def first_block_replay(directory, device, output):
    """Isolate model.2.cv1 using the same saved model.1 output for both paths, including saved folded weights."""
    output.mkdir(parents=True, exist_ok=False)
    source = torch.load(directory / "source.pt", weights_only=True, map_location="cpu")
    folded = torch.load(directory / "fused_state.pt", weights_only=True, map_location="cpu")["state"]
    layers = torch.load(directory / "layer_outputs.pt", weights_only=True, map_location="cpu")
    name = "model.2.cv1."
    state = source["state"]
    eps = source["attributes"][name + "bn"]["eps"]
    x = layers[0]["model.1"].to(device).clone()
    factor = state[name + "bn.weight"].double() / (state[name + "bn.running_var"].double() + eps).sqrt()
    formula = dict(
        weight=(state[name + "conv.weight"].double() * factor[:, None, None, None]).float(),
        bias=(state[name + "bn.bias"].double() - state[name + "bn.running_mean"].double() * factor).float(),
    )
    report = dict(
        passed=False,
        scope="isolated server first-block fixture; not whole-model or server-runtime certification",
        device=str(device),
        torch=torch.__version__,
        cudnn=torch.backends.cudnn.version(),
        gpu=torch.cuda.get_device_name(device) if x.is_cuda else None,
        capability=torch.cuda.get_device_capability(device) if x.is_cuda else None,
        source_sha256={
            p.name: common.sha256(p)
            for p in (directory / name for name in ("source.pt", "fused_state.pt", "layer_outputs.pt"))
        },
        arms={},
        formula={
            key: diagnostic(value, folded[name + "conv." + key], FUSE_ATOL, FUSE_RTOL, key)
            for key, value in formula.items()
        },
    )
    try:
        for allow in (True, False):
            arm = report["arms"][str(allow)] = {}
            print(f"Server model.2.cv1 replay: {device}, cudnn.allow_tf32={allow}", flush=True)
            with fuse_precision(device, allow) as conditions, torch.no_grad():
                arm["conditions"] = conditions
                s = {k: v.to(device) for k, v in state.items() if k.startswith(name)}
                before = F.silu(
                    F.batch_norm(
                        F.conv2d(x.clone(), s[name + "conv.weight"]),
                        s[name + "bn.running_mean"],
                        s[name + "bn.running_var"],
                        s[name + "bn.weight"],
                        s[name + "bn.bias"],
                        training=False,
                        eps=eps,
                    )
                )
                after = F.silu(
                    F.conv2d(x.clone(), folded[name + "conv.weight"].to(device), folded[name + "conv.bias"].to(device))
                )
                torch.save(
                    dict(input=snapshot(x), before=snapshot(before), after=snapshot(after)), output / f"{allow}.pt"
                )
                arm["comparison"] = diagnostic(snapshot(before), snapshot(after), FUSE_ATOL, FUSE_RTOL, "model.2.cv1")
        assert all(value["passed"] for value in report["formula"].values()), "Saved folding parameters differ"
        assert report["arms"]["False"]["comparison"]["passed"], "Explicit FP32 first-block failed"
        report["passed"] = True
    except BaseException:
        report["traceback"] = traceback.format_exc()
        raise
    finally:
        common.write_json(output / "first_block.json", report)
    return report


def main():
    """Replay all three full server fixtures with the original precision policy, explicit FP32 and native controls."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evidence-root", type=Path, required=True, help="Validated extraction root containing preflight/"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA is unavailable; no CPU fallback or server PASS")
    init_seeds(42, deterministic=True)
    args.output.mkdir(parents=True, exist_ok=False)
    directory = args.evidence_root / "preflight/cuda_0"
    recorded = json.loads((directory / "fuse_native/audit.json").read_text(encoding="utf-8"))
    report = dict(
        passed=False,
        commit=common.git("rev-parse", "HEAD"),
        original_server=recorded,
        actual_device=str(device),
        actual_torch=torch.__version__,
        actual_python=sys.version,
        actual_cudnn=torch.backends.cudnn.version(),
        actual_backend=common.computation_conditions(),
        actual_gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        formal_training="NOT STARTED",
        native_B32="NOT RUN",
    )
    report["matches_recorded_server"] = (
        device.type == "cuda"
        and report["actual_torch"] == recorded["torch"]
        and report["actual_python"].split()[0] == recorded["python"].split()[0]
        and report["actual_cudnn"] == recorded["cudnn"]
        and report["actual_gpu"] == recorded["gpu"]
        and report["actual_backend"] == recorded["backend"]
    )
    report["recorded_server_gpu_ab"] = (
        "RUNNING" if report["matches_recorded_server"] else "NOT RUN: different runtime/device"
    )
    try:
        report["first_block"] = first_block_replay(directory / "fuse_native", device, args.output / "first_block")
        models, inputs = {}, []
        for variant in ("native", "lbi_zero", "lbi_nonzero"):
            model, x = restore_fixture(directory / f"fuse_{variant}")
            models[variant] = model
            inputs.append(x)
        for x in inputs[1:]:
            common.assert_close_tree(inputs[0], x, 0, 0, "same_server_input")
        with fuse_precision(device, recorded["backend"]["cudnn_allow_tf32"]) as conditions:
            report["replay_conditions"] = conditions
            report["full_model"] = fuse_precision_checks(
                models, inputs[0].to(device), args.output / "full_model", FUSE_ATOL, FUSE_RTOL
            )
        report["passed"] = True
        if report["matches_recorded_server"]:
            report["recorded_server_gpu_ab"] = "COMPLETED; consult native and explicit_fp32 verdicts separately"
    except BaseException:
        report["traceback"] = traceback.format_exc()
        if report["matches_recorded_server"]:
            report["recorded_server_gpu_ab"] = "FAILED"
        raise
    finally:
        common.write_json(args.output / "replay.json", report)


if __name__ == "__main__":
    main()
