"""Observe SICR best.pt on a fixed validation subset without modifying its state or selecting a design."""

# ruff: noqa: E402 - Direct script entry must prioritize this worktree before importing Ultralytics.

import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch

from tools.experiments import b19_common as common
from ultralytics.data.augment import LetterBox
from ultralytics.data.utils import IMG_FORMATS, check_det_dataset
from ultralytics.nn.modules import SICRSPPF
from ultralytics.nn.tasks import load_checkpoint


def state_digest(model):
    """Fingerprint all learned parameters and buffers before and after observation."""
    digest = hashlib.sha256()
    for key, tensor in model.state_dict().items():
        digest.update(key.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def stage_statistics(z, increments, refinements, alpha):
    """Return per-sample increment, correction and branch-contribution scalars only."""
    rows = []
    for i, (zi, di, ri) in enumerate(zip(z, increments, refinements)):
        zi, di, ri = [value.detach().float().flatten(1) for value in (zi, di, ri)]
        ci = alpha[i] * ri
        norm = zi.norm(dim=1) + 1e-12
        row = {
            "increment_mean_abs": di.abs().mean(1),
            "increment_rms": di.square().mean(1).sqrt(),
            "increment_over_stage": di.norm(dim=1) / norm,
            "correction_mean_abs": ci.abs().mean(1),
            "correction_rms": ci.square().mean(1).sqrt(),
            "correction_over_stage": ci.norm(dim=1) / norm,
            "refine_over_increment": ri.norm(dim=1) / (di.norm(dim=1) + 1e-12),
        }
        if not all(torch.isfinite(v).all() for v in row.values()):
            raise ValueError(f"Nonfinite SICR diagnostics at stage {i + 1}")
        rows.append({k: v.cpu().tolist() for k, v in row.items()})
    return rows


def diagnose(weights, data, output, device="cuda:0"):
    """Measure the first 16 lexicographically sorted val images at 640 in FP32 eval mode."""
    dataset = check_det_dataset(str(data), autodownload=False)
    images = sorted(p for p in Path(dataset["val"]).rglob("*") if p.suffix[1:].lower() in IMG_FORMATS)[:16]
    if len(images) != 16:
        raise ValueError("The fixed diagnostic subset requires 16 validation images")
    output.mkdir(parents=True, exist_ok=False)
    model, _ = load_checkpoint(weights, device=device)
    model.float().eval()
    module = model.model[9]
    assert type(module) is SICRSPPF
    initial_digest = state_digest(model)
    alpha = (module.alpha_max * module.theta.tanh()).detach()
    batches = []

    def observe(block, inputs):
        with torch.no_grad():
            z = [block.cv1(inputs[0])]
            z.extend(block.m(z[-1]) for _ in range(3))
            increments = [z[i + 1] - z[i] for i in range(3)]
            refined = [branch(di) for branch, di in zip(block.refine, increments)]
            batches.append(stage_statistics(z[1:], increments, refined, alpha))

    hook = module.register_forward_pre_hook(observe)
    letterbox = LetterBox(new_shape=(640, 640), auto=False, stride=32)
    try:
        with torch.inference_mode():
            for path in images:
                image = cv2.imread(str(path))
                if image is None:
                    raise ValueError(f"Cannot decode {path}")
                image = letterbox(image=image)
                tensor = torch.from_numpy(np.ascontiguousarray(image[:, :, ::-1].transpose(2, 0, 1)))
                model(tensor[None].to(device=device, dtype=torch.float32) / 255)
    finally:
        hook.remove()
    assert state_digest(model) == initial_digest, "Diagnostics changed checkpoint parameters/buffers"
    stages = []
    for i in range(3):
        stage = {}
        for key in batches[0][i]:
            values = np.array([v for batch in batches for v in batch[i][key]], dtype=np.float64)
            stage[key] = {
                "mean": float(values.mean()),
                "median": float(np.median(values)),
                "p95": float(np.quantile(values, 0.95)),
                "max": float(values.max()),
            }
        stages.append(stage)
    report = {
        "weights": str(weights),
        "weight_sha256": common.sha256(weights),
        "split": "val",
        "subset_rule": "first 16 lexicographic image paths",
        "imgsz": 640,
        "dtype": "float32",
        "samples": [{"path": p.relative_to(data.parent).as_posix(), "sha256": common.sha256(p)} for p in images],
        "theta": module.theta.detach().cpu().tolist(),
        "alpha": alpha.cpu().tolist(),
        "alpha_max": 0.10,
        "state_unchanged": True,
        "stages": stages,
        "commit": common.git("rev-parse", "HEAD"),
    }
    common.write_json(output / "sicr_sppf_diagnostics.json", report)
    lines = [
        "# SICR-SPPF v1 diagnostics",
        "",
        "Fixed 16-image val observation; no training or model selection.",
        "",
        f"Theta: {report['theta']}",
        f"Alpha: {report['alpha']}",
        "",
        "| Stage | Measurement | Mean | Median | p95 | Max |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for i, stage in enumerate(stages, 1):
        for key, values in stage.items():
            lines.append(f"| {i} | {key} | " + " | ".join(f"{v:.8g}" for v in values.values()) + " |")
    lines += [
        "",
        "Small alpha alone does not establish a weak branch: inspect correction/stage and refine/increment ratios.",
        "Coefficients are bounded; correction feature magnitudes are not mathematically bounded by 10% of Zi.",
        "Saturation with lower AP75 may suggest excessive correction; only full test comparisons establish performance.",
    ]
    (output / "sicr_sppf_diagnostics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main():
    """Expose the same provenance-checked diagnose stage as the finish entry."""
    from tools.experiments.finish_b19_sicr_sppf import main as finish_main

    finish_main(["--stage", "diagnose", *sys.argv[1:]])


if __name__ == "__main__":
    main()
