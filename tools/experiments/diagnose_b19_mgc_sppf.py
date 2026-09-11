"""Observe MGC best.pt on a fixed validation subset without modifying its state or selecting a design."""

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
from ultralytics.nn.modules import MGC_SPPF
from ultralytics.nn.tasks import load_checkpoint


def state_digest(model):
    """Fingerprint all learned parameters and buffers before and after observation."""
    digest = hashlib.sha256()
    for key, tensor in model.state_dict().items():
        digest.update(key.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def statistics(value):
    """Summarize finite feature values without interpreting them as pixel labels."""
    value = value.detach().double()
    assert value.numel() and torch.isfinite(value).all()
    return dict(
        mean=value.mean().item(),
        std=value.std(unbiased=False).item(),
        min=value.min().item(),
        max=value.max().item(),
        zero_fraction=(value == 0).double().mean().item(),
        rms=value.square().mean().sqrt().item(),
        l2=value.norm().item(),
    )


def diagnose(weights, data, output, device="cuda:0"):
    """Observe a fixed manifest of 16 val images using temporary eval-only hooks and emit JSON/CSV."""
    import csv

    dataset = check_det_dataset(str(data), autodownload=False)
    images = sorted(p for p in Path(dataset["val"]).rglob("*") if p.suffix[1:].lower() in IMG_FORMATS)[:16]
    if len(images) != 16:
        raise ValueError("MISSING: the fixed diagnostics require 16 validation images")
    output.mkdir(parents=True, exist_ok=False)
    samples = [
        dict(index=i, image_id=p.relative_to(data.parent).as_posix(), sha256=common.sha256(p))
        for i, p in enumerate(images)
    ]
    common.write_json(output / "sample_manifest.json", dict(split="val", samples=samples))
    model, _ = load_checkpoint(weights, device=device)
    model.float().eval()
    block = model.model[9]
    assert type(block) is MGC_SPPF
    initial_digest = state_digest(model)
    rows, captured = [], {}

    def observe(module, inputs, out):
        native = captured["native"] + inputs[0] if block.add else captured["native"]
        g3, g5 = captured["gaps"].chunk(2, 1)
        values = dict(u=captured["u"], g3=g3, g5=g5, r=captured["r"], y_native=native)
        boundary = torch.ones(native.shape[-2:], dtype=torch.bool, device=native.device)
        boundary[4:-4, 4:-4] = False
        row = {key: statistics(v) for key, v in values.items()}
        row["residual_over_native_l2"] = values["r"].double().norm().item() / (native.double().norm().item() + 1e-6)
        e3, e5 = g3.double().square().sum().item(), g5.double().square().sum().item()
        row["g3_energy_fraction"] = e3 / (e3 + e5 + 1e-12)
        row["g5_energy_fraction"] = e5 / (e3 + e5 + 1e-12)
        row["boundary_width"] = 4
        row["regions"] = {
            name: {key: statistics(values[key][..., mask]) for key in ("g3", "g5", "r")}
            for name, mask in (("boundary", boundary), ("interior", ~boundary))
        }
        rows.append(row)

    handles = [
        block.gap_in.register_forward_hook(lambda m, ins, out: captured.update(u=out)),
        block.gap_out.register_forward_hook(lambda m, ins, out: captured.update(gaps=ins[0], r=out)),
        block.cv2.register_forward_hook(lambda m, ins, out: captured.update(native=out)),
        block.register_forward_hook(observe),
    ]
    letterbox = LetterBox(new_shape=(640, 640), auto=False, stride=32)
    try:
        with torch.inference_mode():
            for i, path in enumerate(images):
                image = cv2.imread(str(path))
                if image is None:
                    raise ValueError(f"Cannot decode {path}")
                image = letterbox(image=image)
                tensor = torch.from_numpy(np.ascontiguousarray(image[:, :, ::-1].transpose(2, 0, 1)))
                model(tensor[None].to(device=device, dtype=torch.float32) / 255)
                captured.clear()
                print(f"MGC diagnostics {i + 1}/{len(images)}", flush=True)
    finally:
        for handle in handles:
            handle.remove()
        captured.clear()
    assert state_digest(model) == initial_digest, "Diagnostics changed parameters/buffers"
    report = dict(
        weight_sha256=common.sha256(weights),
        split="val",
        subset_rule="first 16 lexicographic val paths",
        imgsz=640,
        dtype="float32",
        samples=samples,
        rows=rows,
        state_unchanged=True,
        weight_norms={name: statistics(p) for name, p in block.named_parameters() if name.startswith("gap_")},
        commit=common.git("rev-parse", "HEAD"),
        interpretation="Feature closing residuals; no pixel labels, probability or recovery guarantee",
    )
    common.write_json(output / "mgc_sppf_diagnostics.json", report)
    with (output / "mgc_sppf_diagnostics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["image_id", "measurement", "value"])

        def emit(prefix, value, image_id):
            if isinstance(value, dict):
                for key, item in value.items():
                    emit(f"{prefix}.{key}" if prefix else key, item, image_id)
            else:
                writer.writerow([image_id, prefix, value])

        for sample, row in zip(samples, rows):
            emit("", row, sample["image_id"])
    return report


def main():
    """Expose the same provenance-checked diagnose stage as the finish entry."""
    from tools.experiments.finish_b19_mgc_sppf import main as finish_main

    finish_main(["--stage", "diagnose", *sys.argv[1:]])


if __name__ == "__main__":
    main()
