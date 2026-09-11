"""Observe fixed validation samples and actual FDV value-path tensors without mutating the model."""

# Direct entry must resolve this worktree before package imports.
# ruff: noqa: E402

import csv
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
from ultralytics.nn.modules import FDV_C2PSA
from ultralytics.nn.tasks import load_checkpoint


def distribution(value):
    """Summarize finite tensors with population std, absolute magnitude and fixed percentiles."""
    v = value.detach().float().flatten()
    assert v.numel() and torch.isfinite(v).all()
    return dict(
        mean=v.mean().item(),
        std=v.std(unbiased=False).item(),
        min=v.min().item(),
        max=v.max().item(),
        abs_mean=v.abs().mean().item(),
        max_abs=v.abs().max().item(),
        percentiles={str(p): v.quantile(p / 100).item() for p in (1, 5, 25, 50, 75, 95, 99)},
    )


def state_digest(model):
    """Reuse b19 diagnostics' complete parameter/buffer fingerprint before and after observation."""
    digest = hashlib.sha256()
    for key, tensor in model.state_dict().items():
        digest.update(key.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def diagnose(weights, data, output, device="cuda:0"):
    """Measure the first 16 sorted validation images at 640, FP32, using forward hooks on native tensors."""
    data, output = Path(data), Path(output)
    dataset = check_det_dataset(str(data), autodownload=False)
    images = sorted(p for p in Path(dataset["val"]).rglob("*") if p.suffix[1:].lower() in IMG_FORMATS)[:16]
    assert len(images) == 16
    output.mkdir(parents=True, exist_ok=False)
    model, _ = load_checkpoint(weights, device=device)
    model.float().eval()
    assert type(model.model[10]) is FDV_C2PSA and len(model.model[10].m) == 1
    attn = model.model[10].m[0].attn
    before = state_digest(model)
    gamma = attn.gamma_c.detach()
    state, rows = {}, []

    def qkv_hook(module, inputs, value):
        state["qkv"] = value.detach()

    def pe_hook(module, inputs, value):
        state["pe"] = value.detach()

    def pool_hook(module, inputs, value):
        state["low"], state["high"] = value.detach(), inputs[0].detach() - value.detach()

    def proj_hook(module, inputs):
        b, c, h, w = inputs[0].shape
        q, k, v = (
            state["qkv"]
            .view(b, attn.num_heads, 2 * attn.key_dim + attn.head_dim, h * w)
            .split([attn.key_dim, attn.key_dim, attn.head_dim], dim=2)
        )
        a = ((q * attn.scale).transpose(-2, -1) @ k).softmax(dim=-1)
        o_att = (v @ a.transpose(-2, -1)).view(b, c, h, w)
        o_high = gamma.view(1, c, 1, 1) * state["high"]
        native = o_att + state["pe"]
        common.assert_close_tree(native + o_high, inputs[0], 0, 0, path="diagnostic.actual_proj_input")
        denominator = native.norm().item()
        rows.append(
            dict(
                V_low=distribution(state["low"]),
                V_high=distribution(state["high"]),
                O_att_norm=o_att.norm().item(),
                O_pe_norm=state["pe"].norm().item(),
                O_high_norm=o_high.norm().item(),
                native_output_norm=denominator,
                high_frequency_ratio=o_high.norm().item() / denominator if denominator else None,
            )
        )

    handles = [
        attn.qkv.register_forward_hook(qkv_hook),
        attn.pe.register_forward_hook(pe_hook),
        attn.lowpass.register_forward_hook(pool_hook),
        attn.proj.register_forward_pre_hook(proj_hook),
    ]
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
        for handle in handles:
            handle.remove()
    assert state_digest(model) == before and len(rows) == len(images)
    positive = [int(i) for i in gamma.argsort(descending=True) if gamma[i] > 0][:10]
    negative = [int(i) for i in gamma.argsort() if gamma[i] < 0][:10]
    report = dict(
        weight_sha256=common.sha256(weights),
        commit=common.git("rev-parse", "HEAD"),
        split="val",
        subset_rule="first 16 lexicographic paths",
        imgsz=640,
        dtype="float32",
        theta_c=distribution(attn.theta_c),
        gamma_c=distribution(gamma),
        theta_values=attn.theta_c.detach().cpu().tolist(),
        gamma_values=gamma.cpu().tolist(),
        top_positive_gamma_channels=[dict(channel=i, gamma=gamma[i].item()) for i in positive],
        top_negative_gamma_channels=[dict(channel=i, gamma=gamma[i].item()) for i in negative],
        near_zero_threshold=1e-4,
        near_zero_channel_proportion=(gamma.abs() < 1e-4).float().mean().item(),
        state_unchanged=True,
        samples=[
            dict(path=p.relative_to(data.parent).as_posix(), sha256=common.sha256(p), **row)
            for p, row in zip(images, rows)
        ],
    )
    csv_path = output / "fdv_c2psa_diagnostics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "path",
                "V_low_mean",
                "V_low_std",
                "V_high_mean",
                "V_high_std",
                "V_high_max_abs",
                "O_att_norm",
                "O_high_norm",
                "high_frequency_ratio",
            ],
        )
        writer.writeheader()
        for sample in report["samples"]:
            writer.writerow(
                dict(
                    path=sample["path"],
                    V_low_mean=sample["V_low"]["mean"],
                    V_low_std=sample["V_low"]["std"],
                    V_high_mean=sample["V_high"]["mean"],
                    V_high_std=sample["V_high"]["std"],
                    V_high_max_abs=sample["V_high"]["max_abs"],
                    **{k: sample[k] for k in ("O_att_norm", "O_high_norm", "high_frequency_ratio")},
                )
            )
    report["csv_sha256"] = common.sha256(csv_path)
    common.write_json(output / "fdv_c2psa_diagnostics.json", report)
    return report
