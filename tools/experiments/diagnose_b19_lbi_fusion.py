"""Observe fixed validation image IDs and actual LBI projections/residuals without updating weights."""

# ruff: noqa: E402 -- Direct entry must resolve this worktree.
import csv
import hashlib
import json
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
from ultralytics.nn.modules import Concat_LBI_Fusion
from ultralytics.nn.tasks import load_checkpoint

NEAR_ZERO = 1e-4


def distribution(value):
    """Describe each finite tensor using fixed population moments and RMS percentiles."""
    v = value.detach().float().flatten()
    assert v.numel() and torch.isfinite(v).all()
    return dict(
        mean=v.mean().item(),
        std=v.std(unbiased=False).item(),
        min=v.min().item(),
        max=v.max().item(),
        rms=v.square().mean().sqrt().item(),
        l2=v.norm().item(),
        percentiles={str(p): v.quantile(p / 100).item() for p in (1, 5, 25, 50, 75, 95, 99)},
    )


def state_digest(model):
    """Fingerprint all parameters and buffers before and after fixed validation observation."""
    digest = hashlib.sha256()
    for key, tensor in model.state_dict().items():
        digest.update(key.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def diagnose(weights, data, output, device="cuda:0"):
    """Measure first 16 sorted validation images in unfused FP32 with image IDs fixed before inference."""
    weights, data, output = Path(weights), Path(data), Path(output)
    dataset = check_det_dataset(str(data), autodownload=False)
    images = sorted(p for p in Path(dataset["val"]).rglob("*") if p.suffix[1:].lower() in IMG_FORMATS)[:16]
    if len(images) != 16:
        raise ValueError("MISSING: the fixed 16-image validation subset is unavailable")
    selection = [dict(id=p.relative_to(data.parent).as_posix(), sha256=common.sha256(p)) for p in images]
    binding = dict(
        weight_sha256=common.sha256(weights),
        commit=common.git("rev-parse", "HEAD"),
        data_sha256=common.sha256(data),
        selection=selection,
        split="val",
    )
    receipt = output / "lbi_fusion_diagnostics.json"
    if receipt.exists():
        report = json.loads(receipt.read_text(encoding="utf-8"))
        assert report["binding"] == binding
        assert common.sha256(output / "lbi_fusion_diagnostics.csv") == report["csv_sha256"]
        return report
    output.mkdir(parents=True, exist_ok=False)
    common.write_json(output / "validation_samples.json", selection)
    model, _ = load_checkpoint(weights, device=device)
    model.float().eval()
    block = model.model[15]
    assert type(block) is Concat_LBI_Fusion
    before = state_digest(model)
    captured, handles, rows = {}, [], []

    def capture(name):
        def hook(module, inputs, value):
            captured[name] = value.detach()

        return hook

    def inputs_hook(module, inputs):
        captured["S"], captured["L"] = (x.detach() for x in inputs[0])

    def dw_hook(module, inputs):
        captured["interaction_actual"] = inputs[0].detach()

    def out_hook(module, inputs):
        captured["t"] = inputs[0].detach()

    handles.append(block.register_forward_pre_hook(inputs_hook))
    handles.append(block.proj_l.register_forward_hook(capture("u")))
    handles.append(block.proj_s.register_forward_hook(capture("v")))
    handles.append(block.dw.register_forward_pre_hook(dw_hook))
    handles.append(block.out.register_forward_pre_hook(out_hook))
    handles.append(block.out.register_forward_hook(capture("r")))
    try:
        with torch.inference_mode():
            for i, path in enumerate(images):
                captured.clear()
                image = cv2.imread(str(path))
                if image is None:
                    raise ValueError(f"Cannot decode {path}")
                image = LetterBox(new_shape=(640, 640), auto=False, stride=32)(image=image)
                x = torch.from_numpy(np.ascontiguousarray(image[:, :, ::-1].transpose(2, 0, 1)))
                model(x[None].to(device=device, dtype=torch.float32) / 255)
                u, v, detail, r = (captured[k].float() for k in ("u", "v", "L", "r"))
                u_rms, v_rms = (x.square().mean(1, keepdim=True).sqrt() for x in (u, v))
                un = u * torch.rsqrt(u.square().mean(1, keepdim=True) + block.eps)
                vn = v * torch.rsqrt(v.square().mean(1, keepdim=True) + block.eps)
                interaction = un * vn
                common.assert_close_tree(
                    interaction, captured["interaction_actual"], 0, 0, "diagnostic.live_interaction"
                )
                native = torch.cat([captured["S"], detail], 1)
                rows.append(
                    dict(
                        **selection[i],
                        u_rms=distribution(u_rms),
                        v_rms=distribution(v_rms),
                        u_near_zero=(u_rms < NEAR_ZERO).float().mean().item(),
                        v_near_zero=(v_rms < NEAR_ZERO).float().mean().item(),
                        un=distribution(un),
                        vn=distribution(vn),
                        interaction=distribution(interaction),
                        t=distribution(captured["t"]),
                        r=distribution(r),
                        L=distribution(detail),
                        r_over_L=r.norm().item() / (detail.norm().item() + 1e-6),
                        r_over_cat=r.norm().item() / (native.norm().item() + 1e-6),
                    )
                )
                print(f"LBI validation diagnosis {i + 1}/16: {selection[i]['id']}", flush=True)
    finally:
        for handle in handles:
            handle.remove()
    assert state_digest(model) == before
    csv_path = output / "lbi_fusion_diagnostics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["id", "u_near_zero", "v_near_zero", "r_over_L", "r_over_cat"])
        writer.writeheader()
        writer.writerows({k: row[k] for k in writer.fieldnames} for row in rows)
    gradient_path = weights.parent.parent / "provenance/preflight/native_gradient_summary.json"
    report = dict(
        binding=binding,
        weight_sha256=binding["weight_sha256"],
        commit=binding["commit"],
        weight_norms={name: distribution(p) for name, p in block.named_parameters()},
        near_zero_threshold=NEAR_ZERO,
        state_unchanged=True,
        precision="unfused FP32",
        samples=rows,
        csv_sha256=common.sha256(csv_path),
        preflight_gradient_summary=json.loads(gradient_path.read_text()) if gradient_path.exists() else "MISSING",
        long_term_gradient_status="NOT MEASURED: one checkpoint and preflight cannot establish long-term behavior",
        projected_output_summary={k: sum(row[k] for row in rows) / len(rows) for k in ("u_near_zero", "v_near_zero")},
        interpretation="Interaction is not probability, correlation coefficient or a reliability gate",
    )
    common.write_json(receipt, report)
    return report
