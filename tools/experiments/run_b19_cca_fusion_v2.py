"""Fixed b19 CCA v2 identity and diagnostics, reusing the audited v1 training lifecycle."""

# ruff: noqa: E402 -- Resolve the worktree before package imports, as in v1.

import math
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.experiments import run_b19_cca_fusion as shared
from ultralytics.nn.modules import Concat_CCA_Fusion_V2
from ultralytics.nn.modules.cca_fusion_v2 import CENTER_PRIOR, RELIABILITY_FLOOR, RESIDUAL_SCALE

import torch
import torch.nn.functional as F

NAME = "yolo26n_b19_cca_fusion_v2"
MODEL = ROOT / "ultralytics/cfg/models/26/yolo26n-cca-fusion-v2.yaml"
MODULE_CONFIG = dict(
    shared.MODULE_CONFIG,
    center_prior=CENTER_PRIOR,
    reliability_floor=RELIABILITY_FLOOR,
    residual_scale=RESIDUAL_SCALE,
    similarity_scale=4.0,
    confidence_detached=True,
    entropy_eps=1e-6,
    formula="a=softmax(mask(4*cos+1[center])); C=1-clamp(H/log(K_valid),0,1); "
    "G=0.25+0.75*stopgrad(C); Y=cat(U+0.50*G*Wo(sum(a*(Vj-Vparent))),L)",
)
SOURCE_FILES = (
    "tools/experiments/run_b19_cca_fusion_v2.py",
    "tools/experiments/finish_b19_cca_fusion_v2.py",
    "tools/experiments/verify_b19_cca_fusion_v2.py",
    "tools/experiments/server_b19_cca_fusion_v2.sh",
    "tools/experiments/deploy_b19_cca_fusion_v2.sh",
    "ultralytics/nn/modules/cca_fusion_v2.py",
    "ultralytics/cfg/models/26/yolo26n-cca-fusion-v2.yaml",
    "docs/experiments/b19_cca_fusion_v2.md",
    "docs/experiments/b19_cca_fusion_v2_server.md",
    "docs/experiments/evidence/cca_v2_source_audit.json",
    "docs/experiments/evidence/cca_v2_reference_sources.json",
    "tests/test_cca_fusion_v2.py",
    "tools/experiments/verify_b19_cca_fusion.py",
)


def distribution(value):
    """Summarize finite values, retaining quantiles and population standard deviation."""
    value = value.detach().float().flatten()
    assert value.numel() and torch.isfinite(value).all()
    return dict(
        count=value.numel(),
        mean=value.mean().item(),
        std=value.std(unbiased=False).item(),
        min=value.min().item(),
        p50=value.quantile(0.5).item(),
        p90=value.quantile(0.9).item(),
        max=value.max().item(),
    )


@torch.no_grad()
def mechanism_diagnostics(module, up, low, high):
    """Measure the actual fixed gate and compare to ungated residuals without changing any parameter."""
    delta, weights, _, _ = module.correspondence(low, high)
    entropy, normalized, confidence, gate, valid = module.reliability(weights)
    raw = module.Wo(delta).float()
    effective = RESIDUAL_SCALE * gate
    residual = effective * raw
    denominator = up.float().norm(dim=1, keepdim=True) + 1e-6
    raw_norm = raw.norm(dim=1, keepdim=True)
    residual_norm = residual.norm(dim=1, keepdim=True)
    offsets = weights.new_tensor([(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)])
    displacement = torch.einsum("bnhw,nd->bdhw", weights, offsets).norm(dim=1, keepdim=True)
    values = dict(
        center_weight=weights[:, 4:5],
        noncenter_weight=1 - weights[:, 4:5],
        entropy=entropy,
        normalized_entropy=normalized,
        raw_confidence=confidence,
        reliability=gate,
        K_valid=valid.expand_as(gate),
        expected_correspondence_displacement=displacement,
        raw_residual_over_U=raw_norm / denominator,
        residual_over_U=residual_norm / denominator,
        effective_gate=effective,
    )
    support = {}
    for count in valid.unique().tolist():
        mask = (valid == count).expand_as(gate)
        label = {4: "corner", 6: "edge", 9: "interior"}.get(int(count), f"degenerate_{int(count)}")
        support[label] = {name: distribution(v[mask]) for name, v in values.items()}
    # Recover the v1 softmax (no center prior) with the SAME projections, not a trained-v1 checkpoint.
    undo = weights.new_tensor([1, 1, 1, 1, math.exp(-CENTER_PRIOR), 1, 1, 1, 1])
    original_weights = weights * undo.view(1, 9, 1, 1)
    original_weights = original_weights / original_weights.sum(1, keepdim=True)
    v = module.Wv(high)
    b, r, h, w = v.shape
    differences = F.unfold(v.float(), 3, padding=1).reshape(b, r, 9, h, w) - v.float().unsqueeze(2)
    original_delta = (
        (original_weights.reshape(b, 1, 9, h, 2, w, 2) * differences[:, :, :, :, None, :, None])
        .sum(2)
        .reshape(b, r, 2 * h, 2 * w)
    )
    original_raw = module.Wo(original_delta.to(v.dtype)).float().norm(dim=1, keepdim=True)
    nonzero = raw_norm > 0
    gate_ratio = distribution((residual_norm[nonzero] / raw_norm[nonzero])) if nonzero.any() else None
    original_nonzero = original_raw > 0
    v1_ratio = (
        distribution(residual_norm[original_nonzero] / original_raw[original_nonzero])
        if original_nonzero.any()
        else None
    )
    return dict(
        reliability_v2=dict(
            all_positions={name: distribution(value) for name, value in values.items()},
            support=support,
            actual_over_ungated_same_correspondence=gate_ratio,
            actual_over_v1_no_prior_same_projections=v1_ratio,
            zero_raw_positions=int((~nonzero).sum()),
            interpretation="Displacement is the norm of the expected coarse-neighbor offset, not GT alignment error. "
            "Gate-only ratio uses v2 correspondence and equals 0.50*G where raw is nonzero. "
            "The v1-mechanism ratio additionally removes the prior; it need not be bounded by 0.5. "
            "Null ratios mean no nonzero denominator. No v1 trained weights are used.",
        )
    )


class AuditedTrainer(shared.AuditedTrainer):
    """Retain every b19 and native lifecycle audit while selecting layer-12 v2."""

    block_type = Concat_CCA_Fusion_V2
    mechanism_diagnostics = staticmethod(mechanism_diagnostics)


EXPERIMENT = SimpleNamespace(
    name=NAME,
    model=MODEL,
    block_type=Concat_CCA_Fusion_V2,
    version=2,
    trainer_type=AuditedTrainer,
    finish_entry=ROOT / "tools/experiments/finish_b19_cca_fusion_v2.py",
    source_files=SOURCE_FILES,
)


def main(argv=None):
    """Run only the fixed v2 experiment through the unchanged full b19 preflight and train gates."""
    return shared.main(
        argv,
        model=MODEL,
        trainer_type=AuditedTrainer,
        entrypoint=Path(__file__).resolve(),
        name=NAME,
        source_files=tuple(ROOT / p for p in SOURCE_FILES),
        module_config=MODULE_CONFIG,
        fixed_name=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
