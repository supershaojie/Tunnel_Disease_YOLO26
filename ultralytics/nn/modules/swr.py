"""Semantic-guided Haar reconstruction and gated P3 fusion for the fixed b19 experiment."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv


def haar_split(x: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Decompose even BCHW features in the fixed [L0, H1, H2, H3] order using FP32 arithmetic."""
    if x.ndim != 4 or x.shape[-2] % 2 or x.shape[-1] % 2:
        raise ValueError(f"Haar requires BCHW with even H/W, received {tuple(x.shape)}")
    a, b = x[..., 0::2, 0::2].float(), x[..., 0::2, 1::2].float()
    c, d = x[..., 1::2, 0::2].float(), x[..., 1::2, 1::2].float()
    return tuple(
        y.to(x.dtype) for y in ((a + b + c + d) / 2, (-a - b + c + d) / 2, (-a + b - c + d) / 2, (a - b - c + d) / 2)
    )


def haar_inverse(bands: tuple[torch.Tensor, ...]) -> torch.Tensor:
    """Reconstruct each channel's four spatial phases, with FP32 sums before dtype restoration."""
    low, h1, h2, h3 = (x.float() for x in bands)
    phases = torch.stack(
        ((low - h1 - h2 + h3) / 2, (low - h1 + h2 - h3) / 2, (low + h1 - h2 - h3) / 2, (low + h1 + h2 + h3) / 2), dim=2
    )
    # B,C,4,h,w -> B,C*4,h,w: phases are contiguous WITHIN each channel.
    return F.pixel_shuffle(phases.flatten(1, 2), 2).to(bands[0].dtype)


class SWRGatedBlock(nn.Module):
    """Apply independent 3/5/7 depthwise fields and multiplicative gating at fixed width 64."""

    def __init__(self):
        """Build one complete residual extraction block with native Conv/BN behavior."""
        super().__init__()
        self.value = Conv(64, 64, 1)
        self.fields = nn.ModuleList(Conv(c, c, k, g=c, act=False) for c, k in ((16, 3), (16, 5), (32, 7)))
        self.gate = Conv(64, 64, 1, act=False)
        self.mix = Conv(64, 64, 1, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the residual sum after gated multi-field extraction."""
        parts = self.value(x).split((16, 16, 32), dim=1)
        u = torch.cat([layer(part) for layer, part in zip(self.fields, parts)], dim=1)
        return x + self.mix(F.silu(self.gate(x)) * F.silu(u))


class SWRFusion(nn.Module):
    """Replace the b19 P3 upsample/concat/C3k2 stage with semantic-conditioned Haar fusion."""

    def __init__(self, c_l: int, c_s: int, c_out: int, hidden: int = 64, blocks: int = 2):
        """Build the fixed first-version module; only parser-scaled c_out=64 is supported."""
        super().__init__()
        if (c_out, hidden, blocks) != (64, 64, 2):
            raise ValueError("SWR-Fusion v1 requires c_out=64, hidden=64 and blocks=2 (scale=n)")
        self.p_l = Conv(c_l, hidden, 1)
        self.p_s = Conv(c_s, hidden, 1)
        self.context_mix = Conv(hidden * 2, hidden, 1)
        self.context_dw = Conv(hidden, hidden, 5, g=hidden)
        self.low_delta = nn.Conv2d(hidden, hidden, 1, bias=False)
        self.band_gates = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(128, 16, 1, bias=False),
                nn.SiLU(),
                nn.Conv2d(16, 16, 3, padding=1, groups=16, bias=False),
                nn.SiLU(),
                nn.Conv2d(16, 8, 1, bias=True),
            )
            for _ in range(3)
        )
        self.fuse = Conv(192, c_out, 1)
        self.blocks = nn.Sequential(*(SWRGatedBlock() for _ in range(blocks)))
        nn.init.zeros_(self.low_delta.weight)
        for gate in self.band_gates:
            nn.init.zeros_(gate[-1].weight)
            nn.init.zeros_(gate[-1].bias)

    def _paths(self, inputs: list[torch.Tensor]) -> tuple:
        """Compute all three fusion paths without retaining activations on the module."""
        low, semantic = inputs
        if (
            low.ndim != 4
            or semantic.ndim != 4
            or low.shape[0] != semantic.shape[0]
            or low.shape[-2] != 2 * semantic.shape[-2]
            or low.shape[-1] != 2 * semantic.shape[-1]
        ):
            raise ValueError(
                f"SWRFusion requires L(B,C,2h,2w), S(B,C,h,w); got {tuple(low.shape)}, {tuple(semantic.shape)}"
            )
        lp, sp = self.p_l(low), self.p_s(semantic)
        l0, *high = haar_split(lp)
        context = self.context_dw(self.context_mix(torch.cat((l0, sp), dim=1)))
        delta = self.low_delta(context)
        gains = [
            ((2 * gate(torch.cat((context, h), dim=1)).float().sigmoid()).to(h.dtype))
            for gate, h in zip(self.band_gates, high)
        ]
        updated = [g.repeat_interleave(8, dim=1) * h for g, h in zip(gains, high)]
        reconstructed = haar_inverse((l0 + delta, *updated))
        sup = F.interpolate(sp, size=lp.shape[-2:], mode="nearest")
        return lp, sup, reconstructed, gains, delta

    def forward(self, inputs: list[torch.Tensor]) -> torch.Tensor:
        """Execute reconstruction, direct paths, and both gated blocks in every model mode."""
        lp, sup, reconstructed, _, _ = self._paths(inputs)
        return self.blocks(self.fuse(torch.cat((lp, sup, reconstructed), dim=1)))

    @torch.no_grad()
    def diagnostics(self, inputs: list[torch.Tensor]) -> dict:
        """Return aggregate statistics for explicitly selected eval samples, without storing activations."""
        if self.training:
            raise ValueError("Call diagnostics on an eval model to avoid updating BatchNorm buffers")
        lp, sup, reconstructed, gains, delta = self._paths(inputs)
        norms = {
            name: float(value.float().norm())
            for name, value in (("lp", lp), ("sup", sup), ("reconstruction", reconstructed))
        }
        low_norm = haar_split(lp)[0].float().norm().clamp_min(1e-12)
        return {
            "gates": [
                {
                    "mean": float(g.float().mean()),
                    "quantiles_0_25_50_75_100": g.float()
                    .quantile(torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], device=g.device))
                    .tolist(),
                }
                for g in gains
            ],
            "low_delta_norm_ratio": float(delta.float().norm() / low_norm),
            "norms": norms,
            "finite": all(bool(torch.isfinite(t).all()) for t in (lp, sup, reconstructed, delta, *gains)),
        }
