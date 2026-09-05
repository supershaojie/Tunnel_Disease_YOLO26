"""Discrete directional contrast residuals for the single P3 C3k2 ablation."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.torch_utils import autocast

from .block import C3k2
from .conv import Conv

__all__ = ("C3k2_DCRStrip", "DCRStrip")


class DiagonalStrip(nn.Module):
    """Depthwise diagonal convolution with seven trainable coefficients per channel."""

    def __init__(self, channels, anti=False):
        """Create a fixed diagonal basis and initialize by effective kernel length."""
        super().__init__()
        self.weight = nn.Parameter(torch.empty(channels, 7))
        mask = torch.zeros(7, 7, 7)
        index = torch.arange(7)
        mask[index, index, 6 - index if anti else index] = 1
        self.register_buffer("mask", mask)
        nn.init.uniform_(self.weight, -1 / math.sqrt(7), 1 / math.sqrt(7))

    def kernel(self):
        """Assemble a dense kernel whose only active entries lie on one diagonal."""
        return torch.einsum("ck,kij->cij", self.weight, self.mask).unsqueeze(1)

    def forward(self, x):
        """Apply replicate padding and the assembled depthwise convolution."""
        return F.conv2d(F.pad(x, (3, 3, 3, 3), mode="replicate"), self.kernel(), groups=x.shape[1])


class DCRStrip(nn.Module):
    """Add a signed, spatially gated four-direction residual; efficacy requires validation."""

    normals = ((1, 0), (0, 1), (1, -1), (1, 1))

    def __init__(self, channels, enabled=True, use_contrast=True, adaptive_fusion=True):
        """Initialize fixed v1 settings (k=7, r=1, alpha=0.05)."""
        super().__init__()
        self.enabled = enabled
        self.use_contrast = use_contrast
        self.adaptive_fusion = adaptive_fusion
        self.d = max(8, math.ceil(channels / 32) * 8)
        self.reduce = Conv(channels, self.d, 1)
        self.horizontal = nn.Conv2d(self.d, self.d, (1, 7), groups=self.d, bias=False)
        self.vertical = nn.Conv2d(self.d, self.d, (7, 1), groups=self.d, bias=False)
        self.diagonal = DiagonalStrip(self.d)
        self.antidiagonal = DiagonalStrip(self.d, anti=True)
        self.gate = nn.Conv2d(2, 1, 1, bias=False)
        nn.init.zeros_(self.gate.weight)
        self.expand = nn.Conv2d(self.d, channels, 1, bias=False)
        self.alpha = nn.Parameter(torch.tensor(0.05))

    @staticmethod
    def contrast(t, normal):
        """Return signed normal contrast and side asymmetry without wraparound."""
        dy, dx = normal
        h, w = t.shape[-2:]
        padded = F.pad(t, (1, 1, 1, 1), mode="replicate")
        plus = padded[..., 1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w]
        minus = padded[..., 1 - dy : 1 - dy + h, 1 - dx : 1 - dx + w]
        return t - 0.5 * (plus + minus), (plus - minus).abs()

    def directional_responses(self, x):
        """Return responses in horizontal, vertical, diagonal, antidiagonal order."""
        z = self.reduce(x)
        return (
            self.horizontal(F.pad(z, (3, 3, 0, 0), mode="replicate")),
            self.vertical(F.pad(z, (0, 0, 3, 3), mode="replicate")),
            self.diagonal(z),
            self.antidiagonal(z),
        )

    def residual(self, x):
        """Return the signed compensation and gates for optional compact diagnostics."""
        responses, logits = [], []
        for t, normal in zip(self.directional_responses(x), self.normals):
            c, asymmetry = self.contrast(t, normal) if self.use_contrast else (t, None)
            responses.append(c)
            if self.adaptive_fusion:
                # Explicit FP32 statistics keep AMP gate logits finite without detaching their gradients.
                with autocast(enabled=False, device=x.device.type):
                    u = c.float().abs().mean(1, keepdim=True)
                    v = asymmetry.float().mean(1, keepdim=True) if asymmetry is not None else torch.zeros_like(u)
                    stats = F.avg_pool2d(F.pad(torch.cat((u, v), 1), (1, 1, 1, 1), mode="replicate"), 3, 1)
                    logits.append(F.conv2d(stats, self.gate.weight.float()))
        if self.adaptive_fusion:
            gates = torch.cat(logits, 1).softmax(1).to(responses[0].dtype)
            fused = sum(gates[:, i : i + 1] * c for i, c in enumerate(responses))
        else:
            gates = x.new_full((x.shape[0], 4, *x.shape[-2:]), 0.25)
            fused = sum(responses) * 0.25
        return self.expand(fused), gates

    def forward(self, x):
        """Return the unmodified baseline feature when the debug bypass is enabled."""
        if not self.enabled:
            return x
        delta, _ = self.residual(x)
        return x + self.alpha * delta


class C3k2_DCRStrip(C3k2):
    """Preserve every original C3k2 key and append one DCR-Strip branch after its output."""

    def __init__(
        self,
        c1,
        c2,
        n=1,
        c3k=False,
        e=0.5,
        attn=False,
        g=1,
        shortcut=True,
        enabled=True,
        use_contrast=True,
        adaptive_fusion=True,
    ):
        """Use the current C3k2 interface and preserve the baseline CPU initialization stream."""
        super().__init__(c1=c1, c2=c2, n=n, c3k=c3k, e=e, attn=attn, g=g, shortcut=shortcut)
        # The native trainer constructs on CPU before moving the model to its device.
        # No CUDA RNG is consumed by this branch construction.
        with torch.random.fork_rng(devices=[]):
            self.dcr = DCRStrip(c2, enabled, use_contrast, adaptive_fusion)

    def forward(self, x):
        """Apply the residual after the original chunk-based C3k2 path."""
        return self.dcr(super().forward(x))

    def forward_split(self, x):
        """Apply the same residual after the original split-based C3k2 path."""
        return self.dcr(super().forward_split(x))
