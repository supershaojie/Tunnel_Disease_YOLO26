"""Content-preserving directional contrast residuals for the single b19 P3 candidate."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.torch_utils import autocast

from .block import C3k2
from .conv import Conv
from .dcr_strip import DCRStrip, DiagonalStrip

__all__ = ("C3k2_DCRStripV2", "DCRStripV2")


class DCRStripV2(nn.Module):
    """Retain ungated strip content and add normalized, signed directional compensation."""

    normals = DCRStrip.normals
    contrast = staticmethod(DCRStrip.contrast)
    directional_responses = DCRStrip.directional_responses

    def __init__(self, channels, enabled=True):
        """Initialize the only v2 candidate with nonzero kernels, alpha=0.05 and beta=0.25."""
        super().__init__()
        self.enabled = enabled
        self.d = max(8, math.ceil(channels / 32) * 8)
        self.reduce = Conv(channels, self.d, 1)
        self.horizontal = nn.Conv2d(self.d, self.d, (1, 7), groups=self.d, bias=False)
        self.vertical = nn.Conv2d(self.d, self.d, (7, 1), groups=self.d, bias=False)
        self.diagonal = DiagonalStrip(self.d)
        self.antidiagonal = DiagonalStrip(self.d, anti=True)
        self.expand = nn.Conv2d(self.d, channels, 1, bias=False)
        self.alpha = nn.Parameter(torch.tensor(0.05))
        self.beta_raw = nn.Parameter(torch.tensor(math.log(0.25 / 0.75)))

    @property
    def beta(self):
        """Return the differentiable contrast fraction inside the new branch."""
        return self.beta_raw.sigmoid()

    @staticmethod
    def direction_weights(contrasts, asymmetries):
        """Compute differentiable FP32 relative contrast scores and temperature-one weights."""
        with autocast(enabled=False, device=contrasts[0].device.type):
            u = torch.cat([c.float().abs().mean(1, keepdim=True) for c in contrasts], 1)
            v = torch.cat([a.float().mean(1, keepdim=True) for a in asymmetries], 1)
            u = F.avg_pool2d(F.pad(u, (1, 1, 1, 1), mode="replicate"), 3, 1)
            v = F.avg_pool2d(F.pad(v, (1, 1, 1, 1), mode="replicate"), 3, 1)
            q = u / (u + v + 1e-6)
            return q.softmax(1), q

    def components(self, x):
        """Return U, beta*V, FP32 weights and q for forward and explicit no-grad diagnostics."""
        responses = self.directional_responses(x)
        contrasts, asymmetries = zip(*(self.contrast(t, n) for t, n in zip(responses, self.normals)))
        weights, q = self.direction_weights(contrasts, asymmetries)
        gates = weights.to(contrasts[0].dtype)
        content = sum(responses) * 0.25
        compensation = self.beta * sum(gates[:, i : i + 1] * c for i, c in enumerate(contrasts))
        return content, compensation, weights, q

    def residual(self, x):
        """Expand the content and bounded contrast mixture without extra BN or activation."""
        content, compensation, weights, _ = self.components(x)
        return self.expand(content + compensation), weights

    def forward(self, x):
        """Apply one residual, or the explicit debug bypass used for baseline equality checks."""
        if not self.enabled:
            return x
        delta, _ = self.residual(x)
        return x + self.alpha * delta


class C3k2_DCRStripV2(C3k2):
    """Preserve native cv1/cv2/m keys and apply v2 only after the original C3k2 output."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, attn=False, g=1, shortcut=True, enabled=True):
        """Keep the native constructor order and the subsequent backbone/head CPU RNG stream."""
        super().__init__(c1=c1, c2=c2, n=n, c3k=c3k, e=e, attn=attn, g=g, shortcut=shortcut)
        with torch.random.fork_rng(devices=[]):
            self.dcr = DCRStripV2(c2, enabled)

    def forward(self, x):
        """Apply v2 after the native chunk path."""
        return self.dcr(super().forward(x))

    def forward_split(self, x):
        """Apply v2 after the native split path."""
        return self.dcr(super().forward_split(x))
