"""Semantic-guided local kernel: fixed b19 nano P3 experiment (project working name)."""

import torch
from torch import nn
from torch.nn import functional as F

from .block import C3k2
from ultralytics.utils.torch_utils import autocast


class SGK(nn.Module):
    """Predict four position-specific 3x3 kernels; aggregate only the second P3 half."""

    def __init__(self):
        """Build the fixed 4596-parameter branch without changing native initialization RNG."""
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            self.Ps = nn.Conv2d(128, 16, 1, bias=False)
            self.DW5 = nn.Conv2d(16, 16, 5, padding=2, groups=16, bias=False)
            self.Pb = nn.Conv2d(32, 16, 1, bias=False)
            self.Wk = nn.Conv2d(16, 36, 1, bias=True)
            self.Po = nn.Conv2d(32, 32, 1, bias=False)
            nn.init.zeros_(self.Wk.bias)
            nn.init.zeros_(self.Po.weight)

    def kernels(self, b, guide):
        """Return FP32 kernels shaped (B, 4, 9, H, W), normalized over neighbors."""
        context = F.interpolate(F.silu(self.DW5(self.Ps(guide))), size=b.shape[-2:], mode="nearest")
        logits = self.Wk(F.silu(self.Pb(b) + context))
        with autocast(False, device=b.device.type):
            return logits.float().reshape(b.shape[0], 4, 9, *b.shape[-2:]).softmax(dim=2)

    @staticmethod
    def local_sum(kernel, b):
        """Accumulate row-major replicate-padded neighbors without a full unfold allocation."""
        with autocast(False, device=b.device.type):
            batch, _, height, width = b.shape
            padded = F.pad(b.float(), (1, 1, 1, 1), mode="replicate").reshape(batch, 4, 8, height + 2, width + 2)
            total = torch.zeros_like(b, dtype=torch.float32).reshape(batch, 4, 8, height, width)
            for j in range(9):
                dy, dx = divmod(j, 3)
                total = total + kernel[:, :, j].float().unsqueeze(2) * padded[..., dy : dy + height, dx : dx + width]
            return total.reshape_as(b)

    def forward(self, features, guide):
        """Leave the first half exact and project an FP32 local aggregation difference."""
        a, b = features.split(32, dim=1)
        kernel = self.kernels(b, guide)
        with autocast(False, device=b.device.type):
            difference = self.local_sum(kernel, b) - b.float()
        residual = self.Po(difference.to(b.dtype)).to(b.dtype)
        return torch.cat((a, b + residual), dim=1)


class C3k2_SGK_P3(C3k2):
    """Native C3k2 state paths plus SGK; inputs are [X15, X13]."""

    def __init__(self, c1, c2, guide_channels, n=1, c3k=False, e=0.5, attn=False, g=1, shortcut=True):
        """Retain native positional flags; the parser inserts guide channels before repeat count."""
        if (c1, c2, guide_channels) != (256, 64, 128):
            raise ValueError("SGK-P3 v1 requires nano main/output/guide channels 256/64/128")
        super().__init__(c1, c2, n, c3k, e, attn, g, shortcut)
        self.sgk = SGK()

    def forward(self, x):
        """Apply SGK after the complete native C3k2 result."""
        return self.sgk(super().forward(x[0]), x[1])

    def forward_split(self, x):
        """Preserve the same SGK postprocessing for native split execution."""
        return self.sgk(super().forward_split(x[0]), x[1])
