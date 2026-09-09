"""Normalized distribution pooling as a parallel residual of the pinned native SPPF."""

import torch
from torch import nn
from torch.nn import functional as F

from ultralytics.utils.torch_utils import autocast

from .block import SPPF


def ndp_window(u, k, diagnostics=False):
    """Aggregate centered observations with bounded standardized scores over valid window positions.

    Args:
        u (torch.Tensor): BCHW floating point features.
        k (int): Odd spatial window size.
        diagnostics (bool): Also return detached FP32 aggregation statistics for auditing.

    Returns:
        (torch.Tensor | tuple): BCHW response in the input dtype, optionally with diagnostic tensors.
    """
    b, c, h, w = u.shape
    with autocast(enabled=False, device=u.device.type):
        patches = F.unfold(u.float(), k, padding=k // 2).reshape(b, c, k * k, h * w)
        valid = F.unfold(u.new_ones((1, 1, h, w), dtype=torch.float32), k, padding=k // 2).bool()[None]
        count = valid.sum(2, keepdim=True)
        mean = patches.sum(2, keepdim=True) / count
        centered = (patches - mean).masked_fill(~valid, 0)
        sigma = (centered.square().sum(2, keepdim=True) / count + 1e-4).sqrt()
        logits = (2.0 * (centered / sigma).tanh()).masked_fill(~valid, -torch.inf)
        weights = logits.softmax(2)
        response = (weights * centered).sum(2).reshape(b, c, h, w)
        if diagnostics:
            # Avoid log(0) at invalid positions; valid weights are strictly positive.
            entropy = -(weights * weights.clamp_min(torch.finfo(torch.float32).tiny).log()).sum(2)
            m = count.squeeze(2).expand(b, c, h * w)
            normalized = torch.where(m > 1, entropy / m.clamp_min(2).log(), torch.zeros_like(entropy))
            stats = {
                "count": m.reshape(b, c, h, w).detach(),
                "entropy": entropy.reshape(b, c, h, w).detach(),
                "normalized_entropy": normalized.reshape(b, c, h, w).detach(),
                "max_weight": weights.amax(2).reshape(b, c, h, w).detach(),
                "weight_sum": weights.sum(2).reshape(b, c, h, w).detach(),
                "weight_ratio": (weights.amax(2) / weights.masked_fill(~valid, torch.inf).amin(2)).detach(),
            }
            return response.to(u.dtype), stats
    return response.to(u.dtype)


class SPPF_NDP(SPPF):
    """Retain native SPPF and add fixed r=16 parallel 5/9/13 distribution summaries."""

    def __init__(self, c1: int, c2: int, k: int = 5, n: int = 3, shortcut: bool = False):
        """Preserve native parameter paths and isolate initialization of the two new CPU projections."""
        super().__init__(c1, c2, k, n, shortcut)
        with torch.random.fork_rng(devices=[]):
            self.ndp_in = nn.Conv2d(c1 // 2, 16, 1, bias=False)
            self.ndp_out = nn.Conv2d(48, c2, 1, bias=False)
            nn.init.zeros_(self.ndp_out.weight)

    def forward(self, x):
        """Compute cv1 once and add Delta after the complete native path, including its shortcut."""
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(self.n))
        base = self.cv2(torch.cat(y, 1))
        base = base + x if self.add else base
        u = self.ndp_in(y[0])
        return base + self.ndp_out(torch.cat([ndp_window(u, k) for k in (5, 9, 13)], 1))
