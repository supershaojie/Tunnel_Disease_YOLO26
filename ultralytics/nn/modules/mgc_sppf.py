"""Low-rank morphological gap context alongside the unchanged native SPPF path."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .block import SPPF


class MGC_SPPF(SPPF):
    """Supplement native SPPF with two grayscale closing residuals, initially projected to zero."""

    def __init__(self, c1: int, c2: int, k: int = 5, n: int = 3, shortcut: bool = False, rank: int = 16):
        """Reuse native construction and isolate only new convolution initialization from the shared RNG stream."""
        super().__init__(c1, c2, k, n, shortcut)
        with torch.random.fork_rng(devices=[]):
            self.gap_in = nn.Conv2d(c1 // 2, rank, 1, bias=False)
            self.gap_out = nn.Conv2d(2 * rank, c2, 1, bias=False)
            nn.init.zeros_(self.gap_out.weight)

    @staticmethod
    def gap(u: torch.Tensor, k: int) -> torch.Tensor:
        """Compute closing minus input using implicit negative-infinity MaxPool padding."""
        d = F.max_pool2d(u, k, stride=1, padding=k // 2, ceil_mode=False)
        closing = -F.max_pool2d(-d, k, stride=1, padding=k // 2, ceil_mode=False)
        return closing - u

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Share one cv1 evaluation and preserve the native pooling, shortcut and checkpoint semantics."""
        z = self.cv1(x)
        levels = [z]
        levels.extend(self.m(levels[-1]) for _ in range(getattr(self, "n", 3)))
        y_native = self.cv2(torch.cat(levels, 1))
        if getattr(self, "add", False):
            y_native = y_native + x
        u = self.gap_in(z)
        r = self.gap_out(torch.cat([self.gap(u, 3), self.gap(u, 5)], dim=1))
        return y_native + r.to(dtype=y_native.dtype)
