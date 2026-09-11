"""Dual-statistic contrast-saliency residual for the native YOLO26 SPPF."""

import torch
from torch import nn

from .block import SPPF
from .conv import Conv


class DCS_SPPF(SPPF):
    """Preserve native SPPF and add three MaxPool-minus-AvgPool refinements through one bounded scalar."""

    def __init__(self, c1: int, c2: int, k: int = 5, n: int = 3, shortcut: bool = False):
        """Reuse native cv1/cv2 and isolate new initialization from all subsequent shared layers.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            k (int): Fixed native max-pooling kernel, 5.
            n (int): Fixed number of native pooling stages, 3.
            shortcut (bool): Preserve the native input shortcut when channels match.
        """
        if (k, n) != (5, 3):
            raise ValueError("DCS-SPPF v1 requires k=5 and n=3")
        super().__init__(c1, c2, k, n, shortcut)
        c = c1 // 2
        self.avg = nn.ModuleList(nn.AvgPool2d(size, 1, size // 2) for size in (5, 9, 13))
        with torch.random.fork_rng(devices=[]):
            self.refine = nn.ModuleList(Conv(c, c, 3, g=c, d=dilation) for dilation in (1, 2, 3))
            self.fuse = Conv(3 * c, c2, 1, act=False)
        self.theta = nn.Parameter(torch.zeros(()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Keep the native pooling/concat/cv2/shortcut sequence; reuse its raw stages for the added branch."""
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(self.n))
        native = self.cv2(torch.cat(y, 1))
        native = native + x if self.add else native
        refined = [branch(maximum - average(y[0])) for maximum, average, branch in zip(y[1:], self.avg, self.refine)]
        residual = self.fuse(torch.cat(refined, 1))
        return native + (0.10 * self.theta.tanh()) * residual
