"""Stage-increment context refinement of the native YOLO26 SPPF pyramid."""

import torch
from torch import nn

from .block import SPPF
from .conv import Conv


class SICRIncrementRefine(nn.Sequential):
    """Refine one pooling increment with directional depthwise convolutions and a linear projection."""

    def __init__(self, c: int, k: int):
        """Preserve channels and spatial size with (1, k), (k, 1), and pointwise Conv primitives."""
        super().__init__(
            Conv(c, c, (1, k), g=c),
            Conv(c, c, (k, 1), g=c),
            Conv(c, c, 1, act=False),
        )


class SICRSPPF(SPPF):
    """Preserve native SPPF and independently inject bounded refinements of its three stage increments."""

    def __init__(self, c1: int, c2: int, k: int = 5, n: int = 3, shortcut: bool = False, alpha_max: float = 0.10):
        """Reuse native cv1/cv2 and isolate added initialization from subsequent backbone/head layers.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            k (int): Fixed MaxPool kernel, 5.
            n (int): Fixed number of pooling stages, 3.
            shortcut (bool): Add the original input when channels match.
            alpha_max (float): Fixed residual coefficient bound, 0.10.
        """
        if (k, n, alpha_max) != (5, 3, 0.10):
            raise ValueError("SICR-SPPF v1 requires k=5, n=3, alpha_max=0.10")
        super().__init__(c1, c2, k, n, shortcut)
        with torch.random.fork_rng(devices=[]):
            self.refine = nn.ModuleList(SICRIncrementRefine(c1 // 2, size) for size in (3, 5, 7))
        self.theta = nn.Parameter(torch.zeros(3))
        self.alpha_max = float(alpha_max)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute all raw stages before independently correcting each increment's corresponding stage."""
        z = [self.cv1(x)]
        z.extend(self.m(z[-1]) for _ in range(self.n))
        alpha = self.alpha_max * self.theta.tanh()
        refined = [z[0]] + [z[i + 1] + alpha[i] * branch(z[i + 1] - z[i]) for i, branch in enumerate(self.refine)]
        y = self.cv2(torch.cat(refined, 1))
        return y + x if self.add else y
