"""Pooling-kernel complementary SPPF with a fixed 32-channel convolution pyramid."""

import torch
from torch import nn

from .block import SPPF
from .conv import Conv


class SPPF_PKC(SPPF):
    """Preserve native SPPF and add a zero-projected 5/9/13 convolution pyramid."""

    def __init__(self, c1: int, c2: int, k: int = 5, n: int = 3, shortcut: bool = False):
        """Build v1 with fixed r=32; retain native keys and the caller's CPU initialization stream."""
        if (k, n) != (5, 3):
            raise ValueError("PKC-SPPF v1 requires k=5 and n=3")
        super().__init__(c1, c2, k, n, shortcut)
        # Native parser constructs modules on CPU. New parameters must not perturb later Detect initialization.
        with torch.random.fork_rng(devices=[]):
            self.pkc = nn.ModuleDict(
                {
                    "reduce": Conv(c1 // 2, 32, 1),
                    "l5": Conv(32, 32, 5, p=2, g=32),
                    "l9": Conv(32, 32, 3, p=2, g=32, d=2),
                    "l13": Conv(32, 32, 3, p=2, g=32, d=2),
                    "project": nn.Conv2d(96, c2, 1, bias=False),
                }
            )
            nn.init.zeros_(self.pkc["project"].weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add Delta after the complete native output, including its conditional outer residual."""
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(self.n))
        base = self.cv2(torch.cat(y, 1))
        base = base + x if self.add else base
        u = self.pkc["reduce"](y[0])
        l5 = self.pkc["l5"](u)
        l9 = self.pkc["l9"](l5)
        l13 = self.pkc["l13"](l9)
        return base + self.pkc["project"](torch.cat((l5, l9, l13), 1))
