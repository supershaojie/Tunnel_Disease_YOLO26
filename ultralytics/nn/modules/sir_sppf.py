"""Scale-increment routing for the original YOLO26 SPPF pyramid."""

import torch
import torch.nn as nn

from .block import SPPF


class SPPF_SIR(SPPF):
    """Jointly route raw pooling increments and accumulate bounded corrections along the pyramid."""

    def __init__(self, c1: int, c2: int, k: int = 5, n: int = 3, shortcut: bool = False):
        """Preserve SPPF parameters and RNG consumption; initialize the extra router to an identity path.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            k (int): Pooling kernel size.
            n (int): Pooling iterations, independent of model depth scaling.
            shortcut (bool): Add the input when input and output channels match.
        """
        super().__init__(c1, c2, k, n, shortcut)
        c = c1 // 2
        with torch.random.fork_rng(devices=[]):
            self.router = nn.Sequential(
                nn.Conv2d((n + 1) * c, 16, 1, bias=True),
                nn.SiLU(),
                nn.Conv2d(16, 16, 3, padding=1, groups=16, bias=True),
                nn.SiLU(),
                nn.Conv2d(16, n * c, 1, bias=True),
            )
            nn.init.zeros_(self.router[-1].weight)
            nn.init.zeros_(self.router[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pool the raw pyramid first, then add cumulative corrections without modifying its tensors."""
        z = [self.cv1(x)]
        z.extend(self.m(z[-1]) for _ in range(self.n))
        increments = [current - previous for previous, current in zip(z, z[1:])]
        logits = self.router(torch.cat([z[0], *increments], 1)).chunk(self.n, 1)
        corrected = [z[0]]
        correction = 0
        for raw, delta, logit in zip(z[1:], increments, logits):
            correction = correction + 0.5 * logit.tanh() * delta
            corrected.append(raw + correction)
        y = self.cv2(torch.cat(corrected, 1))
        return y + x if self.add else y
