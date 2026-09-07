"""Independent scale-increment routing for the original YOLO26 SPPF pyramid."""

import torch

from .sir_sppf import SPPF_SIR


class SPPF_SIR_V2(SPPF_SIR):
    """Preserve the v1 router and initialization, applying each correction only to its own raw scale."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pool the unmodified pyramid, then independently correct each scale before cv2."""
        z = [self.cv1(x)]
        z.extend(self.m(z[-1]) for _ in range(self.n))
        increments = [current - previous for previous, current in zip(z, z[1:])]
        logits = self.router(torch.cat([z[0], *increments], 1)).chunk(self.n, 1)
        corrected = [z[0]]
        for raw, delta, logit in zip(z[1:], increments, logits):
            corrected.append(raw + 0.5 * logit.tanh() * delta)
        y = self.cv2(torch.cat(corrected, 1))
        return y + x if self.add else y
