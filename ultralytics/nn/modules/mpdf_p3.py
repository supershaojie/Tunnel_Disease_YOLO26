"""Mean-preserving, detail-guided fusion at the existing P3 concatenation node."""

import torch
from torch import nn
from torch.nn import functional as F


class MPDFP3(nn.Module):
    """Refine nearest-upsampled semantics with three zero-mean Haar detail coefficients.

    Args:
        c_low (int): Channels in the shallow P3 feature L.
        c_high (int): Channels in the coarse P4 feature H and its native upsample U.
        hidden (int): Actual projection width, independent of model width scaling.
    """

    def __init__(self, c_low, c_high, hidden=32):
        """Initialize only the new branch without consuming the native model's RNG sequence."""
        super().__init__()
        self.c_low, self.c_high = c_low, c_high
        with torch.random.fork_rng(devices=[]):
            self.reduce = nn.Conv2d(c_high + 4 * c_low, hidden, 1, bias=False)
            self.dw = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False)
            self.project = nn.Conv2d(hidden, 3 * c_high, 1, bias=True)
            nn.init.zeros_(self.project.weight)
            nn.init.zeros_(self.project.bias)

    def forward(self, x):
        """Accept [U, L, H], preserve each semantic block mean and return [U + R, L]."""
        if not isinstance(x, (list, tuple)) or len(x) != 3:
            raise ValueError("MPDFP3 requires [U, L, H]")
        u, low, high = x
        if any(t.ndim != 4 for t in x):
            raise ValueError("MPDFP3 inputs must be BCHW tensors")
        batch, channels, h, w = high.shape
        if (
            channels != self.c_high
            or u.shape != (batch, self.c_high, 2 * h, 2 * w)
            or low.shape != (batch, self.c_low, 2 * h, 2 * w)
            or any(t.device != high.device for t in x)
        ):
            raise ValueError("MPDFP3 requires matching batch/device/channels and exact 2x spatial alignment")
        a, b = low[:, :, 0::2, 0::2], low[:, :, 0::2, 1::2]
        c, d = low[:, :, 1::2, 0::2], low[:, :, 1::2, 1::2]
        l0 = (a + b + c + d) * 0.5
        lx = (a - b + c - d) * 0.5
        ly = (a + b - c - d) * 0.5
        ld = (a - b - c + d) * 0.5
        q = torch.cat((high, l0, lx, ly, ld), dim=1)
        s = F.silu(self.reduce(q), inplace=False)
        bx, by, bd = self.project(F.silu(self.dw(s), inplace=False)).split(self.c_high, dim=1)
        ra = (bx + by + bd) * 0.5
        rb = (-bx + by - bd) * 0.5
        rc = (bx - by - bd) * 0.5
        rd = (-bx - by + bd) * 0.5
        phase = torch.stack((ra, rb, rc, rd), dim=2)
        correction = F.pixel_shuffle(phase.reshape(batch, 4 * channels, h, w), 2)
        return torch.cat((u + correction, low), dim=1)
