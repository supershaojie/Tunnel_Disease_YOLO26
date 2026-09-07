"""Dynamic constrained symmetric differences for YOLO26n P3 box regression only."""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.torch_utils import autocast

from .head import Detect


class DSDAdapter(nn.Module):
    """Adapt 64 channels in eight contiguous groups with four signed second differences.

    Only complete 3x3 neighborhoods contribute. The fixed 0.1/4 scale is part of v1,
    and the zero final projection makes construction an exact identity.
    """

    directions = ((0, 1), (1, 0), (1, 1), (1, -1))

    def __init__(self):
        """Create the 1744-parameter coefficient network without advancing the caller's CPU RNG."""
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            self.coeff = nn.Sequential(
                nn.Conv2d(64, 16, 1, bias=True),
                nn.SiLU(),
                nn.Conv2d(16, 16, 3, padding=1, groups=16, bias=True),
                nn.SiLU(),
                nn.Conv2d(16, 32, 1, bias=True),
            )
            nn.init.zeros_(self.coeff[-1].weight)
            nn.init.zeros_(self.coeff[-1].bias)

    def forward(self, x):
        """Accumulate interior differences and the residual in FP32, retaining all gradients."""
        b, c, h, w = x.shape
        if h < 3 or w < 3:
            return x
        z = self.coeff(x)
        with autocast(enabled=False, device=x.device.type):
            grouped = x.float().reshape(b, 8, 8, h, w)
            center = grouped[..., 1:-1, 1:-1]
            coefficients = z.float().reshape(b, 8, 4, h, w).tanh()
            delta = torch.zeros_like(center)
            for d, (dy, dx) in enumerate(self.directions):
                difference = (
                    grouped[..., 1 + dy : h - 1 + dy, 1 + dx : w - 1 + dx]
                    + grouped[..., 1 - dy : h - 1 - dy, 1 - dx : w - 1 - dx]
                    - 2 * center
                ) / (dy * dy + dx * dx)
                delta = delta + coefficients[:, :, d, 1:-1, 1:-1].unsqueeze(2) * difference
            delta = F.pad((0.025 * delta).reshape(b, c, h - 2, w - 2), (1, 1, 1, 1))
            return (x.float() + delta).to(x.dtype)


class DSDDetect(Detect):
    """Retain native Detect routing, detach, decode and weight names; adapt only P3 regression."""

    def __init__(self, nc=80, reg_max=1, end2end=True, ch=()):
        """Construct the native head first, then two independent identically initialized adapters."""
        if tuple(ch) != (64, 128, 256) or reg_max != 1 or not end2end:
            raise ValueError("DSD-Head v1 requires YOLO26n channels (64,128,256), reg_max=1, end2end=True")
        super().__init__(nc, reg_max, end2end, ch)
        self.reg_adapter = DSDAdapter()
        self.one2one_reg_adapter = copy.deepcopy(self.reg_adapter)

    @property
    def one2many(self):
        """Pass the independent many adapter to the native forward owner."""
        return dict(box_head=self.cv2, cls_head=self.cv3, reg_adapter=self.reg_adapter)

    @property
    def one2one(self):
        """The native forward calls this head with already detached backbone inputs."""
        return dict(box_head=self.one2one_cv2, cls_head=self.one2one_cv3, reg_adapter=self.one2one_reg_adapter)

    def forward_head(self, x, box_head=None, cls_head=None, reg_adapter=None):
        """Keep classification inputs and returned feats native, without mutating the feature list."""
        if box_head is None or cls_head is None:  # Same native fused-head contract.
            return {}
        bs = x[0].shape[0]
        boxes = torch.cat(
            [box_head[i](reg_adapter(x[i]) if i == 0 else x[i]).view(bs, 4 * self.reg_max, -1) for i in range(self.nl)],
            dim=-1,
        )
        scores = torch.cat([cls_head[i](x[i]).view(bs, self.nc, -1) for i in range(self.nl)], dim=-1)
        return dict(boxes=boxes, scores=scores, feats=x)

    def fuse(self):
        """Remove the training-only adapter with the native many head; retain the inference adapter."""
        super().fuse()
        self.reg_adapter = None
