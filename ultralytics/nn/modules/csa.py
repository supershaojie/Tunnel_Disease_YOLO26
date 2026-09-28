"""Curve-selective CSP replacement for the fixed YOLO26n b19 experiment."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.torch_utils import autocast, fuse_conv_and_bn

from .conv import Conv


class CurveSampler(nn.Module):
    """Unmodulated strip deformable convolution with four independent continuous offset groups."""

    def __init__(self, channels=32, kernel_size=7, groups=4, direction="horizontal"):
        """Create an independent offset predictor, full convolution kernel, and branch BN."""
        super().__init__()
        if (channels, kernel_size, groups) != (32, 7, 4) or direction not in {"horizontal", "vertical"}:
            raise ValueError("CSA v1 requires channels=32, kernel_size=7, groups=4 and horizontal/vertical direction")
        self.groups = groups
        self.direction = direction
        shape = (1, kernel_size) if direction == "horizontal" else (kernel_size, 1)
        self.kernel = nn.Conv2d(channels, channels, shape, padding=tuple(k // 2 for k in shape), bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.offset = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.SiLU(),
            nn.Conv2d(channels, groups * (kernel_size - 1), 1, bias=True),
        )
        nn.init.zeros_(self.offset[-1].weight)
        nn.init.zeros_(self.offset[-1].bias)

    def curve_offsets(self, logits):
        """Return FP32 steps, [n3,n2,n1,0,p1,p2,p3], and group/point/[dy,dx] offsets."""
        b, _, h, w = logits.shape
        steps = logits.float().reshape(b, self.groups, 6, h, w).tanh()
        negative = steps[:, :, :3].cumsum(2).flip(2)
        positive = steps[:, :, 3:].cumsum(2)
        sequence = torch.cat((negative, torch.zeros_like(negative[:, :, :1]), positive), dim=2)
        zero = torch.zeros_like(sequence)
        pair = (sequence, zero) if self.direction == "horizontal" else (zero, sequence)
        return steps, sequence, torch.stack(pair, dim=3).reshape(b, self.groups * 14, h, w)

    def forward(self, z):
        """Keep coordinate construction and deform_conv2d in FP32, including with half model parameters."""
        from torchvision.ops import deform_conv2d

        logits = self.offset(z)
        with autocast(enabled=False, device=z.device.type):
            _, _, offsets = self.curve_offsets(logits)
            bias = self.kernel.bias.float() if self.kernel.bias is not None else None
            y = deform_conv2d(
                z.float(), offsets, self.kernel.weight.float(), bias, padding=self.kernel.padding, mask=None
            )
        y = y.to(z.dtype)
        return self.bn(y) if hasattr(self, "bn") else y

    def fuse(self):
        """Fold branch BN once while preserving learned offsets and the deformable forward path."""
        if hasattr(self, "bn"):
            self.kernel = fuse_conv_and_bn(self.kernel, self.bn)
            del self.bn
        return self


class CSAUnit(nn.Module):
    """Select local, horizontal and vertical responses per spatial position and consecutive channel group."""

    def __init__(self, hidden=64, latent=32, kernel_size=7, groups=4):
        """Build the fixed three-branch residual block; only prediction terminal layers start at zero."""
        super().__init__()
        if (hidden, latent, kernel_size, groups) != (64, 32, 7, 4):
            raise ValueError("CSA v1 requires hidden=64, latent=32, kernel_size=7, groups=4")
        self.reduce = Conv(hidden, latent, 1)
        self.local = Conv(latent, latent, 3, act=False)
        self.horizontal = CurveSampler(latent, kernel_size, groups, "horizontal")
        self.vertical = CurveSampler(latent, kernel_size, groups, "vertical")
        self.selector = nn.Sequential(
            nn.Conv2d(3 * latent, 16, 1, bias=False),
            nn.SiLU(),
            nn.Conv2d(16, 16, 3, padding=1, groups=16, bias=False),
            nn.SiLU(),
            nn.Conv2d(16, groups * 3, 1, bias=True),
        )
        nn.init.zeros_(self.selector[-1].weight)
        nn.init.zeros_(self.selector[-1].bias)
        self.project = Conv(latent, hidden, 1, act=False)

    @staticmethod
    def aggregate(features, logits):
        """Softmax only across the three branches, accumulating four groups of eight channels in FP32."""
        b, _, h, w = features[0].shape
        weights = logits.float().reshape(b, 4, 3, h, w).softmax(2)
        fused = sum(f.float().reshape(b, 4, 8, h, w) * weights[:, :, i, None] for i, f in enumerate(features))
        return fused.reshape(b, 32, h, w).to(features[0].dtype)

    def forward(self, t):
        """Apply branch selection, projection and the unscaled residual addition."""
        z = self.reduce(t)
        features = (self.local(z), self.horizontal(z), self.vertical(z))
        logits = self.selector(torch.cat(features, dim=1))
        return t + self.project(F.silu(self.aggregate(features, logits)))


class CSAC3k2(nn.Module):
    """Complete layer-4 CSP replacement with two independent curve-selective units."""

    def __init__(self, c1, c2, hidden=64, latent=32, blocks=2, kernel_size=7, groups=4):
        """Accept native parser c1/c2 scaling while retaining fixed internal experiment widths."""
        super().__init__()
        if (c1, c2, hidden, latent, blocks, kernel_size, groups) != (64, 128, 64, 32, 2, 7, 4):
            raise ValueError("CSAC3k2 v1 only supports the specified YOLO26n layer-4 configuration")
        self.stem = Conv(c1, 2 * hidden, 1)
        self.blocks = nn.ModuleList(CSAUnit(hidden, latent, kernel_size, groups) for _ in range(blocks))
        self.merge = Conv((2 + blocks) * hidden, c2, 1)

    def forward(self, x):
        """Merge A, B, B1 and B2 without retaining the old C3k2."""
        features = list(self.stem(x).chunk(2, dim=1))
        for block in self.blocks:
            features.append(block(features[-1]))
        return self.merge(torch.cat(features, dim=1))
