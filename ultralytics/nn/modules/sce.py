"""Source-preserving, scale-complementary exchange for the three YOLO26n neck outputs."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv


def _resize(x, size):
    """Resize in FP32, preserving dtype and the differentiable source path."""
    if x.shape[-2:] == size:
        return x
    if x.shape[-2] >= size[0] and x.shape[-1] >= size[1]:
        return F.adaptive_avg_pool2d(x.float(), size).to(x.dtype)
    return F.interpolate(x.float(), size=size, mode="bilinear", align_corners=False).to(x.dtype)


class SCEContextBlock(nn.Module):
    """Encode independent height and width context within a single source."""

    def __init__(self, hidden=64, axis_kernel=7):
        """Build depthwise local, two axial, and pointwise residual transforms."""
        super().__init__()
        self.local = Conv(hidden, hidden, 3, g=hidden)
        self.height = nn.Sequential(
            nn.Conv2d(2 * hidden, hidden, 1, bias=False),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, (axis_kernel, 1), padding=(axis_kernel // 2, 0), groups=hidden),
        )
        self.width = nn.Sequential(
            nn.Conv2d(2 * hidden, hidden, 1, bias=False),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, (1, axis_kernel), padding=(0, axis_kernel // 2), groups=hidden),
        )
        self.channel_mix = Conv(hidden, 2 * hidden, 1)
        self.channel_out = Conv(2 * hidden, hidden, 1, act=False)

    def forward(self, z):
        """Pool the opposite spatial dimension and broadcast each axial gate."""
        u = self.local(z)
        stats = u.float()
        dh = torch.cat((stats.mean(3, keepdim=True), stats.max(3, keepdim=True).values), 1)
        dw = torch.cat((stats.mean(2, keepdim=True), stats.max(2, keepdim=True).values), 1)
        ah = self.height(dh.to(u.dtype)).float().sigmoid().to(u.dtype)
        aw = self.width(dw.to(u.dtype)).float().sigmoid().to(u.dtype)
        return z + self.channel_out(self.channel_mix(u + u * ah + u * aw))


class SCERouter(nn.Module):
    """Route two external sources and a constant zero candidate, per contiguous channel group."""

    def __init__(self, hidden=64, groups=8):
        """Construct target-conditioned logits, initially uniform over the three candidates."""
        super().__init__()
        self.groups = groups
        self.logits = nn.Sequential(
            nn.Conv2d(3 * hidden, hidden // 4, 1, bias=False),
            nn.SiLU(),
            nn.Conv2d(hidden // 4, hidden // 4, 3, padding=1, groups=hidden // 4, bias=False),
            nn.SiLU(),
            nn.Conv2d(hidden // 4, groups * 3, 1),
        )
        nn.init.zeros_(self.logits[-1].weight)
        nn.init.zeros_(self.logits[-1].bias)

    def probabilities(self, target, first, second):
        """Return FP32 probabilities with axes (batch, group, candidate, height, width)."""
        logits = self.logits(torch.cat((target, first, second), 1))
        b, _, h, w = logits.shape
        return logits.float().reshape(b, self.groups, 3, h, w).softmax(2)

    def forward(self, target, first, second):
        """Mix external sources without renormalizing away the zero candidate."""
        p = self.probabilities(target, first, second)
        b, c, h, w = first.shape
        shape = (b, self.groups, c // self.groups, h, w)
        message = p[:, :, 0:1] * first.float().reshape(shape) + p[:, :, 1:2] * second.float().reshape(shape)
        return message.reshape(b, c, h, w).to(target.dtype)


class SCEFusion(nn.Module):
    """Exchange separately encoded sources and add refined messages at each original resolution."""

    sources = ((1, 2), (0, 2), (0, 1))

    def __init__(self, channels, hidden=64, blocks=2, groups=8, axis_kernel=7, residual_init=0.1):
        """Build the fixed n-scale experiment without width/depth rescaling."""
        super().__init__()
        if tuple(channels) != (64, 128, 256) or (hidden, blocks, groups, axis_kernel, residual_init) != (
            64,
            2,
            8,
            7,
            0.1,
        ):
            raise ValueError("SCE-Fusion requires channels=[64,128,256] and configuration [64,2,8,7,0.1]")
        self.channels = tuple(channels)
        self.proj = nn.ModuleList(Conv(c, hidden, 1) for c in channels)
        self.contexts = nn.ModuleList(
            nn.Sequential(*(SCEContextBlock(hidden, axis_kernel) for _ in range(blocks))) for _ in channels
        )
        self.routers = nn.ModuleList(SCERouter(hidden, groups) for _ in channels)
        self.refine = nn.ModuleList(Conv(hidden, hidden, 3, g=hidden) for _ in channels)
        self.out = nn.ModuleList(Conv(hidden, c, 1, act=False) for c in channels)
        self.lambdas = nn.Parameter(torch.full((3,), residual_init))

    def forward(self, features):
        """Return (Y3, Y4, Y5) without mutating the supplied features or retaining activations."""
        if len(features) != 3 or any(x.ndim != 4 for x in features):
            raise ValueError("SCE-Fusion expects three BCHW tensors in P3/P4/P5 order")
        x3, x4, x5 = features
        if any(x.shape[1] != c for x, c in zip(features, self.channels)):
            raise ValueError(f"SCE-Fusion expects channels {self.channels}")
        if any(x.shape[0] != x3.shape[0] or x.device != x3.device or x.dtype != x3.dtype for x in features):
            raise ValueError("SCE-Fusion inputs must share batch, device and dtype")
        if any(x3.shape[d] != 2 * x4.shape[d] or x4.shape[d] != 2 * x5.shape[d] for d in (-2, -1)):
            raise ValueError("SCE-Fusion requires H3=2H4=4H5 and W3=2W4=4W5")
        z = tuple(_resize(proj(x), x4.shape[-2:]) for proj, x in zip(self.proj, features))
        v = tuple(context(source) for context, source in zip(self.contexts, z))
        outputs = []
        for i, (j, k) in enumerate(self.sources):
            message = self.routers[i](z[i], v[j], v[k])
            delta = self.out[i](self.refine[i](_resize(message, features[i].shape[-2:])))
            outputs.append((features[i].float() + self.lambdas[i].float() * delta.float()).to(features[i].dtype))
        return tuple(outputs)
