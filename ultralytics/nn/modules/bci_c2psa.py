"""Fixed nano Branch-Conditioned Channel Interaction candidate, retaining native PSA."""

import torch
from torch import nn
from torch.nn import functional as F

from .block import C2PSA


class BCI(nn.Module):
    """Mix 32 channel coordinates conditioned on the untouched bypass branch."""

    def __init__(self):
        """Create six bias-free projections; only the output projection starts at zero."""
        super().__init__()
        self.pq = nn.Conv2d(128, 32, 1, bias=False)
        self.dwq = nn.Conv2d(32, 32, 3, padding=1, groups=32, bias=False)
        self.pk = nn.Conv2d(128, 32, 1, bias=False)
        self.dwk = nn.Conv2d(32, 32, 3, padding=1, groups=32, bias=False)
        self.pv = nn.Conv2d(128, 32, 1, bias=False)
        self.po = nn.Conv2d(32, 128, 1, bias=False)
        nn.init.zeros_(self.po.weight)

    @staticmethod
    def interaction(q, k, v):
        """Compute centered channel attention and AV minus V entirely outside autocast."""
        with torch.autocast(device_type=q.device.type, enabled=False):
            q, k, v = q.float(), k.float(), v.float()
            q = F.normalize(q - q.mean(-1, keepdim=True), p=2, dim=-1, eps=1e-6)
            k = F.normalize(k - k.mean(-1, keepdim=True), p=2, dim=-1, eps=1e-6)
            correlation = q @ k.transpose(-1, -2)
            attention = (4.0 * correlation).softmax(dim=-1)
            return attention @ v - v, attention, correlation

    def forward(self, a, b0):
        """Return only the new residual, leaving both native input branches unmodified."""
        q = self.dwq(self.pq(a)).flatten(2)
        k = self.dwk(self.pk(b0)).flatten(2)
        v = self.pv(b0).flatten(2)
        delta, _, _ = self.interaction(q, k, v)
        return self.po(delta.reshape(b0.shape[0], 32, *b0.shape[-2:]).to(b0.dtype))


class C2PSA_BCI(C2PSA):
    """Extend only the verified 256-channel, one-block native nano C2PSA."""

    def __init__(self, c1, c2, n=1, e=0.5):
        """Keep native state paths and isolate new initialization from later native layers."""
        if (c1, c2, n, e) != (256, 256, 1, 0.5):
            raise ValueError("BCI v1 requires nano c1=c2=256, n=1, e=0.5, r=32")
        super().__init__(c1, c2, n, e)
        with torch.random.fork_rng(devices=[]):
            self.bci = BCI()

    def forward(self, x):
        """Apply BCI once after the complete native PSA stack and before native cv2."""
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b0 = self.m(b)
        return self.cv2(torch.cat((a, b0 + self.bci(a, b0)), 1))
