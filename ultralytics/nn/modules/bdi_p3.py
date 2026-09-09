"""Fixed local band-pass P2 detail injection at the native P3 concatenation."""

import torch
import torch.nn.functional as F
from torch import nn


class Concat_BDI_P3(nn.Module):
    """Concatenate [S, L + R] using a fixed r=16 P2 residual, with native initialization RNG preserved."""

    def __init__(self, cS: int, cL: int, cP2: int):
        """Build three bias-free convolutions and normalized nontrainable binomial kernels."""
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            self.pin = nn.Conv2d(cP2, 16, 1, bias=False)
            self.dw = nn.Conv2d(16, 16, 3, padding=1, groups=16, bias=False)
            self.po = nn.Conv2d(16, cL, 1, bias=False)
            nn.init.zeros_(self.po.weight)
        self.act = nn.SiLU()
        for size, values in ((3, [1, 2, 1]), (5, [1, 4, 6, 4, 1])):
            b = torch.tensor(values, dtype=torch.float32)
            b /= b.sum()
            self.register_buffer(f"k{size}", torch.outer(b, b)[None, None].repeat(16, 1, 1, 1))

    @staticmethod
    def blur(x: torch.Tensor, kernel: torch.Tensor, stride: int = 1) -> torch.Tensor:
        """Accumulate grouped binomial filtering in FP32 even inside native autocast."""
        pad = kernel.shape[-1] // 2
        with torch.autocast(device_type=x.device.type, enabled=False):
            return F.conv2d(
                F.pad(x.float(), (pad, pad, pad, pad), mode="replicate"),
                kernel.float(),
                stride=stride,
                groups=x.shape[1],
            ).to(x.dtype)

    def bandpass(self, u: torch.Tensor) -> torch.Tensor:
        """Compute exactly 4(B3-B5), retaining FP32 through subtraction before restoring dtype."""
        with torch.autocast(device_type=u.device.type, enabled=False):
            d = 4 * (self.blur(u.float(), self.k3) - self.blur(u.float(), self.k5))
        return d.to(u.dtype)

    def details(self, f2: torch.Tensor):
        """Expose actual branch intermediates for independent same-weight inference diagnostics."""
        u = self.pin(f2)
        d = self.bandpass(u)
        v3 = self.blur(self.act(self.dw(d)), self.k3, stride=2)
        return u, d, v3, self.po(v3)

    def forward(self, x: list) -> torch.Tensor:
        """Inject the residual only into L and preserve the S-first concatenation order."""
        s, l, f2 = x
        _, _, _, r = self.details(f2)
        if r.shape != l.shape:
            raise ValueError(f"BDI P2 down2 grid {tuple(r.shape)} != native L {tuple(l.shape)}; P2={tuple(f2.shape)}")
        return torch.cat((s, l + r), dim=1)
