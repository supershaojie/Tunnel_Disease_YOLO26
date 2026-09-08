"""MSI-C2PSA v1: an additive multiscale FFN branch with native attention and parameter paths."""

import torch
from torch import nn

from .block import C2PSA, PSABlock
from .conv import Conv


class MSI(nn.Module):
    """Mix the two halves of the existing FFN expansion without another input projection."""

    def __init__(self, c: int):
        """Initialize every depthwise channel to identity and the output projection to zero."""
        super().__init__()
        # Only these new CPU convolutions consume random numbers; restore the native construction stream.
        with torch.random.fork_rng(devices=[]):
            self.dw3 = nn.Conv2d(c, c, 3, padding=1, groups=c, bias=False)
            self.dw5 = nn.Conv2d(c, c, 5, padding=2, groups=c, bias=False)
            self.project = nn.Conv2d(c, c, 1, bias=True)
        self.gelu = nn.GELU(approximate="none")
        with torch.no_grad():
            for conv in (self.dw3, self.dw5):
                conv.weight.zero_()
                center = conv.kernel_size[0] // 2
                conv.weight[:, 0, center, center] = 1
            self.project.weight.zero_()
            self.project.bias.zero_()

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """Return Wo(GELU(DW3(Ua)) * DW5(Ub))."""
        ua, ub = u.chunk(2, dim=1)
        return self.project(self.gelu(self.dw3(ua)) * self.dw5(ub))


class PSABlock_MSI(PSABlock):
    """Keep native Attention and FFN; evaluate the FFN expansion once for both outputs."""

    def __init__(self, c: int, attn_ratio: float = 0.5, num_heads: int = 4, shortcut: bool = True):
        """Construct native parameters first, then the RNG-isolated additive branch."""
        super().__init__(c, attn_ratio, num_heads, shortcut)
        self.msi = MSI(c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Preserve both native shortcut modes and add Delta after the native FFN result."""
        z = x + self.attn(x) if self.add else self.attn(x)
        u = self.ffn[0](z)
        v = self.ffn[1](u)
        return (z + v if self.add else v) + self.msi(u)


class C2PSA_MSI(C2PSA):
    """Use the native C2PSA outer forward with MSI only inside the processing branch's FFN."""

    def __init__(self, c1: int, c2: int, n: int = 1, e: float = 0.5):
        """Retain native construction order and accept already-scaled channels and repeats."""
        nn.Module.__init__(self)
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)
        self.m = nn.Sequential(*(PSABlock_MSI(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))
