"""Query-dependent, trace-free directional geometry for the b19 C2PSA experiment."""

import torch
import torch.nn as nn

from ultralytics.utils.torch_utils import autocast

from .block import Attention, C2PSA, PSABlock
from .conv import Conv


class DGAAttention(Attention):
    """Keep native attention and add -delta.T @ [[a, b], [b, -a]] @ delta to its scaled logits."""

    def __init__(self, dim: int, num_heads: int = 8, attn_ratio: float = 0.5):
        """Build native projections once, isolating the CPU predictor initialization from the model RNG."""
        super().__init__(dim, num_heads, attn_ratio)
        # Model parsing constructs on CPU. Restore its RNG so later native layers initialize identically.
        with torch.random.fork_rng(devices=[]):
            self.geometry_predictor = nn.Sequential(
                nn.Conv2d(dim, 16, 1, bias=False),
                nn.SiLU(),
                nn.Conv2d(16, 16, 3, stride=1, padding=1, groups=16, bias=False),
                nn.SiLU(),
                nn.Conv2d(16, 2 * num_heads, 1, bias=True),
            )
            nn.init.zeros_(self.geometry_predictor[-1].weight)
            nn.init.zeros_(self.geometry_predictor[-1].bias)

    def geometry_bias(self, x: torch.Tensor) -> torch.Tensor:
        """Predict query coefficients in normal AMP, then compute the geometry in local FP32.

        Args:
            x (torch.Tensor): Original attention input of shape (batch, channels, height, width).

        Returns:
            (torch.Tensor): FP32 bias of shape (batch, heads, query, key).
        """
        batch, _, height, width = x.shape
        geometry = self.geometry_predictor(x).reshape(batch, self.num_heads, 2, height * width)
        with autocast(enabled=False, device=x.device.type):
            # Interleaved channels [u0, v0, u1, v1, ...]; v_geom is unrelated to attention values.
            u_geom, v_geom = geometry.float().unbind(dim=2)
            radius = (u_geom.square() + v_geom.square() + 1e-6).sqrt()
            factor = 0.8 * radius.tanh() / radius
            a, b = factor * u_geom, factor * v_geom
            # Flatten in feature order: x is column, y is row. Both axes use the same scale.
            positions = torch.arange(height * width, device=x.device, dtype=torch.float32)
            columns, rows = positions % width, (positions / width).floor()
            scale = max(height - 1, width - 1, 1)
            dx = (columns[None, :] - columns[:, None]) / scale
            dy = (rows[None, :] - rows[:, None]) / scale
            axial, diagonal = dx.square() - dy.square(), 2 * dx * dy
            return -a.unsqueeze(-1) * axial - b.unsqueeze(-1) * diagonal

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add geometry before softmax, retaining b19 Q scaling, AV, original-value PE, and projection."""
        batch, channels, height, width = x.shape
        q, k, value = (
            self.qkv(x)
            .view(batch, self.num_heads, self.key_dim * 2 + self.head_dim, height * width)
            .split([self.key_dim, self.key_dim, self.head_dim], dim=2)
        )
        logits = (q * self.scale).transpose(-2, -1) @ k
        bias = self.geometry_bias(x)
        attn = (logits + bias.to(dtype=logits.dtype)).softmax(dim=-1)
        out = (value @ attn.transpose(-2, -1)).view(batch, channels, height, width)
        out = out + self.pe(value.reshape(batch, channels, height, width))
        return self.proj(out)


class PSABlock_DGA(PSABlock):
    """Reuse native shortcut and FFN forward with DGA attention and unchanged parameter paths."""

    def __init__(self, c: int, attn_ratio: float = 0.5, num_heads: int = 4, shortcut: bool = True):
        """Construct each native component once in the original order; do not replace initialized modules."""
        nn.Module.__init__(self)
        self.attn = DGAAttention(c, attn_ratio=attn_ratio, num_heads=num_heads)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut


class C2PSA_DGA(C2PSA):
    """Reuse native CSP forward, split, concatenation, projections, and repeat count."""

    def __init__(self, c1: int, c2: int, n: int = 1, e: float = 0.5):
        """Construct the C2PSA components once, with only the attention logits extended by DGA."""
        nn.Module.__init__(self)
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)
        self.m = nn.Sequential(*(PSABlock_DGA(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))
