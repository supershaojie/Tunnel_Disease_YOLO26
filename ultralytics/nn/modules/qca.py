"""Query contrast attention for the independent b19 QCA-C2PSA experiment."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.torch_utils import autocast

from .block import Attention, C2PSA, PSABlock
from .conv import Conv

__all__ = ("C2PSA_QCA", "QCAAttention", "QCAPSABlock")


class QCAAttention(Attention):
    """Preserve native attention and add a per-head query-context contrast correction.

    The effective aggregation weights are signed and retain the native quadratic attention matrices. Under AMP, the
    native message retains its original rounding; only the contrast correction uses FP32 arithmetic.
    """

    def __init__(self, dim: int, num_heads: int = 8, attn_ratio: float = 0.5):
        """Initialize the native projections and one zero-initialized contrast parameter per actual head."""
        super().__init__(dim, num_heads=num_heads, attn_ratio=attn_ratio)
        self.theta = nn.Parameter(torch.zeros(self.num_heads))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Aggregate native values with the native message plus the FP32 query-context correction."""
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )

        native_logits = (q * self.scale).transpose(-2, -1) @ k
        attn = native_logits.softmax(dim=-1)
        message_native = v @ attn.transpose(-2, -1)

        with autocast(False, device=x.device.type):
            q_context = F.avg_pool2d(
                q.float().reshape(B * self.num_heads, self.key_dim, H, W),
                kernel_size=3,
                stride=1,
                padding=1,
                count_include_pad=False,
            ).reshape(B, self.num_heads, self.key_dim, N)
            context_logits = (q_context.transpose(-2, -1) @ k.float()) * self.scale
            attn_reference = native_logits.float().softmax(dim=-1)
            attn_context = context_logits.softmax(dim=-1)
            difference = v.float() @ (attn_reference - attn_context).transpose(-2, -1)
            strength = 0.25 * self.theta.float().tanh()
            delta = strength.view(1, self.num_heads, 1, 1) * difference

        message = message_native + delta.to(message_native.dtype)
        x = message.view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        return self.proj(x)


class QCAPSABlock(PSABlock):
    """Use query contrast attention with the unchanged native PSABlock forward and parameter paths."""

    def __init__(self, c: int, attn_ratio: float = 0.5, num_heads: int = 4, shortcut: bool = True):
        """Construct native layers once, in native order, without consuming extra random initialization."""
        nn.Module.__init__(self)
        self.attn = QCAAttention(c, attn_ratio=attn_ratio, num_heads=num_heads)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut


class C2PSA_QCA(C2PSA):
    """Use QCA blocks while inheriting the native C2PSA split, concatenation, and forward semantics."""

    def __init__(self, c1: int, c2: int, n: int = 1, e: float = 0.5):
        """Construct native layers once and retain the depth-scaled number of internal PSA blocks."""
        nn.Module.__init__(self)
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)
        self.m = nn.Sequential(*(QCAPSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))
