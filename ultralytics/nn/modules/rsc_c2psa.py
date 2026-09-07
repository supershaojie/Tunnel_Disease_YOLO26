"""Reciprocal support calibration of native dense YOLO26 attention."""

import math

import torch.nn as nn

from .block import Attention, C2PSA

__all__ = ("Attention_RSC", "C2PSA_RSC")


def reciprocal_probabilities(scores):
    """Return the row-normalized geometric symmetrization in FP32, without probability products."""
    log_a = scores.float().log_softmax(dim=-1)
    return (0.5 * (log_a + log_a.transpose(-2, -1))).softmax(dim=-1)


class Attention_RSC(Attention):
    """Reuse an existing Attention's tensors and add one bounded mixing scalar per head.

    Args:
        native (Attention): Native module whose QKV, PE and projection are transferred without RNG use.

    Attributes:
        theta (nn.Parameter): Head logits; beta = 0.2 * sigmoid(theta), initially 0.01.
        enabled (bool): Explicit diagnostic switch to the inherited native forward.
    """

    def __init__(self, native):
        """Take ownership of the native modules without initializing another QKV or changing state paths."""
        nn.Module.__init__(self)
        self.num_heads, self.head_dim = native.num_heads, native.head_dim
        self.key_dim, self.scale = native.key_dim, native.scale
        self.qkv, self.proj, self.pe = native.qkv, native.proj, native.pe
        self.theta = nn.Parameter(native.qkv.conv.weight.new_full((self.num_heads,), math.log(0.05 / 0.95)))
        self.enabled = True

    @property
    def beta(self):
        """Return FP32 mixing coefficients in (0, 0.2), including when the model is half precision."""
        return 0.2 * self.theta.float().sigmoid()

    def forward(self, x):
        """Keep native Q scaling, softmax, V aggregation, PE(V) and projection order."""
        if not self.enabled:
            return super().forward(x)
        B, C, H, W = x.shape
        q, k, v = (
            self.qkv(x)
            .view(B, self.num_heads, self.key_dim * 2 + self.head_dim, H * W)
            .split([self.key_dim, self.key_dim, self.head_dim], dim=2)
        )
        scores = (q * self.scale).transpose(-2, -1) @ k
        a = scores.softmax(dim=-1)
        r = reciprocal_probabilities(scores)
        beta = self.beta.view(1, self.num_heads, 1, 1)
        mixed = ((1 - beta) * a.float() + beta * r).to(v.dtype)
        x = (v @ mixed.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        return self.proj(x)


class C2PSA_RSC(C2PSA):
    """Preserve native C2PSA construction, repeats, split/concat, FFN and residuals; calibrate its attention."""

    def __init__(self, c1, c2, n=1, e=0.5):
        """Build the native block once, then transfer each attention without consuming random numbers."""
        super().__init__(c1, c2, n, e)
        for block in self.m:
            block.attn = Attention_RSC(block.attn)
