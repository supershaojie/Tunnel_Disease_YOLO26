"""Bounded reciprocal-imbalance logit correction for native dense YOLO26 attention."""

from .block import Attention, C2PSA
from .rsc_c2psa import Attention_RSC

__all__ = ("Attention_RSC_V2", "C2PSA_RSC_V2")


def reciprocal_logit_correction(scores, theta):
    """Return FP32 corrected probabilities, antisymmetric imbalance and bounded per-head logit changes.

    Args:
        scores (torch.Tensor): Native scaled logits of shape (batch, heads, tokens, tokens).
        theta (torch.Tensor): One learnable scalar per head.
    """
    logits = scores.float()
    log_a = logits.log_softmax(dim=-1)
    imbalance = (0.5 * (log_a.transpose(-2, -1) - log_a)).tanh()
    beta = 0.2 * theta.float().sigmoid()
    delta_logits = beta.view(1, theta.numel(), 1, 1) * imbalance
    return (logits + delta_logits).softmax(dim=-1), imbalance, delta_logits


class Attention_RSC_V2(Attention_RSC):
    """Reuse v1's RNG-free native tensor transfer and theta initialization with a distinct forward and class path.

    This bounded correction is an engineering hypothesis, not a guarantee of better detection or a novelty claim.
    Dense quadratic attention and the native QKV/PE/projection parameter paths are retained.
    """

    def forward(self, x):
        """Compute native logits in their original order, correct in FP32, then resume native value aggregation."""
        if not self.enabled:
            return Attention.forward(self, x)
        B, C, H, W = x.shape
        q, k, v = (
            self.qkv(x)
            .view(B, self.num_heads, self.key_dim * 2 + self.head_dim, H * W)
            .split([self.key_dim, self.key_dim, self.head_dim], dim=2)
        )
        scores = (q * self.scale).transpose(-2, -1) @ k
        attention, _, _ = reciprocal_logit_correction(scores, self.theta)
        x = (v @ attention.to(v.dtype).transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        return self.proj(x)


class C2PSA_RSC_V2(C2PSA):
    """Keep native C2PSA initialization, bypass, concat, FFN and residuals; replace only its attention calculation."""

    def __init__(self, c1, c2, n=1, e=0.5):
        """Construct the native block once; adding constant theta tensors consumes no random numbers."""
        super().__init__(c1, c2, n, e)
        for block in self.m:
            block.attn = Attention_RSC_V2(block.attn)
