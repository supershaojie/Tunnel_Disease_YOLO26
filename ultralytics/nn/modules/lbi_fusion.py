"""Local normalized multiplicative interaction on the detail half of a two-input concatenation."""

import torch
from torch import nn

from ultralytics.utils.torch_utils import autocast

__all__ = ("Concat_LBI_Fusion",)


class Concat_LBI_Fusion(nn.Module):
    """Preserve S and add a zero-initialized rank-16 local residual to L before concatenation.

    Args:
        channels (list[int]): Ordered input channels [c_s, c_l].
        rank (int): Fixed interaction width, 16.
        eps (float): Fixed channel RMS stabilizer inside the square root, 1e-4.
        dim (int): Fixed concatenation dimension, 1.
    """

    def __init__(self, channels, rank=16, eps=1e-4, dim=1):
        """Isolate only new Conv2d initialization from the subsequent native layer RNG stream."""
        super().__init__()
        if len(channels) != 2 or any(not isinstance(c, int) or c <= 0 for c in channels):
            raise ValueError("Concat_LBI_Fusion requires ordered positive channels [c_s, c_l]")
        if rank != 16 or eps != 1e-4 or dim != 1:
            raise ValueError("LBI-Fusion v1 fixes rank=16, eps=1e-4 and dim=1")
        self.channels = tuple(channels)
        self.rank, self.eps, self.dim = rank, eps, dim
        c_s, c_l = channels
        with torch.random.fork_rng(devices=[]):
            self.proj_l = nn.Conv2d(c_l, rank, 1, bias=False)
            self.proj_s = nn.Conv2d(c_s, rank, 1, bias=False)
            self.dw = nn.Conv2d(rank, rank, 3, 1, 1, groups=rank, bias=False)
            self.out = nn.Conv2d(rank, c_l, 1, bias=False)
            nn.init.zeros_(self.out.weight)

    def forward(self, inputs):
        """Return cat(S, L + R), with only RMS statistics and interaction explicitly in FP32."""
        if not isinstance(inputs, (tuple, list)) or len(inputs) != 2:
            raise ValueError("Concat_LBI_Fusion expects [S from layer14, L from layer4], exactly two tensors")
        semantic, detail = inputs
        if any(not isinstance(x, torch.Tensor) or x.ndim != 4 for x in inputs):
            raise ValueError("Concat_LBI_Fusion inputs must be NCHW tensors in [S, L] order")
        if (semantic.shape[1], detail.shape[1]) != self.channels:
            raise ValueError(f"Expected [S,L] channels {self.channels}, got {(semantic.shape[1], detail.shape[1])}")
        if (semantic.shape[0], *semantic.shape[2:]) != (detail.shape[0], *detail.shape[2:]):
            raise ValueError(f"S/L batch and spatial dimensions must match: S={semantic.shape}, L={detail.shape}")
        if semantic.device != detail.device or semantic.device != self.proj_s.weight.device:
            raise ValueError(
                f"S/L/module devices must match: {semantic.device}, {detail.device}, {self.proj_s.weight.device}"
            )
        if not semantic.is_floating_point() or not detail.is_floating_point():
            raise ValueError(f"S/L require floating dtypes: {semantic.dtype}, {detail.dtype}")
        u, v = self.proj_l(detail), self.proj_s(semantic)
        with autocast(enabled=False, device=u.device.type):
            uf, vf = u.float(), v.float()
            un = uf * torch.rsqrt(uf.square().mean(dim=1, keepdim=True) + self.eps)
            vn = vf * torch.rsqrt(vf.square().mean(dim=1, keepdim=True) + self.eps)
            interaction = un * vn
        # Functional SiLU keeps inplace=False across native initialize_weights/load_checkpoint policy updates.
        t = nn.functional.silu(self.dw(interaction.to(dtype=u.dtype)), inplace=False)
        r = self.out(t).to(dtype=detail.dtype)
        return torch.cat([semantic, detail + r], dim=self.dim)
