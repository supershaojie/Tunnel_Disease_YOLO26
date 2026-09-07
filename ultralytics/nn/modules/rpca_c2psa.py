"""Region-probability calibrated dense attention, retaining native fine-grained values and PE."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .block import C2PSA, Attention, PSABlock


def region_group(x, height, width):
    """Group the final spatial token axis into genuine 2x2 regions, padding only temporary storage."""
    x = F.pad(x.reshape(*x.shape[:-1], height, width), (0, width % 2, 0, height % 2))
    return (
        x.reshape(*x.shape[:-2], (height + 1) // 2, 2, (width + 1) // 2, 2)
        .transpose(-3, -2)
        .flatten(-2)
        .flatten(-3, -2)
    )


def region_ungroup(x, height, width):
    """Restore row-major pixels and discard the temporary boundary padding."""
    x = x.reshape(*x.shape[:-2], (height + 1) // 2, (width + 1) // 2, 2, 2).transpose(-3, -2)
    return x.reshape(*x.shape[:-4], 2 * ((height + 1) // 2), 2 * ((width + 1) // 2))[..., :height, :width].flatten(-2)


def calibrated_probabilities(logits, gamma, height, width, diagnostics=False):
    """Compute FP32 A'=(1-gamma)A+gamma*B*C; optional small-sample details retain autograd."""
    with torch.autocast(device_type=logits.device.type, enabled=False):
        logits, gamma = logits.float(), gamma.float()
        grouped = region_group(logits, height, width)
        valid = region_group(torch.ones(height * width, device=logits.device), height, width).bool()
        area = valid.sum(-1).float()
        coarse = (grouped.sum(-1) / area + area.log()).softmax(-1)
        conditional = grouped.masked_fill(~valid, -torch.inf).softmax(-1)
        original = logits.softmax(-1)
        routed = region_ungroup(coarse.unsqueeze(-1) * conditional, height, width)
        calibrated = (1 - gamma) * original + gamma * routed
        if diagnostics:
            mass = region_group(original, height, width).sum(-1)
            return calibrated, dict(
                A=original,
                B=coarse,
                C=conditional,
                P=mass,
                M=(1 - gamma) * mass + gamma * coarse,
                area=area,
                valid=valid,
            )
        return calibrated


class Attention_RPCA(Attention):
    """Native QKV/PE/projection with an explicit original-attention diagnostic bypass."""

    def __init__(self, dim, num_heads=8, attn_ratio=0.5):
        """Keep native construction and parameter names; calibration is enabled by default."""
        super().__init__(dim, num_heads, attn_ratio)
        self.enabled = True

    def forward(self, x, gamma=None):
        """Aggregate original V in FP32, then restore its dtype before native spatial PE and projection."""
        if not self.enabled:
            return super().forward(x)
        batch, channels, height, width = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.view(batch, self.num_heads, 2 * self.key_dim + self.head_dim, height * width).split(
            (self.key_dim, self.key_dim, self.head_dim), dim=2
        )
        with torch.autocast(device_type=x.device.type, enabled=False):
            logits = (q.float() * self.scale).transpose(-2, -1) @ k.float()
            probabilities = calibrated_probabilities(logits, gamma, height, width)
            attended = (v.float() @ probabilities.transpose(-2, -1)).reshape(batch, channels, height, width)
        return self.proj(attended.to(v.dtype) + self.pe(v.reshape(batch, channels, height, width)))


class PSABlock_RPCA(PSABlock):
    """Native attention/FFN residual order, with a distinct gate reading the C2PSA bypass half."""

    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True):
        """Reuse native state and isolate all added or replacement construction from the main CPU RNG."""
        super().__init__(c, attn_ratio, num_heads, shortcut)
        with torch.random.fork_rng(devices=[]):
            attention = Attention_RPCA(c, num_heads, attn_ratio)
            attention.load_state_dict(self.attn.state_dict())
            self.attn = attention
            self.gate = nn.Sequential(
                nn.Conv2d(c, 16, 1, bias=True),
                nn.SiLU(),
                nn.Conv2d(16, 16, 3, padding=1, groups=16, bias=True),
                nn.SiLU(),
                nn.Conv2d(16, num_heads, 1, bias=True),
            )
            nn.init.zeros_(self.gate[-1].weight)
            nn.init.constant_(self.gate[-1].bias, -math.log(9))

    def forward(self, x, bypass):
        """Read the same deep bypass features in every block, without detaching any gradient path."""
        gamma = 0.5 * self.gate(bypass).float().sigmoid().flatten(2).unsqueeze(-1)
        attended = self.attn(x, gamma)
        x = x + attended if self.add else attended
        return x + self.ffn(x) if self.add else self.ffn(x)


class C2PSA_RPCA(C2PSA):
    """Replace only C2PSA attention; preserve cv1/cv2, repetitions, native state names and RNG sequence."""

    def __init__(self, c1, c2, n=1, e=0.5):
        """Construct the complete native block first and transfer its state to serializable RPCA blocks."""
        super().__init__(c1, c2, n, e)
        with torch.random.fork_rng(devices=[]):
            for index, native in enumerate(self.m):
                block = PSABlock_RPCA(self.c, 0.5, self.c // 64, native.add)
                result = block.load_state_dict(native.state_dict(), strict=False)
                assert set(result.missing_keys) == {f"gate.{i}.{p}" for i in (0, 2, 4) for p in ("weight", "bias")}
                assert not result.unexpected_keys
                self.m[index] = block

    def forward(self, x):
        """Keep a as the native bypass and supply it to every independently gated PSA block."""
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        for block in self.m:
            b = block(b, a)
        return self.cv2(torch.cat((a, b), 1))
