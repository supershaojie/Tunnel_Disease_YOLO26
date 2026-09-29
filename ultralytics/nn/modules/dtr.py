"""Dual-representative region interaction for the isolated b19 DTR-C2PSA experiment."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.torch_utils import autocast

from .conv import Conv


class ChannelLayerNorm(nn.Module):
    """Normalize channels at each NCHW position using optimizer-recognizable LayerNorm parameters."""

    def __init__(self, channels):
        """Initialize affine channel normalization with epsilon 1e-6."""
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=1e-6)

    def forward(self, x):
        """Preserve the spatial dimensions and normalize only the final, permuted channel axis."""
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class DualRegionTokenizer(nn.Module):
    """Pool interleaved mean/content representatives from disjoint, dynamically sized regions."""

    def __init__(self, channels, region_grid=5):
        """Create the independent content scorer without batch statistics or hard routing."""
        super().__init__()
        self.region_grid = region_grid
        self.avg = nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False)
        self.score_in = nn.Conv2d(2 * channels, 16, 1, bias=False)
        self.score_dw = nn.Conv2d(16, 16, 3, padding=1, groups=16, bias=False)
        self.score_out = nn.Conv2d(16, 1, 1, bias=True)
        nn.init.normal_(self.score_out.weight, std=0.01)
        nn.init.zeros_(self.score_out.bias)

    def regions(self, height, width):
        """Return row-major half-open (y0, y1, x0, x1) bounds with exact non-overlapping coverage."""
        gh, gw = min(self.region_grid, height), min(self.region_grid, width)
        return [
            (r * height // gh, (r + 1) * height // gh, c * width // gw, (c + 1) * width // gw)
            for r in range(gh)
            for c in range(gw)
        ]

    def pool(self, u, scores, regions, diagnostics=False):
        """Reduce each region in FP32, retaining gradients to both scores and normalized features."""
        tokens, entropies, differences = [], [], []
        with autocast(False, u.device.type):
            for y0, y1, x0, x1 in regions:
                values = u[:, :, y0:y1, x0:x1].flatten(2).float()
                weights = scores[:, :, y0:y1, x0:x1].flatten(2).float().softmax(-1)
                mean = values.mean(-1)
                detail = (values * weights).sum(-1)
                tokens.extend((mean, detail))
                if diagnostics:
                    n = values.shape[-1]
                    entropy = -(weights * weights.clamp_min(1e-30).log()).sum(-1)
                    entropies.append(entropy / math.log(n) if n > 1 else torch.zeros_like(entropy))
                    differences.append((detail - mean).norm(dim=1) / mean.norm(dim=1).clamp_min(1e-6))
        stats = {}
        if diagnostics:
            stats = {
                "region_entropy_mean": torch.stack(entropies).detach().mean().item(),
                "mean_detail_relative_difference": torch.stack(differences).detach().mean().item(),
                "singleton_regions": sum((y1 - y0) * (x1 - x0) == 1 for y0, y1, x0, x1 in regions),
            }
        return torch.stack(tokens, dim=1).to(u.dtype), stats

    def forward(self, u, diagnostics=False):
        """Predict region-local scores and return representatives, geometry, and optional aggregate statistics."""
        difference = u - self.avg(u)
        scores = self.score_out(F.silu(self.score_dw(F.silu(self.score_in(torch.cat((u, difference), 1))))))
        regions = self.regions(*u.shape[-2:])
        tokens, stats = self.pool(u, scores, regions, diagnostics)
        return tokens, regions, stats


class DTRBlock(nn.Module):
    """Apply local 3/5/7 extraction, region attention, gated context, and a convolutional GLU."""

    def __init__(self, channels=128, heads=2, region_grid=5):
        """Initialize one complete interaction block; all parameters are independent between blocks."""
        super().__init__()
        if channels % 4 or channels % heads or region_grid < 1:
            raise ValueError("DTR requires channels divisible by four and heads, and a positive region grid")
        self.heads, self.head_dim = heads, channels // heads
        self.splits = (channels // 4, channels // 4, channels // 2)
        self.local_in = Conv(channels, channels, 1)
        self.local_dw = nn.ModuleList(Conv(c, c, k, g=c) for c, k in zip(self.splits, (3, 5, 7)))
        self.local_out = Conv(channels, channels, 1, act=False)
        self.ln_attn = ChannelLayerNorm(channels)
        self.tokenizer = DualRegionTokenizer(channels, region_grid)
        self.q_proj = nn.Conv2d(channels, channels, 1, bias=False)
        self.k_proj = nn.Linear(channels, channels, bias=False)
        self.v_proj = nn.Linear(channels, channels, bias=False)
        self.ctx_proj = Conv(channels, channels, 1, act=False)
        self.pos_mlp = nn.Sequential(nn.Linear(2, 16), nn.SiLU(), nn.Linear(16, heads))
        nn.init.zeros_(self.pos_mlp[-1].weight)
        nn.init.zeros_(self.pos_mlp[-1].bias)
        self.type_bias = nn.Parameter(torch.zeros(heads, 2))
        self.context_gate = nn.Sequential(
            nn.Conv2d(2 * channels, 16, 1, bias=False), nn.SiLU(), nn.Conv2d(16, channels, 1, bias=True)
        )
        nn.init.zeros_(self.context_gate[-1].weight)
        nn.init.zeros_(self.context_gate[-1].bias)
        self.ln_ffn = ChannelLayerNorm(channels)
        self.ffn_in = nn.Conv2d(channels, 4 * channels, 1, bias=True)
        self.ffn_dw = nn.Conv2d(2 * channels, 2 * channels, 3, padding=1, groups=2 * channels, bias=True)
        self.ffn_out = nn.Conv2d(2 * channels, channels, 1, bias=True)
        self.diagnostics_enabled = False
        self.diagnostics = {}

    def position_bias(self, height, width, regions, device):
        """Compute shared head/query/token bias in differentiable FP32, including for pure-half models."""
        with autocast(False, device.type):
            y, x = torch.meshgrid(
                torch.arange(height, device=device, dtype=torch.float32),
                torch.arange(width, device=device, dtype=torch.float32),
                indexing="ij",
            )
            queries = torch.stack((x.flatten(), y.flatten()), -1)
            centers = torch.tensor(
                [((x0 + x1 - 1) / 2, (y0 + y1 - 1) / 2) for y0, y1, x0, x1 in regions],
                device=device,
                dtype=torch.float32,
            ).repeat_interleave(2, 0)
            relative = (centers[None] - queries[:, None]) / max(height - 1, width - 1, 1)
            first, last = self.pos_mlp[0], self.pos_mlp[-1]
            hidden = F.silu(F.linear(relative, first.weight.float(), first.bias.float()))
            bias = F.linear(hidden, last.weight.float(), last.bias.float()).permute(2, 0, 1)
            return bias + self.type_bias.float().repeat(1, len(regions))[:, None, :]

    def forward(self, z):
        """Run both the direct spatial path and all learned region/context paths in train and inference."""
        batch, channels, height, width = z.shape
        parts = self.local_in(z).split(self.splits, dim=1)
        zl = z + self.local_out(torch.cat([conv(x) for conv, x in zip(self.local_dw, parts)], 1))
        u = self.ln_attn(zl)
        tokens, regions, stats = self.tokenizer(u, self.diagnostics_enabled)
        q = self.q_proj(u).reshape(batch, self.heads, self.head_dim, height * width).transpose(2, 3)
        k = self.k_proj(tokens).reshape(batch, -1, self.heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(tokens).reshape(batch, -1, self.heads, self.head_dim).transpose(1, 2)
        logits = (q * self.head_dim**-0.5) @ k.transpose(-2, -1)
        with autocast(False, z.device.type):
            attention = (logits.float() + self.position_bias(height, width, regions, z.device)).softmax(-1)
        context = (attention.to(v.dtype) @ v).transpose(2, 3).reshape(batch, channels, height, width)
        context = self.ctx_proj(context)
        gate_logits = self.context_gate(torch.cat((zl, context), 1))
        with autocast(False, z.device.type):
            gate = gate_logits.float().sigmoid().to(context.dtype)
        f = zl + gate * context
        a, b = self.ffn_in(self.ln_ffn(f)).chunk(2, dim=1)
        output = f + self.ffn_out(F.gelu(self.ffn_dw(a)) * b)
        if self.diagnostics_enabled:
            with torch.no_grad():
                gate_float = gate.float()
                self.diagnostics = {
                    **stats,
                    "attention_shape": list(attention.shape),
                    "attention_finite": bool(torch.isfinite(attention).all()),
                    "attention_mass_mean": attention[..., 0::2].sum(-1).mean((0, 2)).tolist(),
                    "attention_mass_detail": attention[..., 1::2].sum(-1).mean((0, 2)).tolist(),
                    "gate_mean": gate_float.mean().item(),
                    "gate_std": gate_float.std(unbiased=False).item(),
                    "gate_min": gate_float.min().item(),
                    "gate_max": gate_float.max().item(),
                }
        return output


class C2PSA_DTR(nn.Module):
    """Retain the native split/fuse convolutions and replace the entire PSA processing branch."""

    def __init__(self, c1, c2, e=0.5, blocks=2, heads=2, region_grid=5):
        """Keep fixed internal block count separate from parser depth scaling."""
        super().__init__()
        if c1 != c2:
            raise ValueError("C2PSA_DTR requires equal input/output channels")
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)
        self.dtr_blocks = nn.Sequential(*(DTRBlock(self.c, heads, region_grid) for _ in range(blocks)))

    def forward(self, x):
        """Process the second split through both blocks before the original outer fusion."""
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        return self.cv2(torch.cat((a, self.dtr_blocks(b)), dim=1))
