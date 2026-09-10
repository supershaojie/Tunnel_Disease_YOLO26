"""Fixed local cross-scale correspondence correction for the first P5/P4 concatenation."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.torch_utils import autocast


class Concat_CCA_Fusion(nn.Module):
    """Return cat(U + Wo(sum(a * (V_neighbor - V_parent))), L) for inputs [U, L, H]."""

    def __init__(self, cU, cL, cH):
        """Register four bias-free projections without consuming the native initialization RNG."""
        super().__init__()
        assert cU == cH
        # Explicit CPU construction also preserves CUDA RNG when the caller uses a default CUDA device.
        with torch.random.fork_rng(devices=[]):
            self.Wq = nn.Conv2d(cL, 16, 1, bias=False, device="cpu")
            self.Wk = nn.Conv2d(cH, 16, 1, bias=False, device="cpu")
            self.Wv = nn.Conv2d(cH, 16, 1, bias=False, device="cpu")
            self.Wo = nn.Conv2d(16, cU, 1, bias=False, device="cpu")
            nn.init.zeros_(self.Wo.weight)

    def correspondence(self, low, high):
        """Compute FP32 nine-neighbor probabilities and explicit parent-referenced value differences."""
        q, k, v = self.Wq(low), self.Wk(high), self.Wv(high)
        b, r, h, w = v.shape
        with autocast(False, device=high.device.type):
            qhat = F.normalize(q.float(), dim=1, eps=1e-6)
            khat = F.normalize(k.float(), dim=1, eps=1e-6)
            # Keep neighbors at coarse resolution; broadcast four fine phases without a dense N x N matrix.
            neighbors = F.unfold(khat, 3, padding=1).reshape(b, r, 9, h, w)
            queries = qhat.reshape(b, r, h, 2, w, 2)
            scores = 4.0 * (queries.unsqueeze(2) * neighbors[:, :, :, :, None, :, None]).sum(1)
            valid = F.unfold(v.new_ones(1, 1, h, w, dtype=torch.float32), 3, padding=1)
            valid = valid.reshape(1, 9, h, 1, w, 1).bool()
            weights = self.correspondence_scores(scores).masked_fill(~valid, -torch.inf).softmax(1)
            values = F.unfold(v.float(), 3, padding=1).reshape(b, r, 9, h, w)
            differences = values - v.float().unsqueeze(2)
            delta = (weights.unsqueeze(1) * differences[:, :, :, :, None, :, None]).sum(2)
            delta = delta.reshape(b, r, 2 * h, 2 * w)
        return delta.to(v.dtype), weights.reshape(b, 9, 2 * h, 2 * w), q, k

    def correspondence_scores(self, scores):
        """Use the original cosine scores without a spatial prior in v1."""
        return scores

    def residual(self, delta, weights):
        """Apply the original ungated value-difference projection in v1."""
        return self.Wo(delta)

    def forward(self, inputs):
        """Preserve supplied nearest features and concatenation order; only the semantic branch is corrected."""
        up, low, high = inputs
        assert up.shape[0] == low.shape[0] == high.shape[0]
        assert up.shape[1] == high.shape[1] == self.Wo.out_channels
        assert low.shape[1] == self.Wq.in_channels
        assert up.shape[-2:] == low.shape[-2:] == (2 * high.shape[-2], 2 * high.shape[-1])
        delta, weights, _, _ = self.correspondence(low, high)
        residual = self.residual(delta, weights)
        return torch.cat((up + residual.to(up.dtype), low), 1)
