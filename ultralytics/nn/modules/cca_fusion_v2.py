"""Reliability-constrained local CCA, retaining the v1 projections, indexing and initialization."""

import torch

from ultralytics.utils.torch_utils import autocast

from .cca_fusion import Concat_CCA_Fusion

CENTER_PRIOR = 1.0
RELIABILITY_FLOOR = 0.25
RESIDUAL_SCALE = 0.50


class Concat_CCA_Fusion_V2(Concat_CCA_Fusion):
    """Respect the nearest parent and detach entropy confidence before scaling the CCA residual."""

    def correspondence_scores(self, scores):
        """Add a fixed center-index-4 prior out of place, preserving autograd and the border mask."""
        prior = scores.new_tensor([0, 0, 0, 0, CENTER_PRIOR, 0, 0, 0, 0]).view(1, 9, 1, 1, 1, 1)
        return scores + prior

    @staticmethod
    def reliability(weights):
        """Return entropy, normalized entropy, confidence, detached gate and geometric valid counts."""
        with autocast(False, device=weights.device.type):
            weights = weights.float()
            h, w = weights.shape[-2:]
            y = torch.arange(h, device=weights.device) // 2
            x = torch.arange(w, device=weights.device) // 2
            ny = 1 + (y > 0).float() + (y < h // 2 - 1).float()
            nx = 1 + (x > 0).float() + (x < w // 2 - 1).float()
            valid_count = (ny[:, None] * nx[None, :])[None, None]
            entropy = -(weights * weights.clamp_min(1e-6).log()).sum(1, keepdim=True)
            normalized = torch.where(
                valid_count > 1, entropy / valid_count.clamp_min(2).log(), torch.zeros_like(entropy)
            ).clamp(0, 1)
            confidence = (1 - normalized).clamp(0, 1)
            gate = RELIABILITY_FLOOR + (1 - RELIABILITY_FLOOR) * confidence.detach()
        return entropy, normalized, confidence, gate, valid_count

    def residual(self, delta, weights):
        """Keep the weights-to-delta gradient path while applying 0.50 * detached reliability."""
        raw = self.Wo(delta)
        gate = self.reliability(weights)[3]
        with autocast(False, device=raw.device.type):
            residual = RESIDUAL_SCALE * gate * raw.float()
        return residual.to(raw.dtype)
