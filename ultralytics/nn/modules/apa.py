"""Axis-paired prediction alignment for the b19 DFL-free end-to-end detection head."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.torch_utils import autocast

from .head import Detect


class AxisPairedAlignment(nn.Module):
    """Learn two bounded sampling positions over absolute, feature-grid box boundary pairs."""

    def __init__(self, reg_channels: int, cls_channels: int):
        """Construct independent projections without consuming the native model's initialization RNG."""
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            self.reg_proj = nn.Sequential(nn.Conv2d(reg_channels, 8, 1, bias=True), nn.SiLU())
            self.cls_proj = nn.Sequential(nn.Conv2d(cls_channels, 8, 1, bias=True), nn.SiLU())
            self.offsets = nn.Sequential(nn.Conv2d(16, 16, 1, bias=True), nn.SiLU(), nn.Conv2d(16, 4, 1, bias=True))
            nn.init.zeros_(self.offsets[-1].weight)
            nn.init.zeros_(self.offsets[-1].bias)

    def forward(self, raw: torch.Tensor, reg_features: torch.Tensor, cls_features: torch.Tensor) -> torch.Tensor:
        """Return aligned raw ltrb distances while leaving classification logits untouched."""
        logits = self.offsets(torch.cat((self.cls_proj(cls_features), self.reg_proj(reg_features)), dim=1))
        return self.align(raw, 0.5 * logits.float().tanh())

    @staticmethod
    def align(raw: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        """Sample valid absolute edges in FP32, using a zero-grid difference for exact zero-state identity.

        Args:
            raw (torch.Tensor): Native (l, t, r, b) distances, shaped (B, 4, H, W).
            offsets (torch.Tensor): Bounded (dx_horizontal, dy_horizontal, dx_vertical, dy_vertical).

        Returns:
            (torch.Tensor): Aligned distances in the original dtype and feature-grid units.
        """
        with autocast(enabled=False, device=raw.device.type):
            if not torch.isfinite(raw).all():
                location = (~torch.isfinite(raw)).nonzero()[0].tolist()
                raise ValueError(f"APA received non-finite native raw at [batch, ltrb, row, column]={location}")
            values = raw.float()
            offsets = offsets.float()
            batch, _, height, width = raw.shape
            ax = (
                (torch.arange(width, device=raw.device, dtype=torch.float32) + 0.5).view(1, width).expand(height, width)
            )
            ay = (
                (torch.arange(height, device=raw.device, dtype=torch.float32) + 0.5)
                .view(height, 1)
                .expand(height, width)
            )
            edges = torch.stack((ax - values[:, 0], ay - values[:, 1], ax + values[:, 2], ay + values[:, 3]), dim=1)
            valid = (
                torch.isfinite(edges).all(dim=1, keepdim=True)
                & (edges[:, 2:3] > edges[:, 0:1])
                & (edges[:, 3:4] > edges[:, 1:2])
            )
            zero_grid = torch.stack((2 * ax / width - 1, 2 * ay / height - 1), dim=-1)
            zero_grid = zero_grid.unsqueeze(0).expand(batch, -1, -1, -1)
            deltas, masses = [], []
            for channels, offset in (((0, 2), offsets[:, :2]), ((1, 3), offsets[:, 2:])):
                pair = torch.where(valid, edges[:, channels], 0.0)
                source = torch.cat((pair, valid.float()), dim=1)
                grid = torch.stack((2 * (ax + offset[:, 0]) / width - 1, 2 * (ay + offset[:, 1]) / height - 1), dim=-1)
                sampled = F.grid_sample(source, grid, mode="bilinear", padding_mode="border", align_corners=False)
                reference = F.grid_sample(
                    source, zero_grid, mode="bilinear", padding_mode="border", align_corners=False
                )
                mass, ref_mass = sampled[:, 2:3], reference[:, 2:3]
                aligned = torch.where(mass > 1e-6, sampled[:, :2] / mass.clamp_min(1e-6), 0.0)
                origin = torch.where(ref_mass > 1e-6, reference[:, :2] / ref_mass.clamp_min(1e-6), 0.0)
                deltas.append(aligned - origin)
                masses.append(mass > 1e-6)
            dx, dy = deltas
            delta_raw = torch.stack((-dx[:, 0], -dy[:, 0], dx[:, 1], dy[:, 1]), dim=1)
            candidate = raw + delta_raw.to(raw.dtype)
            check = candidate.float()
            usable = (
                valid
                & masses[0]
                & masses[1]
                & torch.isfinite(check).all(dim=1, keepdim=True)
                & (check[:, 0:1] + check[:, 2:3] > 0)
                & (check[:, 1:2] + check[:, 3:4] > 0)
            )
            return torch.where(usable, candidate, raw)


class DetectAPA(Detect):
    """Preserve native towers, decoding and O2O detach; align each scale and branch independently."""

    def __init__(self, nc: int = 80, reg_max: int = 1, end2end: bool = True, ch: tuple = ()):
        """Build six APA groups beside the existing YOLO26 P3/P4/P5 towers."""
        if reg_max != 1 or not end2end or len(ch) != 3:
            raise ValueError("DetectAPA requires b19 reg_max=1, end2end=True and three detection scales")
        super().__init__(nc, reg_max, end2end, ch)
        self.apa = nn.ModuleList(
            AxisPairedAlignment(r[-1].in_channels, c[-1].in_channels) for r, c in zip(self.cv2, self.cv3)
        )
        self.one2one_apa = nn.ModuleList(
            AxisPairedAlignment(r[-1].in_channels, c[-1].in_channels)
            for r, c in zip(self.one2one_cv2, self.one2one_cv3)
        )

    @property
    def one2many(self):
        """Return native O2M towers and their alignment modules, including the fused empty state."""
        return dict(box_head=self.cv2, cls_head=self.cv3, apa_head=self.apa)

    @property
    def one2one(self):
        """Return the native O2O towers and independently registered alignment modules."""
        return dict(box_head=self.one2one_cv2, cls_head=self.one2one_cv3, apa_head=self.one2one_apa)

    def forward_head(
        self, x: list[torch.Tensor], box_head: nn.Module = None, cls_head: nn.Module = None, apa_head: nn.Module = None
    ) -> dict[str, torch.Tensor]:
        """Run each original tower exactly once, retaining its final hidden features for APA."""
        if box_head is None or cls_head is None:
            return dict()
        batch = x[0].shape[0]
        boxes, scores = [], []
        for feature, reg_tower, cls_tower, apa in zip(x, box_head, cls_head, apa_head):
            reg_feature, cls_feature = feature, feature
            for index in range(len(reg_tower) - 1):
                reg_feature = reg_tower[index](reg_feature)
            for index in range(len(cls_tower) - 1):
                cls_feature = cls_tower[index](cls_feature)
            raw = apa(reg_tower[-1](reg_feature), reg_feature, cls_feature)
            boxes.append(raw.view(batch, 4, -1))
            scores.append(cls_tower[-1](cls_feature).view(batch, self.nc, -1))
        return dict(boxes=torch.cat(boxes, dim=-1), scores=torch.cat(scores, dim=-1), feats=x)

    def fuse(self) -> None:
        """Remove O2M alignment together with the native O2M towers; preserve O2O alignment."""
        super().fuse()
        self.apa = None
