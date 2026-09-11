"""DCS v1 features with a smooth per-image bound on the actual injected residual."""

import torch

from .dcs_sppf import DCS_SPPF


def relative_residual(
    raw: torch.Tensor, reference: torch.Tensor, residual_budget: float = 0.05, residual_eps: float = 1e-6
):
    """Return the bounded residual and one scale per image; detach only the reference budget.

    The caller supplies FP32 tensors for inference/training. Preserving the input dtype also permits an
    independent FP64 gradient check of this stateless arithmetic. The bound applies to C,H,W together.

    Args:
        raw (torch.Tensor): Residual before the norm controller, shaped B,C,H,W.
        reference (torch.Tensor): Native output supplying the detached per-image budget.
        residual_budget (float): Relative RMS budget.
        residual_eps (float): Positive denominator stabilizer, squared inside the square root.

    Returns:
        (tuple[torch.Tensor, torch.Tensor]): Injection and nonnegative scale of shape B,1,1,1.
    """
    budget = residual_budget * reference.detach().square().mean(dim=(1, 2, 3), keepdim=True).sqrt()
    raw_ms = raw.square().mean(dim=(1, 2, 3), keepdim=True)
    scale = budget / (budget.square() + raw_ms + residual_eps**2).sqrt()
    return scale * raw, scale


class DCS_SPPF_V2(DCS_SPPF):
    """Keep all v1 parameters and limit the FP32 injection to 5% of each native output's L2 norm."""

    def __init__(
        self,
        c1: int,
        c2: int,
        k: int = 5,
        n: int = 3,
        shortcut: bool = False,
        residual_budget: float = 0.05,
        residual_eps: float = 1e-6,
    ):
        """Reuse the native/v1 initialization and accept only the fixed formal architecture constants."""
        if residual_budget != 0.05 or residual_eps != 1e-6:
            raise ValueError("DCS v2 formal architecture constants are fixed")
        super().__init__(c1, c2, k, n, shortcut)
        self.residual_budget = float(residual_budget)
        self.residual_eps = float(residual_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Execute cv1 once, preserve v1 contrasts/BN, and bound the residual in a local FP32 region."""
        levels = [self.cv1(x)]
        levels.extend(self.m(levels[-1]) for _ in range(self.n))
        y0 = self.cv2(torch.cat(levels, dim=1))
        y0 = y0 + x if self.add else y0
        refined = [
            branch(maximum - average(levels[0])) for maximum, average, branch in zip(levels[1:], self.avg, self.refine)
        ]
        features = self.fuse(torch.cat(refined, dim=1))
        # Meta has no autocast backend; its shape-only execution uses the CPU policy without allocating data.
        device_type = x.device.type if x.device.type != "meta" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            raw = (0.10 * self.theta.float().tanh()) * features.float()
            injected, _ = relative_residual(raw, y0.float(), self.residual_budget, self.residual_eps)
        return y0 + injected.to(dtype=y0.dtype)
