"""Evaluate, diagnose and archive v2 through the verified shared lifecycle, retaining v1 history."""

# ruff: noqa: E402 -- Direct entry must import this worktree.
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.experiments import finish_b19_dcs_sppf as shared
from tools.experiments.run_b19_dcs_sppf_v2 import MODEL, NAME
from ultralytics.nn.modules import DCS_SPPF_V2
from ultralytics.nn.modules.dcs_sppf_v2 import relative_residual


def residual_control(raw, native, block):
    """Use the same controller and checkpoint constants as the real forward."""
    return relative_residual(raw, native, block.residual_budget, block.residual_eps)


if __name__ == "__main__":
    historical = json.loads(
        (ROOT / "docs/experiments/evidence/dcs_sppf_v2/v1_evidence.json").read_text(encoding="utf-8")
    )
    shared.main(
        name=NAME, model=MODEL, block_type=DCS_SPPF_V2, residual_control=residual_control, historical=historical
    )
