"""Run the fixed v2 using the verified b19 trainer and independent preflight lifecycle."""

# ruff: noqa: E402 -- Direct entry must import this worktree.
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.experiments import run_b19_dcs_sppf as shared
from ultralytics.nn.modules import DCS_SPPF_V2

NAME = "yolo26n_b19_dcs_sppf_v2"
MODEL = ROOT / "ultralytics/cfg/models/26/yolo26n-dcs-sppf-v2.yaml"


class AuditedTrainer(shared.AuditedTrainer):
    """Bind the existing audited native trainer to the exact v2 class and YAML."""

    block_type = DCS_SPPF_V2
    model_yaml = MODEL


if __name__ == "__main__":
    shared.main(NAME, MODEL, AuditedTrainer, "verify_b19_dcs_sppf_v2.py")
