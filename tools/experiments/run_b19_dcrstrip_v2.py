"""Run the sole b19 DCR-Strip v2 candidate through the shared native-trainer audits."""

# ruff: noqa: E402 -- Establish worktree imports before loading Ultralytics.

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.experiments import run_b19_dcrstrip as shared
from ultralytics.nn.modules import C3k2_DCRStripV2
from ultralytics.utils.torch_utils import unwrap_model

MODEL = ROOT / "ultralytics/cfg/models/26/yolo26n-dcrstrip-v2.yaml"
NAME = "yolo26n_b19_a1_dcrstrip_v2"
MODULE_CONFIG = dict(
    k=7,
    r=1,
    reduction="max(8,ceil(C/32)*8)",
    alpha=0.05,
    beta=0.25,
    temperature=1.0,
    enabled=True,
    layer=4,
    scale="n",
    directions=["H", "V", "D", "AD"],
    fusion="mean(T) + sigmoid(beta_raw) * sum(softmax(u/(u+v+1e-6)) * C)",
)


def record_scalars(trainer, event, epoch):
    """Read only scalar parameters; do not forward data or consume any random numbers."""
    block = unwrap_model(trainer.model).model[4].dcr
    record = dict(event=event, epoch=epoch, alpha=block.alpha.item(), beta=block.beta.item())
    with (trainer.save_dir / "dcr_v2_scalars.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record) + "\n")


def record_start(trainer):
    """Record the fresh initialized model after the final native optimizer audit."""
    record_scalars(trainer, "start", 0)


def record_epoch(trainer):
    """Sample the requested 50/100/150 epoch checkpoints without extra inference."""
    if trainer.epoch + 1 in (50, 100, 150):
        record_scalars(trainer, "epoch", trainer.epoch + 1)


def record_end(trainer):
    """Record final live-model scalars; best-EMA diagnostics are a separate explicit command."""
    record_scalars(trainer, "end_live_model", trainer.epoch + 1)


def main(argv=None):
    """Keep v1 CLI semantics while binding every model-specific audit and child process to v2."""
    return shared.main(
        argv,
        model=MODEL,
        block_type=C3k2_DCRStripV2,
        module_config=MODULE_CONFIG,
        entrypoint=__file__,
        name=NAME,
        callbacks={
            "on_pretrain_routine_end": record_start,
            "on_train_epoch_end": record_epoch,
            "on_train_end": record_end,
        },
    )


if __name__ == "__main__":
    raise SystemExit(main())
