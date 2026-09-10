"""Verify v2 with the shared synthetic lifecycle suite; never claim server batch32 preflight."""

import argparse
from pathlib import Path

from tools.experiments.run_b19_cca_fusion_v2 import AuditedTrainer, EXPERIMENT
from tools.experiments.verify_b19_cca_fusion import verify as verify_lifecycle


def verify(weights, output):
    """Exercise structural, shared-weight, gradient, EMA, new-process reload, fuse and Validator checks."""
    return verify_lifecycle(weights, output, experiment=EXPERIMENT, trainer_type=AuditedTrainer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    verify(options.weights.resolve(), options.output.resolve())
