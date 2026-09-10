"""Independent CCA v2 val/test, diagnosis and packaging entry; all evaluation rules are shared with v1."""

# ruff: noqa: E402 -- Resolve this worktree before package imports.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.experiments.finish_b19_cca_fusion import main as finish
from tools.experiments.run_b19_cca_fusion_v2 import EXPERIMENT


def main(argv=None):
    """Bind the validated completion infrastructure to the v2 experiment identity."""
    return finish(argv, experiment=EXPERIMENT)


if __name__ == "__main__":
    main()
