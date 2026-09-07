"""Run the sole b19 SIR-SPPF v2 experiment through explicit, shared training contracts."""

# ruff: noqa: E402 -- Initialize the worktree and native Ultralytics environment before torch.

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.experiments import run_b19_sir_sppf as shared

import torch

from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.modules import SPPF_SIR, SPPF_SIR_V2

MODEL = ROOT / "ultralytics/cfg/models/26/yolo26n-sir-sppf-v2.yaml"
NAME = "yolo26n_b19_d1_sir_sppf_v2"
SOURCE_FILES = tuple(
    ROOT / name
    for name in (
        "tools/experiments/run_b19_sir_sppf_v2.py",
        "tools/experiments/finish_b19_sir_sppf_v2.py",
        "tools/experiments/server_b19_sir_sppf_v2.sh",
    )
)
MODULE_CONFIG = dict(shared.MODULE_CONFIG, version=2, formula="corrected_i = z_i + 0.5*tanh(L_i)*d_i")


class AuditedTrainer(shared.AuditedTrainer):
    """Construct v2 natively and compare the entire initial state against v1 and b19."""

    block_type = SPPF_SIR_V2

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Keep v1 construction outside the candidate's RNG stream; compare all router and BN tensors."""
        with torch.random.fork_rng(devices=[]):
            v1 = DetectionTrainer.get_model(self, str(shared.MODEL), weights, verbose=False)
        candidate = super().get_model(cfg, weights, verbose)
        shared.assert_close_tree(v1.state_dict(), candidate.state_dict(), 0, 0, path="v1_v2_initial_state")
        self.weight_audit["v1_v2_all_state_equal"] = True
        self.weight_audit["block_type"] = self.block_type.__name__
        return candidate


@torch.random.fork_rng(devices=[])
@torch.no_grad()
def independent_correction_check():
    """Activate only r1 and inspect cv2 inputs: v2 must not propagate r1 into scales 2 or 3."""
    v2 = SPPF_SIR_V2(32, 32, 5, 3, True).eval()
    v1 = SPPF_SIR(32, 32, 5, 3, True).eval()
    v2.router[-1].bias[:16].fill_(0.75)
    v1.load_state_dict(v2.state_dict())
    x = torch.randn(2, 32, 17, 23)
    raw = [v2.cv1(x)]
    for _ in range(3):
        raw.append(v2.m(raw[-1]))
    r1 = 0.5 * torch.tensor(0.75).tanh() * (raw[1] - raw[0])
    assert torch.count_nonzero(r1) > 0
    observed = []
    for block in (v1, v2):
        hook = block.cv2.register_forward_pre_hook(lambda module, args: observed.append(args[0].clone()))
        try:
            block(x)
        finally:
            hook.remove()
    for i, (old, new) in enumerate(zip(observed[0].chunk(4, 1), observed[1].chunk(4, 1))):
        shared.assert_close_tree(raw[i] if i == 0 else raw[i] + r1, old, 0, 0)
        shared.assert_close_tree(raw[i] + r1 if i == 1 else raw[i], new, 0, 0)
    assert not torch.equal(observed[0], observed[1])
    return dict(passed=True, only_r1_nonzero=True, v2_scales_2_3_equal_raw=True, v1_scales_2_3_include_r1=True)


def structural_checks(directory, model=MODEL, block_type=SPPF_SIR_V2):
    """Check the full single-class nano graph and an independently observable nonzero correction."""
    assert model == MODEL and block_type is SPPF_SIR_V2
    report = shared.structural_checks(directory, model, block_type, nc=1)
    report["independent_correction"] = independent_correction_check()
    shared.write_json(Path(directory) / "structural.json", report)
    return report


def main(argv=None):
    """Bind every version-specific default explicitly, including the preflight child entrypoint."""
    return shared.main(
        argv,
        model=MODEL,
        trainer_type=AuditedTrainer,
        entrypoint=Path(__file__).resolve(),
        name=NAME,
        source_files=SOURCE_FILES,
        structure_check=structural_checks,
        module_config=MODULE_CONFIG,
    )


if __name__ == "__main__":
    raise SystemExit(main())
