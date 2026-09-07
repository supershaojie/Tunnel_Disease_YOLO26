"""Run the single b19 + RPCA-C2PSA v1 candidate through the audited native b19 trainer."""

# ruff: noqa: E402 -- Import the worktree and offline settings before torch.

import copy
import math
import sys
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.experiments import run_b19_sir_sppf as shared

import torch

from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.modules import C2PSA_RPCA, C3k2, SPPF
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import YAML

MODEL = ROOT / "ultralytics/cfg/models/26/yolo26n-rpca-c2psa-v1.yaml"
NAME = "yolo26n_b19_e1_rpca_c2psa_v1"
SOURCE_FILES = tuple(
    ROOT / name
    for name in (
        "tools/experiments/run_b19_rpca_c2psa.py",
        "tools/experiments/finish_b19_rpca_c2psa.py",
        "tools/experiments/finish_b19_sir_sppf_v2.py",
        "tools/experiments/server_b19_rpca_c2psa_v1.sh",
        "tools/experiments/server_b19_sir_sppf_v2.sh",
        "docs/experiments/b19_rpca_c2psa_v1.md",
    )
)
MODULE_CONFIG = dict(
    version=1,
    layer=10,
    scale="n",
    region=[2, 2],
    gate_channels=16,
    gamma="0.5*sigmoid(G(a))",
    gamma_initial=0.05,
    formula="A'=(1-gamma)*A+gamma*B*C",
    dense_attention=True,
)


@contextmanager
def bypass(model):
    """Temporarily select exact native attention; restore every block even if diagnosis fails."""
    blocks = [block.attn for module in model.modules() if isinstance(module, C2PSA_RPCA) for block in module.m]
    states = [block.enabled for block in blocks]
    try:
        for block in blocks:
            block.enabled = False
        yield
    finally:
        for block, enabled in zip(blocks, states):
            block.enabled = enabled


def audit(baseline, candidate, weights):
    """Compare every common parameter/buffer and confirm the sole replacement and six added tensors."""
    report = shared.audit_weights(baseline, candidate, weights, "model.10.m.", 10)
    expected = {
        f"model.10.m.{i}.gate.{j}.{p}"
        for i in range(len(candidate.model[10].m))
        for j in (0, 2, 4)
        for p in ("weight", "bias")
    }
    assert set(report["new_parameters"]) == expected
    assert report["baseline_parameters"] == 2504190 and report["candidate_parameters"] == 2506448
    assert report["added_parameters"] == 2258
    assert [i for i, (a, b) in enumerate(zip(baseline.model, candidate.model)) if type(a) is not type(b)] == [10]
    assert type(candidate.model[4]) is C3k2 and type(candidate.model[9]) is SPPF
    assert type(candidate.model[10]) is C2PSA_RPCA and len(candidate.model[10].m) == 1
    assert candidate.model[10].c == 128 and candidate.model[10].m[0].attn.num_heads == 2
    assert candidate.model[21].f == [-1, 10]
    assert candidate.stride.tolist() == baseline.stride.tolist() == [8, 16, 32]
    return report


class AuditedTrainer(shared.AuditedTrainer):
    """Reuse native dataset/optimizer/lifecycle and fixed-budget ownership with RPCA-specific audits."""

    block_type = C2PSA_RPCA
    layer = 10
    new_marker = ".gate."
    new_parameters = 2258
    gradient_markers = ("model.10.m.0.gate.", "model.10.m.0.attn.qkv.", "model.10.m.0.ffn.")

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Build both nc-adapted models at the same RNG state, then audit actual native weight migration."""
        with torch.random.fork_rng(devices=[]):
            baseline = DetectionTrainer.get_model(self, copy.deepcopy(shared.baseline_architecture()), weights, False)
        candidate = DetectionTrainer.get_model(self, cfg, weights, verbose)
        self.weight_audit = audit(baseline, candidate, weights)
        self.initial_common = {
            k: v.detach().cpu().clone() for k, v in candidate.state_dict().items() if ".gate." not in k
        }
        with torch.random.fork_rng(devices=[]), torch.no_grad(), bypass(candidate):
            baseline.eval()
            candidate.eval()
            x = torch.randn(1, 3, 64, 96)
            shared.assert_close_tree(baseline(x), candidate(x), 0, 0)
        candidate.train()
        return candidate

    def validate_new(self):
        """Require enabled attention and the exact bounded gate initialization at the formal start."""
        for block in self.model.model[10].m:
            assert block.attn.enabled
            assert torch.count_nonzero(block.gate[-1].weight) == 0
            assert torch.equal(block.gate[-1].bias, torch.full_like(block.gate[-1].bias, -math.log(9)))
            assert all(p.requires_grad for p in block.gate.parameters())


@torch.random.fork_rng(devices=[])
@torch.no_grad()
def structural_checks(directory, model=MODEL, block_type=C2PSA_RPCA):
    """Verify the full b19 graph, shared initialization, and exact native-bypass outputs on square/rectangular inputs."""
    assert model == MODEL and block_type is C2PSA_RPCA
    original = YAML.load(ROOT / "ultralytics/cfg/models/26/yolo26.yaml")
    expected = copy.deepcopy(original)
    expected["nc"] = 1
    expected["backbone"][10][2] = "C2PSA_RPCA"
    assert YAML.load(model) == expected
    torch.manual_seed(42)
    baseline = DetectionModel(shared.baseline_architecture(), verbose=False).eval()
    rng = torch.get_rng_state()
    torch.manual_seed(42)
    candidate = DetectionModel(str(model), nc=1, verbose=False).eval()
    assert torch.equal(rng, torch.get_rng_state())
    report = audit(baseline, candidate, None)
    with bypass(candidate):
        for height, width in ((640, 640), (640, 512)):
            x = torch.randn(1, 3, height, width)
            shared.assert_close_tree(baseline(x), candidate(x), 0, 0)
    report.update(full_network_bypass_exact=True, constructor_rng_equal=True, layer=10)
    shared.write_json(Path(directory) / "structural.json", report)
    return report


def main(argv=None):
    """Bind the RPCA architecture, trainer, child entry and fingerprints explicitly without replacing globals."""
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
