"""Run the single b19 + DSD-Head v1 experiment with the recorded native training recipe."""

# ruff: noqa: E402 -- Resolve the worktree and offline settings before importing torch.

import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.experiments import b19_common as shared

import torch

from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.modules import DSDAdapter, DSDDetect
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import YAML

MODEL = ROOT / "ultralytics/cfg/models/26/yolo26n-dsd-head-v1.yaml"
NAME = "yolo26n_b19_dsd_head_v1"
SOURCE_FILES = tuple(
    ROOT / name
    for name in (
        "tools/experiments/run_b19_dsd_head.py",
        "tools/experiments/b19_common.py",
        "tools/experiments/b19_finish.py",
        "tools/experiments/dsd_preflight.py",
        "tools/experiments/dsd_validator.py",
        "tools/experiments/finish_b19_dsd_head.py",
        "tools/experiments/server_b19_dsd_head_v1.sh",
        "tools/experiments/b19_launcher_expanded.txt",
        "tools/experiments/dsd_sources.json",
        "tests/test_dsd_head.py",
        "docs/experiments/b19_dsd_head_v1.md",
    )
)
MODULE_CONFIG = dict(
    version=1,
    layer=23,
    feature_layer=16,
    inputs=[16, 19, 22],
    channels=64,
    groups=8,
    directions=DSDAdapter.directions,
    diagonal_denominator=2,
    scale=0.1,
    formula="X_box = X + (0.1/4)*sum_d tanh(z[group(c),d])*D_d(X)",
    boundary="one-pixel exterior correction is zero; H<3 or W<3 is identity",
    parameters_per_adapter=1744,
    added_training_parameters=3488,
    added_fused_parameters=1744,
    detach="native Detect.forward detaches backbone inputs before one2one_reg_adapter",
)


def audit(baseline, candidate, weights):
    """Check every common initialization/source value and the exact graph and parameter delta."""
    report = shared.audit_weights(baseline, candidate, weights)
    expected = {
        f"model.23.{branch}.coeff.{i}.{p}"
        for branch in ("reg_adapter", "one2one_reg_adapter")
        for i in (0, 2, 4)
        for p in ("weight", "bias")
    }
    assert set(report["new_parameters"]) == expected
    assert report["baseline_parameters"] == 2504190
    assert report["candidate_parameters"] == 2507678 and report["added_parameters"] == 3488
    assert [i for i, (a, b) in enumerate(zip(baseline.model, candidate.model)) if type(a) is not type(b)] == [23]
    head = candidate.model[-1]
    assert type(head) is DSDDetect and head.f == [16, 19, 22]
    assert head.reg_max == 1 and head.end2end
    assert candidate.stride.tolist() == baseline.stride.tolist() == [8, 16, 32]
    for left, right in zip(head.reg_adapter.parameters(), head.one2one_reg_adapter.parameters()):
        assert left is not right and left.data_ptr() != right.data_ptr() and torch.equal(left, right)
    for adapter in (head.reg_adapter, head.one2one_reg_adapter):
        assert sum(p.numel() for p in adapter.parameters()) == 1744
    return report


class AuditedTrainer(shared.AuditedTrainer):
    """Use native trainer initialization, names, losses, optimizer and EMA with DSD-specific audits."""

    gradient_markers = ("reg_adapter.", "model.0.conv.weight", "model.23.cv2.0.", "model.23.one2one_cv2.0.")

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Audit the nc-adapted native b19 reconstruction without consuming a second RNG stream."""
        with torch.random.fork_rng(devices=[]):
            baseline = DetectionTrainer.get_model(self, copy.deepcopy(shared.baseline_architecture()), weights, False)
        candidate = DetectionTrainer.get_model(self, cfg, weights, verbose)
        self.weight_audit = audit(baseline, candidate, weights)
        self.initial_common = {
            k: v.detach().cpu().clone() for k, v in candidate.state_dict().items() if self.new_marker not in k
        }
        with torch.random.fork_rng(devices=[]), torch.no_grad():
            baseline.eval()
            candidate.eval()
            x = torch.randn(1, 3, 64, 96)
            shared.assert_close_tree(baseline(x), candidate(x), 0, 0)
        candidate.train()
        return candidate

    def validate_new(self):
        """Require exact identity initialization at both preflight and formal training setup."""
        head = self.model.model[-1]
        for adapter in (head.reg_adapter, head.one2one_reg_adapter):
            assert torch.count_nonzero(adapter.coeff[-1].weight) == 0
            assert torch.count_nonzero(adapter.coeff[-1].bias) == 0
            assert all(p.requires_grad for p in adapter.parameters())


@torch.random.fork_rng(devices=[])
@torch.no_grad()
def structural_checks(directory, model=MODEL, block_type=DSDDetect):
    """Check square/rectangular full-network identity, graph, RNG and measured parameter counts."""
    assert model == MODEL and block_type is DSDDetect
    expected = YAML.load(ROOT / "ultralytics/cfg/models/26/yolo26.yaml")
    expected["nc"] = 1
    expected["head"][-1][2] = "DSDDetect"
    assert YAML.load(model) == expected
    torch.manual_seed(42)
    baseline = DetectionModel(shared.baseline_architecture(), verbose=False).eval()
    rng = torch.get_rng_state()
    torch.manual_seed(42)
    candidate = DetectionModel(str(model), verbose=False).eval()
    assert torch.equal(rng, torch.get_rng_state())
    report = audit(baseline, candidate, None)
    shapes = []
    for height, width in ((640, 640), (384, 640)):
        x = torch.randn(1, 3, height, width)
        left, right = baseline(x), candidate(x)
        shared.assert_close_tree(left, right, 0, 0)
        feats = right[1]["one2many"]["feats"]
        assert feats[0].shape == (1, 64, height // 8, width // 8)
        shapes.append([list(f.shape) for f in feats])
    # Exact integer-valued affine features remove roundoff ambiguity in the mathematical nullspace check.
    adapter = candidate.model[-1].reg_adapter
    adapter.coeff[-1].bias.copy_(torch.linspace(-2, 2, 32))
    for h, w in ((7, 11), (3, 3), (1, 9), (9, 2)):
        ramp = (torch.arange(h)[:, None] * 2 + torch.arange(w)[None, :] * 3).float()
        for x in (torch.ones(1, 64, h, w), ramp.expand(1, 64, h, w)):
            assert torch.equal(adapter(x), x)
        x = torch.randn(1, 64, h, w)
        y = adapter(x)
        assert torch.equal(y[..., (0, -1), :], x[..., (0, -1), :])
        assert torch.equal(y[..., :, (0, -1)], x[..., :, (0, -1)])
    report.update(
        constructor_rng_equal=True,
        square_rectangle_exact=True,
        feature_shapes=shapes,
        boundary_and_affine_nullspace=True,
    )
    from tools.experiments.dsd_preflight import routing_checks

    report["routing"] = routing_checks()
    shared.write_json(Path(directory) / "structural.json", report)
    return report


def main(argv=None):
    """Bind independent DSD checks to the audited b19 recipe and process lifecycle."""
    from tools.experiments.dsd_preflight import preflight_batches

    return shared.main(
        argv,
        model=MODEL,
        trainer_type=AuditedTrainer,
        entrypoint=Path(__file__).resolve(),
        name=NAME,
        source_files=SOURCE_FILES,
        structure_check=structural_checks,
        module_config=MODULE_CONFIG,
        batch_check=preflight_batches,
    )


if __name__ == "__main__":
    raise SystemExit(main())
