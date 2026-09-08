"""Numerical, initialization and serialization checks for the isolated RSC candidate."""

import argparse
import copy
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

import torch

from tools.experiments import b19_common as common
from tools.experiments.rsc_experiment import EXPERIMENTS, V1
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.nn.modules import SPPF
from ultralytics.nn.modules.rsc_c2psa import Attention_RSC, reciprocal_probabilities
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import YAML


@contextmanager
def bypass(model):
    """Temporarily execute native Attention on this instance only, including after an exception."""
    blocks = [m for m in model.modules() if isinstance(m, Attention_RSC)]
    states = [m.enabled for m in blocks]
    try:
        for m in blocks:
            m.enabled = False
        yield
    finally:
        for m, state in zip(blocks, states):
            m.enabled = state


def audit(baseline, candidate, weights, experiment=V1):
    """Verify all shared names/shapes/values, the native graph and precisely two added scalars."""
    report = common.audit_weights(baseline, candidate, weights)
    block = candidate.model[10]
    assert type(block) is experiment.block_type and type(candidate.model[9]) is SPPF
    assert [i for i, (a, b) in enumerate(zip(baseline.model, candidate.model)) if type(a) is not type(b)] == [10]
    assert candidate.model[21].f == [-1, 10]
    assert candidate.model[-1].f == [16, 19, 22]
    assert candidate.stride.tolist() == baseline.stride.tolist() == [8, 16, 32]
    assert len(block.m) == 1 and block.c == 128
    attn = block.m[0].attn
    assert (attn.num_heads, attn.key_dim, attn.head_dim) == (2, 32, 64)
    assert set(report["new_parameters"]) == {"model.10.m.0.attn.theta"}
    assert report["added_parameters"] == 2
    assert report["baseline_parameters"] == 2504190 and report["candidate_parameters"] == 2504192
    report["dimensions"] = dict(
        channels=block.cv1.conv.in_channels,
        half_channels=block.c,
        repeats=len(block.m),
        heads=attn.num_heads,
        key_dim=attn.key_dim,
        value_dim=attn.head_dim,
    )
    report["shared_tensors"] = {
        k: dict(shape=list(v.shape), equal=True, pretrained=k in report["loaded_keys"])
        for k, v in baseline.state_dict().items()
    }
    return report


def probability_checks(device="cpu"):
    """Check batch/head isolation, stability, fixed points, nonnegative unit rows and the L1 bound."""
    torch.manual_seed(42)
    scores = torch.randn(3, 2, 17, 17, device=device, requires_grad=True)
    a = scores.softmax(-1)
    r = reciprocal_probabilities(scores)
    direct = (a * a.transpose(-2, -1)).sqrt()
    direct = direct / direct.sum(-1, keepdim=True)
    torch.testing.assert_close(r, direct)
    for batch in range(3):
        for head in range(2):
            torch.testing.assert_close(r[batch, head], reciprocal_probabilities(scores[batch, head]))
    beta = torch.tensor([0.01, 0.199], device=device).view(1, 2, 1, 1)
    mixed = (1 - beta) * a + beta * r
    for p in (r, mixed):
        assert (p >= 0).all() and torch.isfinite(p).all()
        torch.testing.assert_close(p.sum(-1), torch.ones_like(p.sum(-1)))
    assert ((mixed - a).abs().sum(-1) <= 2 * beta.squeeze(-1) + 1e-6).all()
    symmetric = torch.full((2, 2, 17, 17), 0.4 / 17, device=device)
    symmetric += torch.eye(17, device=device) * 0.6
    torch.testing.assert_close(reciprocal_probabilities(symmetric.log()), symmetric)
    assert not torch.allclose(r, r.transpose(-2, -1))
    assert not torch.allclose(r.sum(-2), torch.ones_like(r.sum(-2)))
    (r * torch.randn_like(r)).sum().backward()
    assert scores.grad is not None and torch.isfinite(scores.grad).all() and scores.grad.count_nonzero()
    extreme = torch.tensor([[0.0, -1000.0, -200.0], [-400.0, 0.0, -900.0], [-700.0, -300.0, 0.0]], device=device)
    extreme = extreme.half().requires_grad_()
    stable = reciprocal_probabilities(extreme)
    assert stable.dtype == torch.float32 and torch.isfinite(stable).all()
    torch.testing.assert_close(stable.sum(-1), torch.ones(3, device=device))
    stable.square().sum().backward()
    assert torch.isfinite(extreme.grad).all()
    return dict(
        nonnegative=True,
        unit_rows=True,
        symmetric_fixed_point=True,
        l1_bound=True,
        batch_head_isolation=True,
        fp16_underflow_safe=True,
        calibration_gradient_connected=True,
    )


def structural_checks(directory, experiment=V1):
    """Run complete 640-square and rectangular model forward/backward, preserving initialization evidence."""
    original = YAML.load(common.ROOT / "ultralytics/cfg/models/26/yolo26.yaml")
    expected = copy.deepcopy(original)
    expected["nc"] = 1
    expected["backbone"][10][2] = experiment.block_type.__name__
    assert YAML.load(experiment.model) == expected
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        baseline = DetectionModel(common.baseline_architecture(), verbose=False).eval()
        rng = torch.get_rng_state()
        torch.manual_seed(42)
        candidate = DetectionModel(str(experiment.model), nc=1, verbose=False).eval()
        assert torch.equal(rng, torch.get_rng_state())
        report = audit(baseline, candidate, None, experiment)
        if experiment.version == 2:
            from tools.experiments.verify_b19_rsc_c2psa_v2 import probability_checks as check_probabilities
        else:
            check_probabilities = probability_checks
        report.update(constructor_rng_equal=True, probabilities=check_probabilities(), shapes=[])
        for h, w in ((640, 640), (640, 512)):
            x = torch.randn(1, 3, h, w)
            with torch.no_grad(), bypass(candidate):
                common.assert_close_tree(baseline(x), candidate(x), 0, 0)
            shape = []
            hook = candidate.model[10].register_forward_hook(lambda m, inputs, output: shape.append(list(output.shape)))
            try:
                candidate.train()
                candidate.zero_grad(set_to_none=True)
                output = candidate(x)
                raw = output["one2many"]
                if experiment.version == 2:
                    candidate.args = get_cfg(overrides=common.REFERENCE["args"])
                    task_batch = dict(
                        img=x,
                        batch_idx=torch.tensor([0.0]),
                        cls=torch.tensor([[0.0]]),
                        bboxes=torch.tensor([[0.5, 0.5, 0.4, 0.3]]),
                    )
                    loss, _ = candidate.loss(task_batch, output)
                    loss = loss.sum()
                else:
                    loss = raw["boxes"].square().mean() + raw["scores"].square().mean()
                loss.backward()
                theta = candidate.model[10].m[0].attn.theta
                assert theta.grad is not None and torch.isfinite(theta.grad).all()
                if experiment.version == 2:
                    assert theta.grad.count_nonzero(), "Synthetic labeled task probe must reach theta"

                assert shape == [[1, 256, h // 32, w // 32]]
                assert raw["boxes"].shape == (1, 4, (h // 8) * (w // 8) * 21 // 16)
                report["shapes"].append(
                    dict(
                        input=list(x.shape),
                        layer10=shape[0],
                        boxes=list(raw["boxes"].shape),
                        scores=list(raw["scores"].shape),
                        theta_gradient=theta.grad.tolist(),
                    )
                )
            finally:
                hook.remove()
            # The training probe updated BN; restore the shared baseline before the next exact bypass comparison.
            candidate.load_state_dict(baseline.state_dict(), strict=False)
            candidate.eval()
    common.write_json(Path(directory) / "structural.json", report)
    return report


def save_reload_check(model, directory, experiment=V1):
    """Compare a native FP16 EMA checkpoint in a fresh process against the pre-save independent snapshot."""
    directory = Path(directory)
    snapshot = copy.deepcopy(model).cpu().half().eval()
    snapshot.criterion = None
    path = directory / "preflight.pt"
    args = snapshot.args if isinstance(snapshot.args, dict) else vars(snapshot.args)
    torch.save(dict(model=None, ema=snapshot, train_args=args), path)
    snapshot.float()
    with torch.no_grad():
        x = torch.randn(1, 3, 64, 96)
        reference = dict(x=x, state=snapshot.state_dict(), raw=snapshot(x))
        torch.save(reference, directory / "reload_reference.pt")
    command = [
        sys.executable,
        "-m",
        "tools.experiments.verify_b19_rsc_c2psa",
        str(path),
        "--version",
        str(experiment.version),
    ]
    result = subprocess.run(
        command, cwd=common.ROOT, env={**os.environ, "PYTHONPATH": str(common.ROOT)}, capture_output=True, text=True
    )
    (directory / "reload.log").write_text(result.stdout + result.stderr, encoding="utf-8")
    result.check_returncode()
    return dict(
        checkpoint=str(path),
        sha256=common.sha256(path),
        fresh_process=True,
        checkpoint_precision="FP16 EMA, reloaded FP32",
        state_exact=True,
        fuse=True,
    )


def reload_in_process(path, experiment=V1):
    """Validate all serialized tensors, raw outputs, retained fused branch and anchor-aligned predictions."""
    path = Path(path)
    reference = torch.load(path.with_name("reload_reference.pt"), map_location="cpu", weights_only=False)
    model = YOLO(path)
    assert type(model.model.model[10]) is experiment.block_type
    assert model.model.model[10].m[0].attn.enabled
    common.assert_close_tree(reference["state"], model.model.state_dict(), 0, 0)
    report = []
    with torch.no_grad():
        before = model.model(reference["x"])
        common.assert_close_tree(reference["raw"], before, 1e-5, 1e-5, report=report)
        # Both branches start from the same complete v2 state, including any updated theta/shared weights.
        fused = copy.deepcopy(model)
        common.assert_close_tree(model.model.state_dict(), fused.model.state_dict(), 0, 0)
        fused.fuse()
        after = fused.model(reference["x"])
        assert after[1]["one2many"] == {}
        common.assert_close_tree(before[1]["one2one"], after[1]["one2one"], 1e-4, 1e-4, report=report)
        head = fused.model.model[-1]
        decoded = [head._inference(v[1]["one2one"]).permute(0, 2, 1) for v in (before, after)]
        common.assert_close_tree(decoded[0], decoded[1], 1e-4, 1e-4, report=report)
        aligned = []
        for raw, dense in zip((before, after), decoded):
            index = head.get_topk_index(dense[..., 4:], head.max_det)[2].squeeze(-1)
            assert dense.shape[1] <= head.max_det
            assert torch.equal(index.sort(-1).values, torch.arange(dense.shape[1]).expand_as(index))
            common.assert_close_tree(head.postprocess(dense), raw[0], 0, 0)
            aligned.append(raw[0].gather(1, index.argsort(-1).unsqueeze(-1).expand_as(raw[0])))
        common.assert_close_tree(*aligned, atol=1e-4, rtol=1e-4, report=report)
    common.write_json(path.with_name("reload_check.json"), dict(passed=True, comparisons=report))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--version", type=int, choices=(1, 2), default=1)
    options = parser.parse_args()
    reload_in_process(options.checkpoint, EXPERIMENTS[options.version])
