"""Run the single b19 + RPCA-C2PSA v1 candidate through the audited native b19 trainer."""

# ruff: noqa: E402 -- Import the worktree and offline settings before torch.

import copy
import math
import sys
import time
import warnings
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.experiments import run_b19_sir_sppf as shared

import torch
import numpy as np

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


def tensor_stats(tensor):
    """Keep small, JSON-finite summaries, including the distinction between absent and zero gradients."""
    if tensor is None:
        return dict(is_none=True, finite=None, norm=None, nonzero=0, dtype=None)
    tensor = tensor.detach()
    finite = bool(torch.isfinite(tensor).all())
    return dict(
        is_none=False,
        finite=finite,
        norm=tensor.double().norm().item() if finite else None,
        nonzero=tensor.count_nonzero().item(),
        dtype=str(tensor.dtype),
        max_abs=tensor.abs().max().item() if finite else None,
        elements=tensor.numel(),
    )


def preflight_batches(trainer, report, directory):
    """Observe at most 16 real batches using native first-epoch AMP/MuSGD ordering, saving failures too."""
    state = report["gate_preflight"] = dict(
        status="running",
        max_batches=16,
        attempted_steps=0,
        completed_steps=0,
        gate_updates=0,
        first_task_weight_update=None,
        first_nonzero_gradient={},
        first_effective_gradient={},
    )
    handles = []
    try:
        gate = trainer.model.model[10].m[0].gate
        params = {k: p for k, p in trainer.model.named_parameters() if ".gate." in k}
        required = {
            k: p for k, p in trainer.model.named_parameters() if any(marker in k for marker in trainer.gradient_markers)
        }
        groups = {id(p): i for i, g in enumerate(trainer.optimizer.param_groups) for p in g["params"]}
        state["optimizer_membership"] = {k: groups.get(id(p)) for k, p in params.items()}
        if len(params) != 6 or any(id(p) not in groups or not p.requires_grad for p in params.values()):
            raise AssertionError("RPCA gate parameters missing from optimizer or not trainable")
        if trainer.optimizer.state or gate[-1].weight.count_nonzero():
            raise AssertionError("RPCA preflight must start with a fresh optimizer and zero last weight")

        def completed_step(optimizer, args, kwargs):
            state["completed_steps"] += 1  # The hook is not called for GradScaler's skipped steps.

        def last_conv(module, inputs, output):
            row["last_conv"] = dict(
                input_dtype=str(inputs[0].dtype),
                output_dtype=str(output.dtype),
                weight=tensor_stats(module.weight),
                weight_in_compute_dtype=tensor_stats(module.weight.to(output.dtype)),
            )
            for name, tensor in (("input", inputs[0]), ("output", output)):
                if tensor.requires_grad:

                    def record_gradient(gradient, name=name):
                        row["last_conv"][name + "_scaled_gradient"] = tensor_stats(gradient)

                    tensor.register_hook(record_gradient)

        handles = [trainer.optimizer.register_step_post_hook(completed_step), gate[-1].register_forward_hook(last_conv)]
        nb = len(trainer.train_loader)
        nw = max(round(trainer.args.warmup_epochs * nb), 100) if trainer.args.warmup_epochs > 0 else -1
        state["warmup_batches"] = nw
        trainer.epoch = 0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            trainer.scheduler.step()  # Same first-epoch scheduling order as BaseTrainer._do_train.
        trainer._model_train()
        trainer.optimizer.zero_grad()
        loader = iter(trainer.train_loader)
        last_opt_step = -1
        for ni in range(min(16, nb)):
            if trainer.device.type == "cuda":
                torch.cuda.synchronize(trainer.device)
            started = time.perf_counter()
            row = dict(
                batch=ni + 1,
                iteration=ni,
                scale_before=trainer.scaler.get_scale(),
                skipped=None,
                attempt=state["attempted_steps"],
                optimizer_update=state["completed_steps"],
            )
            report["real_batches"].append(row)
            try:
                current = dict(trainer.model.named_parameters())
                if any(current.get(k) is not p for k, p in params.items()):
                    raise AssertionError("RPCA gate was replaced during preflight")
                if ni <= nw:
                    trainer.accumulate = max(
                        1, int(np.interp(ni, [0, nw], [1, trainer.args.nbs / trainer.batch_size]).round())
                    )
                    for group in trainer.optimizer.param_groups:
                        start = trainer.args.warmup_bias_lr if group.get("param_group") == "bias" else 0.0
                        group["lr"] = float(np.interp(ni, [0, nw], [start, group["initial_lr"] * trainer.lf(0)]))
                        if "momentum" in group:
                            group["momentum"] = float(
                                np.interp(ni, [0, nw], [trainer.args.warmup_momentum, trainer.args.momentum])
                            )
                row["accumulate"] = trainer.accumulate
                row["groups"] = [
                    dict(
                        index=i,
                        kind=g.get("param_group"),
                        lr=g["lr"],
                        momentum=g.get("momentum"),
                        weight_decay=g["weight_decay"],
                        gate_parameters=[k for k, p in params.items() if groups[id(p)] == i],
                    )
                    for i, g in enumerate(trainer.optimizer.param_groups)
                ]
                forward_parameters = {k: p.detach().clone() for k, p in params.items()}
                with shared.autocast(enabled=trainer.amp, device=trainer.device.type):
                    batch = trainer.preprocess_batch(next(loader))
                    loss, items = trainer.model(batch)
                    total = loss.sum()
                row.update(
                    loss=tensor_stats(total),
                    components=items.detach().cpu().tolist(),
                    image_shape=list(batch["img"].shape),
                    targets=batch["cls"].numel(),
                )
                if not torch.isfinite(total) or not row["targets"]:
                    raise AssertionError("Nonfinite task loss or unlabeled RPCA preflight batch")
                trainer.scaler.scale(total).backward()
                attempted = ni - last_opt_step >= trainer.accumulate
                row["attempted_step"] = attempted
                if attempted:
                    trainer.scaler.unscale_(trainer.optimizer)  # Exactly once, before recording and clipping.
                row["gradient_units"] = "unscaled" if attempted else "scaled_accumulation"
                gradients = row["gradients"] = {k: tensor_stats(p.grad) for k, p in required.items()}
                if attempted:
                    for key in params:
                        if gradients[key]["finite"] and gradients[key]["nonzero"]:
                            state["first_nonzero_gradient"].setdefault(key, ni + 1)
                nonfinite = [
                    k
                    for k, p in trainer.model.named_parameters()
                    if p.grad is not None and not torch.isfinite(p.grad).all()
                ]
                row["nonfinite_gradient_parameters"] = nonfinite
                if any(g["is_none"] for g in gradients.values()):
                    raise AssertionError("Disconnected required RPCA task gradient")
                if any(not torch.equal(p, forward_parameters[k]) for k, p in params.items()):
                    raise AssertionError("RPCA gate parameters changed outside the optimizer step")
                before = {k: p.detach().clone() for k, p in params.items()}
                if attempted:
                    state["attempted_steps"] += 1
                    completed_before = state["completed_steps"]
                    # Match BaseTrainer.optimizer_step; task statistics above exclude clipping/decay/momentum.
                    row["global_norm_before_clip"] = tensor_stats(
                        torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), 10.0)
                    )
                    trainer.scaler.step(trainer.optimizer)
                    trainer.scaler.update()
                    row["skipped"] = state["completed_steps"] == completed_before
                    trainer.optimizer.zero_grad()
                    if trainer.ema:
                        trainer.ema.update(trainer.model)
                    last_opt_step = ni  # Native accumulation counts attempted updates, including scaler skips.
                row.update(
                    attempt=state["attempted_steps"],
                    optimizer_update=state["completed_steps"],
                    scale_after=trainer.scaler.get_scale(),
                )
                row["parameter_deltas"] = {k: tensor_stats(p - before[k]) for k, p in params.items()}
                row["last_weight_after"] = tensor_stats(gate[-1].weight)
                row["last_weight_after_compute_cast"] = tensor_stats(
                    gate[-1].weight.to(getattr(torch, row["last_conv"]["output_dtype"].split(".")[-1]))
                )
                if not all(torch.isfinite(p).all() for p in trainer.model.parameters()):
                    raise AssertionError("Nonfinite parameters after RPCA optimizer update")
                if attempted and nonfinite and not (row["skipped"] and row["scale_after"] < row["scale_before"]):
                    raise AssertionError("Nonfinite task gradients without a native GradScaler overflow skip")
                if attempted and not row["skipped"]:
                    if any(s["nonzero"] for s in row["parameter_deltas"].values()):
                        state["gate_updates"] += 1
                    last_key = "model.10.m.0.gate.4.weight"
                    if (
                        state["first_task_weight_update"] is None
                        and row["parameter_deltas"][last_key]["nonzero"]
                        and gradients[last_key]["nonzero"]
                    ):
                        # Zero-origin weight + fresh optimizer + task gradient proves this is not decay-only motion.
                        state["first_task_weight_update"] = ni + 1
                    for key in params:
                        if gradients[key]["finite"] and gradients[key]["nonzero"]:
                            state["first_effective_gradient"].setdefault(key, ni + 1)
                    first_update = state["first_task_weight_update"]
                    if (
                        first_update is not None
                        and ni + 1 > first_update
                        and all(k in state["first_effective_gradient"] for k in params)
                    ):
                        if not all(
                            any(g["nonzero"] for k, g in gradients.items() if marker in k)
                            for marker in trainer.gradient_markers
                        ):
                            raise AssertionError("Required RPCA attention/FFN task gradients remain zero")
                        state["status"] = "passed"
                        return state["attempted_steps"]
            finally:
                if trainer.device.type == "cuda":
                    torch.cuda.synchronize(trainer.device)
                row["diagnostic_batch_seconds"] = (
                    time.perf_counter() - started
                )  # Includes statistics, excludes JSON I/O.
                shared.write_json(directory / "checks.json", report)
        raise AssertionError(
            "RPCA gate did not establish finite task gradients and effective updates within 16 real batches"
        )
    except Exception as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        for handle in handles:
            handle.remove()
        shared.write_json(directory / "checks.json", report)


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
        batch_check=preflight_batches,
    )


if __name__ == "__main__":
    raise SystemExit(main())
