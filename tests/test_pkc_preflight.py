"""CPU fault injection for the real preflight observer; no claim about server CUDA batch32/640."""

import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from contextlib import redirect_stderr
from unittest.mock import patch

from tools.experiments import run_b19_pkc_sppf as run

import torch
from torch import nn
from ultralytics.optim.muon import MuSGD
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.utils.torch_utils import ModelEMA


class ToyModel(nn.Module):
    """Expose tiny gamma gradients beside a large gradient that triggers native global clipping."""

    def __init__(self, fault=None, tiny=True):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(32))
        self.large = nn.Parameter(torch.ones(1))
        self.fault, self.tiny = fault, tiny
        self.batch = 0

    def forward(self, batch):
        """Return a finite loss even for the deliberate early backward overflow."""
        coefficient = 0.0003 if self.tiny else 1.0
        gamma = self.gamma.detach() if self.fault == "detached" else self.gamma
        loss = gamma.sum() * (0 if self.fault == "decay_only" else coefficient) + self.large.sum() * 100
        if self.fault == "overflow" and self.batch < 8:
            loss.register_hook(lambda g: g * float("inf"))
        self.batch += 1
        return loss.reshape(1), loss.detach().reshape(1)


def toy_trainer(fault=None, tiny=True):
    """Use native MuSGD and CPU GradScaler with the same clipping and warmup arithmetic."""
    model = ToyModel(fault, tiny)
    params = [model.large] if fault == "missing" else list(model.parameters())
    optimizer = MuSGD(
        [dict(params=params, param_group="bn", lr=0.01, initial_lr=0.01)],
        momentum=0.937,
        nesterov=True,
        weight_decay=0.01 if fault == "decay_only" else 0,
    )
    if fault == "duplicate":
        optimizer.param_groups[0]["params"].append(model.gamma)
    trainer = SimpleNamespace(
        model=model,
        optimizer=optimizer,
        scaler=torch.amp.GradScaler("cpu"),
        ema=None,
        args=SimpleNamespace(warmup_epochs=3, nbs=64, warmup_bias_lr=0.1, warmup_momentum=0.8, momentum=0.937),
        start_epoch=0,
        batch_size=32,
        accumulate=2,
        amp=False,
        device=torch.device("cpu"),
        train_loader=range(330),
        preprocess_batch=lambda b: b,
        _model_train=model.train,
        lf=lambda epoch: 0.0 if fault == "no_update" else 1.0,
    )
    trainer.scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=trainer.lf)
    return trainer


class PreflightObservationTests(unittest.TestCase):
    """Require actual updates, fixed budgets, faithful overflow accounting and durable failed reports."""

    def observe(self, fault=None, tiny=True, budget=128):
        trainer = toy_trainer(fault, tiny)
        params = {"reduce.bn.weight": trainer.model.gamma}
        report = run.observation_report(params, budget)
        batches = iter([dict(img=torch.zeros(32, 3, 1, 1), cls=torch.ones(1))] * budget)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "checks.json"
            try:
                run.observe_batches(trainer, params, batches, report, path)
            except AssertionError:
                if fault not in ("missing", "duplicate", "detached", "no_update", "decay_only") and budget != 16:
                    raise
            saved = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(saved, report)
        return trainer, report

    def test_tiny_gamma_requires_real_change_after_old_window(self):
        _, short = self.observe(budget=16)
        self.assertEqual(short["failure_stage"], "observation_budget")
        self.assertIsNotNone(short["first_finite_nonzero_gradient_step"]["reduce.bn.weight"])
        trainer, report = self.observe()
        self.assertGreater(report["batches_observed"], 16)
        self.assertLess(report["batches_observed"], 128)
        self.assertFalse(report["missing_effective_parameters"])
        self.assertFalse(torch.equal(trainer.model.gamma, torch.ones(32)))
        for row in report["real_batches"]:
            info = row["parameters"]["reduce.bn.weight"]
            self.assertTrue(info["native_copy_replay"]["matches_actual_exactly"])
            self.assertLess(row["clip_coefficient"], 1)
        first = report["real_batches"][1]["parameters"]["reduce.bn.weight"]
        self.assertFalse(first["changed"])
        self.assertGreater(first["native_copy_replay"]["nonzero_proposals_below_half_spacing"], 0)

    def test_eight_overflows_are_skips_not_successes(self):
        _, report = self.observe("overflow", tiny=False)
        self.assertEqual(report["overflow_skips"], 8)
        self.assertEqual(report["successful_steps"], 1)
        self.assertEqual(report["attempted_steps"], 9)
        self.assertEqual(report["real_batches"][7]["scale_after"], 256)
        self.assertEqual(report["first_effective_update_step"]["reduce.bn.weight"], 9)

    def test_registration_and_disconnected_faults_fail(self):
        for fault, stage in (
            ("missing", "optimizer_membership"),
            ("duplicate", "optimizer_membership"),
            ("detached", "optimizer_step"),
        ):
            with self.subTest(fault=fault):
                _, report = self.observe(fault)
                self.assertFalse(report["passed"])
                self.assertEqual(report["failure_stage"], stage)
                self.assertTrue(report["missing_effective_parameters"])

    def test_no_update_and_decay_only_exhaust_budget(self):
        for fault in ("no_update", "decay_only"):
            with self.subTest(fault=fault):
                _, report = self.observe(fault)
                self.assertEqual(report["batches_observed"], 128)
                self.assertEqual(report["failure_stage"], "observation_budget")
                self.assertTrue(report["missing_effective_parameters"])
                if fault == "decay_only":
                    self.assertTrue(any(r["parameters"]["reduce.bn.weight"]["changed"] for r in report["real_batches"]))
                    self.assertIsNone(report["first_finite_nonzero_gradient_step"]["reduce.bn.weight"])

    def test_setup_failure_is_durable(self):
        def broken(**kwargs):
            raise RuntimeError("injected setup failure")

        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp) / "attempt"
            with self.assertRaisesRegex(RuntimeError, "injected setup"):
                run.preflight({}, {}, directory, trainer_type=broken)
            report = json.loads((directory / "checks.json").read_text(encoding="utf-8"))
            self.assertEqual(report["failure_stage"], "setup")
            self.assertEqual(report["max_batches"], 128)
            self.assertEqual(len(report["missing_effective_parameters"]), 13)
            self.assertEqual(report["successful_steps"], 0)

    def test_observer_matches_native_step_and_does_not_pollute_state(self):
        observed = toy_trainer("overflow")
        native = toy_trainer("overflow")
        observed.ema, native.ema = ModelEMA(observed.model), ModelEMA(native.model)
        report = run.observation_report(["reduce.bn.weight"])
        for _ in range(12):
            for trainer in (observed, native):
                loss, _ = trainer.model({})
                trainer.scaler.scale(loss.sum()).backward()
            run.observed_optimizer_step(observed, {"reduce.bn.weight": observed.model.gamma}, report, {})
            BaseTrainer.optimizer_step(native)
            run.assert_close_tree(observed.model.state_dict(), native.model.state_dict(), 0, 0)
            run.assert_close_tree(observed.ema.ema.state_dict(), native.ema.ema.state_dict(), 0, 0)
            run.assert_close_tree(observed.optimizer.state_dict(), native.optimizer.state_dict(), 0, 0)
            self.assertEqual(observed.scaler.state_dict(), native.scaler.state_dict())
            self.assertEqual(observed.ema.updates, native.ema.updates)
            self.assertEqual(len(observed.optimizer.state), len(native.optimizer.state))

    def test_launch_failure_writes_unique_checks_and_returns_nonzero(self):
        def broken(*args):
            raise RuntimeError("injected structural failure")

        with tempfile.TemporaryDirectory() as temp, patch.object(run, "git", return_value=""), redirect_stderr(
            io.StringIO()
        ):
            arguments = ["--baseline-root", str(run.ROOT), "--stage", "preflight", "--project", temp]
            for _ in range(2):
                self.assertEqual(run.main(arguments, structure_check=broken), 1)
            paths = list(Path(temp).rglob("checks.json"))
            self.assertEqual(len(paths), 2)
            for path in paths:
                report = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(report["failure_stage"], "source_structure_recipe")
                self.assertEqual(report["max_batches"], 128)
                self.assertEqual(report["successful_steps"], 0)
                self.assertEqual(len(report["missing_effective_parameters"]), 13)
            self.assertFalse(list(Path(temp).rglob("passed.json")))

    def test_receipts_bind_source_and_preserve_observation_counts(self):
        class Probe:
            block_type = run.SPPF_PKC

            def get_model(self, *args, **kwargs):
                self.weight_audit = {}

        head, calls = ["old-source"], []

        def recipe(*args):
            return (
                {},
                dict(seed=42, deterministic=True),
                dict(commit=head[0], initial_path="unused.pt", initial_sha256="test", config_differences={}),
            )

        def passed_preflight(config, evidence, directory, *args):
            calls.append(head[0])
            directory.mkdir(parents=True)
            report = run.observation_report()
            report.update(
                passed=True,
                stage="complete",
                attempted_steps=33,
                overflow_skips=8,
                successful_steps=25,
                batches_observed=33,
                missing_effective_parameters=[],
            )
            for key in ("first_finite_nonzero_gradient_step", "first_effective_update_step"):
                report[key] = {k: 33 for k in run.PKC_PARAMETERS}
            run.write_json(directory / "checks.json", report)

        with tempfile.TemporaryDirectory() as temp, patch.multiple(
            run,
            git=lambda *a: "",
            resolve_recipe=recipe,
            launcher_evidence=lambda *a: {},
            source_hashes=lambda *a: {"runner": head[0]},
            load_checkpoint=lambda *a: (None, None),
            runtime_issues=lambda *a: [],
            preflight=passed_preflight,
        ):
            arguments = ["--baseline-root", str(run.ROOT), "--stage", "preflight", "--project", temp]
            for source in ("old-source", "old-source", "repaired-source"):
                head[0] = source
                self.assertEqual(run.main(arguments, trainer_type=Probe, structure_check=lambda *a: None), 0)
            self.assertEqual(calls, ["old-source", "repaired-source"])
            self.assertEqual(len(list(Path(temp).rglob("passed.json"))), 2)
            for path in Path(temp).glob("*/preflight-invocation-*/checks.json"):
                report = json.loads(path.read_text(encoding="utf-8"))
                self.assertTrue(report["passed"])
                self.assertEqual(report["successful_steps"], 25)
                self.assertEqual(report["overflow_skips"], 8)
                self.assertEqual(report["missing_effective_parameters"], [])


if __name__ == "__main__":
    unittest.main()
