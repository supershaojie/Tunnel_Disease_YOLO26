"""Prepare or train b19 + RSC-C2PSA with the unchanged native training lifecycle."""

# ruff: noqa: E402 -- Resolve the worktree and offline policy before importing Ultralytics.

import argparse
import copy
import gc
import json
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("YOLO_AUTOINSTALL", "false")
os.environ.setdefault("YOLO_OFFLINE", "true")
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import ultralytics
import torch

from tools.experiments import b19_common as common
from tools.experiments import verify_b19_rsc_c2psa as verify
from ultralytics.cfg import DEFAULT_CFG_DICT, get_cfg
from ultralytics.data.utils import check_det_dataset
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.modules import C2PSA_RSC
from ultralytics.nn.tasks import load_checkpoint
from ultralytics.utils.torch_utils import init_seeds

NAME = "yolo26n_b19_rsc_c2psa_v1"
MODEL = common.MODEL


class AuditedTrainer(DetectionTrainer):
    """Own only fixed-batch output and initialization audits; retain native training and optimization."""

    new_marker = ".attn.theta"

    def __init__(self, overrides, _callbacks=None):
        """Reserve the exact experiment directory without auto-incrementing or overwriting another run."""
        output = Path(overrides["project"]) / overrides["name"]
        output.mkdir(parents=True, exist_ok=False)
        super().__init__(
            cfg={**DEFAULT_CFG_DICT, "save_dir": str(output)}, overrides=overrides.copy(), _callbacks=_callbacks
        )
        self.add_callback("on_pretrain_routine_end", final_model_audit)

    @property
    def _oom_retries(self):
        """The pinned native loop has no fixed-batch OOM hook; this experiment owns no retries."""
        return 0

    @_oom_retries.setter
    def _oom_retries(self, value):
        """Re-raise the active native memory exception before its first batch-size mutation."""
        if value:
            error = sys.exc_info()[1]
            if error is None:
                raise RuntimeError("Fixed batch=32 experiment forbids memory recovery retries")
            raise error

    def get_dataset(self):
        """Use b19's dataset checker without replacement downloads."""
        return check_det_dataset(self.args.data, autodownload=False)

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Audit the actual native reconstruction, including unmatched single-class head initialization."""
        with torch.random.fork_rng(devices=[]):
            baseline = super().get_model(copy.deepcopy(common.baseline_architecture()), weights, False)
        candidate = super().get_model(cfg, weights, verbose)
        self.weight_audit = verify.audit(baseline, candidate, weights)
        self.initial_common = {
            k: v.detach().cpu().clone() for k, v in candidate.state_dict().items() if self.new_marker not in k
        }
        return candidate


def final_model_audit(trainer):
    """Verify common values after AMP setup, optimizer membership and the EMA copy at the real boundary."""
    assert trainer.amp == trainer.args.amp
    assert trainer.batch_size == trainer.args.batch == 32
    current = trainer.model.state_dict()
    assert all(torch.equal(v, current[k].cpu()) for k, v in trainer.initial_common.items())
    attn = trainer.model.model[10].m[0].attn
    assert attn.enabled and attn.theta.requires_grad
    torch.testing.assert_close(attn.beta, torch.full_like(attn.beta, 0.01), atol=1e-8, rtol=1e-6)
    assert type(trainer.ema.ema.model[10]) is C2PSA_RSC
    torch.testing.assert_close(trainer.ema.ema.model[10].m[0].attn.theta, attn.theta, atol=0, rtol=0)
    common.write_json(trainer.save_dir / "provenance/weights.json", trainer.weight_audit)
    common.write_json(trainer.save_dir / "provenance/optimizer.json", common.audit_optimizer(trainer))
    del trainer.initial_common


def runtime_evidence():
    """Report b19 runtime differences and current memory without rejecting other experiments on the GPU."""
    reference = common.REFERENCE["environment"]
    actual = dict(
        torch=str(torch.__version__),
        python=sys.version.split()[0],
        ultralytics=ultralytics.__version__,
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    )
    issues = {k: dict(b19=v, current=actual[k]) for k, v in reference.items() if actual[k] != v}
    info = dict(actual=actual, mismatches=issues, backend=common.computation_conditions())
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info(0)
        info.update(
            free_bytes=free,
            total_bytes=total,
            cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
            gpu_uuid=str(getattr(torch.cuda.get_device_properties(0), "uuid", "unavailable")),
        )
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader"],
            capture_output=True,
            text=True,
        )
        info["concurrent_processes"] = result.stdout.strip()
    return info


class PreflightComplete(Exception):
    """End the disposable native loop at its batch callback without declaring a training run complete."""


def preflight(config, evidence, directory):
    """Observe 32 real native batches: AMP, GradScaler, warmup, accumulation, clipping, MuSGD and EMA."""
    directory.mkdir(parents=True, exist_ok=False)
    common.write_json(directory / "environment.json", evidence)
    trainer = AuditedTrainer(dict(config, project=str(directory), name="check"))
    report = dict(
        passed=False,
        batches=[],
        optimizer_attempts=0,
        completed_steps=0,
        theta_updates=0,
        nonzero_task_gradient_steps=0,
        max_batches=32,
        loop="unmodified BaseTrainer._do_train and optimizer_step",
    )
    handles = []
    state = {}

    def setup(t):
        assert len(t.train_loader.dataset) == 8414 and len(t.test_loader.dataset) == 2404
        assert t.amp and t.scaler.is_enabled() and type(t.optimizer).__name__ == "MuSGD"
        assert not t.optimizer.state and t.ema.updates == 0
        attn = t.model.model[10].m[0].attn
        state["initial_ema"] = t.ema.ema.model[10].m[0].attn.theta.detach().clone()
        torch.cuda.reset_peak_memory_stats(0)

        def observe_batch(module, inputs):
            batch = inputs[0]
            state["image_shape"] = list(batch["img"].shape)
            state["targets"] = batch["cls"].numel()
            assert state["image_shape"] == [32, 3, 640, 640] and state["targets"] > 0

        def gradient(g):
            state["scaled_gradient"] = dict(
                finite=bool(torch.isfinite(g).all()), norm=float(g.float().norm()) if torch.isfinite(g).all() else None
            )

        def before_step(optimizer, args, kwargs):
            # Native optimizer_step has already unscaled and clipped once; GradScaler invokes this only on real steps.
            assert attn.theta.grad is not None and torch.isfinite(attn.theta.grad).all()
            state["before"] = attn.theta.detach().clone()
            nonzero = bool(attn.theta.grad.count_nonzero())
            report["nonzero_task_gradient_steps"] += int(nonzero)
            state["effective_gradient"] = attn.theta.grad.detach().cpu().tolist()
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in t.model.parameters())

        def after_step(optimizer, args, kwargs):
            report["completed_steps"] += 1
            report["theta_updates"] += int(not torch.equal(state["before"], attn.theta))
            assert torch.isfinite(attn.theta).all()

        handles.extend(
            [
                attn.theta.register_hook(gradient),
                t.optimizer.register_step_pre_hook(before_step),
                t.optimizer.register_step_post_hook(after_step),
                t.model.register_forward_pre_hook(observe_batch),
            ]
        )

    def batch_start(t):
        state["scale_before"] = t.scaler.get_scale()
        state["steps_before"] = report["completed_steps"]
        state["ema_before"] = t.ema.updates
        state.pop("effective_gradient", None)
        state.pop("scaled_gradient", None)

    def batch_end(t):
        assert torch.isfinite(t.loss) and t.batch_size == 32
        attempts = t.ema.updates - state["ema_before"]
        report["optimizer_attempts"] += attempts
        steps = report["completed_steps"] - state["steps_before"]
        if attempts and not steps:
            assert t.scaler.get_scale() < state["scale_before"]
        row = dict(
            batch=len(report["batches"]) + 1,
            epoch=t.epoch,
            loss=float(t.loss),
            image_shape=state["image_shape"],
            targets=state["targets"],
            scale_before=state["scale_before"],
            scale_after=t.scaler.get_scale(),
            accumulate=t.accumulate,
            optimizer_attempt=attempts,
            optimizer_step=steps,
            scaled_theta_gradient=state.get("scaled_gradient"),
            clipped_unscaled_theta_gradient=state.get("effective_gradient"),
            theta=t.model.model[10].m[0].attn.theta.detach().cpu().tolist(),
            beta=t.model.model[10].m[0].attn.beta.detach().cpu().tolist(),
            groups=[
                {k: g.get(k) for k in ("param_group", "lr", "momentum", "weight_decay")}
                for g in t.optimizer.param_groups
            ],
            ema_updates=t.ema.updates,
        )
        report["batches"].append(row)
        common.write_json(directory / "checks.json", report)
        if len(report["batches"]) == report["max_batches"]:
            raise PreflightComplete

    trainer.add_callback("on_pretrain_routine_end", setup)
    trainer.add_callback("on_train_batch_start", batch_start)
    trainer.add_callback("on_train_batch_end", batch_end)
    try:
        try:
            trainer.train()
        except PreflightComplete:
            pass
        assert len(report["batches"]) == 32
        assert report["completed_steps"] >= 2 and report["theta_updates"] > 0
        assert report["nonzero_task_gradient_steps"] > 0  # No per-batch nonzero requirement; R=A can give zero.
        ema = trainer.ema.ema
        assert trainer.ema.updates == report["optimizer_attempts"]
        assert not torch.equal(state["initial_ema"], ema.model[10].m[0].attn.theta)
        assert all(torch.isfinite(v).all() for v in ema.state_dict().values())
        report["ema"] = dict(updates=trainer.ema.updates, theta_updated=True)
        # The formal trainer validates with its own loader/AMP policy, including its native batch size.
        trainer.ema.update_attr(trainer.model, include=["yaml", "nc", "args", "names", "stride", "class_weights"])
        metrics = trainer.validator(trainer=trainer)
        assert trainer.validator.seen == 2404
        report["native_validator"] = dict(
            metrics=metrics,
            images=trainer.validator.seen,
            batch=trainer.test_loader.batch_size,
            args=vars(trainer.validator.args),
        )
        report["reload"] = verify.save_reload_check(ema, directory)
        report.update(
            peak_allocated_bytes=torch.cuda.max_memory_allocated(0),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(0),
            free_bytes=torch.cuda.mem_get_info(0)[0],
        )
    finally:
        for h in handles:
            h.remove()
        for name in ("train_loader", "test_loader"):
            loader = getattr(trainer, name, None)
            if loader is not None:
                loader.close()
        common.write_json(directory / "checks.json", report)
    # Release the disposable training allocations before the independent evaluation process starts.
    del trainer, ema
    state.clear()
    gc.collect()
    torch.cuda.empty_cache()
    subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.experiments.finish_b19_rsc_c2psa",
            "--checkpoint-check",
            str(directory / "preflight.pt"),
            "--data",
            config["data"],
            "--output",
            str(directory / "checkpoint_val"),
        ],
        cwd=ROOT,
        check=True,
    )
    report["passed"] = True
    common.write_json(directory / "checks.json", report)
    return report


def resolve(options):
    """Resolve all original b19 fields and bind code, data and initialization evidence to this execution."""
    raw, config, evidence = common.resolve_recipe(options, MODEL)
    evidence["launch_evidence"] = common.launcher_evidence(options, raw)
    evidence["source_sha256"] = common.source_hashes()
    evidence["runtime"] = runtime_evidence()
    print(json.dumps(evidence["config_differences"], indent=2, ensure_ascii=False))
    return config, evidence


def main(argv=None):
    """Run preflight in a disposable process and always rebuild formal training from seed 42."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("preflight", "train"), required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--baseline-args", type=Path, required=True)
    parser.add_argument("--baseline-launcher", type=Path)
    parser.add_argument("--pretrained", type=Path)
    parser.add_argument("--pretrained-sha256", default=common.PRETRAINED_SHA256)
    parser.add_argument("--project", type=Path, default=ROOT / "runs/detect")
    parser.add_argument("--name", default=NAME, choices=(NAME,))
    options = parser.parse_args(argv)
    os.chdir(ROOT)
    project = options.project.resolve()
    if project != ROOT / "runs/detect":
        raise ValueError("Server stages must use this experiment worktree's runs/detect directory")
    project.mkdir(parents=True, exist_ok=True)
    if options.stage == "train" and (project / NAME).exists():
        raise FileExistsError(f"Preserving existing run: {project / NAME}")
    attempt = Path(tempfile.mkdtemp(prefix=f"{NAME}_{options.stage}_audit_", dir=project))
    try:
        config, evidence = resolve(options)
        common.write_json(attempt / "resolved.json", dict(config=config, evidence=evidence))
        verify.structural_checks(attempt)
        probe = object.__new__(AuditedTrainer)
        probe.args = get_cfg(overrides=config)
        probe.data = dict(nc=1, channels=3, names={0: "crack"})
        weights, _ = load_checkpoint(evidence["initial_path"])
        init_seeds(42, deterministic=True)
        probe.get_model(str(MODEL), weights, False)
        common.write_json(attempt / "initialization.json", probe.weight_audit)
        del probe, weights
        if evidence["runtime"]["mismatches"]:
            raise RuntimeError(f"Pending server validation; b19 runtime differs: {evidence['runtime']['mismatches']}")
        if options.stage == "preflight":
            checks = preflight(config, evidence, attempt / "preflight")
            common.write_json(
                attempt / "passed.json",
                dict(
                    passed=True,
                    commit=evidence["commit"],
                    checks_sha256=common.sha256(attempt / "preflight/checks.json"),
                    peak_reserved_bytes=checks["peak_reserved_bytes"],
                ),
            )
            common.write_json(project / f"{NAME}_preflight_latest.json", dict(directory=str(attempt)))
            print(f"Preflight passed: {attempt}")
            return
        # Always remeasure at training launch, so GPU sharing is assessed against current occupancy.
        arguments = list(sys.argv[1:] if argv is None else argv)
        arguments[arguments.index("--stage") + 1] = "preflight"
        subprocess.run([sys.executable, str(Path(__file__).resolve()), *arguments], cwd=ROOT, check=True)
        receipt_dir = Path(json.loads((project / f"{NAME}_preflight_latest.json").read_text())["directory"])
        passed = json.loads((receipt_dir / "passed.json").read_text())
        assert passed["passed"] and passed["commit"] == evidence["commit"]
        assert passed["checks_sha256"] == common.sha256(receipt_dir / "preflight/checks.json")
        child = json.loads((receipt_dir / "resolved.json").read_text())
        assert child["config"] == config
        for key in ("source_sha256", "initial_sha256", "args_sha256", "data_sha256", "dataset_manifest"):
            assert child["evidence"][key] == evidence[key], f"Evidence changed during preflight: {key}"
        assert evidence["source_sha256"] == common.source_hashes()
        # Formal training starts only after the child has exited and released its model, optimizer and CUDA allocations.
        torch.cuda.empty_cache()
        if torch.cuda.mem_get_info(0)[0] < passed["peak_reserved_bytes"]:
            raise RuntimeError("Available memory is below the measured fixed-batch preflight peak; no batch reduction")
        init_seeds(42, deterministic=True)
        trainer = AuditedTrainer(config)
        provenance = trainer.save_dir / "provenance"
        shutil.copytree(attempt, provenance / "launch")
        shutil.copytree(receipt_dir, provenance / "preflight")
        for key, name in (("args_path", "b19_original_args.yaml"), ("data_path", "original_data.yaml")):
            shutil.copyfile(evidence[key], provenance / name)
        shutil.copyfile(evidence["launch_evidence"]["path"], provenance / "b19_launcher_expanded.txt")
        common.write_json(provenance / "resolved.json", dict(config=config, evidence=evidence))
        trainer.train()
        common.write_json(
            trainer.save_dir / "completed.json",
            dict(
                completed=True,
                run=str(trainer.save_dir),
                epoch=trainer.epoch + 1,
                best_sha256=common.sha256(trainer.best),
                last_sha256=common.sha256(trainer.last),
                commit=common.git("rev-parse", "HEAD"),
            ),
        )
    except Exception:
        common.write_json(attempt / "failed.json", dict(passed=False, error=traceback.format_exc()))
        raise


if __name__ == "__main__":
    main()
