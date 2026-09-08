"""Prepare or train b19 + MSI-C2PSA with the unchanged native training lifecycle."""

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
from tools.experiments import verify_b19_msi_c2psa as verify
from tools.experiments.msi_experiment import V1
from ultralytics.cfg import DEFAULT_CFG_DICT, get_cfg
from ultralytics.data.utils import check_det_dataset
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import load_checkpoint
from ultralytics.utils.torch_utils import init_seeds

NAME = V1.name
MODEL = V1.model


class AuditedTrainer(DetectionTrainer):
    """Own only fixed-batch output and initialization audits; retain native training and optimization."""

    new_marker = ".msi."
    experiment = V1

    def __init__(self, overrides, _callbacks=None, experiment=V1):
        """Reserve the exact experiment directory without auto-incrementing or overwriting another run."""
        self.experiment = experiment
        self.requested_config = overrides.copy()
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
        self.weight_audit = verify.audit(baseline, candidate, weights, self.experiment)
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
    branch = trainer.model.model[10].m[0].msi
    assert not branch.project.weight.count_nonzero() and not branch.project.bias.count_nonzero()
    assert type(trainer.ema.ema.model[10]) is trainer.experiment.block_type
    common.assert_close_tree(branch.state_dict(), trainer.ema.ema.model[10].m[0].msi.state_dict(), 0, 0)
    effective = vars(trainer.args)
    changed = {k: [v, effective.get(k)] for k, v in trainer.requested_config.items() if effective.get(k) != v}
    common.write_json(
        trainer.save_dir / "provenance/effective_config.json", dict(config=effective, setup_differences=changed)
    )
    assert not changed, f"Native setup altered the requested b19 recipe: {changed}"
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


def preflight(config, evidence, directory, experiment=V1):
    """Bound native training to <=32 batches with >=2 real steps, then validate 32 fixed val images."""
    from tools.experiments.finish_b19_msi_c2psa import fixed_subset

    directory.mkdir(parents=True, exist_ok=False)
    common.write_json(directory / "environment.json", evidence)
    trainer = AuditedTrainer(dict(config, project=str(directory), name="check"), experiment=experiment)
    report = dict(
        passed=False,
        batches=[],
        optimizer_attempts=0,
        completed_steps=0,
        project_updates=0,
        gradient_steps={},
        max_batches=32,
        loop="unmodified BaseTrainer._do_train and optimizer_step",
    )
    handles, state = [], {}

    def setup(t):
        assert len(t.train_loader.dataset) == 8414 and len(t.test_loader.dataset) == 2404
        assert t.amp and t.scaler.is_enabled() and type(t.optimizer).__name__ == "MuSGD"
        assert not t.optimizer.state and t.ema.updates == 0
        branch = t.model.model[10].m[0].msi
        state["initial_ema"] = t.ema.ema.model[10].m[0].msi.project.weight.detach().clone()
        torch.cuda.reset_peak_memory_stats(0)

        def observe_batch(module, inputs):
            batch = inputs[0]
            state["image_shape"] = list(batch["img"].shape)
            state["targets"] = batch["cls"].numel()
            assert state["image_shape"] == [32, 3, 640, 640] and state["targets"] > 0

        def before_step(optimizer, args, kwargs):
            # GradScaler invokes these hooks only on actual steps, after native unscale and clipping.
            state["before"] = branch.project.weight.detach().clone()
            row = {}
            for name, parameter in branch.named_parameters():
                assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
                row[name] = float(parameter.grad.float().norm())
                report["gradient_steps"][name] = report["gradient_steps"].get(name, 0) + int(row[name] > 0)
            state["effective_gradient"] = row
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in t.model.parameters())

        def after_step(optimizer, args, kwargs):
            report["completed_steps"] += 1
            report["project_updates"] += int(not torch.equal(state["before"], branch.project.weight))
            assert all(torch.isfinite(p).all() for p in branch.parameters())

        handles.extend(
            [
                t.optimizer.register_step_pre_hook(before_step),
                t.optimizer.register_step_post_hook(after_step),
                t.model.register_forward_pre_hook(observe_batch),
            ]
        )

    def batch_start(t):
        state.update(
            scale_before=t.scaler.get_scale(), steps_before=report["completed_steps"], ema_before=t.ema.updates
        )
        state.pop("effective_gradient", None)

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
            ema_updates=t.ema.updates,
            clipped_unscaled_data_gradients=state.get("effective_gradient"),
            project_norm=float(t.model.model[10].m[0].msi.project.weight.norm()),
            groups=[
                {k: g.get(k) for k in ("param_group", "lr", "momentum", "weight_decay")}
                for g in t.optimizer.param_groups
            ],
        )
        report["batches"].append(row)
        common.write_json(directory / "checks.json", report)
        print("MSI preflight", json.dumps(row), flush=True)
        enough = report["completed_steps"] >= 2 and all(
            report["gradient_steps"].get(k, 0) > 0 for k in ("dw3.weight", "dw5.weight", "project.weight")
        )
        if enough or len(report["batches"]) >= report["max_batches"]:
            raise PreflightComplete

    trainer.add_callback("on_pretrain_routine_end", setup)
    trainer.add_callback("on_train_batch_start", batch_start)
    trainer.add_callback("on_train_batch_end", batch_end)
    try:
        try:
            trainer.train()
        except PreflightComplete:
            pass
        assert 2 <= len(report["batches"]) <= report["max_batches"]
        assert report["completed_steps"] >= 2 and report["project_updates"] > 0
        assert all(report["gradient_steps"].get(k, 0) > 0 for k in ("dw3.weight", "dw5.weight", "project.weight"))
        for h in handles:
            h.remove()
        handles.clear()
        ema = trainer.ema.ema
        assert trainer.ema.updates == report["optimizer_attempts"] >= 2
        assert not torch.equal(state["initial_ema"], ema.model[10].m[0].msi.project.weight)
        assert all(torch.isfinite(v).all() for v in ema.state_dict().values())
        report["ema"] = dict(updates=trainer.ema.updates, project_updated=True)
        trainer.ema.update_attr(trainer.model, include=["yaml", "nc", "args", "names", "stride", "class_weights"])
        subset, images = fixed_subset(config["data"], directory / "fixed_val", 32)
        trainer.test_loader.close()
        trainer.test_loader = trainer.get_dataloader(
            str(subset.with_suffix(".txt")), batch_size=32, rank=-1, mode="val"
        )
        trainer.validator.dataloader = trainer.test_loader
        metrics = trainer.validator(trainer=trainer)
        assert trainer.validator.seen == len(images) == 32
        report["native_validator"] = dict(
            metrics=metrics, images=32, image_ids=[str(p) for p in images], batch=32, args=vars(trainer.validator.args)
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
    # Release all GPU tensors before the checkpoint's separate FP32 AutoBackend/Validator process.
    del trainer, ema
    state.clear()
    gc.collect()
    torch.cuda.empty_cache()
    subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.experiments.finish_b19_msi_c2psa",
            "--checkpoint-check",
            str(directory / "preflight.pt"),
            "--data",
            str(subset),
            "--output",
            str(directory / "checkpoint_val"),
            "--expected-images",
            "32",
        ],
        cwd=ROOT,
        check=True,
    )
    report["passed"] = True
    common.write_json(directory / "checks.json", report)
    return report


def resolve(options, experiment=V1):
    """Resolve all original b19 fields and bind code, data and initialization evidence to this execution."""
    raw, config, evidence = common.resolve_recipe(options, experiment.model)
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
    parser.add_argument("--name")
    options = parser.parse_args(argv)
    experiment = V1
    name = experiment.name
    if options.name is None:
        options.name = name
    if options.name != name:
        parser.error(f"Version {experiment.version} requires --name {name}")
    os.chdir(ROOT)
    project = options.project.resolve()
    if project != ROOT / "runs/detect":
        raise ValueError("Server stages must use this experiment worktree's runs/detect directory")
    project.mkdir(parents=True, exist_ok=True)
    if options.stage == "train" and (project / name).exists():
        raise FileExistsError(f"Preserving existing run: {project / name}")
    attempt = Path(tempfile.mkdtemp(prefix=f"{name}_{options.stage}_audit_", dir=project))
    if os.environ.get("B19_STAGE") == options.stage and os.environ.get("B19_STAGE_ATTEMPT"):
        (Path(os.environ["B19_STAGE_ATTEMPT"]) / "audit_path.txt").write_text(str(attempt) + "\n")
    print(f"Audit attempt: {attempt}", flush=True)
    try:
        config, evidence = resolve(options, experiment)
        common.write_json(attempt / "resolved.json", dict(config=config, evidence=evidence))
        if evidence["runtime"]["mismatches"]:
            raise RuntimeError(f"Pending server validation; b19 runtime differs: {evidence['runtime']['mismatches']}")
        if options.stage == "preflight":
            # These checks belong to the disposable child, never repeated in the formal training parent.
            verify.structural_checks(attempt, experiment)
            probe = object.__new__(AuditedTrainer)
            probe.experiment = experiment
            probe.args = get_cfg(overrides=config)
            probe.data = dict(nc=1, channels=3, names={0: "crack"})
            weights, _ = load_checkpoint(evidence["initial_path"])
            init_seeds(42, deterministic=True)
            probe.get_model(str(experiment.model), weights, False)
            common.write_json(attempt / "initialization.json", probe.weight_audit)
            del probe, weights
            verify.gpu_initialization_check(evidence["initial_path"], attempt)
            init_seeds(42, deterministic=True)
            checks = preflight(config, evidence, attempt / "preflight", experiment)
            common.write_json(
                attempt / "passed.json",
                dict(
                    passed=True,
                    commit=evidence["commit"],
                    checks_sha256=common.sha256(attempt / "preflight/checks.json"),
                    peak_reserved_bytes=checks["peak_reserved_bytes"],
                ),
            )
            common.write_json(project / f"{name}_preflight_latest.json", dict(directory=str(attempt)))
            print(f"Preflight passed: {attempt}")
            return
        # Always remeasure at training launch, so GPU sharing is assessed against current occupancy.
        arguments = list(sys.argv[1:] if argv is None else argv)
        arguments[arguments.index("--stage") + 1] = "preflight"
        child_log = attempt / "preflight.log"
        with child_log.open("w", encoding="utf-8") as stream:
            child_process = subprocess.Popen(
                [sys.executable, "-u", str(Path(__file__).resolve()), *arguments],
                cwd=ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            common.write_json(attempt / "preflight_process.json", dict(pid=child_process.pid, state="running"))
            for line in child_process.stdout:
                print(line, end="", flush=True)
                stream.write(line)
                stream.flush()
            code = child_process.wait()
        (attempt / "preflight.exit_status").write_text(str(code) + "\n")
        common.write_json(
            attempt / "preflight_process.json", dict(pid=child_process.pid, state="exited", exit_status=code)
        )
        if code:
            raise subprocess.CalledProcessError(code, child_process.args)
        receipt_dir = Path(
            json.loads((project / f"{name}_preflight_latest.json").read_text(encoding="utf-8"))["directory"]
        )
        passed = common.verify_preflight(receipt_dir)
        child = json.loads((receipt_dir / "resolved.json").read_text(encoding="utf-8"))
        assert child["config"] == config
        for key in ("source_sha256", "initial_sha256", "args_sha256", "data_sha256", "dataset_manifest"):
            assert child["evidence"][key] == evidence[key], f"Evidence changed during preflight: {key}"
        assert evidence["source_sha256"] == common.source_hashes()
        common.require_clean_source()
        # Formal training starts only after the child has exited and released its model, optimizer and CUDA allocations.
        torch.cuda.empty_cache()
        if torch.cuda.mem_get_info(0)[0] < passed["peak_reserved_bytes"]:
            raise RuntimeError("Available memory is below the measured fixed-batch preflight peak; no batch reduction")
        init_seeds(42, deterministic=True)
        trainer = AuditedTrainer(config, experiment=experiment)
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
