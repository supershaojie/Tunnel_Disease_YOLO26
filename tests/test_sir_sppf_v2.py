"""Bounded v2 formula, real training, reload, and completion contract checks."""

import copy
import json
import os
import shutil
import subprocess
import tarfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch.nn as nn

from tests import test_sir_sppf as v1_tests
from tools.experiments import finish_b19_sir_sppf as common_finish
from tools.experiments import finish_b19_sir_sppf_v2 as finish
from tools.experiments import run_b19_rpca_c2psa as rpca
from tools.experiments import run_b19_sir_sppf as shared
from tools.experiments import run_b19_sir_sppf_v2 as run
from ultralytics import YOLO
from ultralytics.cfg import get_cfg, get_save_dir
from ultralytics.data.dataset import YOLODataset
from ultralytics.nn.modules import SPPF, SPPF_SIR, SPPF_SIR_V2
from ultralytics.utils.metrics import ConfusionMatrix


@pytest.fixture(autouse=True)
def cpu_threads():
    """Restore local test thread settings; reference/reload owns its stricter CPU context."""
    previous = torch.get_num_threads()
    torch.set_num_threads(4)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize(
    "shape,shortcut,n", [((2, 32, 20, 20), True, 3), ((2, 32, 17, 23), False, 3), ((1, 32, 9, 13), True, 2)]
)
def test_identity_rng_and_independent_formula(shape, shortcut, n):
    """Match all three constructors and compare nonzero v2 features against an independent stacked reference."""
    blocks, states = [], []
    for cls in (SPPF, SPPF_SIR, SPPF_SIR_V2):
        torch.manual_seed(42)
        blocks.append(cls(32, 32, 5, n, shortcut).eval())
        states.append(torch.get_rng_state())
    original, v1, v2 = blocks
    assert all(torch.equal(states[0], s) for s in states)
    shared.assert_close_tree(v1.state_dict(), v2.state_dict(), 0, 0)
    for key, value in original.state_dict().items():
        assert torch.equal(value, v2.state_dict()[key])
    assert len(v2.router) == 5
    assert v2.router[0].out_channels == v2.router[2].groups == 16
    assert torch.count_nonzero(v2.router[-1].weight) == torch.count_nonzero(v2.router[-1].bias) == 0
    x = torch.randn(shape)
    with torch.no_grad():
        assert torch.equal(original(x), v1(x)) and torch.equal(original(x), v2(x))
        nn.init.normal_(v2.router[-1].weight, std=0.2)
        nn.init.normal_(v2.router[-1].bias, std=0.2)
        raw = [v2.cv1(x)]
        for _ in range(n):
            raw.append(v2.m(raw[-1]))
        increments = torch.stack(raw[1:]) - torch.stack(raw[:-1])
        gates = v2.router(torch.cat([raw[0], *increments.unbind()], 1)).reshape(shape[0], n, 16, *shape[2:])
        gates = gates.permute(1, 0, 2, 3, 4).tanh()
        corrected = torch.stack(raw[1:]) + 0.5 * gates * increments
        expected = v2.cv2(torch.cat([raw[0], *corrected.unbind()], 1))
        if shortcut:
            expected = expected + x
        torch.testing.assert_close(v2(x), expected, atol=0, rtol=0)
        assert not torch.allclose(v2(x), original(x))


def test_graph_nonzero_and_diagnostic_formula(tmp_path):
    """Check full nano dimensions, an r1-only counterexample, and the diagnosis uses the same independent formula."""
    rng = torch.get_rng_state()
    report = run.structural_checks(tmp_path)
    assert torch.equal(rng, torch.get_rng_state())
    assert report["added_parameters"] == 14896 and report["candidate_parameters"] == 2519086
    assert report["independent_correction"]["passed"]
    model = YOLO(str(run.MODEL))
    assert model.model.model[-1].nc == 1 and type(model.model.model[9]) is SPPF_SIR_V2
    module = SPPF_SIR_V2(32, 32, 5, 3, True).eval()
    with torch.no_grad():
        module.router[-1].bias[:16].fill_(0.75)
        x = torch.randn(2, 32, 17, 23)
        stats = finish.correction_statistics(module, x, module(x))
    assert stats["scales"][0]["correction_norm"] > 0
    assert all(s["correction_norm"] == 0 for s in stats["scales"][1:])


def test_pretrained_real_amp_ema_and_fresh_reload(tmp_path):
    """Use three batch=2 real augmented updates locally, explicitly not a server preflight receipt."""
    torch.set_num_threads(1)
    trainer = v1_tests.native_probe(
        "cuda:0" if torch.cuda.is_available() else "cpu", trainer_type=run.AuditedTrainer, model=run.MODEL
    )
    assert trainer.weight_audit["v1_v2_all_state_equal"]
    assert trainer.weight_audit["common_keys"] == 708 and len(trainer.weight_audit["loaded_keys"]) == 606
    ids = {id(p) for group in trainer.optimizer.param_groups for p in group["params"]}
    assert all(id(p) in ids for p in trainer.model.model[9].router.parameters())
    images = v1_tests.sample_images(tmp_path)
    listing = tmp_path / "images.txt"
    listing.write_text("\n".join(map(str, images)), encoding="utf-8")
    data = YOLODataset(
        img_path=str(listing), imgsz=640, batch_size=2, augment=True, hyp=trainer.args, data=trainer.data, cache=False
    )
    rows = [
        shared.gradient_check(trainer, data.collate_fn([data[0], data[1]]), trainer.device.type == "cuda")
        for _ in range(3)
    ]
    assert all(value == 0 for key, value in rows[0]["new_gradient_norms"].items() if ".router.4." not in key)
    assert all(value > 0 for value in rows[-1]["new_gradient_norms"].values())
    assert type(trainer.ema.ema.model[9]) is SPPF_SIR_V2
    original = {k: v.clone() for k, v in trainer.ema.ema.state_dict().items()}
    rng = torch.get_rng_state()
    shared.save_reload_check(trainer.ema.ema, tmp_path, SPPF_SIR_V2)
    assert torch.equal(rng, torch.get_rng_state()) and torch.get_num_threads() == 1
    shared.assert_close_tree(original, trainer.ema.ema.state_dict(), 0, 0)
    reload_report = json.loads((tmp_path / "reload_check.json").read_text())
    assert reload_report["state_exact"] and reload_report["state_keys"] == 714
    assert max(row["max_abs"] for row in reload_report["raw"]) == 0
    checkpoint = torch.load(tmp_path / "preflight.pt", weights_only=False)
    # Corrupt a router parameter and a BN buffer separately: the independent reference must reject both.
    for key in ("model.9.router.4.weight", "model.9.cv1.bn.num_batches_tracked"):
        damaged = copy.deepcopy(checkpoint)
        damaged["ema"].state_dict()[key].view(-1)[0] += 1
        torch.save(damaged, tmp_path / "preflight.pt")
        with pytest.raises(AssertionError, match=rf"state\.{key}.*outside_tolerance"):
            shared.reload_in_process(tmp_path / "preflight.pt", SPPF_SIR_V2)
    shared.write_json(
        run.ROOT / "runs/sir_v2_development/local_real_gradient.json",
        dict(
            formal_preflight=False,
            batch=2,
            imgsz=640,
            device=str(trainer.device),
            rows=rows,
            weights=trainer.weight_audit,
            reload=reload_report,
        ),
    )


def test_native_optimizer_oom_and_rng(tmp_path):
    """Exercise native setup and its real OOM catch boundary with the v2 subclass, without an epoch."""
    v1_tests.test_native_setup_oom_and_audit_rng(tmp_path, trainer_type=run.AuditedTrainer, model=run.MODEL)


@pytest.mark.parametrize("experiment", [run, rpca])
def test_fp32_val_test_reuse_and_conflicting_evidence(tmp_path, experiment):
    """Exercise actual Model.val orchestration twice, stub only inference, and never consume held-out test images."""
    data = tmp_path / "data.yaml"
    data.write_text("val: images/val\ntest: images/test\n", encoding="utf-8")
    evidence = dict(weight="fixture.pt", version=2, data_sha256=shared.sha256(data))
    calls = []
    metrics = SimpleNamespace(
        results_dict={"metrics/precision(B)": 0.123456789},
        box=SimpleNamespace(all_ap=np.linspace(0.1, 0.9, 10).reshape(1, 10), mp=0.123456789, mr=0.987654321),
    )

    class Validator:
        def __init__(self, args, _callbacks):
            self.args = get_cfg(overrides=args)
            self.save_dir = get_save_dir(self.args)
            self.callbacks = _callbacks
            self.metrics = metrics
            self.seen, targets = finish.COUNTS[self.args.split]
            self.metrics.nt_per_class = np.array([targets])
            self.confusion_matrix = SimpleNamespace(matrix=np.zeros((2, 2)))
            self.jdict, self.speed = [], {}

        def __call__(self, model):
            self.model = SimpleNamespace(model=model)
            assert self.args.batch == 32 and self.args.imgsz == 640 and self.args.workers == 8
            assert self.args.conf == 0.001 and self.args.iou == 0.7 and self.args.max_det == 300
            assert self.args.quantize is None and self.args.rect and not self.args.augment
            calls.append(self.args.split)
            for name in finish.EVAL_ARTIFACTS:
                if name.endswith(".png"):
                    (self.save_dir / name).write_bytes(b"Synthetic plot contract fixture, not a real evaluation")
            for callback in self.callbacks["on_val_end"]:
                callback(self)

    def model_factory(path):
        model = YOLO(str(experiment.MODEL))
        model._smart_load = lambda key: Validator
        return model

    with patch.object(finish, "provenance", return_value=evidence), patch.object(
        common_finish, "YOLO", side_effect=model_factory
    ):
        finish.evaluate_best(tmp_path, data, experiment)
        first = json.loads((tmp_path / "evaluation.json").read_text())
        finish.evaluate_best(tmp_path, data, experiment)
        assert calls == ["val", "test"]
        for split, item in first["reports"].items():
            record = json.loads((tmp_path / item["path"]).read_text())
            assert record["metrics"]["P"] == 0.123456789 and len(record["ap_by_iou"]) == 10
            assert record["images"] == finish.COUNTS[split][0]
            assert record["confusion_matrix"]["confidence"] == 0.25
            assert (tmp_path / item["path"]).with_name("predictions.json").read_text().strip() == "[]"
        evidence["weight_sha256"] = "different checkpoint"
        finish.evaluate_best(tmp_path, data, experiment)
        assert calls == ["val", "test", "val", "test"]
        assert len(list((tmp_path / "evaluation").glob("*/metrics.json"))) == 4
        current = json.loads((tmp_path / "evaluation.json").read_text())
        (tmp_path / current["reports"]["val"]["path"]).with_name("BoxPR_curve.png").write_bytes(b"damaged plot")
        finish.evaluate_best(tmp_path, data, experiment)
        assert calls == ["val", "test", "val", "test", "val"]
        assert len(list((tmp_path / "evaluation").glob("*/metrics.json"))) == 5


def test_native_confusion_matrix_thresholds():
    """Confirm the recorded native defaults differ from AP conf/IoU: 0.2 drops, 0.3 and IoU=0.6 match."""
    matrix = ConfusionMatrix(names={0: "crack"})
    target = dict(cls=torch.tensor([0]), bboxes=torch.tensor([[0.0, 0.0, 1.0, 1.0]]))
    prediction = dict(cls=torch.tensor([0]), conf=torch.tensor([0.2]), bboxes=torch.tensor([[0.25, 0.0, 1.25, 1.0]]))
    matrix.process_batch(prediction, target, conf=0.001)
    assert matrix.matrix[0, 0] == 0
    prediction["conf"] = torch.tensor([0.3])
    matrix.process_batch(prediction, target, conf=0.001)
    assert matrix.matrix[0, 0] == 1


def test_v2_entry_binds_all_version_dependencies():
    """The preflight child, model, trainer, checks and hashed files all point to v2 without changing v1 globals."""
    with patch.object(shared, "main", return_value=0) as entry:
        assert run.main(["--help"]) == 0
    options = entry.call_args.kwargs
    assert options["model"] == run.MODEL and options["trainer_type"].block_type is SPPF_SIR_V2
    assert options["entrypoint"] == Path(run.__file__).resolve() and options["name"] == run.NAME
    assert options["structure_check"] is run.structural_checks
    assert set(run.SOURCE_FILES) <= set(options["source_files"])
    assert shared.AuditedTrainer.block_type is SPPF_SIR and shared.MODEL.name.endswith("v1.yaml")


@pytest.mark.parametrize("experiment", [run, rpca])
def test_v2_package_current_reports_and_source(tmp_path, experiment):
    """Verify a synthetic light archive includes both FP32 reports and v2 inheritance, excluding old outputs."""
    fixture = tmp_path / "synthetic_run"
    evidence = dict(status="", version=2)
    reports = {}
    for split in finish.COUNTS:
        folder = fixture / "evaluation" / split
        folder.mkdir(parents=True)
        for name in finish.EVAL_ARTIFACTS:
            (folder / name).write_bytes(b"Synthetic artifact; not a training or evaluation result")
        shared.write_json(
            folder / "metrics.json",
            dict(evidence=evidence, artifacts={name: shared.sha256(folder / name) for name in finish.EVAL_ARTIFACTS}),
        )
        reports[split] = dict(path=f"evaluation/{split}/metrics.json", sha256=shared.sha256(folder / "metrics.json"))
    shared.write_json(fixture / "evaluation.json", dict(evidence=evidence, reports=reports))
    shared.write_json(fixture / "evaluation/diagnostic/metrics.json", dict(evidence=dict(evidence, samples=[])))
    shared.write_json(
        fixture / "diagnostics.json",
        dict(
            path="evaluation/diagnostic/metrics.json",
            sha256=shared.sha256(fixture / "evaluation/diagnostic/metrics.json"),
        ),
    )
    for name in (
        "args.yaml",
        "results.csv",
        "train.log",
        "completed.json",
        "val_metrics.json",
        "weights/best.pt",
        "weights/last.pt",
        "weights/epoch20.pt",
        "evaluation/old/duplicate.png",
    ):
        path = fixture / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"Synthetic archive contract fixture, not a training result")
    output = tmp_path / "synthetic-v2.tar.gz"
    with patch.object(finish, "provenance", return_value=evidence), patch.object(
        finish, "evaluate_best", side_effect=AssertionError("package must not evaluate")
    ):
        finish.package(
            fixture,
            tmp_path / "data.yaml",
            output,
            experiment=experiment,
            include_last=experiment is rpca,
            source_files=("ultralytics/nn/modules/rpca_c2psa.py",) if experiment is rpca else (),
        )
    with tarfile.open(output) as archive:
        names = archive.getnames()
        assert "run/weights/best.pt" in names
        assert "run/evaluation/val/metrics.json" in names and "run/evaluation/test/metrics.json" in names
        assert "source/ultralytics/nn/modules/sir_sppf.py" in names
        assert "source/ultralytics/nn/modules/sir_sppf_v2.py" in names
        assert all("evaluation/old" not in name and not name.endswith("epoch20.pt") for name in names)
        assert ("run/weights/last.pt" in names) == (experiment is rpca)
        if experiment is rpca:
            assert "source/ultralytics/nn/modules/rpca_c2psa.py" in names
    assert output.with_name(output.name + ".sha256").read_text().split()[0] == shared.sha256(output)


def test_shell_keeps_python_tee_status_and_attempts(tmp_path):
    """Use the real shell wrapper with disposable failing commands, without starting a model or using server paths."""
    bash = os.environ.get("SIR_BASH") or shutil.which("bash")
    if not bash:
        pytest.skip("Bash required for wrapper contract")
    script = tmp_path / "work/tools/experiments/server_b19_sir_sppf_v2.sh"
    script.parent.mkdir(parents=True)
    shutil.copy2(run.ROOT / "tools/experiments/server_b19_sir_sppf_v2.sh", script)
    binaries = tmp_path / "bin"
    binaries.mkdir()

    def executable(name, code):
        path = binaries / name
        path.write_bytes(("#!/usr/bin/env bash\n" + code + "\n").encode())
        path.chmod(0o755)

    executable("flock", "exit 0")
    executable("python", 'printf "%s\\n" "$@"; exit 7')
    command = 'fixture="$(cd -- "$1" && pwd -P)"; export PATH="$fixture/bin:/usr/bin:/bin:$PATH"; export B19_PYTHON="$fixture/bin/python"; bash "$fixture/work/tools/experiments/server_b19_sir_sppf_v2.sh" preflight'
    # git rev-parse records identity only; the temporary wrapper is intentionally outside a repository.
    executable("git", "printf 'synthetic-wrapper-fixture\\n'")
    result = subprocess.run([bash, "-c", command, "--", tmp_path.as_posix()], capture_output=True, text=True)
    assert result.returncode == 7, result.stdout + result.stderr
    project = tmp_path / "work/runs/detect"
    status = project / f"{run.NAME}_preflight.process_status.json"
    assert json.loads(status.read_text()) == dict(python=7, tee=0)
    assert "run_b19_sir_sppf_v2.py" in (project / f"{run.NAME}_preflight.console.log").read_text()
    executable("tee", "cat >/dev/null; exit 9")
    result = subprocess.run([bash, "-c", command, "--", tmp_path.as_posix()], capture_output=True, text=True)
    assert result.returncode == 9, result.stdout + result.stderr
    assert json.loads(status.read_text()) == dict(python=7, tee=9)
    assert len(list(project.glob(f"{run.NAME}_preflight.attempt.*/exit_status"))) == 2
    assert not (project / run.NAME).exists() and not list(project.rglob("completed.json"))
