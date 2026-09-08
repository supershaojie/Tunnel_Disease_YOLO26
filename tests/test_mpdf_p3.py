"""Independent formula and native lifecycle regression coverage for MPDF-P3 v1."""

import copy
import os
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from tools.experiments import b19_common as common
from tools.experiments import verify_b19_mpdf_p3 as verify
from tools.experiments.mpdf_experiment import V1
from tools.experiments.run_b19_mpdf_p3 import AuditedTrainer
from ultralytics.nn.modules.mpdf_p3 import MPDFP3
from ultralytics.nn.tasks import load_checkpoint
from ultralytics.utils.torch_utils import ModelEMA


@pytest.fixture(autouse=True, scope="module")
def cpu_threads():
    """Keep small numerical checks reproducible and restore caller thread policy."""
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def test_formula_and_full_graph(tmp_path):
    """Verify explicit phase signs, random coefficients, graph/RNG and two full input shapes."""
    verify.structural_checks(tmp_path)


@pytest.mark.parametrize("bad", ["count", "rank", "batch", "channel", "spatial"])
def test_input_contract(bad):
    """Reject wiring errors at the module boundary rather than repairing the geometry."""
    m = MPDFP3(2, 2)
    x = [torch.randn(1, 2, 6, 10), torch.randn(1, 2, 6, 10), torch.randn(1, 2, 3, 5)]
    if bad == "count":
        x.pop()
    elif bad == "rank":
        x[0] = x[0][0]
    elif bad == "batch":
        x[1] = x[1].repeat(2, 1, 1, 1)
    elif bad == "channel":
        x[2] = x[2][:, :1]
    elif bad == "spatial":
        x[1] = x[1][:, :, :-1]
    with pytest.raises(ValueError, match="MPDFP3"):
        m(x)


def test_input_gradients_and_delayed_data_gradients():
    """Prove W2 updates from data before W1/DW3 become reachable; decay is disabled here."""
    torch.manual_seed(42)
    m = MPDFP3(2, 2)
    optimizer = torch.optim.SGD(m.parameters(), lr=0.1, weight_decay=0)
    h = torch.randn(2, 2, 4, 5, requires_grad=True)
    low = torch.randn(2, 2, 8, 10, requires_grad=True)
    up = F.interpolate(h.detach(), scale_factor=2, mode="nearest").requires_grad_()
    target = torch.randn(2, 4, 8, 10)
    before = m.project.weight.detach().clone()
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        for t in (h, low, up):
            t.grad = None
        (m([up, low, h]) * target).sum().backward()
        assert all(p.grad is not None and p.grad.isfinite().all() for p in m.parameters())
        assert m.project.weight.grad.norm() > 0
        assert up.grad.norm() > 0 and low.grad.norm() > 0
        if step == 0:
            assert (
                m.reduce.weight.grad.count_nonzero() == m.dw.weight.grad.count_nonzero() == h.grad.count_nonzero() == 0
            )
        else:
            assert m.reduce.weight.grad.norm() > 0 and m.dw.weight.grad.norm() > 0 and h.grad.norm() > 0
        optimizer.step()
    assert not torch.equal(before, m.project.weight)


def pretrained_probe():
    """Use the original COCO weight through the actual native single-class Trainer reconstruction."""
    path = Path(os.environ.get("B19_TEST_PRETRAINED", common.ROOT / "yolo26n.pt"))
    if not path.is_file():
        pytest.skip("Original COCO yolo26n.pt needed; set B19_TEST_PRETRAINED")
    assert common.sha256(path) == common.PRETRAINED_SHA256
    source, _ = load_checkpoint(path)
    probe = object.__new__(AuditedTrainer)
    probe.experiment = V1
    probe.args = common.get_cfg(overrides=common.REFERENCE["args"])
    probe.data = dict(nc=1, channels=3, names={0: "crack"})
    torch.manual_seed(42)
    model = probe.get_model(str(V1.model), source, False)
    assert probe.weight_audit["common_keys"] == 708 and len(probe.weight_audit["loaded_keys"]) == 606
    return probe, model


def test_trainer_musgd_ema_checkpoint_and_validator(tmp_path):
    """Exercise native groups, real steps, EMA, fresh-process reload/fuse and an actual CPU Validator."""
    probe, model = pretrained_probe()
    branch = model.model[15]
    torch.save(branch.state_dict(), tmp_path / "initial_mpdf.pt")
    model.args = probe.args
    model.train()
    optimizer = probe.build_optimizer(model, "MuSGD", 0.01, 0.937, 0.0005, 1000)
    ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
    assert all(id(p) in ids for p in branch.parameters())
    ema = ModelEMA(model)
    rows = []
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        x = torch.rand(2, 3, 64, 96)
        batch = dict(
            img=x,
            batch_idx=torch.tensor([0.0, 1.0]),
            cls=torch.zeros(2, 1),
            bboxes=torch.tensor([[0.5, 0.5, 0.2, 0.4], [0.3, 0.3, 0.3, 0.2]]),
        )
        loss, _ = model(batch)
        loss.sum().backward()
        gradients = {k: float(p.grad.norm()) for k, p in branch.named_parameters()}
        assert gradients["project.weight"] > 0
        if step:
            assert gradients["reduce.weight"] > 0 and gradients["dw.weight"] > 0
        before = branch.project.weight.detach().clone()
        optimizer.step()
        ema.update(model)
        assert not torch.equal(before, branch.project.weight)
        rows.append(gradients)
    assert ema.updates == 2 and ema.ema.model[15].project.weight.count_nonzero()
    # Optimizer resume restores existing learned values, including W2, without constructor re-zeroing.
    resumed = copy.deepcopy(model)
    resumed.load_state_dict(model.state_dict(), strict=True)
    resumed_optimizer = probe.build_optimizer(resumed, "MuSGD", 0.01, 0.937, 0.0005, 1000)
    resumed_optimizer.load_state_dict(optimizer.state_dict())
    common.assert_close_tree(model.state_dict(), resumed.state_dict(), 0, 0)
    assert len(resumed_optimizer.state) == len(optimizer.state)
    ema.update_attr(model, include=["yaml", "nc", "args", "names", "stride"])
    common.write_json(tmp_path / "local_training.json", dict(data_gradients=rows, optimizer="MuSGD", ema_updates=2))
    verify.save_reload_check(ema.ema, tmp_path)
    import cv2
    import numpy as np
    from tools.experiments.finish_b19_mpdf_p3 import bind_report, diagnose, evaluate
    from ultralytics.utils import YAML

    images = tmp_path / "images/val"
    labels = tmp_path / "labels/val"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    for i in range(16):
        cv2.imwrite(str(images / f"sample{i}.png"), np.full((96, 128, 3), 80 + i * 4, dtype=np.uint8))
        (labels / f"sample{i}.txt").write_text("0 0.5 0.5 0.2 0.4\n")
    data = tmp_path / "data.yaml"
    YAML.save(
        data, dict(path=str(tmp_path), train="images/val", val="images/val", test="images/val", names={0: "crack"})
    )
    import shutil

    (tmp_path / "weights").mkdir()
    shutil.copyfile(tmp_path / "preflight.pt", tmp_path / "weights/best.pt")
    for split in ("val", "test"):
        folder = tmp_path / f"{split}_evaluation"
        report = evaluate(tmp_path / "weights/best.pt", data, folder, split, device="cpu", batch=2, workers=0)
        assert report["images"] == 16 and report["inference_path"]["one2one"] > 0
        bind_report(tmp_path, f"{split}_fp32", folder)
    common.write_json(
        tmp_path / "provenance/preflight/preflight/checks.json",
        dict(synthetic_local_only=True, batches=rows, gradient_steps={k: sum(r[k] > 0 for r in rows) for k in rows[0]}),
    )
    diagnose(tmp_path, data, tmp_path / "diagnosis", device="cpu")
    assert (tmp_path / "diagnosis/feature_corrections.png").is_file()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable; AMP check belongs on server")
def test_cuda_amp_formula():
    """Check local arithmetic under CUDA autocast separately from the formal batch=32 preflight."""
    verify.formula_checks(device="cuda", amp=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_native_autocast_mixed_inputs():
    """Native nearest may return FP32 while low/high stay FP16; preserve native promotion."""
    _, model = pretrained_probe()
    model.cuda().eval()
    seen = []
    handle = model.model[15].register_forward_hook(lambda m, args, output: seen.append((args[0], output)))
    with torch.no_grad(), torch.autocast("cuda"):
        model(torch.rand(1, 3, 384, 672, device="cuda"))
    handle.remove()
    inputs, output = seen[0]
    torch.testing.assert_close(output, torch.cat(inputs[:2], 1), atol=0, rtol=0)
    assert all(torch.isfinite(t).all() for t in inputs)


def test_attempt_failure_and_retry(tmp_path, monkeypatch):
    """An active automatic preflight must never expose the previous successful exit code."""
    import json
    import subprocess
    import sys

    from tools.experiments.run_b19_mpdf_p3 import execute_preflight

    audit = tmp_path / "audit"
    audit.mkdir()
    status = tmp_path / f"{V1.name}_preflight.exit_status"
    status.write_text("0\n")
    script = (
        "import os,pathlib,sys; p=pathlib.Path(os.environ['B19_STAGE_ATTEMPT']); "
        f"assert not pathlib.Path({str(status)!r}).exists(); "
        "assert (p/'process_status.json').is_file(); print('child executed'); sys.exit(7)"
    )
    with pytest.raises(subprocess.CalledProcessError):
        execute_preflight([sys.executable, "-c", script], tmp_path, audit)
    failed = Path((tmp_path / f"{V1.name}_preflight.current_attempt").read_text().strip())
    assert (failed / "exit_status").read_text().strip() == "7"
    assert (failed / "previous.exit_status").read_text().strip() == "0"
    passed = execute_preflight([sys.executable, "-c", "print('retry executed')"], tmp_path, audit)
    assert passed != failed and (failed / "console.log").is_file()
    assert json.loads((passed / "process_status.json").read_text())["exit_status"] == 0


def test_incomplete_package_is_rejected(tmp_path, monkeypatch):
    """Missing weights, evaluation and diagnosis must produce an explicit incomplete result."""
    from tools.experiments.finish_b19_mpdf_p3 import package

    monkeypatch.setattr(common, "require_clean_source", lambda: None)
    with pytest.raises(FileNotFoundError, match="Incomplete result; missing stages/artifacts"):
        package(tmp_path / V1.name, None)
