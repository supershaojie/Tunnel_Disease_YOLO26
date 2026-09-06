"""Focused v2 geometry, numerical precision, native initialization and persistence checks."""

import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from tools.experiments import run_b19_dcrstrip as shared
from tools.experiments import run_b19_dcrstrip_v2 as v2
from ultralytics.cfg import get_cfg
from ultralytics.nn.modules import C3k2, C3k2_DCRStripV2, DCRStripV2
from ultralytics.nn.tasks import load_checkpoint
from ultralytics.utils.torch_utils import ModelEMA, autocast, init_seeds


@pytest.fixture(autouse=True)
def bounded_threads():
    """Avoid excessive thread startup for small CPU checks."""
    previous = torch.get_num_threads()
    torch.set_num_threads(4)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("shape", [(2, 64, 17, 17), (2, 64, 13, 23), (1, 64, 1, 3)])
def test_main_path_and_all_gradients(shape):
    """Preserve C3k2 tensors/RNG, both forwards, and all nondegenerate new gradient paths."""
    init_seeds(42)
    baseline = C3k2(64, 128, e=0.25).eval()
    after = torch.rand(4)
    init_seeds(42)
    candidate = C3k2_DCRStripV2(64, 128, e=0.25).eval()
    assert torch.equal(after, torch.rand(4))
    assert all(torch.equal(t, candidate.state_dict()[k]) for k, t in baseline.state_dict().items())
    x = torch.randn(shape)
    y = candidate(x)
    assert y.shape == (shape[0], 128, *shape[2:]) and torch.isfinite(y).all()
    torch.testing.assert_close(y, candidate.forward_split(x))
    y.square().mean().backward()
    for name, parameter in candidate.dcr.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        assert torch.count_nonzero(parameter.grad), name
    assert not any("gate" in k for k in candidate.dcr.state_dict())
    assert candidate.dcr.beta.item() == pytest.approx(0.25)
    candidate.dcr.enabled = False
    torch.testing.assert_close(candidate(x), baseline(x), rtol=0, atol=0)
    torch.testing.assert_close(candidate.forward_split(x), baseline.forward_split(x), rtol=0, atol=0)


@pytest.mark.parametrize("device,amp", [("cpu", False), ("cuda:0", True)])
def test_gate_precision_and_monotonicity(device, amp):
    """Check zero statistics, decreasing asymmetry preference, AMP dtype and differentiability."""
    if amp and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    block = DCRStripV2(32).to(device).eval()
    zeros = [torch.zeros(2, 8, 9, 13, device=device)] * 4
    weights, q = block.direction_weights(zeros, zeros)
    assert torch.equal(weights, torch.full_like(weights, 0.25)) and torch.count_nonzero(q) == 0
    c = [torch.ones_like(zeros[0], requires_grad=True) for _ in range(4)]
    a = [torch.full_like(zeros[0], float(i), requires_grad=True) for i in range(4)]
    with autocast(enabled=amp, device=torch.device(device).type):
        weights, q = block.direction_weights(c, a)
        content, compensation, gates, q_real = block.components(torch.randn(2, 32, 9, 13, device=device))
    assert weights.dtype == gates.dtype == q.dtype == torch.float32
    assert content.dtype == compensation.dtype == (torch.float16 if amp else torch.float32)
    assert torch.isfinite(q_real).all() and q_real.min() >= 0 and q_real.max() <= 1
    torch.testing.assert_close(weights.sum(1), torch.ones_like(weights[:, 0]))
    assert (weights[:, :-1] > weights[:, 1:]).all()
    q.sum().backward()
    assert all(t.grad is not None and torch.count_nonzero(t.grad) for t in c + a)
    assert all((t.grad < 0).all() for t in a)


def test_clamped_normals_and_ungated_content():
    """The reused geometry has no wraparound; constant strips retain U when their contrast vanishes."""
    block = DCRStripV2(32)
    t = torch.arange(35, dtype=torch.float32).reshape(1, 1, 5, 7).square()
    for dy, dx in block.normals:
        c, a = block.contrast(t, (dy, dx))
        for y in range(5):
            for x in range(7):
                plus = t[..., min(4, max(0, y + dy)), min(6, max(0, x + dx))]
                minus = t[..., min(4, max(0, y - dy)), min(6, max(0, x - dx))]
                assert c[..., y, x] == t[..., y, x] - (plus + minus) / 2
                assert a[..., y, x] == (plus - minus).abs()
    responses = [torch.full((1, 8, 5, 7), float(i)) for i in range(1, 5)]
    with patch.object(block, "directional_responses", return_value=responses):
        content, compensation, weights, q = block.components(torch.zeros(1, 32, 5, 7))
    assert torch.equal(content, torch.full_like(content, 2.5))
    assert torch.count_nonzero(compensation) == torch.count_nonzero(q) == 0
    assert torch.equal(weights, torch.full_like(weights, 0.25))


def test_full_graph(tmp_path):
    """Check the sole layer-4 replacement at both real image sizes and exact common initialization."""
    report = shared.structural_checks(tmp_path, v2.MODEL, C3k2_DCRStripV2)
    assert report["added_parameters"] == 9154
    assert report["baseline_parameters"] == shared.REFERENCE["model_parameters_nc1"]


@pytest.mark.parametrize("device,amp", [("cpu", False), ("cuda:0", True)])
def test_native_weights_loss_optimizer_reload(tmp_path, device, amp):
    """Audit actual Trainer reconstruction, all optimizer members, detection loss, EMA and child reload."""
    if amp and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    root = os.environ.get("DCR_BASELINE_ROOT")
    if not root:
        pytest.skip("Set DCR_BASELINE_ROOT for original weight audit")
    path = Path(root) / "yolo26n.pt"
    assert shared.sha256(path) == "9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef"
    weights, _ = load_checkpoint(path)
    trainer = object.__new__(shared.AuditedTrainer)
    trainer.args = get_cfg(overrides={k: v for k, v in shared.REFERENCE["args"].items() if k != "save_dir"})
    trainer.args.imgsz = 96  # Disposable small diagnostic; never a formal batch-32 receipt.
    trainer.data = {"nc": 1, "channels": 3, "names": {0: "crack"}}
    trainer.device = torch.device(device)
    init_seeds(42)
    trainer.model = trainer.get_model(str(v2.MODEL), weights, verbose=False).to(device)
    trainer.set_model_attributes()
    trainer.optimizer = trainer.build_optimizer(trainer.model, "MuSGD", 0.01, 0.937, 0.0005)
    ids = {id(p) for group in trainer.optimizer.param_groups for p in group["params"]}
    assert all(id(p) in ids for p in trainer.model.model[4].dcr.parameters())
    batch = dict(
        img=torch.randint(0, 256, (2, 3, 64, 96), dtype=torch.uint8),
        batch_idx=torch.tensor([0, 1]),
        cls=torch.zeros(2, 1),
        bboxes=torch.tensor([[0.5, 0.5, 0.25, 0.35], [0.4, 0.6, 0.1, 0.5]]),
    )
    report = shared.gradient_check(trainer, batch, amp)
    ema = ModelEMA(trainer.model)
    ema.update(trainer.model)
    assert isinstance(ema.ema.model[4], C3k2_DCRStripV2)
    report["weights"] = trainer.weight_audit
    report["reload"] = shared.save_reload_check(trainer.model, tmp_path, C3k2_DCRStripV2)
    shared.write_json(tmp_path / "native_check.json", report)


def test_scalar_callback_before_epoch_exists(tmp_path):
    """The native pretrain event precedes the creation of trainer.epoch and must consume no RNG."""
    trainer = SimpleNamespace(
        model=SimpleNamespace(model=[None] * 4 + [SimpleNamespace(dcr=DCRStripV2(32))]), save_dir=tmp_path
    )
    state = torch.get_rng_state()
    v2.record_start(trainer)
    assert torch.equal(state, torch.get_rng_state())
    assert json.loads((tmp_path / "dcr_v2_scalars.jsonl").read_text())["epoch"] == 0


def test_test_export_is_single_pass_and_fp32(tmp_path):
    """Capture the same val call's precise metrics and refuse a second evaluation into its directory."""
    from unittest.mock import MagicMock

    from tools.experiments import finish_b19_dcrstrip_v2 as finish

    model = MagicMock()
    model.model.model = [None] * 4 + [C3k2_DCRStripV2(64, 128)]
    metrics = SimpleNamespace(results_dict={"metrics/mAP50-95(B)": 0.50123456789}, nt_per_class=torch.tensor([1477]))

    def val(**kwargs):
        callback = model.add_callback.call_args.args[1]
        callback(SimpleNamespace(args=get_cfg(overrides=kwargs), metrics=metrics, speed={"inference": 1.23}, seen=1202))
        (tmp_path / "test/predictions.json").write_text("[]", encoding="utf-8")
        return metrics

    model.val.side_effect = val
    with patch.object(finish, "provenance", return_value={"weight": "best.pt"}), patch.object(
        finish, "YOLO", return_value=model
    ):
        finish.test_best(tmp_path, tmp_path / "data.yaml")
        with pytest.raises(FileExistsError):
            finish.test_best(tmp_path, tmp_path / "data.yaml")
    assert model.val.call_count == 1
    args = model.val.call_args.kwargs
    assert args["quantize"] is None and args["split"] == "test" and args["batch"] == 32
    assert args["rect"] and not args["augment"]
    report = json.loads((tmp_path / "test/metrics.json").read_text())
    assert report["results_dict"] == metrics.results_dict and report["targets"] == 1477


def test_package_excludes_other_checkpoints_and_can_repackage(tmp_path):
    """Exercise actual tar/gzip/hash verification and exclude previous manifests and unwanted weights."""
    import tarfile

    from tools.experiments import finish_b19_dcrstrip_v2 as finish

    run = tmp_path / "fixture"
    for name in (
        "args.yaml",
        "results.csv",
        "train.log",
        "weights/best.pt",
        "weights/last.pt",
        "weights/epoch20.pt",
        "provenance/preflight.pt",
        "test/metrics.json",
        "test/predictions.json",
        "diagnostics.json",
    ):
        path = run / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"test fixture, not experimental evidence")
    with patch.object(finish, "provenance", return_value={"test_fixture": True}), patch.object(
        finish.subprocess, "check_output", return_value="fixture==1"
    ):
        for i in range(2):
            output = tmp_path / f"fixture-{i}.tar.gz"
            finish.package(run, output)
            with tarfile.open(output) as archive:
                names = archive.getnames()
                assert [name for name in names if name.endswith(".pt")] == ["run/weights/best.pt"]
                assert "source.tar" in names and "package_manifest.json" in names
