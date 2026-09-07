"""Independent RPCA probability, 2D boundary, initialization and real native-training contracts."""

import copy
import json
import math
import os
import shutil
import subprocess
from unittest.mock import patch

import pytest
import torch

from tests import test_sir_sppf as fixtures
from tools.experiments import finish_b19_rpca_c2psa as finish
from tools.experiments import finish_b19_sir_sppf as common_finish
from tools.experiments import finish_b19_sir_sppf_v2 as evaluation
from tools.experiments import run_b19_rpca_c2psa as run
from tools.experiments import run_b19_sir_sppf as shared
from ultralytics.data.dataset import YOLODataset
from ultralytics.nn.modules import C2PSA, C2PSA_RPCA
from ultralytics.nn.modules.rpca_c2psa import PSABlock_RPCA, calibrated_probabilities, region_group, region_ungroup
from ultralytics.utils import YAML


@pytest.fixture(autouse=True)
def threads():
    """Bound development threads and restore the caller's state."""
    previous = torch.get_num_threads()
    torch.set_num_threads(4)
    yield
    torch.set_num_threads(previous)


def regions(height, width):
    """Enumerate real 2D keys without reusing the optimized grouping implementation."""
    return [
        [y * width + x for y in range(row, min(row + 2, height)) for x in range(col, min(col + 2, width))]
        for row in range(0, height, 2)
        for col in range(0, width, 2)
    ]


@pytest.mark.parametrize("height,width", [(20, 20), (19, 21), (20, 16), (1, 7), (7, 1), (1, 1)])
def test_probability_reference_and_boundary(height, width):
    """Check A', M, C, B, valid areas, normalization and zero/nontrivial gamma against Python region loops."""
    torch.manual_seed(81)
    n = height * width
    logits = torch.randn(1, 3, 5, n) * 3
    gamma = torch.full((1, 3, 5, 1), 0.05)
    actual, detail = calibrated_probabilities(logits, gamma, height, width, True)
    groups = regions(height, width)
    coarse = torch.stack([logits[..., ids].mean(-1) + math.log(len(ids)) for ids in groups], -1).softmax(-1)
    original = logits.softmax(-1)
    expected = torch.empty_like(original)
    assert detail["area"].tolist() == [len(ids) for ids in groups]
    assert detail["valid"].sum().item() == n
    for index, ids in enumerate(groups):
        mass = (1 - gamma) * original[..., ids].sum(-1, keepdim=True) + gamma * coarse[..., index : index + 1]
        conditional = logits[..., ids].softmax(-1)
        expected[..., ids] = mass * conditional
        torch.testing.assert_close(actual[..., ids].sum(-1, keepdim=True), mass, atol=1e-7, rtol=2e-6)
        torch.testing.assert_close(actual[..., ids], mass * conditional, atol=1e-7, rtol=2e-6)
    torch.testing.assert_close(actual, expected, atol=1e-7, rtol=2e-6)
    torch.testing.assert_close(detail["B"], coarse, atol=1e-7, rtol=2e-6)
    assert actual.min() >= 0
    torch.testing.assert_close(actual.sum(-1), torch.ones_like(actual.sum(-1)), atol=3e-7, rtol=0)
    assert torch.equal(calibrated_probabilities(logits, gamma * 0, height, width), original)
    if n > 4:
        assert (actual - original).abs().max() > 1e-5
        assert not torch.allclose(detail["B"], detail["P"])
    uniform, uniform_detail = calibrated_probabilities(torch.zeros_like(logits), gamma, height, width, True)
    torch.testing.assert_close(uniform, torch.full_like(uniform, 1 / n))
    torch.testing.assert_close(uniform_detail["B"], detail["area"].expand_as(coarse) / n)
    tokens = torch.arange(n)
    assert torch.equal(region_ungroup(region_group(tokens, height, width), height, width), tokens)


def test_mean_keys_equivalence_and_extreme_gradient():
    """Coarse B is mean-K matching; stable conditional softmax remains differentiable under extreme logits."""
    height, width = 3, 5
    q, k = torch.randn(2, 4, 7, 3), torch.randn(2, 4, 3, height * width)
    logits = q @ k / math.sqrt(3)
    _, detail = calibrated_probabilities(logits, torch.tensor(0.05), height, width, True)
    means = torch.stack([k[..., ids].mean(-1) for ids in regions(height, width)], -1)
    expected = (q @ means / math.sqrt(3) + detail["area"].log()).softmax(-1)
    torch.testing.assert_close(detail["B"], expected)
    extreme = (logits * 10000).requires_grad_()
    gamma = torch.full((2, 4, 7, 1), 0.05, requires_grad=True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        result = calibrated_probabilities(extreme, gamma, height, width)
        assert result.dtype == torch.float32
    (result * torch.randn_like(result)).sum().backward()
    assert torch.isfinite(result).all() and torch.isfinite(extreme.grad).all() and torch.isfinite(gamma.grad).all()


@pytest.mark.parametrize("channels,repeats", [(128, 2), (256, 1), (384, 3)])
def test_native_state_rng_repeats_and_bypass(channels, repeats):
    """Vary parsed-compatible heads and repetitions; every block reads the same bypass and preserves native state."""
    torch.manual_seed(42)
    original = C2PSA(channels, channels, repeats).eval()
    rng = torch.get_rng_state()
    torch.manual_seed(42)
    candidate = C2PSA_RPCA(channels, channels, repeats).eval()
    assert torch.equal(rng, torch.get_rng_state())
    for key, tensor in original.state_dict().items():
        assert torch.equal(tensor, candidate.state_dict()[key]), key
    x = torch.randn(1, channels, 3, 5)
    inputs = []
    hooks = [block.gate.register_forward_pre_hook(lambda m, args: inputs.append(args[0])) for block in candidate.m]
    with torch.no_grad():
        output = candidate(x)
        for block in candidate.m:
            torch.testing.assert_close(
                0.5 * block.gate(inputs[0]).sigmoid(), torch.full((1, channels // 128, 3, 5), 0.05)
            )
        assert all(a is inputs[0] for a in inputs[:repeats])
        with run.bypass(candidate):
            assert torch.equal(original(x), candidate(x))
        assert all(block.attn.enabled for block in candidate.m)
        assert (output - original(x)).abs().max() > 0
    for hook in hooks:
        hook.remove()


@pytest.mark.parametrize("shape", [(20, 20), (19, 21), (20, 16), (1, 7), (7, 1), (1, 1)])
def test_amp_native_v_pe_order_and_gradients(shape):
    """Compare the full attention output with a direct QKV/V/PE reference and check gate/FFN gradients."""
    torch.manual_seed(3)
    block = PSABlock_RPCA(128, num_heads=2).eval()
    x = torch.randn(2, 128, *shape, requires_grad=True)
    a = torch.randn_like(x, requires_grad=True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = block(x, a)
        gamma = 0.5 * block.gate(a).float().sigmoid().flatten(2).unsqueeze(-1)
        q, k, v = block.attn.qkv(x).reshape(2, 2, 128, -1).split((32, 32, 64), 2)
        with torch.autocast("cpu", enabled=False):
            logits = (q.float() * block.attn.scale).transpose(-2, -1) @ k.float()
            probabilities = calibrated_probabilities(logits, gamma, *shape)
            attended = (v.float() @ probabilities.transpose(-2, -1)).reshape_as(x).to(v.dtype)
        attention = block.attn.proj(attended + block.attn.pe(v.reshape_as(x)))
        residual = x + attention
        expected = residual + block.ffn(residual)
    assert torch.equal(output, expected)
    output.square().mean().backward()
    assert torch.isfinite(x.grad).all() and torch.isfinite(a.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in block.parameters())


def test_full_graph(tmp_path):
    """The actual parsed nano model changes only layer 10 and adds exactly 2258 parameters."""
    report = run.structural_checks(tmp_path)
    assert report["common_keys"] == 708 and report["candidate_parameters"] == 2506448


def test_real_pretrained_amp_musgd_ema_reload(tmp_path):
    """Three actual augmented batch=2 updates are local evidence, never a formal batch=32 server receipt."""
    torch.set_num_threads(1)
    trainer = fixtures.native_probe(
        "cuda:0" if torch.cuda.is_available() else "cpu", trainer_type=run.AuditedTrainer, model=run.MODEL
    )
    assert trainer.weight_audit["common_keys"] == 708 and len(trainer.weight_audit["loaded_keys"]) == 606
    ids = {id(p) for group in trainer.optimizer.param_groups for p in group["params"]}
    assert all(id(p) in ids for key, p in trainer.model.named_parameters() if ".gate." in key)
    images = fixtures.sample_images(tmp_path)
    listing = tmp_path / "images.txt"
    listing.write_text("\n".join(map(str, images)), encoding="utf-8")
    data = YOLODataset(
        img_path=str(listing), imgsz=640, batch_size=2, augment=True, hyp=trainer.args, data=trainer.data, cache=False
    )
    if trainer.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    rows = [
        shared.gradient_check(trainer, data.collate_fn([data[0], data[1]]), trainer.device.type == "cuda")
        for _ in range(3)
    ]
    assert all(v == 0 for k, v in rows[0]["new_gradient_norms"].items() if ".gate.4." not in k)
    assert all(v > 0 for v in rows[-1]["new_gradient_norms"].values())
    assert trainer.ema.updates == 3 and type(trainer.ema.ema.model[10]) is C2PSA_RPCA
    shared.save_reload_check(trainer.ema.ema, tmp_path, C2PSA_RPCA, 10, ".gate.", 2258)
    report = json.loads((tmp_path / "reload_check.json").read_text())
    assert report["state_keys"] == 714 and max(r["max_abs"] for r in report["raw"]) == 0
    checkpoint = torch.load(tmp_path / "preflight.pt", weights_only=False)
    for key in ("model.10.m.0.gate.4.weight", "model.10.cv1.bn.num_batches_tracked"):
        damaged = copy.deepcopy(checkpoint)
        damaged["ema"].state_dict()[key].view(-1)[0] += 1
        torch.save(damaged, tmp_path / "preflight.pt")
        with pytest.raises(AssertionError, match="outside_tolerance"):
            shared.reload_in_process(tmp_path / "preflight.pt", C2PSA_RPCA, 10, ".gate.", 2258)
    shared.write_json(
        run.ROOT / "runs/rpca_development/local_real_gradient.json",
        dict(
            formal_preflight=False,
            batch=2,
            imgsz=640,
            device=str(trainer.device),
            rows=rows,
            peak_allocated_bytes=torch.cuda.max_memory_allocated() if trainer.device.type == "cuda" else None,
            weights=trainer.weight_audit,
            reload=report,
        ),
    )


def test_diagnostic_summary_and_restoring_bypass():
    """A trained nonconstant gate produces head/spatial summaries; exceptions cannot leave bypass enabled."""
    model = C2PSA_RPCA(256, 256).eval()
    block = model.m[0]
    torch.nn.init.normal_(block.gate[-1].weight, std=0.2)
    a, x = torch.randn(1, 128, 3, 5), torch.randn(1, 128, 3, 5)
    gamma = 0.5 * block.gate(a).float().sigmoid().flatten(2).unsqueeze(-1)
    report, maps = finish.attention_statistics(block.attn, x, gamma)
    assert maps.shape == (2, 3, 5) and report["gamma"]["std"] > 0
    assert report["region_mass_mean_absolute_change"] > 0
    assert report["probability_row_sum_max_error"] < 1e-6
    with pytest.raises(RuntimeError), run.bypass(model):
        assert not block.attn.enabled
        raise RuntimeError("intentional probe")
    assert block.attn.enabled


def test_entry_contracts_and_package_last(tmp_path):
    """No v2 model is accidentally selected, and RPCA explicitly includes last.pt and its own source."""
    with patch.object(shared, "main", return_value=0) as entry:
        assert run.main(["--help"]) == 0
        assert entry.call_args.kwargs["trainer_type"] is run.AuditedTrainer
        assert entry.call_args.kwargs["model"] == run.MODEL
    with patch.object(evaluation, "package") as package:
        finish.package(tmp_path, tmp_path / "data.yaml", tmp_path / "bundle.tar.gz")
        assert package.call_args.kwargs["include_last"]
        assert package.call_args.kwargs["experiment"] is run
        assert "ultralytics/nn/modules/rpca_c2psa.py" in package.call_args.kwargs["source_files"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="The fixed evaluation entry requires CUDA device 0")
def test_real_native_fp32_validator_on_two_copied_training_images(tmp_path):
    """Exercise actual AutoBackend/DetectionValidator on two training images, never claim held-out metrics."""
    fixtures.sample_images(tmp_path)
    data = tmp_path / "data.yaml"
    YAML.save(
        data,
        dict(path=str(tmp_path), train="images/train", val="images/train", test="images/train", names={0: "crack"}),
    )
    trainer = fixtures.native_probe(trainer_type=run.AuditedTrainer, model=run.MODEL)
    weights = tmp_path / "weights/best.pt"
    weights.parent.mkdir()
    torch.save(dict(model=trainer.model.half().eval(), train_args=vars(trainer.args)), weights)
    for split in ("val", "test"):
        report = common_finish.test_best(
            tmp_path,
            data,
            split=split,
            block_type=C2PSA_RPCA,
            layer=10,
            evidence=dict(weight=str(weights), local_training_subset=True),
            output=tmp_path / split,
        )
        assert report["images"] == 2
        assert report["actual_parameter_dtype"] == "torch.float32" and report["actual_split"] == split
        assert report["args"]["quantize"] is None and report["args"]["rect"]
        assert (tmp_path / split / "predictions.json").is_file()


def test_diagnostic_artifact_reuse_rejects_damage(tmp_path):
    """A matching diagnostic receipt cannot certify a missing or changed gamma visualization."""
    evidence = dict(local_fixture=True)
    output = evaluation.report_directory(tmp_path, "diagnostics", evidence)
    output.mkdir(parents=True)
    plot = output / "gamma_0_block_0.png"
    plot.write_bytes(b"Synthetic visualization integrity fixture")
    shared.write_json(output / "metrics.json", dict(evidence=evidence, artifacts={plot.name: shared.sha256(plot)}))
    assert evaluation.report_directory(tmp_path, "diagnostics", evidence) == output
    plot.write_bytes(b"damaged")
    assert evaluation.report_directory(tmp_path, "diagnostics", evidence) != output
    plot.unlink()
    assert evaluation.report_directory(tmp_path, "diagnostics", evidence) != output


def test_evaluation_identity_tracks_backend_and_device_mapping(tmp_path, monkeypatch):
    """FP32 matmul policy or CUDA visibility changes must invalidate evaluation reuse identity."""
    weights = tmp_path / "weights/best.pt"
    weights.parent.mkdir()
    weights.write_bytes(b"Hash-only provenance fixture")
    precision = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("highest")
        first = common_finish.provenance(tmp_path)
        torch.set_float32_matmul_precision("high")
        second = common_finish.provenance(tmp_path)
        assert first["execution_conditions"] != second["execution_conditions"]
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "fixture-mapping")
        third = common_finish.provenance(tmp_path)
        assert second["execution_conditions"] != third["execution_conditions"]
    finally:
        torch.set_float32_matmul_precision(precision)


def test_shell_attempt_identity_and_exit_codes(tmp_path):
    """Exercise the actual RPCA wrapper, stale statuses and both PIPESTATUS entries without any GPU process."""
    bash = os.environ.get("SIR_BASH") or shutil.which("bash")
    if not bash:
        pytest.skip("Bash unavailable")
    scripts = tmp_path / "work/tools/experiments"
    scripts.mkdir(parents=True)
    for name in ("server_b19_rpca_c2psa_v1.sh", "server_b19_sir_sppf_v2.sh"):
        shutil.copy2(run.ROOT / "tools/experiments" / name, scripts / name)
    binaries = tmp_path / "bin"
    binaries.mkdir()
    for name, code in dict(flock="exit 0", git="echo fixture", python='printf "%s\\n" "$@"; exit 7').items():
        p = binaries / name
        p.write_bytes(("#!/usr/bin/env bash\n" + code + "\n").encode())
        p.chmod(0o755)
    command = 'fixture="$(cd -- "$1" && pwd -P)"; export PATH="$fixture/bin:/usr/bin:/bin:$PATH"; export B19_PYTHON="$fixture/bin/python"; bash "$fixture/work/tools/experiments/server_b19_rpca_c2psa_v1.sh" preflight'
    for _ in range(2):
        result = subprocess.run([bash, "-c", command, "--", tmp_path.as_posix()], capture_output=True, text=True)
        assert result.returncode == 7, result.stdout + result.stderr
    project = tmp_path / "work/runs/detect"
    assert json.loads((project / f"{run.NAME}_preflight.process_status.json").read_text()) == dict(python=7, tee=0)
    assert len(list(project.glob(f"{run.NAME}_preflight.attempt.*/previous.exit_status"))) == 1
    assert "run_b19_rpca_c2psa.py" in (project / f"{run.NAME}_preflight.console.log").read_text()
