"""Regressions for ordered LBI interaction, native parser/loading and auditable experiment lifecycle."""

import io
import json
import subprocess
import tarfile
from pathlib import Path

import pytest
import torch

from tools.experiments import b19_common as common
from tools.experiments.finish_b19_lbi_fusion import archive_package, package
from tools.experiments.run_b19_lbi_fusion import audit_arguments, require_runtime
from tools.experiments.verify_b19_lbi_fusion import (
    build_pair,
    module_checks,
    staged_gradient_audit,
    topology_checks,
    whole_identity,
)
from ultralytics.nn.modules import Concat_LBI_Fusion


@pytest.fixture(autouse=True)
def small_cpu_pool():
    """Bound CPU test overhead without changing production threading or precision settings."""
    previous = torch.get_num_threads()
    torch.set_num_threads(4)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("device,amp", [("cpu", False), ("cuda:0", False), ("cuda:0", True)])
def test_module_formula_and_staged_updates(device, amp):
    """Check hand-derived channel RMS and nonzero task updates after output projection unlock."""
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    result = module_checks(device, amp)
    assert result["identity"][0]["max_abs"] == 0


def test_order_noncontiguous_and_shared_inputs():
    """S stays first and untouched; L is never mutated even with a nonzero residual and shared views."""
    block = Concat_LBI_Fusion([3, 5])
    semantic = torch.full((1, 3, 17, 13), 3.0).transpose(2, 3)
    detail = torch.full((1, 5, 17, 13), 7.0).transpose(2, 3)
    assert not semantic.is_contiguous()
    assert torch.equal(block([semantic, detail]), torch.cat([semantic, detail], 1))
    with torch.no_grad():
        block.out.weight.fill_(0.2)
    before = [semantic.clone(), detail.clone()]
    output = block([semantic, detail])
    assert torch.equal(output[:, :3], semantic) and torch.equal(semantic, before[0]) and torch.equal(detail, before[1])
    assert output.shape == (1, 8, 13, 17) and torch.isfinite(output).all()
    with pytest.raises(ValueError, match="channels"):
        block([detail, semantic])
    # Equal channel counts cannot reveal semantic origins from tensor metadata; graph f=[-1,4] is authoritative.
    equal = Concat_LBI_Fusion([3, 3])
    assert not torch.equal(equal([semantic, semantic + 1]), equal([semantic + 1, semantic]))


@pytest.mark.parametrize(
    "inputs,message",
    [
        ([], "exactly two"),
        ([torch.zeros(1, 3, 4, 5)], "exactly two"),
        ([torch.zeros(1, 3, 4, 5)] * 3, "exactly two"),
        ([torch.zeros(1, 3, 4), torch.zeros(1, 5, 4, 5)], "NCHW"),
        ([torch.zeros(2, 3, 4, 5), torch.zeros(1, 5, 4, 5)], "batch and spatial"),
        ([torch.zeros(1, 3, 4, 5), torch.zeros(1, 5, 5, 4)], "batch and spatial"),
        ([torch.zeros(1, 5, 4, 5), torch.zeros(1, 3, 4, 5)], "channels"),
        ([torch.zeros(1, 3, 4, 5, dtype=torch.int64), torch.zeros(1, 5, 4, 5, dtype=torch.int64)], "floating dtype"),
    ],
)
def test_invalid_inputs(inputs, message):
    """Reject metadata errors at the module interface rather than interpolating, cropping or reordering."""
    with pytest.raises(ValueError, match=message):
        Concat_LBI_Fusion([3, 5])(inputs)


def test_invalid_device():
    """Report differing input/module devices before any convolution starts."""
    with pytest.raises(ValueError, match="devices"):
        Concat_LBI_Fusion([3, 5])([torch.empty(1, 3, 4, 5, device="meta"), torch.zeros(1, 5, 4, 5)])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_native_amp_mixed_input_dtypes():
    """Native AMP can produce FP32 S and FP16 L; preserve native cat promotion and never coerce S."""
    module = Concat_LBI_Fusion([128, 128]).cuda()
    semantic = torch.randn(2, 128, 13, 17, device="cuda", dtype=torch.float32)
    detail = torch.randn(2, 128, 13, 17, device="cuda", dtype=torch.float16)
    with torch.autocast("cuda"):
        y = module([semantic, detail])
        expected = torch.cat([semantic, detail], 1)
    assert y.dtype == expected.dtype == torch.float32 and torch.equal(y, expected)


@pytest.mark.parametrize("kwargs", [dict(rank=8), dict(eps=1e-5), dict(dim=0)])
def test_fixed_design(kwargs):
    """The v1 interface cannot silently enable alternative ranks, normalization or channel layouts."""
    with pytest.raises(ValueError, match="fixes"):
        Concat_LBI_Fusion([128, 128], **kwargs)


def test_parameter_budget_rng_and_nonzero_reload():
    """Only four ordinary bias-free conv weights are added, with no subsequent RNG drift or reset on reload."""
    torch.manual_seed(42)
    state = torch.get_rng_state().clone()
    block = Concat_LBI_Fusion([128, 128])
    assert torch.equal(state, torch.get_rng_state())
    assert len(list(block.parameters())) == 4 and sum(p.numel() for p in block.parameters()) == 6288
    assert not list(block.buffers())
    for layer in (block.proj_l, block.proj_s, block.dw, block.out):
        assert type(layer) is torch.nn.Conv2d and layer.bias is None
    assert block.dw.groups == 16 and block.dw.kernel_size == (3, 3)
    assert block.dw.padding == block.dw.stride == block.dw.dilation == (1, 1)
    assert block.dw.padding_mode == "zeros"
    assert not block.out.weight.count_nonzero()
    with torch.no_grad():
        block.out.weight.normal_()
    clone = Concat_LBI_Fusion([128, 128])
    clone.load_state_dict(block.state_dict())
    assert torch.equal(clone.out.weight, block.out.weight)


def test_parser_topology_and_native_shared_identity():
    """All 24 parsed layers and both raw Detect branches remain b19-equivalent at step zero."""
    baseline, candidate = build_pair()
    assert topology_checks(baseline, candidate)["changed_layers"] == [15]
    audit = common.audit_weights(baseline, candidate, None)
    assert audit["exact_matches"] == audit["common_keys"]
    whole_identity(baseline, candidate, "cpu", size=96)


def test_recipe_rejects_nonidentity_overrides():
    """No tweak to patience, batch, AMP, optimizer or augmentation is accepted as an identity difference."""
    raw = common.REFERENCE["args"]
    for name, value in dict(patience=0, batch=16, amp=False, optimizer="SGD", mosaic=0.0).items():
        with pytest.raises(ValueError, match="Non-identity"):
            audit_arguments(raw, {**raw, name: value})
    assert set(audit_arguments(raw, {**raw, "name": "lbi"})) == {"name"}


def test_local_environment_cannot_authorize_server():
    """Local runtime results cannot substitute for the exact b19 server environment."""
    if torch.__version__ != common.REFERENCE["environment"]["torch"]:
        with pytest.raises(RuntimeError, match="environment mismatch"):
            require_runtime()


def test_staged_audit_does_not_accept_decay_or_grad_objects():
    """Finite grad objects and weight decay motion alone never count as task-driven effective updates."""
    with pytest.raises(AssertionError):
        staged_gradient_audit([])
    rows = [
        dict(
            batch=0,
            out_zero_before=True,
            out_zero_after=True,
            parameters={
                k: dict(unscaled_before_clip=dict(finite=True, max_abs=0), effective_update=False)
                for k in ("out.weight", "proj_l.weight", "proj_s.weight", "dw.weight")
            },
        )
    ]
    assert len(staged_gradient_audit(rows, False)["missing"]) == 4
    with pytest.raises(AssertionError, match="Missing gradient-backed"):
        staged_gradient_audit(rows)


def test_archive_dry_run(tmp_path):
    """Round-trip a minimal fixture archive including the committed LBI implementation and its dependencies."""
    run = tmp_path / "fixture"
    (run / "weights").mkdir(parents=True)
    for name in ("weights/best.pt", "weights/last.pt", "args.yaml"):
        (run / name).write_bytes(b"LOCAL FIXTURE ONLY; no training results")
    (run / "args_hard.txt").hardlink_to(run / "args.yaml")
    (run / "failed.attempt.1").mkdir()
    (run / "failed.attempt.1/console.log").write_text("excluded")
    result = archive_package(run, tmp_path / "dry_run.tar.gz", ["weights/best.pt", "weights/last.pt", "args.yaml"])
    assert result["gzip_crc_verified"]
    with tarfile.open(result["path"]) as archive:
        assert not any(".attempt." in n for n in archive.getnames())
        assert archive.getmember("run/args_hard.txt").isfile()
        assert archive.extractfile("run/args_hard.txt").read() == (run / "args.yaml").read_bytes()
        with tarfile.open(fileobj=io.BytesIO(archive.extractfile("run/source.tar").read())) as source:
            required = [
                "ultralytics/nn/modules/lbi_fusion.py",
                "tools/experiments/b19_common.py",
                "tools/experiments/run_b19_lbi_fusion.py",
                "tools/experiments/verify_b19_lbi_fusion.py",
                "tools/experiments/diagnose_b19_lbi_fusion.py",
                "tools/experiments/finish_b19_lbi_fusion.py",
            ]
            assert all(n in source.getnames() for n in required), "Package dry-run requires committed LBI sources"
            shell = source.extractfile("tools/experiments/server_b19_lbi_fusion_v1.sh").read()
            assert b"\r\n" not in shell
    common.write_json(common.ROOT / "runs/lbi_local/package_dry_run.json", {**result, "fixture_only": True})
    with pytest.raises(ValueError):
        archive_package(run, Path(result["path"]), [])
    with pytest.raises(FileNotFoundError):
        archive_package(run, tmp_path / "missing.tar.gz", ["results.csv"])


def test_archive_real_links(tmp_path):
    """Exercise actual link handling where supported; never substitute a mock Linux symlink claim."""
    run = tmp_path / "run"
    run.mkdir()
    (run / "regular.txt").write_text("data")
    (run / "hard.txt").hardlink_to(run / "regular.txt")
    try:
        (run / "dangling.log").symlink_to(run / "absent.attempt.1/console.log")
    except OSError:
        pytest.skip("Windows does not grant real symlink creation; Linux symlink integration NOT RUN")
    result = archive_package(run, tmp_path / "links.tar.gz", ["regular.txt", "hard.txt"])
    with tarfile.open(result["path"]) as archive:
        assert archive.getmember("run/hard.txt").isfile()
        assert "run/dangling.log" not in archive.getnames()
        excluded = json.load(archive.extractfile("run/package_exclusions.json"))
        assert any(x["path"] == "dangling.log" for x in excluded["excluded"])


def test_package_rejects_incomplete_formal_results(tmp_path):
    """No final experiment package may be produced from fixture weights or missing evaluation evidence."""
    with pytest.raises(FileNotFoundError):
        package(tmp_path, tmp_path / "invalid.tar.gz", {})


def test_source_core_unchanged():
    """Training math, losses, dependencies and all native block implementations remain canonical b19."""
    paths = [
        "ultralytics/engine/trainer.py",
        "ultralytics/utils/loss.py",
        "ultralytics/optim/muon.py",
        "ultralytics/nn/modules/block.py",
        "ultralytics/nn/modules/conv.py",
        "ultralytics/nn/modules/head.py",
        "ultralytics/utils/torch_utils.py",
        "ultralytics/cfg/default.yaml",
        "pyproject.toml",
    ]
    for name in paths:
        original = subprocess.check_output(
            [
                "git",
                "-c",
                f"safe.directory={common.ROOT.as_posix()}",
                "show",
                f"{common.REFERENCE['source_commit']}:{name}",
            ],
            cwd=common.ROOT,
        )
        assert original.replace(b"\r\n", b"\n") == (common.ROOT / name).read_bytes().replace(b"\r\n", b"\n")


@pytest.mark.parametrize("nonfinite,expected_batches", [(True, 1), (False, 64)])
def test_preflight_failure_receipts_and_fixed_bound(tmp_path, monkeypatch, nonfinite, expected_batches):
    """Test only callback failure ownership with a stub; this is not native trainer or server validation."""
    from tools.experiments import verify_b19_lbi_fusion as verifier

    class CallbackDriver:
        def __init__(self, config):
            self.callbacks = {}
            self.loss = torch.tensor(float("nan") if nonfinite else 1.0)
            self.audit_images, self.accumulate = ["fixture-only"], 1

        def add_callback(self, name, callback):
            self.callbacks[name] = callback

        def train(self):
            for _ in range(65):
                self.callbacks["on_train_batch_start"](self)
                self.callbacks["on_train_batch_end"](self)

    monkeypatch.setattr(verifier, "PreflightTrainer", CallbackDriver)
    with pytest.raises(AssertionError):
        verifier.native_preflight({}, tmp_path)
    batches = json.loads((tmp_path / "native_batches.json").read_text())
    assert len(batches) == expected_batches and batches[-1]["batch"] == expected_batches - 1
    assert batches[-1]["loss"] == ("nan" if nonfinite else 1.0)
    assert (tmp_path / "native_steps.json").is_file() and (tmp_path / "native_gradient_summary.json").is_file()
