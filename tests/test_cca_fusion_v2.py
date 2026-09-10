"""Independent mechanism and lifecycle regressions for fixed reliability-constrained CCA v2."""

import copy
import gzip
import json
import math
import os
import tarfile
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from ultralytics.nn.modules import Concat_CCA_Fusion, Concat_CCA_Fusion_V2
from ultralytics.nn.modules.cca_fusion_v2 import CENTER_PRIOR, RELIABILITY_FLOOR, RESIDUAL_SCALE


@pytest.fixture(autouse=True)
def small_thread_pool():
    """Keep small CPU fixtures independent of host core counts."""
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def reference(module, up, low, high):
    """Enumerate valid coarse candidates independently of unfold and phase broadcasting."""
    q = F.normalize(module.Wq(low).float(), dim=1, eps=1e-6)
    k = F.normalize(module.Wk(high).float(), dim=1, eps=1e-6)
    v = module.Wv(high).float()
    rows = []
    for y in range(low.shape[2]):
        columns = []
        for x in range(low.shape[3]):
            cy, cx = y // 2, x // 2
            neighbors = [
                (j, i)
                for j in range(cy - 1, cy + 2)
                for i in range(cx - 1, cx + 2)
                if 0 <= j < high.shape[2] and 0 <= i < high.shape[3]
            ]
            scores = torch.stack(
                [4 * (q[:, :, y, x] * k[:, :, j, i]).sum(1) + float((j, i) == (cy, cx)) for j, i in neighbors], 1
            )
            weights = scores.softmax(1)
            entropy = -(weights * weights.clamp_min(1e-6).log()).sum(1)
            normalized = (entropy / math.log(len(neighbors))).clamp(0, 1) if len(neighbors) > 1 else entropy * 0
            gate = (0.25 + 0.75 * (1 - normalized).detach())[:, None]
            differences = torch.stack([v[:, :, j, i] - v[:, :, cy, cx] for j, i in neighbors], 2)
            raw = F.linear((weights[:, None] * differences).sum(2), module.Wo.weight[:, :, 0, 0])
            columns.append(0.5 * gate * raw)
        rows.append(torch.stack(columns, -1))
    return torch.cat((up + torch.stack(rows, -2), low), 1)


@pytest.mark.parametrize("shape", [(1, 1), (1, 3), (3, 1), (3, 4)])
def test_reference_forward_and_gradients(shape):
    """Match a pixel oracle with nonzero residual, including every boundary and gradient path."""
    torch.manual_seed(3)
    module = Concat_CCA_Fusion_V2(6, 5, 6)
    torch.nn.init.normal_(module.Wo.weight, std=0.2)
    other = copy.deepcopy(module)
    h, w = shape
    inputs = [torch.randn(2, c, h * s, w * s, requires_grad=True) for c, s in ((6, 2), (5, 2), (6, 1))]
    clones = [v.detach().clone().requires_grad_() for v in inputs]
    actual, expected = module(inputs), reference(other, *clones)
    assert actual.shape == (2, 11, 2 * h, 2 * w)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
    probe = torch.randn_like(actual)
    ga = torch.autograd.grad((actual * probe).sum(), [*module.parameters(), *inputs])
    gb = torch.autograd.grad((expected * probe).sum(), [*other.parameters(), *clones])
    for a, b in zip(ga, gb):
        assert torch.isfinite(a).all()
        torch.testing.assert_close(a, b, atol=1e-5, rtol=2e-5)


def test_fixed_prior_mask_and_noncenter_evidence():
    """Identical evidence favors the parent; a clearly matching neighbor can overcome that prior."""
    module = Concat_CCA_Fusion_V2(2, 2, 2)
    assert (CENTER_PRIOR, RELIABILITY_FLOOR, RESIDUAL_SCALE) == (1.0, 0.25, 0.5)
    with torch.no_grad():
        module.Wq.weight.zero_()
        module.Wk.weight.zero_()
        module.Wq.weight[0, 0] = module.Wk.weight[0, 0] = 1
    low = torch.ones(1, 2, 6, 8)
    high = torch.ones(1, 2, 3, 4)
    _, weights, _, _ = module.correspondence(low, high)
    torch.testing.assert_close(weights.sum(1), torch.ones(1, 6, 8))
    for y in range(6):
        for x in range(8):
            for j, (dy, dx) in enumerate(((dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1))):
                valid = 0 <= y // 2 + dy < 3 and 0 <= x // 2 + dx < 4
                if not valid:
                    assert weights[0, j, y, x] == 0
                elif j != 4:
                    torch.testing.assert_close(weights[0, 4, y, x] / weights[0, j, y, x], torch.tensor(math.e))
    high[:, 0] = -1
    high[:, 0, 1, 2] = 1  # right neighbor of parent (1,1)
    _, weights, _, _ = module.correspondence(low, high)
    assert weights[0, 5, 2, 2] > 0.99 and weights[0, 5, 2, 2] > weights[0, 4, 2, 2]


def test_entropy_geometric_counts_detach_and_equal_raw_gate():
    """Count actual borders, safely handle singleton support and stop only the gate gradient."""
    module = Concat_CCA_Fusion_V2(2, 2, 2)
    low, high = torch.randn(1, 2, 6, 8), torch.randn(1, 2, 3, 4)
    _, weights, _, _ = module.correspondence(low, high)
    entropy, norm, confidence, gate, count = module.reliability(weights)
    assert set(count.unique().tolist()) == {4, 6, 9}
    assert count[0, 0, 0, 0] == 4 and count[0, 0, 0, 2] == 6 and count[0, 0, 2, 2] == 9
    assert confidence.requires_grad and not gate.requires_grad
    assert ((0 <= confidence) & (confidence <= 1)).all()
    assert ((0.25 <= gate) & (gate <= 1)).all()
    assert ((0.125 <= 0.5 * gate) & (0.5 * gate <= 0.5)).all()
    torch.testing.assert_close(entropy / count.log(), norm)
    order = norm.flatten().argsort()
    assert (gate.flatten()[order].diff() <= 0).all()
    # Interior supports are identical; only the distribution changes, with an equal raw residual.
    equal = torch.full((1, 9, 6, 6), 1 / 9, requires_grad=True)
    sharp = equal.detach().clone()
    sharp[:, :, 2:4, 2:4] = 0.001
    sharp[:, 4, 2:4, 2:4] = 0.992
    g0, g1 = (module.reliability(w)[3][:, :, 2:4, 2:4] for w in (equal, sharp))
    assert (0.5 * g1 > 0.5 * g0).all()
    single = module.correspondence(low[:, :, :2, :2], high[:, :, :1, :1])
    stats = module.reliability(single[1])
    assert all(torch.isfinite(t).all() for t in stats)
    assert torch.equal(single[0], torch.zeros_like(single[0]))
    assert torch.equal(stats[-1], torch.ones_like(stats[-1]))
    assert torch.equal(stats[1], torch.zeros_like(stats[1]))


def test_zero_initialization_rng_constant_value_and_unfreezing():
    """Prove exact native concat, constant-V cancellation, equal parameters and staged gradient unlocking."""
    state = torch.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    module = Concat_CCA_Fusion_V2(6, 5, 6)
    assert torch.equal(state, torch.get_rng_state())
    assert all(torch.equal(a, b) for a, b in zip(cuda, torch.cuda.get_rng_state_all()))
    native = Concat_CCA_Fusion(6, 5, 6)
    assert module.state_dict().keys() == native.state_dict().keys()
    assert all(torch.equal(p, q) for p, q in zip(module.parameters(), native.parameters()))
    assert sum(p.numel() for p in module.parameters()) == sum(p.numel() for p in native.parameters())
    up, low, high = [torch.randn(2, c, h, w) for c, h, w in ((6, 6, 8), (5, 6, 8), (6, 3, 4))]
    assert torch.equal(module([up, low, high]), torch.cat((up, low), 1))
    module([up, low, high]).square().sum().backward()
    assert module.Wo.weight.grad.count_nonzero() > 0
    assert all(p.weight.grad.count_nonzero() == 0 for p in (module.Wq, module.Wk, module.Wv))
    torch.nn.init.normal_(module.Wo.weight, std=0.1)
    module.zero_grad()
    assert not torch.equal(module([up, low, high]), torch.cat((up, low), 1))
    module([up, low, high]).square().sum().backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() and p.grad.count_nonzero() > 0 for p in module.parameters()
    )
    high = torch.randn(2, 6, 1, 1).expand(-1, -1, 3, 4).contiguous()
    assert torch.equal(module([up, low, high]), torch.cat((up, low), 1))


@pytest.mark.parametrize("bad", ["batch", "channel", "spatial", "inputs"])
def test_bad_shapes(bad):
    module = Concat_CCA_Fusion_V2(6, 5, 6)
    inputs = [torch.randn(2, c, h, w) for c, h, w in ((6, 6, 8), (5, 6, 8), (6, 3, 4))]
    if bad == "batch":
        inputs[0] = inputs[0][:1]
    if bad == "channel":
        inputs[1] = inputs[1][:, :4]
    if bad == "spatial":
        inputs[1] = inputs[1][:, :, :5]
    if bad == "inputs":
        inputs.pop()
    with pytest.raises((AssertionError, ValueError)):
        module(inputs)
    with pytest.raises(AssertionError):
        Concat_CCA_Fusion_V2(6, 5, 7)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_full_resolution_task():
    """Synthetic batch1 at 640 square/rectangle; it is explicitly not formal real batch32 preflight."""
    from tools.experiments import run_b19_cca_fusion_v2 as run
    from ultralytics.nn.tasks import DetectionModel

    model = DetectionModel(str(run.MODEL), nc=1, verbose=False).cuda().train()
    model.args = run.shared.get_cfg(
        overrides={k: v for k, v in run.shared.REFERENCE["args"].items() if k != "save_dir"}
    )
    torch.nn.init.normal_(model.model[12].Wo.weight, std=0.01)
    for h, w in ((640, 640), (640, 960)):
        for amp in (False, True):
            model.zero_grad(set_to_none=True)
            batch = dict(
                img=torch.rand(1, 3, h, w, device="cuda"),
                batch_idx=torch.zeros(1, device="cuda"),
                cls=torch.zeros(1, 1, device="cuda"),
                bboxes=torch.tensor([[0.5, 0.5, 0.2, 0.3]], device="cuda"),
            )
            with torch.autocast("cuda", enabled=amp):
                loss, _ = model(batch)
            loss.sum().backward()
            assert torch.isfinite(loss).all()
            assert all(
                p.grad is not None and torch.isfinite(p.grad).all() and p.grad.count_nonzero()
                for p in model.model[12].parameters()
            )


def test_structure_and_checkpoint(tmp_path):
    from tools.experiments import run_b19_cca_fusion_v2 as run
    from ultralytics.nn.tasks import DetectionModel

    report = run.shared.structural_checks(tmp_path, run.MODEL, Concat_CCA_Fusion_V2)
    assert report["all_common_tensors_equal"] and report["full_network_zero_cca_equal"]
    assert report["common_keys"] == 708 and report["added_parameters"] == 14336
    model = DetectionModel(str(run.MODEL), nc=1, verbose=False).eval()
    model.args = run.shared.get_cfg()
    torch.nn.init.normal_(model.model[12].Wo.weight, std=0.01)
    assert run.shared.save_reload_check(model, tmp_path, Concat_CCA_Fusion_V2)["passed"]


def test_recipe_identity_and_failure_gates(tmp_path):
    from tools.experiments import run_b19_cca_fusion_v2 as run
    from tools.experiments import finish_b19_cca_fusion as finish

    shared = run.shared
    raw = shared.YAML.load(run.ROOT / "tools/experiments/b19_archived_args.yaml")
    assert raw == shared.REFERENCE["args"] and len(raw) == 112
    assert shared.launcher_evidence(
        SimpleNamespace(baseline_launcher=run.ROOT / "tools/experiments/b19_launcher_expanded.txt"), raw
    )["verified"]
    with pytest.raises(FileNotFoundError):
        finish.completed_run(tmp_path)
    trainer = object.__new__(run.AuditedTrainer)
    with pytest.raises(RuntimeError, match="fixed-batch"):
        trainer._oom_retries = 1
    assert raw["batch"] == 32 and raw["imgsz"] == 640 and raw["epochs"] == 200
    v1 = shared.YAML.load(shared.MODEL)
    v2 = shared.YAML.load(run.MODEL)
    v1["head"][1][2] = "Concat_CCA_Fusion_V2"
    assert v1 == v2
    assert run.NAME == "yolo26n_b19_cca_fusion_v2"
    for path in run.SOURCE_FILES:
        assert (run.ROOT / path).is_file()


def test_mechanism_diagnostics():
    from tools.experiments.run_b19_cca_fusion_v2 import mechanism_diagnostics

    module = Concat_CCA_Fusion_V2(6, 5, 6)
    up, low, high = [torch.randn(2, c, h, w) for c, h, w in ((6, 6, 8), (5, 6, 8), (6, 3, 4))]
    zero = mechanism_diagnostics(module, up, low, high)["reliability_v2"]
    assert zero["actual_over_ungated_same_correspondence"] is None
    torch.nn.init.normal_(module.Wo.weight, std=0.1)
    result = mechanism_diagnostics(module, up, low, high)["reliability_v2"]
    assert set(result["support"]) == {"corner", "edge", "interior"}
    assert result["all_positions"]["residual_over_U"]["max"] > 0
    assert 0.125 <= result["actual_over_ungated_same_correspondence"]["p50"] <= 0.5


@pytest.mark.parametrize("link_kind", ["hardlink", "dangling_symlink"])
def test_package_excludes_attempts_and_materializes_links(tmp_path, link_kind):
    """Transient missing targets cannot poison final results; included hardlinks are regular tar members."""
    from tools.experiments.finish_b19_cca_fusion import archive_package

    run = tmp_path / "run"
    (run / "weights").mkdir(parents=True)
    (run / "args.yaml").write_text("nc: 1\n")
    (run / "weights/best.pt").write_bytes(b"synthetic fixture, not a model")
    (run / "train.log").write_bytes(b"complete\n")
    attempt = run / "train.attempt.transient"
    attempt.mkdir()
    if link_kind == "dangling_symlink":
        try:
            (attempt / "console.log").symlink_to(attempt / "missing.log")
            (run / "obsolete.log").symlink_to(attempt / "missing.log")
        except OSError as exc:
            if getattr(exc, "winerror", None) == 1314:
                pytest.skip("Windows account lacks symlink creation privilege")
            raise
    else:
        os.link(run / "train.log", attempt / "console.log")
        os.link(run / "train.log", run / "duplicate.log")
    target = tmp_path / "result.tar.gz"
    archive_package(run, target, required_files=["args.yaml", "weights/best.pt", "train.log"], evidence={})
    with tarfile.open(target) as archive:
        assert all(m.isfile() for m in archive.getmembers())
        assert not any("attempt" in m.name or "obsolete" in m.name for m in archive.getmembers())
        assert archive.extractfile("run/train.log").read() == b"complete\n"
    with gzip.open(target) as stream:
        assert stream.read()
    result = json.loads((run / "package_result.json").read_text())
    assert result["gzip_crc_verified"] and "run/weights/best.pt" in result["key_members"]
    assert (run / "package_manifest.json").is_file()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_finish_v2_synthetic_diagnose_and_curves(tmp_path):
    """Exercise the actual 16-image diagnostics and fixed FP32 evaluation on synthetic data only."""
    import numpy as np
    from PIL import Image
    from tools.experiments import finish_b19_cca_fusion as finish
    from tools.experiments import run_b19_cca_fusion_v2 as run
    from ultralytics.nn.tasks import DetectionModel

    shared = run.shared
    data_root = tmp_path / "synthetic_data"
    (data_root / "images").mkdir(parents=True)
    (data_root / "labels").mkdir()
    for i in range(16):
        Image.fromarray(np.random.default_rng(i).integers(0, 256, (64, 96, 3), dtype=np.uint8)).save(
            data_root / f"images/{i:02}.png"
        )
        (data_root / f"labels/{i:02}.txt").write_text("0 0.5 0.5 0.2 0.3\n")
    data = data_root / "data.yaml"
    shared.YAML.save(data, dict(path=str(data_root), train="images", val="images", test="images", names={0: "crack"}))
    model = DetectionModel(str(run.MODEL), nc=1, verbose=False).eval()
    model.args = shared.get_cfg()
    model.names = {0: "crack"}
    torch.nn.init.normal_(model.model[12].Wo.weight, std=0.01)
    result_run = tmp_path / "synthetic_run"
    (result_run / "weights").mkdir(parents=True)
    torch.save(dict(model=model, train_args=vars(model.args)), result_run / "weights/best.pt")
    shared.write_json(result_run / "provenance/checks.json", dict(scope="synthetic unit fixture; not server preflight"))
    finish.diagnose(result_run, data, experiment=run.EXPERIMENT)
    pointer = json.loads((result_run / "diagnostics.json").read_text())
    report = json.loads((result_run / pointer["path"]).read_text())
    assert report["evidence"]["version"] == 2 and len(report["images"]) == 16
    assert all("reliability_v2" in row for row in report["images"])
    evidence = finish.provenance(result_run, data, run.EXPERIMENT)
    record = finish.test_best(
        result_run, data, split="val", block_type=Concat_CCA_Fusion_V2, evidence=evidence, output=result_run / "val"
    )
    assert record["images"] == 16 and record["args"]["quantize"] is None
    assert all((result_run / "val" / name).is_file() for name in finish.EVAL_ARTIFACTS)


def test_v2_entry_identity_and_source_fingerprint(monkeypatch, tmp_path):
    """Training and completion fingerprint the same source set; subprocess stages stay in v2."""
    from tools.experiments import finish_b19_cca_fusion as finish
    from tools.experiments import run_b19_cca_fusion_v2 as run

    shared = run.shared
    sources = [
        run.ROOT / "tools/experiments/run_b19_cca_fusion_v2.py",
        run.ROOT / "tools/experiments/server_b19_cca_fusion_v1.sh",
        run.ROOT / "tools/experiments/finish_b19_cca_fusion.py",
        *(run.ROOT / p for p in run.SOURCE_FILES),
    ]
    run_dir = tmp_path / "run"
    (run_dir / "weights").mkdir(parents=True)
    (run_dir / "weights/best.pt").write_bytes(b"identity fixture")
    data = tmp_path / "data.yaml"
    data.write_text("fixture")
    monkeypatch.setattr(shared, "dataset_manifest", lambda _: {})
    evidence = finish.provenance(run_dir, data, run.EXPERIMENT)
    assert shared.source_hashes(sources) == evidence["source_sha256"]
    assert evidence["version"] == 2
    with pytest.raises(SystemExit) as error:
        run.main(["--baseline-root", str(tmp_path), "--stage", "train", "--name", shared.NAME])
    assert error.value.code == 2


def test_v2_full_package_requires_comparison(monkeypatch, tmp_path):
    """Exercise package gates with explicit synthetic receipts, including mandatory b19 comparison."""
    from tools.experiments import finish_b19_cca_fusion as finish
    from tools.experiments import run_b19_cca_fusion_v2 as run

    shared = run.shared
    root = tmp_path / "synthetic_receipts"
    root.mkdir()
    write = shared.write_json
    evidence = dict(
        commit="synthetic", data_sha256="synthetic", dataset_manifest={}, source_sha256={}, status="", version=2
    )
    monkeypatch.setattr(finish, "provenance", lambda *args: evidence.copy())
    for stage in ("train", "test", "diagnose"):
        attempt = tmp_path / f"{stage}.attempt.fixture"
        attempt.mkdir()
        (attempt / "exit_status").write_text("0")
        (root.parent / f"{root.name}_{stage}.current_attempt").write_text(str(attempt))
    write(root / "provenance/checks.json", dict(passed=True, missing_effective_parameters=[]))
    write(root / "provenance/resolved.json", evidence)
    write(root / "provenance/preflight_process_status.json", dict(python=0, commit="synthetic"))
    index = dict(evidence=evidence, reports={}, recall_at_precision=dict(attainable=False))
    for split in ("val", "test"):
        folder = root / "evaluation" / split
        folder.mkdir(parents=True)
        for name in finish.EVAL_ARTIFACTS:
            (folder / name).write_bytes(b"synthetic fixture")
        record = dict(
            evidence=evidence, artifacts={name: shared.sha256(folder / name) for name in finish.EVAL_ARTIFACTS}
        )
        write(folder / "metrics.json", record)
        index["reports"][split] = dict(
            path=f"evaluation/{split}/metrics.json", sha256=shared.sha256(folder / "metrics.json")
        )
    write(
        root / "evaluation/val/frozen_operating_point.json",
        dict(evidence=evidence, operating=index["recall_at_precision"]),
    )
    write(root / "evaluation.json", index)
    write(root / "evaluation/diagnostic/metrics.json", dict(evidence=dict(evidence, samples=[]), artifacts={}))
    write(
        root / "diagnostics.json",
        dict(
            path="evaluation/diagnostic/metrics.json", sha256=shared.sha256(root / "evaluation/diagnostic/metrics.json")
        ),
    )
    with pytest.raises(FileNotFoundError, match="same-condition b19"):
        finish.package(root, tmp_path / "unused_data.yaml", tmp_path / "missing.tar.gz", run.EXPERIMENT)
    assert not (tmp_path / "missing.tar.gz").exists()
    baseline_evidence = dict(evidence, weight_sha256="d0b2ca5a5d30de9ed002c64c9238b182ddeccce644c055ae5f2dba566878bd5e")
    baseline = dict(evidence=baseline_evidence, reports={})
    for split in ("val", "test"):
        folder = root / "baseline_comparison" / split
        folder.mkdir(parents=True)
        for name in finish.EVAL_ARTIFACTS:
            (folder / name).write_bytes(b"synthetic fixture")
        write(
            folder / "metrics.json",
            dict(
                evidence=baseline_evidence,
                artifacts={name: shared.sha256(folder / name) for name in finish.EVAL_ARTIFACTS},
            ),
        )
        baseline["reports"][split] = dict(path=f"{split}/metrics.json", sha256=shared.sha256(folder / "metrics.json"))
    write(root / "baseline_comparison/evaluation.json", baseline)
    for name in (
        "args.yaml",
        "results.csv",
        "train.log",
        "weights/best.pt",
        "completed.json",
        "val_metrics.json",
        "provenance/final_weight_audit.json",
        "provenance/final_optimizer.json",
        "provenance/b19_original_args.yaml",
        "provenance/b19_launcher_expanded.txt",
    ):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"synthetic receipt fixture")
    output = tmp_path / "fixture.tar.gz"
    finish.package(root, tmp_path / "unused_data.yaml", output, run.EXPERIMENT)
    with tarfile.open(output) as archive:
        assert archive.getmember("source/ultralytics/nn/modules/cca_fusion_v2.py").isfile()
        assert archive.getmember("run/baseline_comparison/evaluation.json").isfile()
