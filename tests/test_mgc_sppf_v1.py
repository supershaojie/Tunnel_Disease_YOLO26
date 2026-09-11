"""MGC integration and fixed morphology regressions; these are local tests, not server receipts."""

import copy
import json
import os
import tarfile

import pytest
import torch

from tools.experiments import b19_common as common
from tools.experiments.finish_b19_mgc_sppf import archive_package
from tools.experiments.run_b19_mgc_sppf import audit_arguments
from tools.experiments.verify_b19_mgc_sppf import build_pair, module_checks, topology_checks
from ultralytics.nn.modules import MGC_SPPF, SPPF
from ultralytics.nn.tasks import DetectionModel


@pytest.fixture(scope="module", autouse=True)
def small_thread_pool():
    """Keep local CPU tests bounded without changing any server runtime policy."""
    previous = torch.get_num_threads()
    torch.set_num_threads(4)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("shape", [(1, 16, 20, 20), (2, 16, 13, 17), (1, 1, 1, 1), (1, 1, 1, 7)])
@pytest.mark.parametrize("value", [0.0, 2.0, -3.0])
@pytest.mark.parametrize("kernel", [3, 5])
def test_constant_gaps(shape, value, kernel):
    """Include corners, negative inputs and windows larger than a spatial dimension."""
    x = torch.full(shape, value, requires_grad=True)
    gap = MGC_SPPF.gap(x, kernel)
    assert gap.shape == x.shape and gap.dtype == x.dtype
    assert torch.isfinite(gap).all() and gap.count_nonzero() == 0
    gap.sum().backward()
    assert torch.isfinite(x.grad).all()  # Framework tie subgradient, not cross-platform order.


def neighborhood_extreme(x, kernel, maximum):
    """Use explicit valid neighborhoods as an independent small boundary reference."""
    output = torch.empty_like(x)
    h, w = x.shape[-2:]
    r = kernel // 2
    for y in range(h):
        for z in range(w):
            patch = x[..., max(y - r, 0) : min(y + r + 1, h), max(z - r, 0) : min(z + r + 1, w)]
            output[..., y, z] = patch.amax((-2, -1)) if maximum else patch.amin((-2, -1))
    return output


@pytest.mark.parametrize("kernel", [3, 5])
def test_boundary_and_constructed_gap(kernel):
    """Compare valid-neighborhood morphology and two separate, explicitly constructed toy cases."""
    x = -torch.rand(2, 3, 7, 11)
    reference = neighborhood_extreme(neighborhood_extreme(x, kernel, True), kernel, False) - x
    torch.testing.assert_close(MGC_SPPF.gap(x, kernel), reference, rtol=0, atol=0)
    line = torch.zeros(1, 1, 15, 19)
    line[..., 7, 3:16] = 1
    line[..., 7, 9] = 0
    assert MGC_SPPF.gap(line, kernel)[0, 0, 7, 9] == 1
    peak = torch.zeros_like(line)
    peak[..., 7, 9] = 1
    assert MGC_SPPF.gap(peak, kernel).count_nonzero() == 0


def test_identity_bn_and_staged_gradients():
    """Exercise both B values and shapes, train/eval BN state and two actual SGD steps."""
    report = module_checks("cpu")
    assert len(report["identities"]) == 8


def test_parser_rng_and_shared_state():
    """Construct native and candidate with identical seed streams and audit every shared tensor."""
    native, candidate = build_pair()
    topology = topology_checks(native, candidate)
    report = common.audit_weights(native, candidate, None)
    assert topology["changed_layers"] == [9]
    assert report["all_common_tensors_equal"] and report["added_parameters"] == 10240
    assert report["exact_matches"] == len(native.state_dict())
    assert set(report["new_parameters"]) == {"model.9.gap_in.weight", "model.9.gap_out.weight"}
    assert type(native.model[9]) is SPPF


@pytest.mark.parametrize("channels,shortcut", [(32, False), (48, True)])
def test_general_channels_and_native_compatibility(channels, shortcut):
    """Keep the native compatibility fallbacks and support differing input/output channels."""
    native = SPPF(32, channels, 5, 3, shortcut).eval()
    candidate = MGC_SPPF(32, channels, 5, 3, shortcut).eval()
    candidate.load_state_dict(native.state_dict(), strict=False)
    del native.n, candidate.n, native.add, candidate.add
    x = torch.randn(2, 32, 13, 17)
    torch.testing.assert_close(candidate(x), native(x), rtol=0, atol=0)


def test_nonzero_state_reload_and_ema(tmp_path):
    """Loading learned output projections and EMA must preserve a measurable residual."""
    from ultralytics.utils.torch_utils import ModelEMA

    native = SPPF(32, 32, 5, 3, True).eval()
    model = MGC_SPPF(32, 32, 5, 3, True).eval()
    model.load_state_dict(native.state_dict(), strict=False)
    with torch.no_grad():
        model.gap_out.weight.fill_(0.01)
    path = tmp_path / "weights.pt"
    torch.save(model.state_dict(), path)
    restored = MGC_SPPF(32, 32, 5, 3, True).eval()
    restored.load_state_dict(torch.load(path, weights_only=True))
    x = torch.randn(2, 32, 13, 17)
    torch.testing.assert_close(restored(x), model(x), rtol=0, atol=0)
    assert (restored(x) - native(x)).abs().max() > 0
    ema = ModelEMA(model)
    assert torch.equal(ema.ema.gap_out.weight, model.gap_out.weight)


def test_native_head_detach_is_preserved():
    """One2one learns its own head while the original detach keeps its backbone gradient absent."""
    native, candidate = build_pair()
    for model in (native, candidate):
        model.train()
        output = model(torch.rand(1, 3, 64, 64))
        output["one2one"]["scores"].sum().backward()
        assert model.model[0].conv.weight.grad is None
        assert any(p.grad is not None and p.grad.count_nonzero() for p in model.model[-1].one2one_cv3.parameters())


def test_args_whitelist():
    """Identity differences are allowed and training/augmentation changes fail closed."""
    raw = common.REFERENCE["args"]
    candidate = copy.deepcopy(raw)
    candidate["name"] = "yolo26n_b19_mgc_sppf_v1"
    assert set(audit_arguments(raw, candidate)) == {"name"}
    for key in ("batch", "seed", "patience", "mosaic", "amp", "lr0", "nbs"):
        changed = {**candidate, key: "invalid"}
        with pytest.raises(ValueError, match="Non-identity"):
            audit_arguments(raw, changed)


def test_package_dry_run_and_missing_required(tmp_path):
    """Roundtrip a miniature canonical run and reject missing formal artifacts."""
    run = tmp_path / "run"
    (run / "weights").mkdir(parents=True)
    for name in ("weights/best.pt", "weights/last.pt", "args.yaml", "results.csv"):
        (run / name).write_bytes(b"fixture only, not training evidence\n")
    (run / "temporary.attempt.fixture").mkdir()
    (run / "temporary.attempt.fixture" / "old.log").write_text("not canonical")
    os.link(run / "args.yaml", run / "hardlinked_args.yaml")
    required = ["weights/best.pt", "weights/last.pt", "args.yaml", "results.csv", "hardlinked_args.yaml"]
    result = archive_package(run, tmp_path / "dry_run.tar.gz", required)
    assert result["gzip_crc_verified"]
    with tarfile.open(result["path"]) as archive:
        assert all(member.isfile() for member in archive.getmembers())
        assert archive.extractfile("run/hardlinked_args.yaml").read() == (run / "args.yaml").read_bytes()
        assert not any(".attempt." in name for name in archive.getnames())
    with pytest.raises(FileNotFoundError, match="Missing regular"):
        archive_package(run, tmp_path / "missing.tar.gz", ["required_missing.json"])


def test_archive_links(tmp_path):
    """Exercise real links when permitted, explicitly skipping Windows without symlink privileges."""
    run = tmp_path / "run"
    run.mkdir()
    (run / "canonical.log").write_text("canonical")
    try:
        (run / "broken.log").symlink_to(run / "missing")
        (run / "alias.log").symlink_to(run / "canonical.log")
    except OSError as error:
        if os.name == "nt" and getattr(error, "winerror", None) == 1314:
            pytest.skip("Windows lacks real symlink privilege; Linux link test NOT RUN")
        raise
    os.link(run / "canonical.log", run / "hard.log")
    result = archive_package(run, tmp_path / "links.tar.gz", ["canonical.log", "hard.log"])
    with tarfile.open(result["path"]) as archive:
        assert archive.getmember("run/hard.log").isfile()
    exclusions = json.loads((run / "package_exclusions.json").read_text())
    assert {"alias.log", "broken.log"} <= {item["path"] for item in exclusions}


def test_other_native_yaml_interface():
    """A second native model family still builds through the unchanged public parser interface."""
    model = DetectionModel(str(common.ROOT / "ultralytics/cfg/models/11/yolo11.yaml"), nc=1, verbose=False)
    assert any(type(layer) is SPPF for layer in model.model)
