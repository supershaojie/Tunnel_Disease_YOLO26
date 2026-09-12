"""Fixed nc=1 b19 Detect audit, adapted from LBI bc090ac without importing its model or changing inference."""

import copy
import random
import sys
import tempfile
import traceback
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch.overrides import TorchFunctionMode

import ultralytics
from tools.experiments import b19_common as common
from ultralytics.nn.modules import Conv, DCS_SPPF, DCS_SPPF_V2


PROFILES = ("native_precision_diagnostic", "strict_fp32_equivalence", "amp_diagnostic")


class FuseEquivalenceMismatch(AssertionError):
    """A finite cross-path value exceeds the unchanged tolerance, after structural checks pass."""


@contextmanager
def fuse_precision(device, profile):
    """Isolate independent audit copies; restore grouped backend settings, autocast and all RNG on either exit."""
    assert profile in PROFILES, f"Unknown precision profile: {profile}"
    device = torch.device(device)
    before = common.computation_conditions()
    autocast = {
        d: dict(enabled=torch.is_autocast_enabled(d), dtype=str(torch.get_autocast_dtype(d))) for d in ("cpu", "cuda")
    }
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    receipt = dict(profile=profile, before=before, autocast_before=autocast, rng_restored=False)
    try:
        # fork_rng covers CPU and every available CUDA generator, including deliberate failed forwards.
        with torch.random.fork_rng(), torch.autocast(device.type, enabled=False):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            if profile == "strict_fp32_equivalence":
                torch.backends.cudnn.allow_tf32 = False
                torch.set_float32_matmul_precision("highest")
            receipt["during"] = common.computation_conditions()
            receipt["autocast_during_fusion"] = torch.is_autocast_enabled(device.type)
            yield receipt
    finally:
        try:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
        finally:
            torch.backends.cudnn.allow_tf32 = before["cudnn_allow_tf32"]
            torch.backends.cuda.matmul.allow_tf32 = before["matmul_allow_tf32"]
            torch.set_float32_matmul_precision(before["float32_matmul_precision"])
            random.setstate(python_rng)
            np.random.set_state(numpy_rng)
            receipt["restored"] = common.computation_conditions()
            receipt["autocast_restored"] = {
                d: dict(enabled=torch.is_autocast_enabled(d), dtype=str(torch.get_autocast_dtype(d)))
                for d in ("cpu", "cuda")
            }
            current_numpy = np.random.get_state()
            receipt["rng_restored"] = (
                random.getstate() == python_rng
                and current_numpy[0] == numpy_rng[0]
                and np.array_equal(current_numpy[1], numpy_rng[1])
                and current_numpy[2:] == numpy_rng[2:]
                and torch.equal(torch.get_rng_state(), cpu_rng)
                and all(torch.equal(torch.cuda.get_rng_state(i), state) for i, state in enumerate(cuda_rng))
            )
            assert receipt["rng_restored"], "Audit RNG restoration failed"
            assert receipt["restored"] == before and receipt["autocast_restored"] == autocast


def snapshot(value):
    """Detach independent CPU evidence immediately, including outputs used by in-place downstream layers."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: snapshot(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(snapshot(v) for v in value)
    return copy.deepcopy(value)


def attributes(model):
    """Record model/BN mode and the fixed Detect cache, precision and postprocess settings."""
    head = model.model[-1]
    block = model.model[9]
    return snapshot(
        dict(
            training={name: m.training for name, m in model.named_modules()},
            block_class=f"{type(model.model[9]).__module__}.{type(model.model[9]).__qualname__}",
            parameter_dtype=str(next(model.parameters()).dtype),
            bn={
                name: dict(eps=m.eps, momentum=m.momentum, affine=m.affine, track_running_stats=m.track_running_stats)
                for name, m in model.named_modules()
                if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)
            },
            block_config={
                name: getattr(block, name)
                for name in ("theta", "residual_budget", "residual_eps")
                if hasattr(block, name)
            },
            head={
                name: getattr(head, name)
                for name in (
                    "nc",
                    "max_det",
                    "end2end",
                    "export",
                    "format",
                    "xyxy",
                    "agnostic_nms",
                    "dynamic",
                    "stride",
                    "strides",
                    "anchors",
                    "shape",
                    "reg_max",
                )
            },
        )
    )


class TopkTrace(TorchFunctionMode):
    """Observe the two actual Tensor.topk operations only during one temporary instance's native head call."""

    def __init__(self, stages):
        self.stages = stages

    def __torch_function__(self, func, types, args=(), kwargs=None):
        output = func(*args, **(kwargs or {}))
        if func is torch.Tensor.topk:
            self.stages.append(
                dict(
                    input=snapshot(args[0]),
                    values=snapshot(output.values),
                    indices=snapshot(output.indices),
                    k=args[1],
                    dim=args[2] if len(args) > 2 else (kwargs or {}).get("dim", -1),
                    largest=(kwargs or {}).get("largest", True),
                    sorted=(kwargs or {}).get("sorted", True),
                )
            )
        return output


def capture(model, x, result, layers=None):
    """Capture live raw/decode/top-k tensors; all temporary methods and hooks are removed even on failure."""
    head = model.model[-1]
    result["autocast"] = dict(
        enabled=torch.is_autocast_enabled(x.device.type), dtype=str(torch.get_autocast_dtype(x.device.type))
    )
    decode, topk = head._inference, head.get_topk_index
    handles = []

    def decoded(raw):
        result["raw"] = snapshot(raw)
        result["grids"] = [list(v.shape) for v in raw["feats"]]
        output = decode(raw)
        result["decoded"] = snapshot(output.permute(0, 2, 1))
        return output

    def indexed(scores, max_det):
        result["topk_stages"] = []
        with TopkTrace(result["topk_stages"]):
            output = topk(scores, max_det)
        result["selected_scores"], result["classes"], result["indices"] = snapshot(output)
        return output

    block = model.model[9]
    if isinstance(block, DCS_SPPF_V2):
        live = {}
        handles.append(block.register_forward_pre_hook(lambda m, a: live.update(x=a[0].detach().clone())))
        handles.append(block.cv2.register_forward_hook(lambda m, a, o: live.update(cv2=o.detach().clone())))
        handles.append(block.fuse.register_forward_hook(lambda m, a, o: live.update(R=o.detach().clone())))

        def residual(m, inputs, output):
            # Reconstruct the controller from actual operands on their original device, then require exact Y.
            y0 = live["cv2"] + live["x"] if m.add else live["cv2"]
            with torch.autocast(device_type=x.device.type, enabled=False):
                r0 = (0.10 * m.theta.float().tanh()) * live["R"].float()
                b = m.residual_budget * y0.float().square().mean((1, 2, 3), keepdim=True).sqrt()
                q = b / (b.square() + r0.square().mean((1, 2, 3), keepdim=True) + m.residual_eps**2).sqrt()
                r = q * r0
                reconstructed = y0 + r.to(y0.dtype)
            result["residual"] = snapshot(
                dict(
                    y0=y0,
                    R=live["R"],
                    theta=m.theta,
                    r0=r0,
                    b=b,
                    q=q,
                    r=r,
                    Y=output,
                    reconstructed=reconstructed,
                    rho=m.residual_budget,
                    eps=m.residual_eps,
                )
            )

        handles.append(block.register_forward_hook(residual))
    if layers is not None:
        for name, module in model.named_modules():
            if (
                isinstance(module, (Conv, DCS_SPPF))
                or (name.startswith("model.") and name.count(".") == 1 and module is not head)
                or (isinstance(module, torch.nn.Conv2d) and not name.endswith(".conv"))
            ):
                handles.append(
                    module.register_forward_hook(
                        lambda m, inputs, output, name=name: layers.update({name: snapshot(output)})
                    )
                )
    try:
        with patch.object(head, "_inference", decoded), patch.object(head, "get_topk_index", indexed), torch.no_grad():
            output = model(x)
            result["final"] = snapshot(output[0])
            result["output_device"], result["output_dtype"] = str(output[0].device), str(output[0].dtype)
            result["one2many_keys"] = list(output[1]["one2many"])
        result["attributes"] = attributes(model)
        result["input_after"] = snapshot(x)
    finally:
        for handle in handles:
            handle.remove()


def audit_topk_stages(data, k, label, rows):
    """Validate both observed native stages, all selected scores, and the complete local-to-original mapping."""
    stages = data["topk_stages"]
    assert len(stages) == 2, "UNRESOLVED: expected the fixed two-stage nc=1 head"
    stage_input = data["decoded"][..., 4]
    for number, stage in enumerate(stages):
        values, indices = stage["values"], stage["indices"]
        prefix = f"{label}.topk.{number + 1}"
        assert stage["k"] == k and stage["dim"] in (-1, 1) and stage["largest"] and stage["sorted"]
        common.assert_close_tree(stage_input, stage["input"], 0, 0, f"{prefix}.input", rows)
        assert indices.shape == values.shape == (stage_input.shape[0], k)
        assert indices.dtype == torch.int64 and bool(((indices >= 0) & (indices < stage_input.shape[1])).all())
        assert all(len(ids.unique()) == k for ids in indices), "Duplicate stage candidate ID"
        common.assert_close_tree(stage_input.gather(1, indices), values, 0, 0, f"{prefix}.gather", rows)
        common.assert_close_tree(
            stage_input.sort(1, descending=True).values[:, :k], values, 0, 0, f"{prefix}.all_top_scores", rows
        )
        stage_input = stage_input.gather(1, indices)
    # nc=1: stage2 receives exactly k scores and selects all k, so it can only permute stage1.
    mapped = stages[0]["indices"].gather(1, stages[1]["indices"]).unsqueeze(-1)
    common.assert_close_tree(mapped, data["indices"], 0, 0, f"{label}.original_ids", rows)
    return dict(
        stage1_ids=stages[0]["indices"].tolist(),
        stage1_scores=stages[0]["values"].tolist(),
        stage1_cutoff=stages[0]["values"][:, -1].tolist(),
        stage2_local_ids=stages[1]["indices"].tolist(),
        stage2_scores=stages[1]["values"].tolist(),
        stage2_cutoff=stages[1]["values"][:, -1].tolist(),
        stage2_is_full_permutation=True,
    )


def audit_residual(before, after, report):
    """Check the captured live controller operands, exact output reconstruction and the original rounding budget."""
    rows = report["residual_rows"] = []
    report["residual_norms"] = {}
    for label, data in (("before", before), ("after", after)):
        assert data["rho"] == 0.05 and data["eps"] == 1e-6
        assert data["q"].shape == (data["y0"].shape[0], 1, 1, 1)
        assert all(data[key].dtype == torch.float32 for key in ("r0", "b", "q", "r"))
        common.assert_close_tree(data["reconstructed"], data["Y"], 0, 0, f"{label}.controller_output", rows)

        def rms(value):
            return value.double().square().mean((1, 2, 3)).sqrt()

        denominator = rms(data["y0"]).clamp_min(1e-30)
        ratio = rms(data["r"]) / denominator
        observed = data["Y"].float() - data["y0"].float()
        rounding = rms(observed - data["r"]) / denominator
        assert bool((ratio <= 0.05 + 2e-8).all())
        assert bool((rms(observed) / denominator <= ratio + rounding + 1e-12).all())
        report["residual_norms"][label] = dict(
            rms={key: rms(data[key]).tolist() for key in ("y0", "R", "r0", "b", "q", "r", "Y")},
            theta=data["theta"].item(),
            injected_ratio=ratio.tolist(),
            rounding_ratio=rounding.tolist(),
        )
    common.assert_close_tree(before, after, 1e-4, 1e-4, "fusion.residual", rows, mismatch_error=FuseEquivalenceMismatch)


def fuse_audit(model, x, directory, *, profile="strict_fp32_equivalence", report=None):
    """Own unique failure receipts, same-source copies and fixed-tolerance gates; never modify the source model."""
    Path(directory).mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="fuse-evidence-", dir=directory))
    report = {} if report is None else report
    report.update(
        passed=False,
        status="FAIL",
        profile=profile,
        required=profile == "strict_fp32_equivalence",
        stage="capture",
        directory=str(directory),
        commit=common.git("rev-parse", "HEAD"),
        tracked_diff=common.git("diff", "HEAD", "--name-only"),
        source_sha256=common.source_hashes(),
        seed=torch.initial_seed(),
        input_device=str(x.device),
        input_dtype=str(x.dtype),
        input_shape=list(x.shape),
        python=sys.version,
        torch=torch.__version__,
        cuda=torch.version.cuda,
        cudnn=torch.backends.cudnn.version(),
        ultralytics=ultralytics.__version__,
        gpu=torch.cuda.get_device_name(x.device) if x.is_cuda else None,
        capability=torch.cuda.get_device_capability(x.device) if x.is_cuda else None,
        backend=common.computation_conditions(),
        atol=1e-4,
        rtol=1e-4,
        forward_amp=profile == "amp_diagnostic",
        autocast={
            d: dict(enabled=torch.is_autocast_enabled(d), dtype=str(torch.get_autocast_dtype(d)))
            for d in ("cpu", "cuda")
        },
        artifact_trust="Locally generated trusted diagnostic state/tensors; never a training checkpoint",
    )
    try:
        with fuse_precision(x.device, profile) as conditions:
            report["precision"] = conditions
            _fuse_audit(model, x.float(), directory, report)
        report["status"] = "PASS"
    except BaseException:
        report.update(passed=False, status="FAIL", traceback=traceback.format_exc())
        raise
    finally:
        common.write_json(directory / "audit.json", report)
    return report


def _fuse_audit(model, x, directory, report):
    """Fuse and observe only independent FP32 copies inside the caller's declared precision scope."""
    forward_amp = report["forward_amp"]
    before, after = {}, {}
    before_model = after_model = None
    common.write_json(directory / "audit.json", report)
    try:
        torch.save(
            dict(
                input=snapshot(x),
                cpu_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                python_rng=random.getstate(),
                numpy_rng=np.random.get_state(),
            ),
            directory / "input_rng.pt",
        )
        before_model = copy.deepcopy(model).to(x.device).float().eval()
        state = snapshot(before_model.state_dict())
        torch.save(
            dict(state=state, yaml=before_model.yaml, names=before_model.names, attributes=attributes(before_model)),
            directory / "source.pt",
        )
        after_model = copy.deepcopy(before_model)
        common.assert_close_tree(state, snapshot(after_model.state_dict()), 0, 0, "same_source_copy")
        after_model.fuse(verbose=False)
        torch.save(
            dict(state=snapshot(after_model.state_dict()), attributes=attributes(after_model)),
            directory / "fused_state.pt",
        )
        for label, m, data in (("before", before_model, before), ("after", after_model, after)):
            try:
                with torch.autocast(device_type=x.device.type, enabled=forward_amp):
                    capture(m, x.clone(), data)
            finally:
                torch.save(data, directory / f"{label}.pt")
        report["stage"] = "state_and_attributes"
        common.assert_close_tree(state, snapshot(before_model.state_dict()), 0, 0, "unfused_state_unchanged")
        for data in (before, after):
            common.assert_close_tree(snapshot(x), data["input_after"], 0, 0, "input_unchanged")
            assert not any(data["attributes"]["training"].values()), "Non-eval module/BN"
            assert data["output_device"] == str(x.device)
        for key in ("head", "block_class", "block_config", "parameter_dtype"):
            common.assert_close_tree(before["attributes"][key], after["attributes"][key], 0, 0, key)
        assert before["output_dtype"] == after["output_dtype"]
        assert before["autocast"] == after["autocast"]
        assert before["grids"] == after["grids"] and before["one2many_keys"] and not after["one2many_keys"]
        assert after_model.model[-1].cv2 is after_model.model[-1].cv3 is None
        head = before_model.model[-1]
        assert head.end2end and not head.export and not head.agnostic_nms and head.nc == 1
        report["stage"] = "single_model_invariants"
        report["single_model_invariants"] = {}
        for label, data in (("before", before), ("after", after)):
            invariant = report["single_model_invariants"][label] = {}
            audit_candidates(data, data, head.max_det, 1e-4, 1e-4, invariant)
            if "residual" in data:
                audit_residual(data["residual"], data["residual"], invariant)
            invariant["passed"] = True
        audit_candidates(before, after, head.max_det, 1e-4, 1e-4, report)
        if isinstance(before_model.model[9], DCS_SPPF_V2):
            report["stage"] = "residual"
            from tools.experiments.run_b19_dcs_sppf_v2 import MODEL

            report["binding"] = common.model_binding(MODEL, DCS_SPPF_V2, before_model.model[9])
            assert report["binding"] == common.model_binding(MODEL, DCS_SPPF_V2, after_model.model[9])
            audit_residual(before["residual"], after["residual"], report)
        report.update(passed=True, stage="complete")
    except BaseException:
        report["traceback"] = traceback.format_exc()
        if before_model is not None and after_model is not None:
            layers = [{}, {}]
            try:
                for m, outputs in zip((before_model, after_model), layers):
                    with torch.autocast(device_type=x.device.type, enabled=forward_amp):
                        capture(m, x.clone(), {}, outputs)
                torch.save(layers, directory / "layer_outputs.pt")
                report["layers"] = {
                    name: diagnostic(value, layers[1][name], 1e-4, 1e-4, name)
                    for name, value in layers[0].items()
                    if name in layers[1]
                }
                report["first_layer_outside_tolerance"] = next(
                    (name for name, value in report["layers"].items() if not value["passed"]), None
                )
            except BaseException:
                report["localization_traceback"] = traceback.format_exc()
        raise
    finally:
        common.write_json(directory / "audit.json", report)
    return report


def fuse_precision_checks(models, x, directory, report=None):
    """Require strict full-network equivalence; retain finite native/AMP numerical FAIL as named diagnostics."""
    directory = Path(directory)
    report = {} if report is None else report
    report.update(
        strict_gate_passed=False,
        profiles={},
        gate_definition="Strict FP32 plus all single-model invariants are mandatory; finite cross-path native/AMP "
        "tolerance failures are diagnostic FAIL, never training safety or B32 certification.",
    )
    failures = []
    try:
        for profile in PROFILES:
            cases = report["profiles"][profile] = {}
            for label, model in models.items():
                result = cases[label] = {}
                print(f"BEGIN fuse {profile}/{label}", flush=True)
                try:
                    fuse_audit(model, x, directory / profile / label, profile=profile, report=result)
                except FuseEquivalenceMismatch:
                    if profile == "strict_fp32_equivalence":
                        failures.append(f"{profile}/{label}")
                except Exception:
                    failures.append(f"{profile}/{label}")
                finally:
                    print(f"END fuse {profile}/{label}: {result.get('status', 'FAIL')}", flush=True)
            common.write_json(directory / "precision_checks.json", report)
        report["blocking_failures"] = failures
        assert not failures, f"Required fuse checks failed; see per-profile evidence: {failures}"
        # Same input/state/attributes and folding for each variant across profiles, including whole-model updated state.
        for label in models:
            baseline = Path(report["profiles"][PROFILES[0]][label]["directory"])
            for profile in PROFILES[1:]:
                target = Path(report["profiles"][profile][label]["directory"])
                for filename in ("source.pt", "fused_state.pt"):
                    a, b = [torch.load(p / filename, map_location="cpu", weights_only=True) for p in (baseline, target)]
                    common.assert_close_tree(a, b, 0, 0, f"same_profile_{filename}.{label}")
        report["same_source_and_fold_across_profiles"] = True
        report["strict_gate_passed"] = True
    except BaseException:
        report["traceback"] = traceback.format_exc()
        raise
    finally:
        common.write_json(directory / "precision_checks.json", report)
    return report


def diagnostic(a, b, atol, rtol, path):
    """Retain an unsuccessful positional comparison as evidence, separate from the identity-based gate."""
    result = dict(rows=[], passed=False)
    try:
        common.assert_close_tree(a, b, atol, rtol, path, result["rows"])
        result["passed"] = True
    except AssertionError:
        result["traceback"] = traceback.format_exc()
    return result


def audit_candidates(before, after, max_det, atol, rtol, report):
    """Check every candidate, exact gather/ranking, and every changed boundary ID without filtering scores."""
    report["positional"] = diagnostic(before["final"], after["final"], atol, rtol, "fusion.predictions")
    report["positional_units"] = {
        key: diagnostic(before["final"][..., s], after["final"][..., s], atol, rtol, key)
        for key, s in (("boxes", slice(0, 4)), ("scores", slice(4, 5)), ("classes", slice(5, 6)))
    }
    rows = report["candidate_rows"] = []
    report["stage"] = "raw_and_decode"
    for key in ("boxes", "scores", "feats"):
        common.assert_close_tree(
            before["raw"][key],
            after["raw"][key],
            atol,
            rtol,
            f"raw.one2one.{key}",
            rows,
            mismatch_error=FuseEquivalenceMismatch,
        )
    for key, s in (("boxes", slice(0, 4)), ("scores", slice(4, None))):
        common.assert_close_tree(
            before["decoded"][..., s],
            after["decoded"][..., s],
            atol,
            rtol,
            f"decoded.{key}",
            rows,
            mismatch_error=FuseEquivalenceMismatch,
        )
    report["stage"] = "postprocess"
    a, b = before["decoded"], after["decoded"]
    assert a.ndim == 3 and a.shape[-1] == 5, "Audit requires fixed b19 nc=1"
    k = min(max_det, a.shape[1])
    thresholds = []
    report["topk_stages"] = {}
    for label, data in (("before", before), ("after", after)):
        assert "topk_stages" in data, "UNRESOLVED: missing actual two-stage top-k trace"
        report["topk_stages"][label] = audit_topk_stages(data, k, label, rows)
        values, indices, classes = data["decoded"], data["indices"], data["classes"]
        assert indices.shape == classes.shape == (a.shape[0], k, 1)
        assert indices.dtype == torch.int64 and bool(((indices >= 0) & (indices < a.shape[1])).all())
        assert bool((classes == 0).all()), "Unexpected class ID in single-class b19"
        gathered = values.gather(1, indices.expand(-1, -1, 5))
        expected = torch.cat((gathered, classes), -1)
        common.assert_close_tree(expected, data["final"], 0, 0, f"{label}.actual_gather", rows)
        common.assert_close_tree(gathered[..., 4:5], data["selected_scores"], 0, 0, f"{label}.selected_scores", rows)
        # Both stages already proved all top scores, unique IDs, ranking and the original-ID composition.
        threshold = data["topk_stages"][0]["values"][:, -1]
        thresholds.append(threshold.double())
    report["batches"] = []
    for batch in range(a.shape[0]):
        old, new = before["indices"][batch, :, 0], after["indices"][batch, :, 0]
        dropped = sorted(set(old.tolist()) - set(new.tolist()))
        added = sorted(set(new.tolist()) - set(old.tolist()))
        union = sorted(set(old.tolist()) | set(new.tolist()))
        scores_a, scores_b = a[batch, :, 4].double(), b[batch, :, 4].double()
        perturbation = (scores_b - scores_a).abs()
        boundary = []
        for kind, ids in (("dropped", dropped), ("added", added)):
            for i in ids:
                boundary.append(
                    dict(
                        kind=kind,
                        candidate=i,
                        class_id=0,
                        score_before=scores_a[i].item(),
                        score_after=scores_b[i].item(),
                        margin_before=(scores_a[i] - thresholds[0][batch]).item(),
                        margin_after=(scores_b[i] - thresholds[1][batch]).item(),
                        perturbation=perturbation[i].item(),
                    )
                )
        info = dict(
            changed_rank_positions=int((old != new).sum()),
            same_set=not dropped and not added,
            dropped=dropped,
            added=added,
            boundary=boundary,
            kth_before=thresholds[0][batch].item(),
            kth_after=thresholds[1][batch].item(),
            measured_max_score_perturbation=perturbation.max().item(),
            compared_union=len(union),
        )
        report["batches"].append(info)
        if dropped:
            # For EVERY dropped/added pair, the rank reversal must be explained by its measured perturbation.
            gap = scores_a[dropped, None] - scores_a[added][None, :]
            budget = perturbation[dropped, None] + perturbation[added][None, :]
            info["boundary_pair_count"] = gap.numel()
            info["max_boundary_gap"] = gap.max().item()
            info["max_gap_minus_measured_budget"] = (gap - budget).max().item()
            assert bool((gap >= 0).all() and (gap <= budget).all()), "UNRESOLVED: unexplained top-k boundary change"
        # Includes dropped AND added identities, with their counterpart before selection, not just the intersection.
        for key, s in (("boxes", slice(0, 4)), ("scores", slice(4, 5))):
            common.assert_close_tree(
                a[batch, union, s],
                b[batch, union, s],
                atol,
                rtol,
                f"union.{batch}.{key}",
                rows,
                mismatch_error=FuseEquivalenceMismatch,
            )
    report["stage"] = "complete"
