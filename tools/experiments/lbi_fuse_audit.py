"""Candidate-identity fuse audit for the fixed single-class b19 Detect head; no inference overrides."""

import copy
import json
import sys
import traceback
from contextlib import contextmanager
from unittest.mock import patch

import torch

import ultralytics
from tools.experiments import b19_common as common
from ultralytics.nn.modules import Conv, Concat_LBI_Fusion


@contextmanager
def fuse_precision(device, allow_tf32):
    """Scope legacy PyTorch 2.8 convolution precision to diagnostics and restore grouped matmul settings."""
    device = torch.device(device)
    before = common.computation_conditions()
    ambient_amp = torch.is_autocast_enabled(device.type)
    receipt = dict(before=before, autocast_before=ambient_amp)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    try:
        torch.backends.cudnn.allow_tf32 = allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        with torch.autocast(device.type, enabled=False):
            receipt["during"] = common.computation_conditions()
            receipt["autocast_during"] = torch.is_autocast_enabled(device.type)
            yield receipt
    finally:
        try:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
        finally:
            torch.backends.cudnn.allow_tf32 = before["cudnn_allow_tf32"]
            torch.backends.cuda.matmul.allow_tf32 = before["matmul_allow_tf32"]
            torch.set_float32_matmul_precision(before["float32_matmul_precision"])
            receipt["restored"] = common.computation_conditions()
            receipt["autocast_restored"] = torch.is_autocast_enabled(device.type)
            assert receipt["restored"] == before and receipt["autocast_restored"] == ambient_amp


def snapshot(value):
    """Detach independent CPU evidence immediately, including outputs of in-place downstream layers."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: snapshot(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(snapshot(v) for v in value)
    return copy.deepcopy(value)


def capture(model, x, layers=None):
    """Observe the actual decode/top-k calls on a diagnostic copy, restoring bound methods even on failure."""
    head = model.model[-1]
    decode, topk = head._inference, head.get_topk_index
    result, handles = {}, []

    def decoded(raw):
        result["raw"] = snapshot({key: raw[key] for key in ("boxes", "scores")})
        result["grids"] = [list(v.shape) for v in raw["feats"]]
        output = decode(raw)
        result["decoded"] = snapshot(output.permute(0, 2, 1))
        return output

    def indexed(scores, max_det):
        output = topk(scores, max_det)
        result["selected_scores"], result["classes"], result["indices"] = snapshot(output)
        return output

    if layers is not None:
        for name, module in model.named_modules():
            # Compare complete Conv+BN+activation blocks, not pre-BN conv against post-fuse conv.
            if (
                isinstance(module, (Conv, Concat_LBI_Fusion))
                or (name.startswith("model.") and name.count(".") == 1 and name != "model.23")
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
        result["attributes"] = snapshot(common.inference_attributes(model))
        result["input_after"] = snapshot(x)
        return result
    finally:
        for handle in handles:
            handle.remove()


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
    report["positional"] = diagnostic(before["final"], after["final"], atol, rtol, "fused_predictions")
    report["positional_units"] = {
        key: diagnostic(before["final"][..., s], after["final"][..., s], atol, rtol, key)
        for key, s in (("boxes", slice(0, 4)), ("scores", slice(4, 5)), ("classes", slice(5, 6)))
    }
    rows = report["candidate_rows"] = []
    numeric = report["numeric_checks"] = {}
    report["stage"] = "raw_and_decode"
    for key in ("boxes", "scores"):
        numeric[f"raw.one2one.{key}"] = diagnostic(
            before["raw"][key], after["raw"][key], atol, rtol, f"raw.one2one.{key}"
        )
    for key, s in (("boxes", slice(0, 4)), ("scores", slice(4, None))):
        numeric[f"decoded.{key}"] = diagnostic(
            before["decoded"][..., s], after["decoded"][..., s], atol, rtol, f"decoded.{key}"
        )
    for result in numeric.values():
        rows.extend(result["rows"])
    report["raw_close"] = all(numeric[f"raw.one2one.{key}"]["passed"] for key in ("boxes", "scores"))
    report["decoded_close"] = all(numeric[f"decoded.{key}"]["passed"] for key in ("boxes", "scores"))
    report["stage"] = "postprocess"
    a, b = before["decoded"], after["decoded"]
    assert a.ndim == 3 and a.shape[-1] == 5, "Audit requires fixed b19 nc=1"
    k = min(max_det, a.shape[1])
    thresholds = []
    for label, data in (("before", before), ("after", after)):
        values, indices, classes = data["decoded"], data["indices"], data["classes"]
        assert indices.shape == classes.shape == (a.shape[0], k, 1)
        assert indices.dtype == torch.int64 and bool(((indices >= 0) & (indices < a.shape[1])).all())
        assert bool((classes == 0).all()), "Unexpected class ID in single-class b19"
        assert all(len(ids.unique()) == k for ids in indices), "Duplicate candidate ID"
        gathered = values.gather(1, indices.expand(-1, -1, 5))
        expected = torch.cat((gathered, classes.to(values.dtype)), -1)
        common.assert_close_tree(expected, data["final"], 0, 0, f"{label}.actual_gather", rows)
        common.assert_close_tree(gathered[..., 4:5], data["selected_scores"], 0, 0, f"{label}.selected_scores", rows)
        scores = data["final"][..., 4]
        assert bool((scores[:, :-1] >= scores[:, 1:]).all()), "Top-k scores are not descending"
        # Validate the entire excluded set too; equal scores may choose any tied identity.
        threshold = values[..., 4].sort(dim=1, descending=True).values[:, k - 1]
        assert torch.equal(scores[:, -1], threshold), "Top-k omitted a higher-scoring candidate"
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
            assert bool((gap >= 0).all() and (gap <= budget).all()), "Unexplained top-k boundary change"
        # Includes dropped AND added identities, with their counterpart before selection, not just the intersection.
        for key, s in (("boxes", slice(0, 4)), ("scores", slice(4, 5))):
            name = f"union.{batch}.{key}"
            numeric[name] = diagnostic(a[batch, union, s], b[batch, union, s], atol, rtol, name)
            rows.extend(numeric[name]["rows"])
    report["postprocess_valid"] = True
    report["stage"] = "raw_and_decode"
    for result in numeric.values():
        assert result["passed"], result.get("traceback", result)
    report["stage"] = "complete"


def fuse_audit(model, x, directory, atol, rtol, *, include_layers=False):
    """Persist same-source states/input/outputs before assertions; fail closed with reconstructable receipts."""
    directory.mkdir(parents=True, exist_ok=False)
    report = dict(
        passed=False,
        stage="capture",
        commit=common.git("rev-parse", "HEAD"),
        tracked_diff=common.git("diff", "HEAD", "--name-only"),
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
        backend=common.computation_conditions(),
        atol=atol,
        rtol=rtol,
        source_sha256={
            name: common.sha256(common.ROOT / name)
            for name in (
                "tools/experiments/lbi_fuse_audit.py",
                "tools/experiments/verify_b19_lbi_fusion.py",
                "ultralytics/nn/modules/head.py",
                "ultralytics/nn/modules/lbi_fusion.py",
                "ultralytics/nn/tasks.py",
                "ultralytics/utils/torch_utils.py",
            )
        },
    )
    common.write_json(directory / "audit.json", report)
    before_model = copy.deepcopy(model).to(x.device).eval()
    state = snapshot(before_model.state_dict())
    original_input = snapshot(x)
    layers = [{}, {}]
    torch.save(
        dict(
            input=original_input,
            state=state,
            yaml=before_model.yaml,
            names=before_model.names,
            attributes=snapshot(common.inference_attributes(before_model)),
        ),
        directory / "source.pt",
    )
    try:
        after_model = copy.deepcopy(before_model)
        copy_state = snapshot(after_model.state_dict())
        after_model.fuse(verbose=False)
        torch.save(
            dict(
                state=snapshot(after_model.state_dict()), attributes=snapshot(common.inference_attributes(after_model))
            ),
            directory / "fused_state.pt",
        )
        before = capture(before_model, x.clone(), layers[0] if include_layers else None)
        torch.save(before, directory / "before.pt")
        after = capture(after_model, x.clone(), layers[1] if include_layers else None)
        torch.save(after, directory / "after.pt")
        if include_layers:
            torch.save(layers, directory / "layer_outputs.pt")
        report["stage"] = "state_and_attributes"
        common.assert_close_tree(state, copy_state, 0, 0, "same_source_copy")
        common.assert_close_tree(state, snapshot(before_model.state_dict()), 0, 0, "unfused_state_unchanged")
        for data in (before, after):
            common.assert_close_tree(original_input, data["input_after"], 0, 0, "input_unchanged")
            assert all(not v["training"] for v in data["attributes"].values()), "Non-eval module/BN"
            assert data["output_device"] == str(x.device) and data["output_dtype"] == str(x.dtype)
        common.assert_close_tree(before["attributes"]["model.23"], after["attributes"]["model.23"], 0, 0, "head_cache")
        assert before["grids"] == after["grids"] and before["one2many_keys"] and not after["one2many_keys"]
        head = before_model.model[-1]
        assert head.end2end and not head.export and not head.agnostic_nms and head.nc == 1
        if isinstance(before_model.model[15], Concat_LBI_Fusion):
            common.assert_close_tree(
                snapshot(before_model.model[15].state_dict()),
                snapshot(after_model.model[15].state_dict()),
                0,
                0,
                "LBI_weights",
            )
            report["lbi_out_nonzero"] = int(before_model.model[15].out.weight.count_nonzero())
        audit_candidates(before, after, head.max_det, atol, rtol, report)
        report["passed"] = True
    except BaseException:
        report["traceback"] = traceback.format_exc()
        if report["stage"] == "raw_and_decode":
            try:
                if not include_layers:
                    for m, outputs in zip((before_model, after_model), layers):
                        capture(m, x.clone(), outputs)
                    torch.save(layers, directory / "layer_outputs.pt")
                report["layers"] = {
                    name: diagnostic(value, layers[1][name], atol, rtol, name)
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


def fuse_precision_checks(models, x, directory, atol, rtol):
    """Require strict FP32 equivalence plus same-source native-precision controls, preserving both verdicts."""
    directory.mkdir(parents=True, exist_ok=False)
    original = common.computation_conditions()
    report = dict(passed=False, original=original, arms={}, native_precision_raw_close={})

    def run_arm(label, allow_tf32):
        arm = report["arms"][label] = dict(models={})
        print(f"LBI fuse precision: {x.device}, {label}, cudnn.allow_tf32={allow_tf32}", flush=True)
        with fuse_precision(x.device, allow_tf32) as conditions:
            arm["conditions"] = conditions
            for variant, model in models.items():
                path = directory / label / variant
                try:
                    arm["models"][variant] = fuse_audit(model, x, path, atol, rtol, include_layers=True)
                except Exception:
                    # Preserve a failing arm; the protocol below still rejects state, postprocess and strict errors.
                    arm["models"][variant] = json.loads((path / "audit.json").read_text(encoding="utf-8"))
                    arm["models"][variant]["exception"] = traceback.format_exc()
        common.write_json(directory / "precision_checks.json", report)
        return arm["models"]

    def load(arm, variant, filename):
        return torch.load(directory / arm / variant / filename, weights_only=True, map_location="cpu")

    try:
        assert list(models) == ["native", "lbi_zero", "lbi_nonzero"]
        assert x.dtype == torch.float32 and not original["matmul_allow_tf32"], "Expected native b19 FP32 lifecycle"
        native = run_arm("native", original["cudnn_allow_tf32"])
        strict = run_arm("explicit_fp32", False)
        report["native_precision_raw_close"] = {k: v.get("raw_close", False) for k, v in native.items()}
        assert all(v["passed"] for v in strict.values()), "Explicit FP32 fuse failed; no precision waiver"
        assert all(v.get("postprocess_valid", False) for v in native.values()), "Native source/postprocess error"
        assert all(v["passed"] or v["stage"] == "raw_and_decode" for v in native.values()), "Unexpected native failure"
        assert all(
            check["rows"] and all(row["finite"] for row in check["rows"])
            for v in native.values()
            for check in v["numeric_checks"].values()
        ), "Non-finite or malformed native candidate tensors"
        for variant in models:
            a, b = [load(arm, variant, "source.pt") for arm in ("native", "explicit_fp32")]
            common.assert_close_tree(a, b, 0, 0, f"{variant}.immutable_source")
            common.assert_close_tree(snapshot(x), a["input"], 0, 0, f"{variant}.fixed_input")
            common.assert_close_tree(
                load("native", variant, "fused_state.pt"),
                load("explicit_fp32", variant, "fused_state.pt"),
                0,
                0,
                f"{variant}.same_fold",
            )
        baseline = load("native", "native", "source.pt")["state"]
        for variant in ("lbi_zero", "lbi_nonzero"):
            state = load("native", variant, "source.pt")["state"]
            common.assert_close_tree(baseline, {k: state[k] for k in baseline}, 0, 0, f"{variant}.shared_state")
            assert bool(state["model.15.out.weight"].count_nonzero()) == (variant == "lbi_nonzero")
        report["precision_effects"] = {}
        for variant in models:
            effects = report["precision_effects"][variant] = {}
            for side in ("before", "after"):
                a, b = [load(arm, variant, f"{side}.pt") for arm in ("native", "explicit_fp32")]
                effects[side] = {
                    key: diagnostic(a["raw"][key], b["raw"][key], atol, rtol, key) for key in ("boxes", "scores")
                }
        for arm in ("native", "explicit_fp32"):
            for side in ("before", "after"):
                a, b = [load(arm, variant, f"{side}.pt") for variant in ("native", "lbi_zero")]
                for key in ("raw", "decoded"):
                    common.assert_close_tree(a[key], b[key], 0, 0, f"{arm}.{side}.zero_identity.{key}")
        if all(v["passed"] for v in native.values()):
            report["classification"] = "native_and_explicit_fp32_close"
        else:
            assert not native["native"]["passed"] and not native["lbi_zero"]["passed"], "LBI-only native anomaly"
            assert x.is_cuda and torch.cuda.get_device_capability(x.device)[0] >= 8
            assert original["cudnn_allow_tf32"] and original["cudnn_deterministic"] and original["deterministic"]
            first = native["native"].get("first_layer_outside_tolerance")
            assert first and int(first.split(".")[1]) < 15, "Native error lacks shared pre-LBI localization"
            assert all(v["passed"] or v.get("first_layer_outside_tolerance") == first for v in native.values()), (
                "Unexplained nonzero-LBI native anomaly"
            )
            baseline_layers = load("native", "native", "layer_outputs.pt")
            for variant in ("lbi_zero", "lbi_nonzero"):
                layers = load("native", variant, "layer_outputs.pt")
                for side in range(2):
                    prefix = {k: v for k, v in baseline_layers[side].items() if int(k.split(".")[1]) < 15}
                    common.assert_close_tree(
                        prefix, {k: layers[side][k] for k in prefix}, 0, 0, f"{variant}.shared_prefix.{side}"
                    )
            repeated = run_arm("native_repeat", original["cudnn_allow_tf32"])
            for variant in models:
                assert repeated[variant]["passed"] == native[variant]["passed"]
                for side in ("before", "after"):
                    a, b = [load(arm, variant, f"{side}.pt") for arm in ("native", "native_repeat")]
                    for key in ("raw", "decoded"):
                        common.assert_close_tree(a[key], b[key], 0, 0, f"{variant}.{side}.repeat.{key}")
            report["classification"] = "convolution_precision_conditioned; native_strict_close_remains_false"
            report["shared_first_native_error"] = first
        report["restored"] = common.computation_conditions()
        assert report["restored"] == original, "Fuse precision leaked into subsequent native execution"
        report["passed"] = True
    except BaseException:
        report["traceback"] = traceback.format_exc()
        raise
    finally:
        report["restored"] = common.computation_conditions()
        common.write_json(directory / "precision_checks.json", report)
    return report
