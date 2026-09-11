"""Candidate-identity fuse audit for the fixed single-class b19 Detect head; no inference overrides."""

import copy
import sys
import traceback
from unittest.mock import patch

import torch

import ultralytics
from tools.experiments import b19_common as common
from ultralytics.nn.modules import Conv, Concat_LBI_Fusion


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
    report["stage"] = "raw_and_decode"
    for key in ("boxes", "scores"):
        common.assert_close_tree(before["raw"][key], after["raw"][key], atol, rtol, f"raw.one2one.{key}", rows)
    for key, s in (("boxes", slice(0, 4)), ("scores", slice(4, None))):
        common.assert_close_tree(
            before["decoded"][..., s], after["decoded"][..., s], atol, rtol, f"decoded.{key}", rows
        )
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
            common.assert_close_tree(a[batch, union, s], b[batch, union, s], atol, rtol, f"union.{batch}.{key}", rows)
    report["stage"] = "complete"


def fuse_audit(model, x, directory, atol, rtol):
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
        before = capture(before_model, x.clone())
        torch.save(before, directory / "before.pt")
        after = capture(after_model, x.clone())
        torch.save(after, directory / "after.pt")
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
            layers = [{}, {}]
            try:
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
