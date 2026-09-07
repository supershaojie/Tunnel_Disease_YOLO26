"""DSD/native numerical controls on the same batch, state and real Validator path."""

import copy
import hashlib
from contextlib import contextmanager, nullcontext

import torch

from tools.experiments import b19_common as shared
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.nn.tasks import DetectionModel


def model_identity(model):
    """Identify all parameters, BN buffers, modes and gradient flags without retaining GPU tensors."""
    digest = hashlib.sha256()
    for key, value in model.state_dict().items():
        digest.update(key.encode())
        digest.update(str(value.dtype).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return dict(
        state_sha256=digest.hexdigest(),
        head=type(model.model[-1]).__name__,
        training_modules=[name for name, m in model.named_modules() if m.training],
        trainable_parameters=[name for name, p in model.named_parameters() if p.requires_grad],
        dtypes=sorted({str(p.dtype) for p in model.parameters()}),
        devices=sorted({str(p.device) for p in model.parameters()}),
    )


def observe(a, b, path, rows, atol=1e-4, rtol=1e-4):
    """Collect numerical failures before the mandatory gate, so one scale cannot hide the others."""
    start = len(rows)
    try:
        shared.assert_close_tree(a, b, atol, rtol, path, rows)
    except AssertionError:
        # Only defer a recorded numerical mismatch. Shape/type/device errors remain immediate failures.
        if len(rows) == start:
            raise


@contextmanager
def layer_trace(model, samples):
    """Keep selected images from full-batch forwards; include every graph layer and regression/classification stage."""
    layers = {f"model.{i}": m for i, m in enumerate(model.model[:-1])}
    for branch in ("one2one_cv2", "one2one_cv3"):
        for scale, sequence in enumerate(getattr(model.model[-1], branch)):
            for stage, m in enumerate(sequence):
                layers[f"model.23.{branch}.{scale}.{stage}"] = m
    values, handles = {}, []
    try:
        for name, m in layers.items():
            handles.append(
                m.register_forward_hook(
                    lambda m, args, out, key=name: values.update({key: out[samples].detach().cpu().clone()})
                )
            )
        yield values
    finally:
        for handle in handles:
            handle.remove()


def raw_comparison(before, after, rows, prefix):
    """Compare all full-batch scales using actual feature shapes, including scores and shared features."""
    offset = 0
    for i, (a, b) in enumerate(zip(before["feats"], after["feats"])):
        anchors = a.shape[-2] * a.shape[-1]
        for field in ("boxes", "scores"):
            observe(
                before[field][..., offset : offset + anchors],
                after[field][..., offset : offset + anchors],
                f"{prefix}.P{i + 3}.{field}",
                rows,
            )
        observe(a, b, f"{prefix}.P{i + 3}.feats", rows)
        offset += anchors
    assert len(before["feats"]) == len(after["feats"]) == 3
    assert offset == before["boxes"].shape[-1] == after["boxes"].shape[-1]


def decoded_comparison(reference, candidate, original, result, rows, prefix):
    """Propagate the unchanged raw tolerance through native reg_max=1 affine distance decoding."""
    before = reference.model[-1]._inference(original[1]["one2one"])
    after = candidate.model[-1]._inference(result[1]["one2one"])
    raw = original[1]["one2one"]["boxes"]
    bound = (1e-4 + 1e-4 * raw.abs()) * candidate.model[-1].strides
    bound = bound + 4 * torch.finfo(before.dtype).eps * before[:, :4].abs()
    excess = ((before[:, :4] - after[:, :4]).abs() - bound).clamp_min(0)
    observe(torch.zeros_like(excess), excess, f"{prefix}.decoded.box_error_beyond_raw_bound", rows, 0, 0)
    observe(before[:, 4:], after[:, 4:], f"{prefix}.decoded.scores", rows)
    # Each actual output must be EXACTLY its own native top-k result; never compare unstable cross-model tie order.
    observe(candidate.model[-1].postprocess(after.permute(0, 2, 1)), result[0], f"{prefix}.postprocess", rows, 0, 0)


def validator_case(source, batch, trainer, directory, record):
    """Compare one unfused snapshot with CPU fusion and real AutoBackend GPU fusion under caller-owned arithmetic."""

    class OneBatch:
        dataset = range(len(batch["img"]))

        def __len__(self):
            return 1

        def __iter__(self):
            yield copy.deepcopy(batch)

    reference = copy.deepcopy(source).float().eval().requires_grad_(False).to(trainer.device)
    candidate = copy.deepcopy(reference)
    record["before_backend"] = model_identity(reference)
    assert model_identity(candidate) == record["before_backend"]
    assert {p.data_ptr() for p in reference.parameters()}.isdisjoint(p.data_ptr() for p in candidate.parameters())
    record["same_state_and_independent_storage"] = True
    # The checkpoint route fuses on CPU before transfer. Retain this independent control to isolate GPU folding.
    with shared.reload_context():
        cpu_fused = copy.deepcopy(source).cpu().float().eval().fuse(verbose=False)
    cpu_fused = cpu_fused.to(trainer.device).requires_grad_(False)
    handles = []
    rows = record["comparisons"] = []

    def compare(module, inputs, result):
        x = inputs[0]
        assert x.dtype == torch.float32 and not module.training and not reference.training
        record["autocast_enabled"] = torch.is_autocast_enabled(trainer.device.type)
        assert not record["autocast_enabled"]
        record["input"] = shared.reload_tensor_info(x)
        record["conditions_at_forward"] = shared.computation_conditions()
        record["after_backend"] = model_identity(module)
        assert record["after_backend"]["dtypes"] == ["torch.float32"]
        assert record["after_backend"]["trainable_parameters"] == []
        assert record["after_backend"]["training_modules"] == []
        assert result[1]["one2many"] == {}
        original = reference(x)
        error = (original[1]["one2one"]["boxes"] - result[1]["one2one"]["boxes"]).abs()
        score_error = (original[1]["one2one"]["scores"] - result[1]["one2one"]["scores"]).abs()
        samples = sorted({0, error.flatten(1).amax(1).argmax().item(), score_error.flatten(1).amax(1).argmax().item()})
        record["traced_samples"] = samples
        record["trace_scope"] = "Selected images only; every forward still uses the complete unchanged batch."
        record["scale_shapes"] = [list(f.shape) for f in original[1]["one2one"]["feats"]]
        record["scale_box_elements"] = [
            f.shape[0] * 4 * f.shape[-2] * f.shape[-1] for f in original[1]["one2one"]["feats"]
        ]
        with layer_trace(reference, samples) as ref_layers:
            repeated = reference(x)
        for field in ("boxes", "scores"):
            observe(original[1]["one2one"][field], repeated[1]["one2one"][field], f"repeat.unfused.{field}", rows, 0, 0)
        for name, fused in (("gpu_fused", module), ("cpu_fused", cpu_fused)):
            with layer_trace(fused, samples) as layers:
                # Bypass this model-level hook, retaining the exact native forward and all layer hooks.
                output = fused.forward(x)
            if name == "gpu_fused":
                for field in ("boxes", "scores"):
                    observe(
                        result[1]["one2one"][field],
                        output[1]["one2one"][field],
                        f"repeat.validator.{field}",
                        rows,
                        0,
                        0,
                    )
                observe(result[0], output[0], "repeat.validator.postprocess", rows, 0, 0)
            raw_comparison(original[1]["one2one"], output[1]["one2one"], rows, name)
            decoded_comparison(reference, fused, original, output, rows, name)
            for key in ref_layers:
                observe(ref_layers[key], layers[key], f"{name}.layers.{key}", rows)
        assert module.state_dict().keys() == cpu_fused.state_dict().keys()
        for key, value in cpu_fused.state_dict().items():
            observe(value, module.state_dict()[key], f"gpu_vs_cpu_fusion.state.{key}", rows)
        record["reference_unchanged"] = model_identity(reference) == record["before_backend"]
        assert record["reference_unchanged"]
        record["first_divergent_layer"] = {
            name: next(
                (r["path"] for r in rows if r["path"].startswith(f"{name}.layers.") and r["outside_tolerance"]), None
            )
            for name in ("gpu_fused", "cpu_fused")
        }

    def start(validator):
        head = candidate.model[-1]
        if hasattr(head, "one2one_reg_adapter"):
            assert head.reg_adapter is None
            assert sum(p.numel() for p in head.one2one_reg_adapter.parameters()) == 1744
            record["adapter_calls"] = []

            def adapter_call(module, inputs, output):
                delta = output - inputs[0]
                record["adapter_calls"].append(dict(shape=list(output.shape), changed=delta.count_nonzero().item()))

            handles.append(head.one2one_reg_adapter.register_forward_hook(adapter_call))
        handles.append(candidate.register_forward_hook(compare))

    try:
        validator = DetectionValidator(
            dataloader=OneBatch(),
            save_dir=directory,
            args=dict(
                data=trainer.args.data,
                imgsz=640,
                batch=32,
                device=str(trainer.device),
                workers=0,
                conf=0.001,
                iou=0.7,
                max_det=300,
                quantize=None,
                plots=False,
                save_json=False,
                rect=True,
                split="val",
            ),
        )
        validator.add_callback("on_val_start", start)
        validator(model=candidate)
        assert validator.seen == 32 and validator.args.quantize is None
        if hasattr(candidate.model[-1], "one2one_reg_adapter"):
            assert len(record["adapter_calls"]) == 2  # Real Validator and its exact repeated forward.
        record["images"] = validator.seen
        record["equivalent_at_1e_4"] = all(r["finite"] and not r["outside_tolerance"] for r in rows)
    finally:
        for handle in handles:
            handle.remove()


def validator_check(model, trainer, directory):
    """Keep ambient evidence and require strict native/DSD equivalence without changing the training recipe."""
    source_identity = model_identity(model)
    ambient = shared.computation_conditions()
    rng_cpu = torch.get_rng_state()
    rng_cuda = torch.cuda.get_rng_state(trainer.device) if trainer.device.type == "cuda" else None
    report = dict(
        passed=False,
        split="val",
        precision="FP32",
        source=source_identity,
        source_origin="trainer.ema.ema AFTER effective preflight optimizer/EMA updates; never an initial native baseline",
        native_origin="Native b19 Detect graph with EXACT common parameters and BN buffers from the same DSD EMA; numerical control, not independent training",
        initial_equivalence="Separate fresh untrained models in structural_checks and AuditedTrainer.get_model",
        ambient_conditions=ambient,
        cases={},
        acceptance="Strict FP32 (autocast/TF32 off) keeps raw atol=rtol=1e-4; repeats/postprocess exact. Ambient differences remain diagnostic failures, never relabelled equivalent.",
    )
    loader = trainer.get_dataloader(trainer.data["val"], batch_size=32, rank=-1, mode="val")
    try:
        with torch.random.fork_rng(devices=[trainer.device] if trainer.device.type == "cuda" else []):
            batch = next(iter(loader))
            snapshot = copy.deepcopy(model).cpu().float().eval().requires_grad_(False)
            with torch.random.fork_rng(devices=[]):
                native = DetectionModel(shared.baseline_architecture(), verbose=False).eval().requires_grad_(False)
            common = {k: v for k, v in snapshot.state_dict().items() if "reg_adapter." not in k}
            native.load_state_dict(common, strict=True)
            shared.assert_close_tree(common, native.state_dict(), 0, 0)
            native.names = copy.deepcopy(snapshot.names)
            report["native_common_state_exact"] = True
            for mode in ("ambient", "strict"):
                context = shared.reload_context(device=trainer.device.type) if mode == "strict" else nullcontext()
                with context, torch.inference_mode(), shared.autocast(False, device=trainer.device.type):
                    for name, source in (("native", native), ("dsd", snapshot)):
                        key = f"{mode}_{name}"
                        record = report["cases"][key] = dict(conditions_at_fusion=shared.computation_conditions())
                        validator_case(source, batch, trainer, directory / "validator" / key, record)
                        shared.write_json(directory / "validator_check.json", report)
            inputs = [r["input"] for r in report["cases"].values()]
            assert all(i == inputs[0] for i in inputs)
            # Ambient mismatches may be numerical, but exact repeats MUST still hold in the real Validator path.
            required = [
                r
                for k, c in report["cases"].items()
                for r in c["comparisons"]
                if k.startswith("strict") or r["path"].startswith("repeat.")
            ]
            failures = [r for r in required if not r["finite"] or r["outside_tolerance"]]
            assert not failures, f"Validator numerical contract failed; see validator_check.json: {failures[:3]}"
            report.update(
                passed=True,
                images=32,
                input_shapes=[inputs[0]["shape"]],
                fused_one2one_verified=True,
                dense_decoded_close=True,
            )
            return report
    finally:
        loader.close()
        report["source_unchanged"] = model_identity(model) == source_identity
        report["restored_conditions"] = shared.computation_conditions()
        report["settings_restored"] = report["restored_conditions"] == ambient
        report["rng_restored"] = torch.equal(torch.get_rng_state(), rng_cpu) and (
            rng_cuda is None or torch.equal(torch.cuda.get_rng_state(trainer.device), rng_cuda)
        )
        unchanged = report["source_unchanged"] and report["settings_restored"] and report["rng_restored"]
        report["passed"] = report["passed"] and unchanged
        shared.write_json(directory / "validator_check.json", report)
        assert unchanged, "Validator changed source EMA or caller settings"
