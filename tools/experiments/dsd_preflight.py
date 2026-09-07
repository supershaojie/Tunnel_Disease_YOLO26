"""DSD-specific gradient observations on the fixed b19 AMP/MuSGD preflight loop (6f5f1e2)."""

import time
import warnings

import numpy as np
import torch

from tools.experiments import b19_common as shared


@torch.random.fork_rng(devices=[])
@torch.enable_grad()
def routing_checks():
    """Prove native branch ownership on fixed features, including independent adapter gradients."""
    from ultralytics.nn.modules import DSDDetect, Detect

    torch.manual_seed(42)
    native = Detect(1, 1, True, (64, 128, 256)).eval()
    torch.manual_seed(42)
    head = DSDDetect(1, 1, True, (64, 128, 256)).eval()
    features = [torch.randn(2, c, h, w, requires_grad=True) for c, h, w in ((64, 8, 12), (128, 4, 6), (256, 2, 3))]
    with torch.no_grad():
        original = [t.clone() for t in features]
        for branch in ("one2many", "one2one"):
            shared.assert_close_tree(
                native.forward_head(features, **getattr(native, branch)),
                head.forward_head(features, **getattr(head, branch)),
                0,
                0,
            )
        for adapter in (head.reg_adapter, head.one2one_reg_adapter):
            adapter.coeff[-1].weight.normal_(0, 0.03)
            adapter.coeff[-1].bias.copy_(torch.linspace(-0.7, 0.7, 32))
        for branch in ("one2many", "one2one"):
            a = native.forward_head(features, **getattr(native, branch))
            b = head.forward_head(features, **getattr(head, branch))
            assert b["feats"] is features
            assert torch.equal(a["scores"], b["scores"])
            assert torch.equal(a["boxes"][..., 96:], b["boxes"][..., 96:])
            assert not torch.equal(a["boxes"][..., :96], b["boxes"][..., :96])
        assert all(torch.equal(a, b) for a, b in zip(features, original))
    head.training = True  # Retain fixed BN statistics to isolate routing in the native forward.
    for branch, adapter_name, other in (
        ("one2one", "one2one_reg_adapter", "reg_adapter"),
        ("one2many", "reg_adapter", "one2one_reg_adapter"),
    ):
        head.zero_grad(set_to_none=True)
        for x in features:
            x.grad = None
        preds = head(features)[branch]
        (preds["boxes"].square().mean() + preds["scores"].square().mean()).backward()
        assert all(
            p.grad is not None and torch.isfinite(p.grad).all() and p.grad.count_nonzero()
            for p in getattr(head, adapter_name).parameters()
        )
        assert all(p.grad is None for p in getattr(head, other).parameters())
        if branch == "one2one":
            assert all(x.grad is None for x in features)
        else:
            assert all(x.grad is not None and x.grad.count_nonzero() for x in features)
    return dict(
        p3_regression_only=True,
        unchanged_features=True,
        independent_gradients=True,
        one2one_backbone_detached=True,
        one2many_backbone_gradient=True,
    )


def validator_check(model, trainer, directory):
    """Observe the real FP32 Validator's fused forward on one actual validation batch."""
    import copy

    from ultralytics.models.yolo.detect import DetectionValidator

    # The native validation loader may use twice the training batch; explicitly construct the requested batch=32.
    loader = trainer.get_dataloader(trainer.data["val"], batch_size=32, rank=-1, mode="val")
    batch = next(iter(loader))

    class OneBatch:
        dataset = range(len(batch["img"]))

        def __len__(self):
            return 1

        def __iter__(self):
            yield copy.deepcopy(batch)

    reference = copy.deepcopy(model).float().eval().to(trainer.device)
    candidate = copy.deepcopy(reference)
    observed = []

    def compare(module, inputs, result):
        with torch.no_grad(), shared.autocast(False, device=trainer.device.type):
            original = reference(inputs[0])
        assert result[1]["one2many"] == {} and module.model[-1].reg_adapter is None
        assert sum(p.numel() for p in module.model[-1].one2one_reg_adapter.parameters()) == 1744
        shared.assert_close_tree(original[1]["one2one"], result[1]["one2one"], 1e-4, 1e-4)
        # Compare dense decoded anchors; native top-k ordering may swap nearly tied predictions after BN fusion.
        before = reference.model[-1]._inference(original[1]["one2one"])
        after = module.model[-1]._inference(result[1]["one2one"])
        # reg_max=1 decoding is (anchor +/- raw_distance)*stride. Propagate the verified raw tolerance
        # through this affine map; a fixed pixel tolerance would misclassify cancellation near zero.
        raw = original[1]["one2one"]["boxes"]
        stride = module.model[-1].strides
        bound = (1e-4 + 1e-4 * raw.abs()) * stride
        bound = bound + 4 * torch.finfo(before.dtype).eps * before[:, :4].abs()
        assert torch.isfinite(after).all() and ((before[:, :4] - after[:, :4]).abs() <= bound).all()
        shared.assert_close_tree(before[:, 4:], after[:, 4:], 1e-4, 1e-4)
        shared.assert_close_tree(module.model[-1].postprocess(after.permute(0, 2, 1)), result[0], 0, 0)
        assert inputs[0].dtype == torch.float32
        observed.append(list(inputs[0].shape))

    handles = []
    try:
        validator = DetectionValidator(
            dataloader=OneBatch(),
            save_dir=directory / "validator",
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
        # Validator owns the transition from backend warmup to actual image evaluation.
        validator.add_callback("on_val_start", lambda v: handles.append(candidate.register_forward_hook(compare)))
        validator(model=candidate)
        assert validator.seen == 32 and validator.args.quantize is None
        result = dict(
            passed=True,
            images=validator.seen,
            precision="FP32",
            input_shapes=observed,
            fused_one2one_verified=True,
            dense_decoded_close=True,
            split="val",
        )
        shared.write_json(directory / "validator_check.json", result)
        return result
    finally:
        for handle in handles:
            handle.remove()
        loader.close()


def tensor_stats(tensor):
    """Keep small, JSON-finite summaries, including the distinction between absent and zero gradients."""
    if tensor is None:
        return dict(is_none=True, finite=None, norm=None, nonzero=0, dtype=None)
    tensor = tensor.detach()
    finite = bool(torch.isfinite(tensor).all())
    return dict(
        is_none=False,
        finite=finite,
        norm=tensor.double().norm().item() if finite else None,
        nonzero=tensor.count_nonzero().item(),
        dtype=str(tensor.dtype),
        max_abs=tensor.abs().max().item() if finite else None,
        elements=tensor.numel(),
    )


def preflight_batches(trainer, report, directory):
    """Reuse fixed RPCA preflight ordering, requiring later gradients after each DSD final weight learns."""
    state = report["adapter_preflight"] = dict(
        status="running",
        max_batches=64,
        attempted_steps=0,
        completed_steps=0,
        adapter_updates=0,
        first_task_weight_update={},
        first_nonzero_gradient={},
        first_effective_gradient={},
    )
    handles = []
    try:
        params = {k: p for k, p in trainer.model.named_parameters() if "reg_adapter." in k}
        required = {
            k: p for k, p in trainer.model.named_parameters() if any(marker in k for marker in trainer.gradient_markers)
        }
        groups = {id(p): i for i, g in enumerate(trainer.optimizer.param_groups) for p in g["params"]}
        state["optimizer_membership"] = {k: groups.get(id(p)) for k, p in params.items()}
        if len(params) != 12 or any(id(p) not in groups or not p.requires_grad for p in params.values()):
            raise AssertionError("DSD adapter parameters missing from optimizer or not trainable")
        if trainer.optimizer.state or any(p.count_nonzero() for k, p in params.items() if ".4." in k):
            raise AssertionError("DSD preflight must start with a fresh optimizer and zero last weight")

        def completed_step(optimizer, args, kwargs):
            state["completed_steps"] += 1  # The hook is not called for GradScaler's skipped steps.

        handles = [trainer.optimizer.register_step_post_hook(completed_step)]
        nb = len(trainer.train_loader)
        nw = max(round(trainer.args.warmup_epochs * nb), 100) if trainer.args.warmup_epochs > 0 else -1
        state["warmup_batches"] = nw
        trainer.epoch = 0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            trainer.scheduler.step()  # Same first-epoch scheduling order as BaseTrainer._do_train.
        trainer._model_train()
        trainer.optimizer.zero_grad()
        loader = iter(trainer.train_loader)
        last_opt_step = -1
        for ni in range(min(64, nb)):
            if trainer.device.type == "cuda":
                torch.cuda.synchronize(trainer.device)
            started = time.perf_counter()
            row = dict(
                batch=ni + 1,
                iteration=ni,
                scale_before=trainer.scaler.get_scale(),
                skipped=None,
                attempt=state["attempted_steps"],
                optimizer_update=state["completed_steps"],
            )
            report["real_batches"].append(row)
            try:
                current = dict(trainer.model.named_parameters())
                if any(current.get(k) is not p for k, p in params.items()):
                    raise AssertionError("DSD adapter was replaced during preflight")
                if ni <= nw:
                    trainer.accumulate = max(
                        1, int(np.interp(ni, [0, nw], [1, trainer.args.nbs / trainer.batch_size]).round())
                    )
                    for group in trainer.optimizer.param_groups:
                        start = trainer.args.warmup_bias_lr if group.get("param_group") == "bias" else 0.0
                        group["lr"] = float(np.interp(ni, [0, nw], [start, group["initial_lr"] * trainer.lf(0)]))
                        if "momentum" in group:
                            group["momentum"] = float(
                                np.interp(ni, [0, nw], [trainer.args.warmup_momentum, trainer.args.momentum])
                            )
                row["accumulate"] = trainer.accumulate
                row["groups"] = [
                    dict(
                        index=i,
                        kind=g.get("param_group"),
                        lr=g["lr"],
                        momentum=g.get("momentum"),
                        weight_decay=g["weight_decay"],
                        adapter_parameters=[k for k, p in params.items() if groups[id(p)] == i],
                    )
                    for i, g in enumerate(trainer.optimizer.param_groups)
                ]
                forward_parameters = {k: p.detach().clone() for k, p in params.items()}
                with shared.autocast(enabled=trainer.amp, device=trainer.device.type):
                    batch = trainer.preprocess_batch(next(loader))
                    loss, items = trainer.model(batch)
                    total = loss.sum()
                row.update(
                    loss=tensor_stats(total),
                    components=items.detach().cpu().tolist(),
                    image_shape=list(batch["img"].shape),
                    targets=batch["cls"].numel(),
                )
                if not torch.isfinite(total) or not row["targets"]:
                    raise AssertionError("Nonfinite task loss or unlabeled DSD preflight batch")
                trainer.scaler.scale(total).backward()
                attempted = ni - last_opt_step >= trainer.accumulate
                row["attempted_step"] = attempted
                if attempted:
                    trainer.scaler.unscale_(trainer.optimizer)  # Exactly once, before recording and clipping.
                row["gradient_units"] = "unscaled" if attempted else "scaled_accumulation"
                gradients = row["gradients"] = {k: tensor_stats(p.grad) for k, p in required.items()}
                if attempted:
                    for key in params:
                        if gradients[key]["finite"] and gradients[key]["nonzero"]:
                            state["first_nonzero_gradient"].setdefault(key, ni + 1)
                nonfinite = [
                    k
                    for k, p in trainer.model.named_parameters()
                    if p.grad is not None and not torch.isfinite(p.grad).all()
                ]
                row["nonfinite_gradient_parameters"] = nonfinite
                if any(g["is_none"] for g in gradients.values()):
                    raise AssertionError("Disconnected required DSD task gradient")
                if any(not torch.equal(p, forward_parameters[k]) for k, p in params.items()):
                    raise AssertionError("DSD adapter parameters changed outside the optimizer step")
                before = {k: p.detach().clone() for k, p in params.items()}
                if attempted:
                    state["attempted_steps"] += 1
                    completed_before = state["completed_steps"]
                    # Match BaseTrainer.optimizer_step; task statistics above exclude clipping/decay/momentum.
                    row["global_norm_before_clip"] = tensor_stats(
                        torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), 10.0)
                    )
                    trainer.scaler.step(trainer.optimizer)
                    trainer.scaler.update()
                    row["skipped"] = state["completed_steps"] == completed_before
                    trainer.optimizer.zero_grad()
                    if trainer.ema:
                        trainer.ema.update(trainer.model)
                    last_opt_step = ni  # Native accumulation counts attempted updates, including scaler skips.
                row.update(
                    attempt=state["attempted_steps"],
                    optimizer_update=state["completed_steps"],
                    scale_after=trainer.scaler.get_scale(),
                )
                row["parameter_deltas"] = {k: tensor_stats(p - before[k]) for k, p in params.items()}
                if not all(torch.isfinite(p).all() for p in trainer.model.parameters()):
                    raise AssertionError("Nonfinite parameters after DSD optimizer update")
                if attempted and nonfinite and not (row["skipped"] and row["scale_after"] < row["scale_before"]):
                    raise AssertionError("Nonfinite task gradients without a native GradScaler overflow skip")
                if attempted and not row["skipped"]:
                    if any(s["nonzero"] for s in row["parameter_deltas"].values()):
                        state["adapter_updates"] += 1
                    for key in params:
                        if (
                            key.endswith(".4.weight")
                            and row["parameter_deltas"][key]["nonzero"]
                            and gradients[key]["nonzero"]
                        ):
                            state["first_task_weight_update"].setdefault(key, ni + 1)
                        branch = key.split(".coeff.")[0]
                        first_update = state["first_task_weight_update"].get(branch + ".coeff.4.weight")
                        if (
                            first_update is not None
                            and ni + 1 > first_update
                            and gradients[key]["finite"]
                            and gradients[key]["nonzero"]
                        ):
                            state["first_effective_gradient"].setdefault(key, ni + 1)
                    if len(state["first_effective_gradient"]) == len(params):
                        if not all(
                            any(g["nonzero"] and g["finite"] for k, g in gradients.items() if marker in k)
                            for marker in trainer.gradient_markers
                        ):
                            raise AssertionError("Required native backbone/regression task gradients remain zero")
                        state["status"] = "passed"
                        return state["attempted_steps"]
            finally:
                if trainer.device.type == "cuda":
                    torch.cuda.synchronize(trainer.device)
                row["diagnostic_batch_seconds"] = (
                    time.perf_counter() - started
                )  # Includes statistics, excludes JSON I/O.
                shared.write_json(directory / "checks.json", report)
        raise AssertionError(
            "DSD adapter did not establish finite task gradients and effective updates within 64 real batches"
        )
    except Exception as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        for handle in handles:
            handle.remove()
        shared.write_json(directory / "checks.json", report)
