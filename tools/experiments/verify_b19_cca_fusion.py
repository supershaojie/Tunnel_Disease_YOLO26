"""Run local synthetic lifecycle verification; never label it a real-data batch32 preflight."""

import argparse
import copy
import time
from pathlib import Path

import torch

from tools.experiments import run_b19_cca_fusion as shared
from ultralytics import YOLO
from ultralytics.nn.tasks import load_checkpoint
from ultralytics.utils.torch_utils import ModelEMA, autocast


def verify(weights, output, experiment=shared.EXPERIMENT, trainer_type=shared.AuditedTrainer):
    """Check native reconstruction, task gradients, nonzero EMA, reload/fuse and actual Validator calls."""
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    report = dict(scope="local synthetic checks; NOT server real batch32/640 preflight", torch=torch.__version__)
    report["structure"] = shared.structural_checks(output, experiment.model, experiment.block_type)
    assert shared.sha256(weights) == shared.PRETRAINED_SHA256
    source, _ = load_checkpoint(weights)
    trainer = object.__new__(trainer_type)
    trainer.args = shared.get_cfg(overrides=shared.REFERENCE["args"] | {"save_dir": None})
    trainer.data = dict(nc=1, channels=3, names={0: "crack"})
    torch.manual_seed(42)
    model = trainer.get_model(str(experiment.model), source, verbose=False)
    model.args = trainer.args
    report["pretrained"] = trainer.weight_audit
    report["pretrained"]["loaded_count"] = len(trainer.weight_audit["loaded_keys"])
    states = {k: v.clone() for k, v in model.state_dict().items()}
    with torch.no_grad():
        model.eval()
        for shape in ((64, 96), (96, 64), (64, 96)):
            model(torch.randn(1, 3, *shape))
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in states.items())
    report["eval_state_unchanged"] = True
    report["training"] = []
    for device, amp in [("cpu", False)] + ([("cuda:0", False), ("cuda:0", True)] if torch.cuda.is_available() else []):
        candidate = copy.deepcopy(model).to(device).train().requires_grad_(True)
        optimizer = trainer.build_optimizer(candidate, "MuSGD", 0.01, 0.937, 0.0005, 200)
        params = dict(candidate.model[12].named_parameters())
        assert all(sum(p is q for g in optimizer.param_groups for q in g["params"]) == 1 for p in params.values())
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
        ema = ModelEMA(candidate)
        updates, rows = {k: False for k in params}, []
        for step in range(6):
            batch = dict(
                img=torch.rand(2, 3, 64, 96, device=device),
                batch_idx=torch.tensor([0.0, 1.0], device=device),
                cls=torch.zeros(2, 1, device=device),
                bboxes=torch.tensor([[0.5, 0.5, 0.2, 0.3]] * 2, device=device),
            )
            before = {k: p.detach().clone() for k, p in params.items()}
            with autocast(amp, device=torch.device(device).type):
                loss, _ = candidate(batch)
            assert torch.isfinite(loss).all()
            scaler.scale(loss.sum()).backward()
            scaler.unscale_(optimizer)
            gradients = {k: shared.tensor_stats(p.grad) for k, p in params.items()}
            assert all(v["finite"] for v in gradients.values())
            torch.nn.utils.clip_grad_norm_(candidate.parameters(), 10)
            calls = []
            handle = optimizer.register_step_post_hook(lambda *a: calls.append(True))
            scaler.step(optimizer)
            scaler.update()
            handle.remove()
            for k, p in params.items():
                updates[k] |= bool(calls) and bool(gradients[k]["nonzero"]) and not torch.equal(before[k], p)
            rows.append(dict(step=step, gradients=gradients, optimizer_step=bool(calls), scale=scaler.get_scale()))
            optimizer.zero_grad()
            ema.update(candidate)
            if all(updates.values()):
                break
        assert all(updates.values()), updates
        report["training"].append(
            dict(device=device, amp=amp, batch=2, imgsz=[64, 96], task_updates=updates, rows=rows)
        )
        if device == "cpu":
            ema.ema.args = model.args
            report["reload"] = shared.save_reload_check(ema.ema, output, experiment.block_type)
        del candidate, optimizer, ema
    # Tiny generated labeled dataset drives the real standalone Validator after warmup.
    from PIL import Image
    import numpy as np

    for part in ("images", "labels"):
        (output / "data" / part).mkdir(parents=True)
    for i in range(2):
        Image.fromarray(np.random.default_rng(i).integers(0, 256, (64, 96, 3), dtype=np.uint8)).save(
            output / f"data/images/{i}.png"
        )
        (output / f"data/labels/{i}.txt").write_text("0 0.5 0.5 0.2 0.3\n")
    shared.YAML.save(
        output / "data/data.yaml",
        dict(path=str((output / "data").resolve()), train="images", val="images", names={0: "crack"}),
    )
    probe = YOLO(output / "preflight.pt")
    calls, residuals = [], []

    def capture(module, args, value):
        calls.append(True)
        residuals.append((value[:, :256] - args[0][0]).norm().item())

    probe.model.model[12].register_forward_hook(capture)
    probe.add_callback("on_val_start", lambda v: (calls.clear(), residuals.clear()))
    probe.val(
        data=str(output / "data/data.yaml"),
        imgsz=96,
        batch=2,
        workers=0,
        device="cpu",
        quantize=None,
        plots=False,
        project=str(output),
        name="validator",
    )
    assert calls and max(residuals) > 0
    report["validator"] = dict(
        real_validator=True, synthetic_images=2, calls=len(calls), residual_max_norm=max(residuals)
    )
    if torch.cuda.is_available():
        report["gpu"] = torch.cuda.get_device_name(0)
        # Isolated operator cost at the specified batch/feature shape, distinct from full model training.
        module = copy.deepcopy(model.model[12]).cuda()
        inputs = [torch.randn(32, c, h, h, device="cuda") for c, h in ((256, 40), (128, 40), (256, 20))]
        torch.cuda.reset_peak_memory_stats()
        for _ in range(3):
            module(inputs).sum().backward()
            module.zero_grad()
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(10):
            module(inputs).sum().backward()
            module.zero_grad()
        torch.cuda.synchronize()
        report["operator_benchmark"] = dict(
            batch=32,
            feature_shapes=[list(v.shape) for v in inputs],
            fp32_forward_backward_ms=(time.perf_counter() - start) * 100,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            scope="CCA operator only, synthetic tensors; includes other live tensors; not end-to-end training memory",
        )
    report["experiment"] = experiment.name
    report["passed"] = True
    shared.write_json(output / "local_validation.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    verify(options.weights.resolve(), options.output.resolve())
