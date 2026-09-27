"""Finite synthetic engineering checks; never run a dataset training/validation epoch."""

# ruff: noqa: E402 - select the experiment source before importing Ultralytics.

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ["YOLO_AUTOINSTALL"] = "false"

import torch

from experiments.dga_c2psa.train import DGATrainer, MODEL, PREDICTOR, RECIPE, environment_report
from ultralytics.cfg import get_cfg
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.modules import DGAAttention
from ultralytics.nn.tasks import load_checkpoint
from ultralytics.utils import YAML
from ultralytics.utils.torch_utils import ModelEMA, autocast, init_seeds


def tensors(value):
    """Flatten native nested raw predictions for numerical comparison."""
    if isinstance(value, torch.Tensor):
        return [value]
    children = value.values() if isinstance(value, dict) else value
    return [tensor for child in children for tensor in tensors(child)]


def compare(left, right, atol=1e-6, rtol=1e-5):
    """Compare every raw tensor and return the largest absolute error."""
    left, right = tensors(left), tensors(right)
    assert len(left) == len(right)
    for a, b in zip(left, right):
        torch.testing.assert_close(a, b, atol=atol, rtol=rtol)
    return max(float((a - b).abs().max()) for a, b in zip(left, right))


def formula_checks():
    """Use scalar loops and a quadratic-form reference on rectangular and degenerate grids."""
    module = DGAAttention(16, num_heads=2).eval()
    with torch.no_grad():
        module.geometry_predictor[-1].weight.normal_(0, 0.05)
        module.geometry_predictor[-1].bias.copy_(torch.tensor([0.1, -0.2, 0.3, 0.15]))
    errors = {}
    for height, width in ((2, 3), (1, 4), (4, 1), (1, 1)):
        x = torch.randn(2, 16, height, width)
        n = height * width
        with torch.no_grad():
            prediction = module.geometry_predictor(x)
            reference = torch.zeros(2, 2, n, n, dtype=torch.float64)
            for batch in range(2):
                for head in range(2):
                    for i in range(n):
                        yi, xi = divmod(i, width)
                        u, v = [float(prediction[batch, 2 * head + channel, yi, xi]) for channel in (0, 1)]
                        radius = math.sqrt(u * u + v * v + 1e-6)
                        a, b = [0.8 * math.tanh(radius) / radius * z for z in (u, v)]
                        matrix = torch.tensor([[a, b], [b, -a]], dtype=torch.float64)
                        for j in range(n):
                            yj, xj = divmod(j, width)
                            delta = torch.tensor([xj - xi, yj - yi], dtype=torch.float64)
                            delta /= max(height - 1, width - 1, 1)
                            reference[batch, head, i, j] = -(delta @ matrix @ delta)
            bias = module.geometry_bias(x)
            bias_error = compare(bias, reference.float(), atol=5e-8)
            q, k, value = module.qkv(x).reshape(2, 2, 16, n).split([4, 4, 8], dim=2)
            logits = torch.einsum("bhdi,bhdj->bhij", q, k) * module.scale
            probabilities = (logits + reference.float()).softmax(-1)
            aggregate = torch.einsum("bhij,bhdj->bhdi", probabilities, value).reshape(2, 16, height, width)
            expected = module.proj(aggregate + module.pe(value.reshape_as(x)))
            output_error = compare(module(x), expected)
        if n == 1:
            assert torch.count_nonzero(bias) == 0
        else:
            assert torch.count_nonzero(bias) > 0
            assert not torch.allclose(bias, bias.transpose(-2, -1))  # query-varying coefficients
        errors[f"{height}x{width}"] = {"bias_max_abs": bias_error, "output_max_abs": output_error}
    return errors


def optimizer_checks(model, trainer, device):
    """Take three actual MuSGD steps, measuring task gradients before clipping/weight decay."""
    model = deepcopy(model).to(device).train()
    trainer.model = model
    trainer.set_model_attributes()
    optimizer = trainer.build_optimizer(model, "MuSGD", lr=0.01, momentum=0.937, decay=0.0005)
    assert type(optimizer).__name__ == "MuSGD"
    predictor = model.model[10].m[0].attn.geometry_predictor
    membership = {}
    for name, param in predictor.named_parameters():
        groups = [group for group in optimizer.param_groups if any(p is param for p in group["params"])]
        assert len(groups) == 1 and groups[0]["lr"] == 0.01
        assert groups[0]["param_group"] == ("bias" if name.endswith("bias") else "muon")
        membership[name] = groups[0]["param_group"]
    amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp, init_scale=128.0)
    stepped = []
    handle = optimizer.register_step_post_hook(lambda *args: stepped.append(True))
    rows = []
    batch = {
        "img": torch.rand(2, 3, 96, 128, device=device),
        "batch_idx": torch.tensor([0, 1], device=device),
        "cls": torch.zeros(2, 1, device=device),
        "bboxes": torch.tensor([[0.45, 0.5, 0.3, 0.4], [0.6, 0.4, 0.25, 0.3]], device=device),
    }
    for step in range(3):
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=amp, device=device.type):
            loss, _ = model(batch)
            loss = loss.sum()
        assert torch.isfinite(loss)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gradients = {name: float(p.grad.abs().max()) for name, p in predictor.named_parameters()}
        assert all(math.isfinite(v) for v in gradients.values())
        assert gradients["4.weight"] > 0 and gradients["4.bias"] > 0
        if step == 0:
            assert gradients["0.weight"] == gradients["2.weight"] == 0
        else:
            assert gradients["0.weight"] > 0 and gradients["2.weight"] > 0
        before = predictor[-1].weight.detach().clone()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        scaler.step(optimizer)
        scaler.update()
        assert len(stepped) == step + 1, "AMP skipped an optimizer update"
        assert not torch.equal(before, predictor[-1].weight)
        rows.append({"loss": float(loss.detach()), "task_gradient_max_abs": gradients, "scale": scaler.get_scale()})
    handle.remove()
    assert model.criterion.o2m == 0.8 and abs(model.criterion.o2o - 0.2) < 1e-12
    return {
        "device": str(device),
        "amp": amp,
        "backwards": 3,
        "accumulation": 1,
        "optimizer_steps": len(stepped),
        "amp_skips": 0,
        "groups": membership,
        "updates": rows,
    }


def lifecycle_checks(model, native):
    """Exercise nonzero DGA through deepcopy, EMA, checkpoint loading in two processes, and native fusion."""
    model = deepcopy(model).eval()
    attn = model.model[10].m[0].attn
    with torch.no_grad():
        attn.geometry_predictor[-1].weight.normal_(0, 0.02)
        attn.geometry_predictor[-1].bias.copy_(torch.tensor([0.08, -0.04, -0.06, 0.1]))
    x = torch.rand(1, 3, 96, 128)
    seen = []
    hook = attn.register_forward_pre_hook(lambda module, args: seen.append(args[0].detach().clone()))
    with torch.no_grad():
        expected = model(x)
    hook.remove()
    bias = attn.geometry_bias(seen[0])
    assert bias.abs().max() > 0
    zero = deepcopy(model)
    with torch.no_grad():
        zero.model[10].m[0].attn.geometry_predictor[-1].weight.zero_()
        zero.model[10].m[0].attn.geometry_predictor[-1].bias.zero_()
        change = max(float((a - b).abs().max()) for a, b in zip(tensors(expected[1]), tensors(zero(x)[1])))
        assert change > 1e-6
        copy_error = compare(deepcopy(model)(x), expected)
        ema = ModelEMA(model)
        ema.update(model)
        ema_error = compare(ema.ema(x)[1], expected[1], atol=1e-4)
    # Relative checkpoint paths also exercise native loading on Windows user paths containing apostrophes.
    with tempfile.TemporaryDirectory(prefix="dga-lifecycle-", dir=".") as directory:
        directory = Path(directory)
        checkpoint = directory / "nonzero.pt"
        # Native training saves the EMA as FP16; compare against that exact quantized state.
        saved_ema = deepcopy(ema.ema).half()
        torch.save({"model": None, "ema": saved_ema, "train_args": vars(model.args)}, checkpoint)
        with torch.no_grad():
            checkpoint_expected = saved_ema.float()(x)
        loaded, _ = load_checkpoint(checkpoint)
        with torch.no_grad():
            reload_error = compare(loaded(x), checkpoint_expected)
        for key, value in saved_ema.state_dict().items():
            if key.startswith(PREDICTOR):
                assert torch.equal(value, loaded.state_dict()[key])
        torch.save({"x": x, "expected": checkpoint_expected}, directory / "reference.pt")
        code = (
            "import sys,torch; from ultralytics.nn.tasks import load_checkpoint; "
            "from ultralytics.nn.modules import DGAAttention; "
            "from experiments.dga_c2psa.validate import compare; "
            "torch.set_num_threads(2); m,_=load_checkpoint(sys.argv[1]); "
            "assert isinstance(m.model[10].m[0].attn,DGAAttention); "
            "r=torch.load(sys.argv[2],weights_only=False); "
            "torch.set_grad_enabled(False); compare(m(r['x']),r['expected']); print('fresh_process_ok')"
        )
        child = subprocess.run(
            [sys.executable, "-c", code, str(checkpoint), str(directory / "reference.pt")],
            cwd=Path.cwd(),
            env={**os.environ, "PYTHONPATH": str(ROOT)},
            check=True,
            capture_output=True,
            text=True,
        )
        assert "fresh_process_ok" in child.stdout
    fused = deepcopy(model).fuse(verbose=False)
    fused_attn = fused.model[10].m[0].attn
    assert isinstance(fused_attn, DGAAttention)
    assert not any(isinstance(m, torch.nn.BatchNorm2d) for m in fused.modules())
    for before, after in zip(attn.geometry_predictor.parameters(), fused_attn.geometry_predictor.parameters()):
        assert torch.equal(before, after)
    calls = []
    hook = fused_attn.geometry_predictor.register_forward_hook(lambda *args: calls.append(True))
    with torch.no_grad():
        fused_output = fused(x)
    hook.remove()
    assert len(calls) == 1
    # Native end-to-end fusion removes O2M; compare raw O2O tensors before postprocess/top-k ordering.
    fuse_error = compare(fused_output[1]["one2one"], expected[1]["one2one"], atol=2e-4, rtol=2e-4)
    native_fused_count = sum(p.numel() for p in deepcopy(native).fuse(verbose=False).parameters())
    return {
        "nonzero_bias_max_abs": float(bias.detach().abs().max()),
        "nonzero_raw_output_change": change,
        "deepcopy_max_abs": copy_error,
        "ema_max_abs": ema_error,
        "reload_max_abs": reload_error,
        "fresh_process": "PASS",
        "fused_o2o_max_abs": fuse_error,
        "fused_predictor_calls": len(calls),
        "dga_parameters_fused": sum(p.numel() for p in fused.parameters()),
        "native_parameters_fused": native_fused_count,
    }


def main():
    """Run the bounded engineering suite and save its measurements outside tracked source."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output", default="runs/dga_validation/report.json")
    options = parser.parse_args()
    torch.set_num_threads(2)
    init_seeds(42, deterministic=True)
    report = {"environment": environment_report(), "formula": formula_checks()}
    weights, _ = load_checkpoint(options.weights)
    trainer = object.__new__(DGATrainer)
    trainer.data = {"nc": 1, "channels": 3, "names": {0: "crack"}}
    trainer.args = get_cfg(overrides=YAML.load(RECIPE))
    init_seeds(42, deterministic=True)
    with torch.random.fork_rng(devices=[]):
        native = DetectionTrainer.get_model(trainer, ROOT / "ultralytics/cfg/models/26/yolo26n.yaml", weights, False)
    trainer.model = str(MODEL)
    trainer.args.pretrained = options.weights
    trainer.setup_model()  # actual native Trainer YAML/pretrained dispatch
    model = trainer.model
    model.args = trainer.args
    report["construction"] = trainer.dga_audit
    native.eval()
    model.eval()
    shapes = []
    hook = model.model[10].m[0].attn.register_forward_pre_hook(lambda module, args: shapes.append(list(args[0].shape)))
    with torch.no_grad():
        image = torch.rand(1, 3, 640, 640)
        report["zero_init_640_raw_max_abs"] = compare(native(image), model(image), atol=1e-6)
        rectangle = torch.rand(1, 3, 96, 160)
        report["zero_init_rectangle_raw_max_abs"] = compare(native(rectangle), model(rectangle), atol=1e-6)
    hook.remove()
    assert shapes == [[1, 128, 20, 20], [1, 128, 3, 5]]
    report["attention_shapes"] = shapes
    report["lifecycle"] = lifecycle_checks(model, native)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        native_attn = deepcopy(native.model[10].m[0].attn).to(device)
        dga_attn = deepcopy(model.model[10].m[0].attn).to(device)
        probe = torch.rand(1, 128, 3, 5, device=device)
        with torch.no_grad(), autocast(enabled=True, device="cuda"):
            prediction = dga_attn.geometry_predictor(probe)
            bias = dga_attn.geometry_bias(probe)
            output = dga_attn(probe)
            error = compare(native_attn(probe), output, atol=1e-3, rtol=1e-3)
        assert prediction.dtype == output.dtype == torch.float16 and bias.dtype == torch.float32
        report["amp_dtypes"] = {
            "predictor": str(prediction.dtype),
            "bias": str(bias.dtype),
            "output": str(output.dtype),
            "zero_init_attention_max_abs": error,
        }
    report["training_updates"] = optimizer_checks(model, trainer, device)
    report["cuda_amp"] = "PASS" if device.type == "cuda" else "NOT_VERIFIED: CUDA unavailable"
    report["formal_training"] = "NOT_STARTED"
    output = Path(options.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
