"""Finite BDF validation on disposable models; never launches Trainer.train or writes into a formal run."""

import argparse
import json
import subprocess
import sys
import tempfile
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from PIL import Image

from experiments.bdf_fusion.train import MODEL, check_weights, environment, locked_config
from ultralytics.cfg import get_cfg
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.modules import BDF_Fusion
from ultralytics.nn.tasks import load_checkpoint, yaml_model_load
from ultralytics.utils.torch_utils import ModelEMA, init_seeds

ROOT = Path(__file__).resolve().parents[2]


def build(cfg, weights):
    """Exercise native production setup_model, including nc adaptation and original checkpoint loading."""
    trainer = DetectionTrainer.__new__(DetectionTrainer)
    trainer.args = get_cfg(overrides={**locked_config(), "pretrained": str(weights), "resume": False})
    trainer.data = {"nc": 1, "channels": 3, "names": {0: "crack"}}
    trainer.model, trainer.resume = str(cfg), False
    trainer.setup_model()
    trainer.set_model_attributes()
    return trainer.model


def max_error(a, b, atol=1e-5, rtol=1e-5):
    """Assert closeness and report maximum absolute error of a tensor pair."""
    torch.testing.assert_close(a, b, atol=atol, rtol=rtol)
    return (a - b).abs().max().item()


def raw_error(a, b, **kwargs):
    """Compare raw boxes and class logits before decoding/ranking, including O2M when both have it."""
    return {
        f"{branch}.{field}": max_error(a[1][branch][field], b[1][branch][field], **kwargs)
        for branch in ("one2many", "one2one")
        if a[1][branch] and b[1][branch]
        for field in ("boxes", "scores")
    }


def smoke_batch(dataset, device):
    """Read exactly two nonempty training samples and resize to 256 for the three finite updates."""
    images, classes, boxes, indices, selected = [], [], [], [], []
    for label in sorted((dataset / "labels/train").glob("*.txt")):
        rows = np.loadtxt(label, ndmin=2, dtype=np.float32)
        if not rows.size:
            continue
        image_path = next(p for p in (dataset / "images/train").glob(label.stem + ".*") if p.is_file())
        with Image.open(image_path) as im:
            image = np.array(im.convert("RGB").resize((256, 256)), copy=True)
        indices.extend([len(images)] * len(rows))
        images.append(torch.from_numpy(image).permute(2, 0, 1))
        classes.extend(rows[:, :1])
        boxes.extend(rows[:, 1:])
        selected.append(image_path.name)
        if len(images) == 2:
            break
    assert len(images) == 2, "Provide a local b19 training dataset with at least two nonempty labels"
    return {
        "img": torch.stack(images).to(device).float() / 255,
        "cls": torch.tensor(np.array(classes), device=device),
        "bboxes": torch.tensor(np.array(boxes), device=device),
        "batch_idx": torch.tensor(indices, device=device),
    }, selected


def main():
    """Run wiring, RNG, gradients, nonzero lifecycle, and CUDA AMP checks once."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    cli = parser.parse_args()
    torch.set_num_threads(4)
    weights = check_weights(cli.weights)
    report = {"training_status": "NOT_STARTED", "environment": environment()[0]}
    native_cfg = ROOT / "ultralytics/cfg/models/26/yolo26n.yaml"
    a_yaml, b_yaml = yaml_model_load(native_cfg), yaml_model_load(MODEL)
    assert a_yaml["backbone"] == b_yaml["backbone"] and b_yaml["scale"] == "n"
    assert [i + 11 for i, (a, b) in enumerate(zip(a_yaml["head"], b_yaml["head"])) if a != b] == [18]

    init_seeds(42, deterministic=True)
    rng = torch.get_rng_state()
    native = build(native_cfg, weights).eval()
    native_rng = torch.get_rng_state()
    torch.set_rng_state(rng)
    bdf = build(MODEL, weights).eval()
    assert torch.equal(native_rng, torch.get_rng_state()), "BDF construction changed downstream RNG"
    ns, bs = native.state_dict(), bdf.state_dict()
    assert all(k in bs and bs[k].shape == v.shape and torch.equal(v, bs[k]) for k, v in ns.items())
    new_keys = sorted(set(bs) - set(ns))
    assert len(new_keys) == 5 and all(k.startswith("model.18.") for k in new_keys)
    pretrained, _ = load_checkpoint(weights)
    ps = pretrained.state_dict()
    gaps = {k: {"pretrained": list(ps[k].shape), "nc1": list(v.shape)} for k, v in ns.items() if ps[k].shape != v.shape}
    assert all(k.startswith("model.23.") for k in gaps)
    assert all(torch.equal(v, ps[k]) for k, v in ns.items() if k not in gaps)
    report["initialization"] = {
        "native_state_items_equal": len(ns),
        "native_shape_gaps": gaps,
        "pretrained_matching_items": len(ns) - len(gaps),
        "new_keys": new_keys,
        "unexpected_native_missing": [],
        "rng_equal": True,
    }
    assert len(bdf.model) == len(native.model) == 24 and bdf.model[18].f == [17, 13, 6]
    assert bdf.model[-1].f == [16, 19, 22]
    assert sum(isinstance(m, BDF_Fusion) for m in bdf.modules()) == 1
    counts = {
        "native_unfused": sum(p.numel() for p in native.parameters()),
        "bdf_unfused": sum(p.numel() for p in bdf.parameters()),
    }
    assert counts["bdf_unfused"] - counts["native_unfused"] == 21784
    layer = bdf.model[18]
    assert torch.equal(layer.proj.weight[:, :, 0, 0], torch.eye(128))
    assert layer.gate.weight.count_nonzero() == layer.gate.bias.count_nonzero() == 0
    wiring = []
    handle = layer.register_forward_hook(
        lambda m, x, y: wiring.append({"inputs": [list(t.shape) for t in x[0]], "output": list(y.shape)})
    )
    with torch.no_grad():
        report["initial_forward"] = {}
        for h, w in ((640, 640), (384, 640)):
            x = torch.rand(1, 3, h, w)
            out_a, out_b = native(x), bdf(x)
            report["initial_forward"][f"{h}x{w}"] = {
                "raw_max_error": raw_error(out_a, out_b),
                "decoded_max_error": max_error(out_a[0], out_b[0]),
            }
    handle.remove()
    assert wiring[0] == {"inputs": [[1, 64, 40, 40], [1, 128, 40, 40], [1, 128, 40, 40]], "output": [1, 192, 40, 40]}
    report["wiring"] = wiring

    test_layer = deepcopy(layer)
    inputs = [torch.randn(2, c, 5, 7) for c in (64, 128, 128)]
    snapshots = [t.clone() for t in inputs]
    with torch.no_grad():
        test_layer.gate.weight.normal_(std=0.01)
        test_layer.gate.bias.copy_(torch.linspace(-0.07, 0.07, 8))
        a, b, c = inputs
        gate = 0.5 * test_layer.gate(test_layer.act(test_layer.dw(test_layer.reduce(torch.cat(inputs, 1))))).tanh()
        groups = torch.stack([gate[:, j] for j in range(8) for _ in range(16)], dim=1)
        expected = torch.cat((a, b + groups * (test_layer.proj(c) - b)), 1)
        actual = test_layer(inputs)
        formula_error = max_error(actual, expected, atol=0, rtol=0)
        correction = (actual - torch.cat((a, b), 1)).abs().max().item()
        assert correction > 0 and gate.min() < 0 and gate.max() > 0
    assert all(torch.equal(a, b) for a, b in zip(inputs, snapshots))
    for channels in ((64, 128, 64), (64, 127, 127)):
        try:
            BDF_Fusion(channels)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid channels accepted")
    try:
        test_layer([inputs[0], inputs[1], inputs[2][:, :, :-1]])
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid spatial wiring accepted")
    report["nonzero_formula"] = {
        "max_error": formula_error,
        "max_correction": correction,
        "signed_contiguous_groups": True,
        "inputs_unchanged": True,
    }
    print("PASS: native states, RNG, layer wiring, zero and nonzero formulas", flush=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        # Also check construction under a CUDA default device, not just CPU construction followed by .to().
        cuda_rng, cpu_rng = torch.cuda.get_rng_state(), torch.get_rng_state()
        with torch.device(device):
            gpu_layer = BDF_Fusion([64, 128, 128])
        assert gpu_layer.proj.weight.device == device
        assert torch.equal(cuda_rng, torch.cuda.get_rng_state()) and torch.equal(cpu_rng, torch.get_rng_state())
        del gpu_layer
        report["cuda_construction_rng_equal"] = True
    trained = deepcopy(bdf).to(device).train()
    optimizer = DetectionTrainer.build_optimizer(
        DetectionTrainer.__new__(DetectionTrainer), trained, name="MuSGD", lr=0.01, momentum=0.937, decay=0.0005
    )
    new_params = dict(trained.model[18].named_parameters())
    groups_report = {}
    for name, p in new_params.items():
        groups = [g for g in optimizer.param_groups for q in g["params"] if q is p]
        assert len(groups) == 1 and groups[0]["lr"] == 0.01
        assert groups[0]["param_group"] == ("bias" if name.endswith("bias") else "muon")
        groups_report[name] = {k: groups[0][k] for k in ("param_group", "lr", "weight_decay")}
    batch, selected = smoke_batch(cli.dataset, device)
    steps = []
    for step in range(3):
        optimizer.zero_grad(set_to_none=True)
        loss, _ = trained(batch)
        loss.sum().backward()
        grads = {n: p.grad.detach().abs().sum().item() for n, p in new_params.items()}
        assert all(torch.isfinite(p.grad).all() for p in new_params.values())
        assert grads["gate.weight"] > 0 and grads["gate.bias"] > 0
        assert all(grads[n] == 0 if step == 0 else grads[n] > 0 for n in ("proj.weight", "reduce.weight", "dw.weight"))
        before = {n: p.detach().clone() for n, p in new_params.items()}
        torch.nn.utils.clip_grad_norm_(trained.parameters(), max_norm=10.0)
        optimizer.step()  # No accumulation or GradScaler skip: exactly three actual updates.
        changes = {n: (p - before[n]).abs().max().item() for n, p in new_params.items()}
        assert changes["gate.weight"] > 0 and changes["gate.bias"] > 0
        steps.append(
            {"loss": loss.detach().tolist(), "task_gradient_l1_before_decay": grads, "parameter_max_change": changes}
        )
    report["optimization"] = {
        "actual_steps": 3,
        "device": str(device),
        "precision": "FP32",
        "batch": [2, 3, 256, 256],
        "train_samples": selected,
        "groups": groups_report,
        "steps": steps,
        "scope": "two real train samples, resized for smoke; not dataset accuracy validation",
    }
    if device.type == "cuda":
        amp_model = deepcopy(trained).train()
        amp_model.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            amp_loss, _ = amp_model(batch)
        amp_loss.sum().backward()
        assert torch.isfinite(amp_loss).all()
        assert all(torch.isfinite(p.grad).all() for p in amp_model.parameters() if p.grad is not None)
        report["cuda_amp"] = {"loss": amp_loss.detach().tolist(), "finite_backward": True, "optimizer_steps": 0}
        del amp_model
    else:
        report["cuda_amp"] = "UNVERIFIED: no CUDA device"
    print("PASS: three actual MuSGD steps and task gradients", flush=True)

    trained = trained.cpu().eval()
    learned = deepcopy(trained.model[18].state_dict())
    assert learned["gate.weight"].count_nonzero() and not torch.equal(learned["proj.weight"], layer.proj.weight)
    cloned = deepcopy(trained)
    ema = ModelEMA(trained)
    ema.update(trained)
    for other in (cloned, ema.ema):
        for k, v in learned.items():
            max_error(v, other.model[18].state_dict()[k])
    probe = torch.rand(1, 3, 256, 320)
    corrections = []
    handle = trained.model[18].register_forward_hook(
        lambda m, x, y: corrections.append((y - torch.cat(x[0][:2], 1)).abs().max().item())
    )
    with torch.no_grad():
        before_fuse = trained(probe)
        cloned_error = raw_error(before_fuse, cloned(probe))
        ema_error = raw_error(before_fuse, ema.ema(probe), atol=1e-4, rtol=1e-4)
    handle.remove()
    assert corrections[0] > 0
    with tempfile.TemporaryDirectory(prefix="bdf-smoke-", dir=Path.cwd()) as temporary:
        # Relative paths also work with the baseline loader's quote normalization on Windows usernames.
        directory = Path(temporary).relative_to(Path.cwd())
        checkpoint = directory / "nonzero.pt"
        torch.save({"model": cloned, "train_args": locked_config()}, checkpoint)
        loaded, _ = load_checkpoint(checkpoint)
        for k, v in learned.items():
            assert torch.equal(v, loaded.model[18].state_dict()[k])
        with torch.no_grad():
            reload_error = raw_error(before_fuse, loaded(probe))
        torch.save(probe, directory / "probe.pt")
        code = """
import sys, torch
from ultralytics import YOLO
from ultralytics.nn.modules import BDF_Fusion
torch.set_num_threads(4)
model = YOLO(sys.argv[1])
layer = model.model.model[18]
assert isinstance(layer, BDF_Fusion)
state = {k: v.clone() for k, v in layer.state_dict().items()}
corrections = []
layer.register_forward_hook(lambda m,x,y: corrections.append((y-torch.cat(x[0][:2],1)).abs().max().item()))
results = model.predict(torch.load(sys.argv[2],weights_only=True),device='cpu',verbose=False)
assert len(results) == 1 and corrections and max(corrections) > 0
assert all(torch.equal(v,layer.state_dict()[k]) for k,v in state.items())
print('NEW_PROCESS_DEFAULT_PREDICT_OK',max(corrections))
"""
        child = subprocess.run(
            [sys.executable, "-c", code, str(checkpoint), str(directory / "probe.pt")],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    fused = deepcopy(trained).fuse(verbose=False)
    for k, v in learned.items():
        assert torch.equal(v, fused.model[18].state_dict()[k])
    with torch.no_grad():
        after_fuse = fused(probe)
        fuse_error = raw_error(before_fuse, after_fuse, atol=2e-4, rtol=1e-4)
    assert isinstance(fused.model[18], BDF_Fusion) and fused.model[18].forward.__func__ is BDF_Fusion.forward
    counts.update(
        native_fused=sum(p.numel() for p in deepcopy(native).fuse(verbose=False).parameters()),
        bdf_fused=sum(p.numel() for p in fused.parameters()),
    )
    assert counts["bdf_fused"] - counts["native_fused"] == 21784
    report["parameters"] = {
        **counts,
        "added": 21784,
        "added_conv_macs_640": 34841600,
        "added_conv_gflops_640": 0.0696832,
    }
    report["nonzero_lifecycle"] = {
        "max_correction": corrections[0],
        "deepcopy_raw_error": cloned_error,
        "ema_raw_error": ema_error,
        "reload_raw_error": reload_error,
        "fuse_raw_error": fuse_error,
        "learned_state_preserved": True,
        "new_process": child.stdout.strip(),
    }
    cli.report.parent.mkdir(parents=True, exist_ok=True)
    cli.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"PASS: nonzero deepcopy/EMA/reload/fuse/new-process predict; report: {cli.report}", flush=True)


if __name__ == "__main__":
    main()
