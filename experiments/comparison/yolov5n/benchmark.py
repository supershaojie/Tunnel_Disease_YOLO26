"""Independent batch-one latency and nc=1 unfused/fused parameter/FLOP measurement."""

import copy
import importlib.metadata
import time
from pathlib import Path

from common import bootstrap, configuration, parser, print_plan, verify_sources
from comparison_io import digest, now, status, trusted_load, verify_weights, write_json


def main():
    p = parser(__doc__)
    p.add_argument("--image", type=Path, help="One fixed representative image; loaded before all timing")
    p.add_argument("--result-dir", type=Path)
    args = p.parse_args()
    cfg, audit = configuration(args)
    verify_sources()
    if args.dry_run:
        print_plan(cfg, audit)
        return
    import numpy as np
    import torch

    from data import image_read, manifest
    from runtime import environment, gpu_processes

    if verify_weights(cfg["weights"])["kind"] != "comparison_checkpoint":
        raise ValueError("Benchmark requires this experiment's trained checkpoint")
    imports = bootstrap(cfg["cache"])
    from geometry import LetterBox
    from utils.general import non_max_suppression
    from utils.torch_utils import select_device

    device = select_device(cfg["train"]["device"], batch_size=1)
    checkpoint = trusted_load(cfg["weights"])
    model = (
        copy.deepcopy(checkpoint["ema"] if checkpoint.get("ema") is not None else checkpoint["model"])
        .float()
        .to(device)
        .eval()
    )
    unfused = sum(p.numel() for p in model.parameters())
    model.fuse()
    fused = sum(p.numel() for p in model.parameters())
    flops = {
        "state": "NA",
        "reason": "profiler unavailable",
        "input": [1, 3, 640, 640],
        "convention": "1 MAC = 2 FLOPs",
    }
    try:
        import thop
        from thop.profile import register_hooks

        clone = copy.deepcopy(model)
        with torch.inference_mode():
            macs, _ = thop.profile(clone, inputs=(torch.zeros(1, 3, 640, 640, device=device),), verbose=False)
        missing = sorted(
            {type(m).__name__ for m in model.modules() if not list(m.children()) and type(m) not in register_hooks}
        )
        flops = {
            "state": "MEASURED",
            "input": [1, 3, 640, 640],
            "MACs": macs,
            "FLOPs": 2 * macs,
            "convention": "1 MAC = 2 FLOPs",
            "profiler": "ultralytics-thop",
            "version": importlib.metadata.version("ultralytics-thop"),
            "uncovered_leaf_types": missing,
            "uncovered_functional_operations": [
                "anchor-grid decode",
                "sigmoid",
                "tensor concat/reshape",
                "NMS",
                "image preprocessing",
            ],
            "scope": "fused model tensor-operation estimate with listed omissions; not an all-operation hardware FLOP count",
        }
    except (ImportError, importlib.metadata.PackageNotFoundError) as error:
        flops["reason"] = str(error)
    except Exception as error:
        flops["reason"] = f"Profiler failed: {type(error).__name__}: {error}"
    b = cfg["benchmark"]
    if b["half"]:
        if device.type != "cuda":
            raise ValueError("FP16 benchmark requires CUDA")
        model.half()
    dtype = next(model.parameters()).dtype
    image_path = args.image
    if image_path is None:
        data = manifest(cfg["data"])
        image_path = Path(data["splits"]["test"]["records"][0]["image"])
    image = image_read(image_path)
    letterbox = LetterBox((b["imgsz"], b["imgsz"]), auto=False, scaleup=True)
    run = args.result_dir.resolve() if args.result_dir else Path(cfg["weights"]).resolve().parent.parent
    directory = run / "benchmark"
    directory.mkdir(parents=True, exist_ok=False)
    started = now()
    status(directory, "benchmark", "RUNNING", started=started)
    contention_before = gpu_processes() if device.type == "cuda" else {"state": "CPU_SMOKE"}

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    samples = []
    try:
        with torch.inference_mode():
            for iteration in range(b["warmup"] + b["iterations"]):
                sync()
                start = time.perf_counter()
                pixels = letterbox(image=image)
                tensor = (
                    torch.from_numpy(np.ascontiguousarray(pixels[..., ::-1].transpose(2, 0, 1)))
                    .unsqueeze(0)
                    .to(device)
                    .to(dtype)
                    / 255
                )
                sync()
                pre = time.perf_counter()
                output = model(tensor, augment=False)
                sync()
                forward = time.perf_counter()
                detections = non_max_suppression(
                    output[0].float(),
                    conf_thres=cfg["eval"]["conf"],
                    iou_thres=cfg["eval"]["iou"],
                    max_det=cfg["eval"]["max_det"],
                )[0]
                ratio = min(b["imgsz"] / image.shape[0], b["imgsz"] / image.shape[1])
                pad_x, pad_y = (
                    round((b["imgsz"] - round(image.shape[1] * ratio)) / 2 - 0.1),
                    round((b["imgsz"] - round(image.shape[0] * ratio)) / 2 - 0.1),
                )
                detections[:, [0, 2]] = ((detections[:, [0, 2]] - pad_x) / ratio).clamp(0, image.shape[1])
                detections[:, [1, 3]] = ((detections[:, [1, 3]] - pad_y) / ratio).clamp(0, image.shape[0])
                detections.cpu()
                sync()
                end = time.perf_counter()
                if iteration >= b["warmup"]:
                    samples.append(
                        [(pre - start) * 1000, (forward - pre) * 1000, (end - forward) * 1000, (end - start) * 1000]
                    )
        array = np.array(samples)
        contention_after = gpu_processes() if device.type == "cuda" else {"state": "CPU_SMOKE"}
        contention = (
            "CONTENDED"
            if "CONTENDED" in (contention_before["state"], contention_after["state"])
            else "UNVERIFIED"
            if "UNVERIFIED" in (contention_before["state"], contention_after["state"])
            else "UNCONTENDED_OBSERVATION"
            if device.type == "cuda"
            else "CPU_SMOKE"
        )
        values = {
            "state": "COMPLETED",
            "purpose": checkpoint["comparison"]["purpose"],
            "contention": contention,
            "eligible_for_main_paper": contention == "UNCONTENDED_OBSERVATION"
            and checkpoint["comparison"]["purpose"] != "SMOKE"
            and dtype == torch.float32,
            "device": str(device),
            "dtype": str(dtype),
            "input": [1, 3, b["imgsz"], b["imgsz"]],
            "fuse": True,
            "weights_sha256": digest(cfg["weights"]),
            "parameters": {"unfused": unfused, "fused": fused, "unfused_M": unfused / 1e6, "fused_M": fused / 1e6},
            "flops": flops,
            "scope": "decoded image in memory; preprocessing includes LetterBox/RGB/CHW/normalization/H2D; forward excludes disk and loader; post includes objectness*class NMS, coordinate restore and D2H; total is sum, synchronized on CUDA",
            "image_path": str(image_path.resolve()),
            "image_sha256": digest(image_path),
            "warmup": b["warmup"],
            "iterations": b["iterations"],
            "latency_ms": {
                name: {
                    "mean": float(array[:, i].mean()),
                    "median": float(np.median(array[:, i])),
                    "p95": float(np.percentile(array[:, i], 95)),
                }
                for i, name in enumerate(("preprocess", "forward", "postprocess", "total"))
            },
            "FPS_total": 1000 / float(array[:, 3].mean()),
            "FPS_forward": 1000 / float(array[:, 1].mean()),
            "samples_ms": samples,
            "contention_before": contention_before,
            "contention_after": contention_after,
            "environment": environment(imports),
        }
        write_json(directory / "benchmark.json", values)
        write_json(directory / "requested_config.json", cfg)
        status(directory, "benchmark", "COMPLETED", started=started, ended=now(), contention=contention)
        write_json(run / "benchmark_status.json", {"state": "COMPLETED", "contention": contention})
        print(values)
    except BaseException as error:
        status(
            directory, "benchmark", "FAILED", started=started, ended=now(), reason=f"{type(error).__name__}: {error}"
        )
        raise


if __name__ == "__main__":
    main()
