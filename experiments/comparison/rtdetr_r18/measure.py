"""Measure the actual fused nc=1 RT-DETR and keep FPS separate from training."""

import json
import os
import subprocess
import time
from copy import deepcopy
from pathlib import Path

from common import bootstrap, environment, protocol, result_directory, status, write_json
from runtime import checkpoint_info


def profile_model(model):
    """Count fused inference parameters and disclose functional operations outside THOP hooks."""
    import thop
    import torch

    inference = deepcopy(model).float().cpu().eval()
    inference.fuse()
    parameters = sum(parameter.numel() for parameter in inference.parameters())
    with torch.inference_mode():
        macs, _ = thop.profile(inference, inputs=(torch.zeros(1, 3, 640, 640),), verbose=False)
    return {
        "status": "MEASURED_PARTIAL",
        "profiler": "thop",
        "version": getattr(thop, "__version__", "unknown"),
        "input_shape": [1, 3, 640, 640],
        "nc": 1,
        "structure": "inference model after native fuse()",
        "native_fuse_applied": True,
        "fused_actual": inference.is_fused(),
        "remaining_batchnorm_modules": sum(
            isinstance(module, torch.nn.modules.batchnorm._BatchNorm) for module in inference.modules()
        ),
        "parameters": parameters,
        "parameters_M": parameters / 1e6,
        "MACs": float(macs),
        "FLOPs": float(2 * macs),
        "convention": "1 MAC = 2 FLOPs",
        "not_covered": [
            "functional attention matmul/softmax",
            "deformable attention grid_sample",
            "functional concat/add",
            "query box decoding and filtering",
        ],
        "coverage_note": "THOP partial count; native fuse retains ResNet ConvNormLayer BatchNorm; functional operators remain uncounted",
    }


def complexity(cfg, args):
    """Profile the selected checkpoint at nc=1 and 640 without measuring speed."""
    if args.dry_run:
        print(json.dumps({"mode": "complexity", "weights": cfg["weights"], "input_shape": [1, 3, 640, 640]}))
        return
    if not cfg["weights"]:
        raise ValueError("complexity requires --weights <this-experiment-best.pt>")
    run, checkpoint = checkpoint_info(cfg["weights"])
    directory = Path(args.result_dir or run / "complexity").resolve()
    directory.mkdir(parents=True, exist_ok=False)
    status(directory, "complexity", "RUNNING")
    try:
        snapshot = directory / "weight_snapshot.pt"
        _, checkpoint = checkpoint_info(cfg["weights"], snapshot)
        backend = bootstrap(cfg)
        from adapter import verify_model

        model = backend.RTDETR(str(snapshot)).model
        verify_model(model)
        result = {
            "weight_sha256": checkpoint["weights"][Path(cfg["weights"]).name],
            "FLOPs": profile_model(model),
            "FPS": None,
            "environment": environment(),
        }
        write_json(directory / "complexity.json", result)
        write_json(run / "complexity_location.json", {"directory": str(directory)})
        status(directory, "complexity", "COMPLETED")
        print(json.dumps({"directory": str(directory), **result}, indent=2))
    except BaseException as error:
        status(directory, "complexity", "FAILED", reason=f"{type(error).__name__}: {error}")
        raise


def benchmark(cfg, args):
    """Time synchronized in-memory preprocessing, native query forward and filtering separately."""
    if args.dry_run:
        print(json.dumps({"mode": "benchmark", "weights": cfg["weights"], "settings": cfg["benchmark"]}))
        return
    if not cfg["weights"]:
        raise ValueError("benchmark requires --weights <this-experiment-best.pt>")
    run, checkpoint = checkpoint_info(cfg["weights"])
    directory = result_directory(run, args, "benchmark")
    directory.mkdir(parents=True, exist_ok=False)
    status(directory, "benchmark", "RUNNING")
    try:
        captured_weight = directory / "weight_snapshot.pt"
        _, checkpoint = checkpoint_info(cfg["weights"], captured_weight)
        backend = bootstrap(cfg)
        public = protocol(
            cfg, checkpoint["data_sha256"], checkpoint["training_augmentation"][Path(cfg["weights"]).name]
        )
        import numpy as np
        import torch
        from adapter import decode, verify_model

        from ultralytics.data.augment import LetterBox
        from ultralytics.utils.torch_utils import select_device

        setting = cfg["benchmark"]
        device = select_device(cfg["train"]["device"], 1)
        if setting["half"] and device.type != "cuda":
            raise ValueError("FP16 benchmark requires CUDA")
        model = backend.RTDETR(str(captured_weight)).model
        verify_model(model)
        unfused = sum(parameter.numel() for parameter in model.parameters())
        profile = profile_model(model)
        if setting["fuse"]:
            model.fuse()
        model.to(device).eval()
        model.half() if setting["half"] else model.float()
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        letterbox = LetterBox(new_shape=(setting["imgsz"], setting["imgsz"]), auto=False, scaleup=False)

        def sync():
            if device.type == "cuda":
                torch.cuda.synchronize(device)

        def iteration():
            sync()
            start = time.perf_counter()
            resized = letterbox(image=image)
            tensor = (
                torch.from_numpy(np.ascontiguousarray(resized[..., ::-1].transpose(2, 0, 1))).unsqueeze(0).to(device)
            )
            tensor = (tensor.half() if setting["half"] else tensor.float()) / 255.0
            sync()
            after_preprocess = time.perf_counter()
            raw = model(tensor)
            sync()
            after_forward = time.perf_counter()
            decode(raw, tensor.shape[-2:], cfg["eval"]["conf"], cfg["eval"]["max_det"])
            sync()
            end = time.perf_counter()
            return [
                (after_preprocess - start) * 1000,
                (after_forward - after_preprocess) * 1000,
                (end - after_forward) * 1000,
                (end - start) * 1000,
            ]

        contention = []
        eligibility = "CPU_DEVELOPMENT_ONLY"
        if device.type == "cuda":
            query = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                check=False,
            )
            contention = [line for line in query.stdout.splitlines() if line.split(",")[0].strip() != str(os.getpid())]
            eligibility = (
                "CONTENDED" if contention else "UNVERIFIED" if query.returncode else "NO_OTHER_COMPUTE_PROCESS_OBSERVED"
            )
        with torch.inference_mode():
            for _ in range(setting["warmup"]):
                iteration()
            samples = np.asarray([iteration() for _ in range(setting["iterations"])], dtype=np.float64)
        columns = ["preprocess", "forward", "postprocess", "total"]
        inference = sum(parameter.numel() for parameter in model.parameters())
        result = {
            "status": "MEASURED",
            "paper_eligibility": eligibility,
            "formal_result": "SMOKE" if checkpoint["execution_kind"] == "SMOKE" else "MEASURED",
            "weight_sha256": checkpoint["weights"][Path(cfg["weights"]).name],
            "protocol_sha256": public["protocol_sha256"],
            "training_augmentation": public["config"]["augment"],
            "settings": setting,
            "NMS": "not applicable to native RT-DETR queries",
            "query_filter": {key: cfg["eval"][key] for key in ["conf", "max_det"]},
            "parameters": {
                "unfused": unfused,
                "inference": inference,
                "unfused_M": unfused / 1e6,
                "inference_M": inference / 1e6,
            },
            "FLOPs": profile,
            "latency_ms": {
                key: {
                    "mean": float(samples[:, index].mean()),
                    "median": float(np.median(samples[:, index])),
                    "p95": float(np.quantile(samples[:, index], 0.95)),
                }
                for index, key in enumerate(columns)
            },
            "FPS_total": float(1000 / samples[:, 3].mean()),
            "other_compute_processes": contention,
            "scope": "in-memory uint8 LetterBox/RGB/transfer/normalization + native query forward + filtering; CUDA synchronized; no disk/data loader",
            "model_dtype": str(next(model.parameters()).dtype),
            "fused_actual": model.is_fused(),
            "environment": environment(),
        }
        np.savez_compressed(directory / "timing_samples.npz", samples_ms=samples, columns=np.asarray(columns))
        write_json(directory / "benchmark.json", result)
        write_json(run / "benchmark_location.json", {"directory": str(directory)})
        status(directory, "benchmark", "COMPLETED", paper_eligibility=eligibility)
        print(json.dumps({"directory": str(directory), **result}, indent=2))
    except BaseException as error:
        status(directory, "benchmark", "FAILED", reason=f"{type(error).__name__}: {error}")
        raise
