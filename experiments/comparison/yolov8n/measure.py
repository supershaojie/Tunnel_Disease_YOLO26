"""Independent batch-one synchronized inference benchmark and profiler provenance."""

import json
import subprocess
import time
from copy import deepcopy
from pathlib import Path

from common import bootstrap, environment, file_hash, status, write_json
from runtime import checkpoint_info, preflight


def benchmark(cfg, args):
    """Measure preprocessing, forward and native NMS separately without disk I/O in timers."""
    if args.dry_run:
        return preflight(cfg, args)
    if not cfg["weights"]:
        raise ValueError("benchmark requires --weights <this-experiment-best.pt>")
    run, checkpoint = checkpoint_info(cfg["weights"])
    directory = Path(args.result_dir or run / "benchmark")
    if directory.exists():
        raise FileExistsError(f"Benchmark directory exists: {directory}")
    directory.mkdir(parents=True)
    status(directory, "benchmark", "RUNNING")
    try:
        backend = bootstrap(cfg)
        import numpy as np
        import torch
        from ultralytics.data.augment import LetterBox
        from ultralytics.utils import ops
        from ultralytics.utils.torch_utils import select_device
        from adapter import verify_model

        setting = cfg["benchmark"]
        device = select_device(cfg["train"]["device"], 1)
        if setting["half"] and device.type != "cuda":
            raise ValueError("FP16 benchmark requires CUDA")
        model = backend.YOLO(str(Path(cfg["weights"]).resolve()), task="detect").model
        verify_model(model)
        unfused = sum(p.numel() for p in model.parameters())
        original = deepcopy(model).float().cpu()
        if setting["fuse"]:
            model.fuse()
        model.to(device).eval()
        model.half() if setting["half"] else model.float()
        inference = sum(p.numel() for p in model.parameters())
        profile = {"status": "NA", "reason": "thop not installed", "MAC_to_FLOPs": "1 MAC = 2 FLOPs"}
        try:
            import thop

            # Native THOP hooks cover convolutions/BN and report uncovered functional operations.
            macs, _ = thop.profile(original.eval(), inputs=(torch.zeros(1, 3, 640, 640),), verbose=False)
            profile = {
                "status": "MEASURED_PARTIAL",
                "profiler": "thop",
                "version": getattr(thop, "__version__", "unknown"),
                "input_shape": [1, 3, 640, 640],
                "MACs": macs,
                "FLOPs": 2 * macs,
                "convention": "1 MAC = 2 FLOPs",
                "structure": "unfused nc=1",
                "not_covered": ["functional concat/add", "native DFL/TAL tensor arithmetic", "NMS"],
                "coverage_note": "THOP hook-based partial count, not a complete operation census",
            }
        except ImportError:
            pass
        # In-memory non-square uint8 image exercises real letterbox preprocessing without disk reads.
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        letterbox = LetterBox(new_shape=(setting["imgsz"], setting["imgsz"]), auto=False, scaleup=False)

        def sync():
            if device.type == "cuda":
                torch.cuda.synchronize(device)

        def iteration():
            sync()
            a = time.perf_counter()
            resized = letterbox(image=image)
            tensor = (
                torch.from_numpy(np.ascontiguousarray(resized[..., ::-1].transpose(2, 0, 1))).unsqueeze(0).to(device)
            )
            tensor = (tensor.half() if setting["half"] else tensor.float()) / 255.0
            sync()
            b = time.perf_counter()
            raw = model(tensor)
            sync()
            c = time.perf_counter()
            ops.non_max_suppression(raw, cfg["eval"]["conf"], cfg["eval"]["iou"], max_det=cfg["eval"]["max_det"])
            sync()
            d = time.perf_counter()
            return [(b - a) * 1000, (c - b) * 1000, (d - c) * 1000, (d - a) * 1000]

        contention = []
        if device.type == "cuda":
            query = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                check=False,
            )
            contention = [
                line
                for line in query.stdout.splitlines()
                if line.split(",")[0].strip() != str(__import__("os").getpid())
            ]
            contention_status = (
                "CONTENDED" if contention else "UNVERIFIED" if query.returncode else "NO_OTHER_COMPUTE_PROCESS_OBSERVED"
            )
        else:
            contention_status = "CPU_DEVELOPMENT_ONLY"
        with torch.inference_mode():
            for _ in range(setting["warmup"]):
                iteration()
            samples = np.asarray([iteration() for _ in range(setting["iterations"])], dtype=np.float64)
        columns = ["preprocess", "forward", "postprocess", "total"]
        result = {
            "status": "MEASURED",
            "paper_eligibility": contention_status,
            "formal_result": "SMOKE" if "/.development/" in run.as_posix() else "MEASURED",
            "weight_sha256": file_hash(cfg["weights"]),
            "protocol_sha256": checkpoint["protocol_sha256"],
            "settings": setting,
            "parameters": {
                "unfused": unfused,
                "inference": inference,
                "unfused_M": unfused / 1e6,
                "inference_M": inference / 1e6,
            },
            "FLOPs": profile,
            "latency_ms": {
                k: {
                    "mean": float(samples[:, i].mean()),
                    "median": float(np.median(samples[:, i])),
                    "p95": float(np.quantile(samples[:, i], 0.95)),
                }
                for i, k in enumerate(columns)
            },
            "FPS_total": float(1000 / samples[:, 3].mean()),
            "other_compute_processes": contention,
            "scope": "in-memory uint8 letterbox/RGB/transfer/normalization + forward + native NMS; CUDA synchronized; no data loader/disk/plotting",
            "model_dtype": str(next(model.parameters()).dtype),
            "fused_actual": model.is_fused(),
            "environment": environment(),
        }
        np.savez_compressed(directory / "timing_samples.npz", samples_ms=samples, columns=np.asarray(columns))
        write_json(directory / "benchmark.json", result)
        status(directory, "benchmark", "COMPLETED", paper_eligibility=contention_status)
        print(json.dumps(result, indent=2))
    except BaseException as e:
        status(directory, "benchmark", "FAILED", reason=f"{type(e).__name__}: {e}")
        raise
