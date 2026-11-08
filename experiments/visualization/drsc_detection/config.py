"""Fixed official runs and isolated runtime locations for the DRSC detection figure."""

from pathlib import Path

BASE = "/root/autodl-tmp/projects/Tunnel_Disease_YOLO26"
DATA = BASE + "/datasets/Tunnel_Crack_AugFirst_Diverse5x_RandomSplit_7_2_1_seed42/data.yaml"
MODELS = {
    "yolov5n": (
        "YOLOv5n",
        "runs/comparison/yolov5n/compare_yolov5n_b19_e200_i640_b32_sgd_s42_retry1",
        "014381620030f337ec6c45dce1d22d0d5fa9e69f754f2713e7172430c2725939",
        "-cmp-y5n",
        ".venv/bin/python",
    ),
    "yolov8n": (
        "YOLOv8n",
        "runs/comparison/yolov8n/compare_yolov8n_b19_e200_i640_b32_sgd_s42_retry1",
        "aa03edafd8fbc005874425ed03038fbdc5ea340ff8958b906b8ab6bfad2a2c1c",
        "-cmp-y8n",
        ".venv-y8n/bin/python",
    ),
    "yolo11n": (
        "YOLOv11n",
        "runs/comparison/yolo11n/compare_yolo11n_b19_e200_i640_b32_s42",
        "7fe93f6e1c55d11eb0ed39199d38f7545b48c05b7fa14b423aa4aa290c504065",
        "/.worktrees/compare-yolo11n-b19",
        ".venv-y11n/bin/python",
    ),
    "yolov13n": (
        "YOLOv13n",
        "runs/comparison/yolov13n/compare_yolov13n_mid120140_e200_i640_b32_sgd_s42",
        "d128157849eedd10727584acc622c286418e0ca0003b0fa7d7d0663e349faf46",
        "-y13n",
        ".venv-y13n/bin/python",
    ),
    "yolo26n": (
        "YOLOv26n",
        "runs/detect/b19_y26n_diverse5x_e200_i640_b32_musgd_b8b9hybrid_s42",
        "d0b2ca5a5d30de9ed002c64c9238b182ddeccce644c055ae5f2dba566878bd5e",
        "",
        "/root/miniconda3/bin/python",
    ),
    "rtdetr_r18": (
        "RT-DETR-R18",
        "runs/comparison/rtdetr_r18/compare_rtdetr_r18_v4_midv3v2_e200_i640_b32_s42",
        "cc1050fc142034c8425b5c704c852fd9affc0cc07536fe43b427ca5c9407d746",
        "-cmp-r18",
        ".venv-r18/bin/python",
    ),
    "drsc": (
        "DRSC-YOLO",
        "runs/detect/dtr_sce_b19_e200_i640_b32_s42",
        "a9908b8197862714057c30376bf93e0802e1b877fde5533492c9c54caf2158e4",
        "",
        "/root/miniconda3/bin/python",
    ),
}


def profiles(base=BASE, overrides=None):
    """Resolve the known paths; overrides only change explicit model locations."""
    result = {}
    tool_root = Path(__file__).resolve().parents[3]
    for key, (name, run, sha, suffix, python) in MODELS.items():
        source = Path(str(base) + suffix) if suffix else tool_root
        result[key] = {
            "name": name,
            "run": str(Path(base) / run),
            "sha256": sha,
            "source": str(source),
            "python": str(source / python),
            "weights": str(Path(base) / run / "weights/best.pt"),
        }
    for key, update in (overrides or {}).items():
        if key not in result or set(update) - {"source", "python", "weights", "run", "cache"}:
            raise ValueError(f"Unknown model/path override: {key}: {update}")
        result[key].update(update)
    return result
