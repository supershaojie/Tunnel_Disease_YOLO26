"""Train APA-Head with the frozen b19 recipe and original public YOLO26n initialization."""

import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WEIGHTS_SHA256 = "9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef"


def main():
    """Resolve experiment paths, verify initialization, and invoke the native detection trainer."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Existing b19 dataset YAML; no dataset generation.")
    parser.add_argument(
        "--weights", type=Path, required=True, help="Original public yolo26n.pt with the locked SHA256."
    )
    parser.add_argument("--project", type=Path, default=ROOT / "runs" / "apa_head")
    parser.add_argument("--name", default="apa_head_b19_e200_s42")
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    data = args.data.resolve(strict=True)
    weights = args.weights.resolve(strict=True)
    digest = hashlib.sha256(weights.read_bytes()).hexdigest()
    if digest != WEIGHTS_SHA256:
        raise ValueError(f"Expected original yolo26n.pt SHA256 {WEIGHTS_SHA256}, got {digest}: {weights}")

    sys.path.insert(0, str(ROOT))
    import ultralytics
    from ultralytics import YOLO

    imported = Path(ultralytics.__file__).resolve()
    expected = ROOT / "ultralytics" / "__init__.py"
    if imported != expected:
        raise RuntimeError(f"Wrong ultralytics import: {imported}; expected {expected}")
    print(f"Experiment root: {ROOT}\nultralytics.__file__: {imported}\nOriginal weights SHA256: {digest}")

    YOLO(str(ROOT / "ultralytics/cfg/models/26/yolo26n-apa.yaml")).train(
        cfg=str(Path(__file__).with_name("b19_train.yaml")),
        data=str(data),
        pretrained=str(weights),
        project=str(args.project.resolve()),
        name=args.name,
        device=args.device,
        resume=False,
    )


if __name__ == "__main__":
    main()
