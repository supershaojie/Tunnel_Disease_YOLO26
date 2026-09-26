"""Start QCA-C2PSA from the public weights using the recovered b19 training recipe."""

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path


def main():
    """Check the fixed experiment inputs and run the native YOLO trainer."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Original b19 dataset YAML")
    parser.add_argument("--weights", type=Path, required=True, help="Original public yolo26n.pt")
    parser.add_argument("--project", type=Path, required=True, help="Independent experiment output directory")
    parser.add_argument("--dry-run", action="store_true", help="Print verified inputs and recipe without training")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    import ultralytics
    from ultralytics import YOLO
    from ultralytics.data.utils import IMG_FORMATS
    from ultralytics.utils import YAML

    if Path(ultralytics.__file__).resolve() != root / "ultralytics" / "__init__.py":
        raise RuntimeError(f"Expected Ultralytics in {root}, imported {ultralytics.__file__}")
    weights = args.weights.resolve(strict=True)
    weights_sha256 = hashlib.sha256(weights.read_bytes()).hexdigest()
    if weights_sha256 != "9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef":
        raise ValueError(f"Weights are not the verified public yolo26n.pt: {weights_sha256}")

    data_file = args.data.resolve(strict=True)
    data = YAML.load(data_file)
    data_root = Path(data.get("path") or data_file.parent)
    if not data_root.is_absolute():
        data_root = data_file.parent / data_root
    counts = {
        split: sum(p.is_file() and p.suffix[1:].lower() in IMG_FORMATS for p in (data_root / data[split]).rglob("*"))
        for split in ("train", "val", "test")
    }
    if data.get("nc") != 1 or data["names"] not in ({0: "crack"}, ["crack"]):
        raise ValueError("The b19 dataset must have nc=1 and names={0: crack}.")
    if counts != {"train": 8414, "val": 2404, "test": 1202}:
        raise ValueError(f"Dataset image counts differ from b19: {counts}")

    recipe = YAML.load(Path(__file__).with_name("b19_train.yaml"))
    recipe.update(data=str(data_file), project=str(args.project.resolve()), resume=False)
    recipe["model"] = str((root / recipe["model"]).resolve(strict=True))
    has_albumentations = importlib.util.find_spec("albumentations") is not None
    print(
        json.dumps(
            {
                "training": "NOT_STARTED",
                "ultralytics_file": ultralytics.__file__,
                "weights": str(weights),
                "weights_sha256": weights_sha256,
                "dataset_image_counts": counts,
                "albumentations_present": has_albumentations,
                "historical_b19_albumentations_present": False,
                "recipe": recipe,
            },
            indent=2,
        )
    )
    if has_albumentations:
        message = (
            "Historical b19 pip_freeze and training logs show no Albumentations. This environment has it, which "
            "activates extra native augmentations. Align a separate server environment before formal training; "
            "this script does not modify the shared environment or disable transforms."
        )
        if not args.dry_run:
            raise RuntimeError(message)
        print(f"FORMAL_TRAINING_BLOCKED: {message}", file=sys.stderr)
    if args.dry_run:
        return

    model = YOLO(recipe["model"]).load(str(weights))
    model.train(**recipe)


if __name__ == "__main__":
    main()
