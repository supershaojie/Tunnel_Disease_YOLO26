"""Read standard artifacts from independent models and check the total byte limit."""

import argparse
import csv
import gzip
import io
from pathlib import Path

from common import UniqueStore
from comparison_io import canonical_hash, read_json
from package import archive


def evaluation_directory(root):
    root = root.resolve()
    candidates = [p for p in (root, root / "evaluation/test", root / "evaluate_test") if (p / "metrics.json").is_file()]
    candidates = [p for p in candidates if read_json(p / "metrics.json").get("split") == "test"]
    if len(candidates) != 1:
        raise ValueError(f"Choose one explicit test evaluation directory: {root}")
    return candidates[0]


def normalized_gt(path):
    import numpy as np

    with np.load(path, allow_pickle=False) as gt:
        ids = gt["image_ids"].tolist()
        order = sorted(range(len(ids)), key=lambda i: ids[i])
        arrays = [gt["boxes"][gt["image_offsets"][i] : gt["image_offsets"][i + 1]] for i in order]
        classes = [gt["classes"][gt["image_offsets"][i] : gt["image_offsets"][i + 1]] for i in order]
        output = io.BytesIO()
        np.savez_compressed(
            output,
            boxes=np.concatenate(arrays),
            classes=np.concatenate(classes),
            image_offsets=np.cumsum([0] + [len(a) for a in arrays], dtype=np.int64),
            image_ids=np.array([ids[i] for i in order]),
        )
        return output.getvalue()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--result-dirs",
        action=UniqueStore,
        nargs="+",
        type=Path,
        required=True,
        help="Run roots or standard test evaluation directories",
    )
    p.add_argument("--output", action=UniqueStore, type=Path, required=True)
    p.add_argument("--max-bytes", action=UniqueStore, type=int, default=100000000)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    directories = [evaluation_directory(root) for root in args.result_dirs]
    metrics = [read_json(root / "metrics.json") for root in directories]
    if len({m["protocol_sha256"] for m in metrics}) != 1:
        raise ValueError("Different protocols cannot be merged as one comparison")
    names = [m["model"] for m in metrics]
    if len(set(names)) != len(names) or any(not n or Path(n).name != n for n in names):
        raise ValueError("Each model must have a unique directory-safe metrics.model identity")
    values = {}
    table = io.StringIO(newline="")
    columns = ["model", "split", "units", "AP50", "AP75", "mAP50-95", "protocol_sha256", "weight_sha256"]
    writer = csv.DictWriter(table, fieldnames=columns)
    writer.writeheader()
    shared_gt = None
    for requested, directory, metric in zip(args.result_dirs, directories, metrics):
        contract = metric.get("public")
        if contract is None:
            contract = read_json(directory / "protocol.json")["config"]
        if canonical_hash(contract) != metric["protocol_sha256"]:
            raise ValueError("Declared protocol hash does not match saved public configuration")
        gt = normalized_gt(directory / "ground_truth.npz")
        if shared_gt is not None and shared_gt != gt:
            raise ValueError(
                "GT pixel coordinates differ after image-ID alignment; reconcile coordinate restoration first"
            )
        shared_gt = gt
        writer.writerow({k: metric[k] for k in columns})
        for path in sorted(directory.rglob("*")):
            if (
                not path.is_file()
                or path.name == "ground_truth.npz"
                or path.suffix not in (".json", ".csv", ".npz", ".svg", ".jpg")
            ):
                continue
            if path.is_symlink() or not path.resolve().is_relative_to(directory):
                raise ValueError(f"Artifact outside selected evaluation: {path}")
            values[f"{metric['model']}/test/{path.relative_to(directory).as_posix()}"] = path.read_bytes()
        if requested.resolve() != directory:
            for path in sorted(requested.glob("*")):
                if path.is_file() and path.suffix in (".json", ".csv", ".yaml", ".svg"):
                    values[f"{metric['model']}/run/{path.name}"] = path.read_bytes()
            log = requested / "train.log"
            if log.is_file():
                values[f"{metric['model']}/run/train.log.gz"] = gzip.compress(log.read_bytes(), mtime=0)
    values["shared/ground_truth.npz"] = shared_gt
    values["summary.csv"] = table.getvalue().encode()
    if args.dry_run:
        print({"state": "DRY_RUN", "models": names, "members": len(values), "max_bytes": args.max_bytes})
        return
    print(
        archive(
            values,
            args.output,
            args.max_bytes,
            {"models": names, "shared_GT_once": True, "strict_bytes": args.max_bytes},
        )
    )


if __name__ == "__main__":
    main()
