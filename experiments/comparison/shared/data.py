"""Read-only split auditing shared by independent comparison backends."""

import hashlib
import json
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

FORMATS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
HISTORICAL_YAML_SHA256 = "1f18760508e9dbf2332cd7102ee9c15e11e08f3d15fc4202c8ee0ed1bb785b12"


def sha256(path):
    """Hash effective file bytes, independent of installation paths."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def canonical_hash(value):
    """Hash sorted portable data indices."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def label_path(image):
    """Use the native images-to-labels directory convention."""
    text = image.as_posix()
    if "/images/" not in text:
        raise ValueError(f"Expected images directory in dataset path: {image}")
    left, right = text.rsplit("/images/", 1)
    return Path(left + "/labels/" + right).with_suffix(".txt")


def split_files(value, root):
    """Resolve directories or file lists without downloading or writing data."""
    files = []
    for item in value if isinstance(value, list) else [value]:
        p = Path(item)
        p = p if p.is_absolute() else root / p
        if p.is_dir():
            files.extend(x.resolve() for x in p.rglob("*") if x.suffix.lower() in FORMATS)
        elif p.is_file() and p.suffix.lower() == ".txt":
            for line in p.read_text(encoding="utf-8").splitlines():
                q = Path(line)
                files.append((q if q.is_absolute() else (p.parent if line.startswith("./") else root) / q).resolve())
        else:
            raise FileNotFoundError(f"Data split missing: {p}")
    files = sorted(files)
    if not files or len(set(files)) != len(files):
        raise ValueError("Empty data split or duplicate image paths")
    return files


def audit(path, read_yaml=None):
    """Validate full effective contents and save portable split hashes in memory."""
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Dataset YAML missing: {path}; no substitute dataset is downloaded")
    raw = read_yaml(path) if read_yaml else yaml.safe_load(path.read_text(encoding="utf-8"))
    names = raw.get("names")
    names = dict(enumerate(names)) if isinstance(names, list) else names
    if names != {0: "crack"} or raw.get("nc", 1) != 1:
        raise ValueError(f"Expected original single class crack=0, got {names}")
    root = Path(raw.get("path", path.parent))
    root = root if root.is_absolute() else path.parent / root
    root = root.resolve()
    dataset = {**raw, "path": str(root), "names": names, "nc": 1, "yaml_file": str(path)}
    manifests, used = {}, set()
    for split in ["train", "val", "test"]:
        if split not in raw:
            raise ValueError(f"Missing split {split}; test cannot substitute for val")
        files = split_files(raw[split], root)
        if used.intersection(files):
            raise ValueError(f"Image appears in multiple splits: {split}")
        used.update(files)
        entries = []
        for image in files:
            with Image.open(image) as im:
                width, height = im.size
                if min(width, height) < 10:
                    raise ValueError(f"Image too small: {image}")
                im.verify()
            if image.suffix.lower() in [".jpg", ".jpeg"]:
                with image.open("rb") as f:
                    f.seek(-2, 2)
                    if f.read() != b"\xff\xd9":
                        raise ValueError(f"JPEG needs repair outside the read-only dataset: {image}")
            label = label_path(image)
            labels = []
            if label.exists():
                for line in label.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    row = np.array([float(x) for x in line.split()], dtype=np.float64)
                    if row.shape != (5,) or not np.isfinite(row).all() or row[0] != 0:
                        raise ValueError(f"Invalid detection class/label: {label}")
                    if (row[1:] < 0).any() or (row[1:] > 1).any() or (row[3:] <= 0).any():
                        raise ValueError(f"Invalid normalized box: {label}")
                    labels.append(row.tolist())
                if len(set(map(tuple, labels))) != len(labels):
                    raise ValueError(f"Duplicate labels would be silently removed by the backend: {label}")
            entries.append(
                {
                    "image_id": image.relative_to(root).as_posix(),
                    "image_sha256": sha256(image),
                    "label_sha256": sha256(label) if label.exists() else None,
                    "width": width,
                    "height": height,
                    "boxes": len(labels),
                }
            )
        manifests[split] = {
            "images": len(files),
            "boxes": sum(x["boxes"] for x in entries),
            "entries": entries,
            "sha256": canonical_hash(entries),
        }
        # Explicit lists preserve the audited files and avoid subsequent split-path guessing.
        dataset[split] = [str(root / x["image_id"]) for x in entries]
    public = {"names": names, "splits": {k: v["sha256"] for k, v in manifests.items()}}
    report = {
        "yaml_sha256": sha256(path),
        "historical_yaml_sha256": HISTORICAL_YAML_SHA256,
        "historical_yaml_matches": sha256(path) == HISTORICAL_YAML_SHA256,
        "identity_rule": "class mapping and sorted image/label effective contents; YAML bytes recorded separately",
        "root": str(root),
        "splits": manifests,
        "data_sha256": canonical_hash(public),
    }
    return dataset, report


def backend_paths(dataset, directory):
    """Write split lists into the experiment cache, never beside original data."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    result = dict(dataset)
    for split in ["train", "val", "test"]:
        path = directory / (split + ".txt")
        path.write_text("\n".join(dataset[split]) + "\n", encoding="utf-8")
        result[split] = str(path)
    return result
