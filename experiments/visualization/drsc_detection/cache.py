"""Read the project's lossless caches, aligning by image paths rather than numeric positions."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image
from tqdm import tqdm

from .config import profiles


def read_json(path):
    """Read UTF-8 provenance without importing a model backend."""
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha256(path):
    """Hash the actual official checkpoint bytes once per model."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset(data_yaml):
    """Read only the unchanged official test split and original detection labels."""
    path = Path(data_yaml).resolve(strict=True)
    cfg = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    names = cfg.get("names")
    if names not in (["crack"], {0: "crack"}, {"0": "crack"}) or cfg.get("nc", 1) != 1:
        raise ValueError(f"Expected single-class crack dataset: {path}")
    root = Path(cfg.get("path", path.parent))
    root = (root if root.is_absolute() else path.parent / root).resolve()
    files = []
    formats = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    for entry in cfg["test"] if isinstance(cfg["test"], list) else [cfg["test"]]:
        entry = Path(entry)
        entry = entry if entry.is_absolute() else root / entry
        if entry.is_dir():
            files.extend(p.resolve() for p in entry.rglob("*") if p.suffix.lower() in formats)
        elif entry.is_file() and entry.suffix == ".txt":
            for line in entry.read_text(encoding="utf-8-sig").splitlines():
                if line.strip():
                    item = Path(line.strip())
                    files.append(
                        (
                            item if item.is_absolute() else (entry.parent if line.startswith("./") else root) / item
                        ).resolve()
                    )
        else:
            raise FileNotFoundError(f"Official test source missing: {entry}")
    if len(files) != 1202 or len(set(files)) != 1202:
        raise ValueError(f"Expected 1202 unique official test images; found {len(files)}")
    images = []
    for image in tqdm(sorted(files), desc="Read test labels", unit="image"):
        with Image.open(image) as source:
            width, height = source.size
        left, separator, right = image.as_posix().rpartition("/images/")
        if not separator:
            raise ValueError(f"Expected images/labels dataset convention: {image}")
        label = Path(left + "/labels/" + right).with_suffix(".txt")
        rows = np.array(
            [line.split() for line in label.read_text(encoding="utf-8").splitlines() if line.strip()]
            if label.exists()
            else [],
            dtype=np.float64,
        ).reshape(-1, 5)
        if (
            not np.isfinite(rows).all()
            or (rows[:, 0] != 0).any()
            or (rows[:, 1:] < 0).any()
            or (rows[:, 1:] > 1).any()
            or (rows[:, 3:] <= 0).any()
        ):
            raise ValueError(f"Invalid normalized crack detection labels: {label}")
        if len(np.unique(rows, axis=0)) != len(rows):
            raise ValueError(f"Duplicate GT labels: {label}")
        boxes = np.column_stack((rows[:, 1:3] - rows[:, 3:5] / 2, rows[:, 1:3] + rows[:, 3:5] / 2))
        boxes *= [width, height, width, height]
        boxes = np.clip(boxes, 0, [width, height, width, height])
        images.append(
            {
                "id": image.relative_to(root).as_posix(),
                "path": str(image),
                "width": width,
                "height": height,
                "gt": np.column_stack((boxes, rows[:, 0])).tolist(),
                "predictions": {},
            }
        )
    count = sum(len(row["gt"]) for row in images)
    if count != 1477:
        raise ValueError(f"Expected 1477 official test GT boxes, got {count}")
    return images, str(root)


def canonical_id(value, ids):
    """Resolve exact relative paths or their full absolute suffix, never bare stems."""
    value = str(value).replace("\\", "/")
    if value in ids:
        return value
    matches = [key for key in ids if value.endswith("/" + key)]
    if len(matches) != 1:
        raise ValueError(f"Cache image path does not identify one official test image: {value}")
    return matches[0]


def unpack(path, images, prediction=True):
    """Read the actual shared.evaluation.save_predictions NPZ schema, including empty images."""
    with np.load(path, allow_pickle=False) as arrays:
        required = {"boxes", "classes", "image_offsets", "image_ids"} | ({"scores"} if prediction else set())
        if not required.issubset(arrays.files):
            raise ValueError(f"Unsupported NPZ schema in {path}: {arrays.files}")
        boxes, classes, offsets, raw_ids = (arrays[k] for k in ("boxes", "classes", "image_offsets", "image_ids"))
        count = len(boxes)
        if boxes.shape != (count, 4) or classes.shape != (count,) or offsets.shape != (len(raw_ids) + 1,):
            raise ValueError(f"Invalid NPZ shapes: {path}")
        if (
            not np.issubdtype(offsets.dtype, np.integer)
            or offsets[0] != 0
            or offsets[-1] != count
            or (np.diff(offsets) < 0).any()
        ):
            raise ValueError(f"Invalid image_offsets: {path}")
        scores = arrays["scores"] if prediction else None
        if prediction and (
            scores.shape != (count,) or not np.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any()
        ):
            raise ValueError(f"Invalid native scores: {path}")
        if not np.isfinite(boxes).all() or (classes != 0).any() or (boxes[:, 2:] < boxes[:, :2]).any():
            raise ValueError(f"Invalid crack boxes/classes: {path}")
        keys = [canonical_id(value, images) for value in raw_ids]
        if len(set(keys)) != len(keys) or set(keys) != set(images):
            raise ValueError(f"Incomplete/duplicate official test coverage: {path}")
        rows = np.column_stack((boxes, scores, classes)) if prediction else np.column_stack((boxes, classes))
        return {key: rows[offsets[i] : offsets[i + 1]].tolist() for i, key in enumerate(keys)}


def read_cache(directory, key, profile, images):
    """Validate identity, collection floor, coverage and GT correspondence once, then reuse native boxes."""
    directory = Path(directory)
    meta_path = directory / "visualization_source.json"
    if meta_path.is_file():
        metadata = read_json(meta_path)
        if (
            metadata["model"] != key
            or metadata["weight_sha256"] != profile["sha256"]
            or metadata["coordinate_space"] != "original_xyxy"
        ):
            raise ValueError(f"Wrong visualization cache identity: {meta_path}")
        floor = metadata["collection_conf"]
    else:
        meta_path = directory / "metrics.json"
        metadata = read_json(meta_path)
        if (
            metadata.get("model") != key
            or metadata.get("weight_sha256") != profile["sha256"]
            or metadata.get("split") != "test"
        ):
            raise ValueError(f"Wrong formal cache model/weight/split: {meta_path}")
        cfg_path = next(
            (
                directory / name
                for name in ("run_config.json", "requested_config.json", "protocol.json")
                if (directory / name).is_file()
            ),
            None,
        )
        if cfg_path is None:
            raise ValueError(f"Missing cache collection configuration: {directory}")
        cfg = read_json(cfg_path)
        floor = cfg.get("config", cfg)["eval"]["conf"]
    if floor is None or not 0 <= float(floor) <= 0.25:
        raise ValueError(f"Cache collection confidence {floor} omits display detections: {directory}")
    predictions = unpack(directory / "predictions.npz", images)
    ground_truth = unpack(directory / "ground_truth.npz", images, prediction=False)
    deviation = 0.0
    for image_id, image in images.items():
        actual = np.array(ground_truth[image_id]).reshape(-1, 5)
        expected = np.array(image["gt"]).reshape(-1, 5)
        if len(actual) != len(expected):
            raise ValueError(f"GT count mismatch: {directory}: {image_id}")
        # Native exporters reverse rounded letterbox transforms differently; raw labels own canonical GT.
        remaining = list(range(len(actual)))
        for row in expected:
            errors = np.max(np.abs(actual[remaining, :4] - row[:4]), axis=1)
            best = int(errors.argmin())
            tolerance = 2 * max(image["width"], image["height"]) / 640 + 0.05
            if errors[best] > tolerance:
                raise ValueError(f"GT coordinates disagree beyond two input pixels: {directory}: {image_id}")
            deviation = max(deviation, float(errors[best]))
            remaining.pop(best)
    environment = read_json(directory / "environment.json") if (directory / "environment.json").is_file() else {}
    return predictions, {
        "cache": str(directory),
        "metadata": str(meta_path),
        "collection_conf": float(floor),
        "coordinate_space": "original_xyxy",
        "GT_source": "original test labels",
        "max_cached_GT_coordinate_difference_px": deviation,
        "geometry": metadata.get("geometry", metadata.get("actual", {})),
        "cache_source": metadata,
        "python": metadata.get("python", environment.get("sys_executable", profile["python"])),
        "backend_source": environment.get("backend_file", environment.get("imports", {})),
    }


def prepare(data_yaml, base, output, device="0", overrides=None, reuse_only=False):
    """Reuse complete official caches and export only missing/insufficient models in isolated processes."""
    images, root = dataset(data_yaml)
    index = {row["id"]: row for row in images}
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "test_index.json"
    manifest.write_text(json.dumps({"root": root, "images": images}, ensure_ascii=False), encoding="utf-8")
    sources = {}
    for key, profile in profiles(base, overrides).items():
        if sha256(profile["weights"]) != profile["sha256"]:
            raise ValueError(f"Official best.pt SHA256 mismatch: {key}: {profile['weights']}")
        run = Path(profile["run"])
        candidates = [Path(profile["cache"])] if "cache" in profile else []
        candidates += [output / "cache" / key]
        candidates += sorted({p.parent for p in run.rglob("predictions.npz")})
        reasons, found = [], None
        for directory in candidates:
            if not (directory / "predictions.npz").is_file():
                continue
            try:
                predictions, source = read_cache(directory, key, profile, index)
            except (ValueError, KeyError, FileNotFoundError) as error:
                reasons.append(f"{directory}: {error}")
                continue
            found = directory
            break
        if found is None:
            reason = "; ".join(reasons) or f"No full prediction cache under {run}"
            print(f"{profile['name']}: exporting because {reason}", flush=True)
            if reuse_only:
                raise ValueError(f"{key} requires export: {reason}")
            target = output / "cache" / key
            target.mkdir(parents=True, exist_ok=True)
            request = {
                "key": key,
                "profile": profile,
                "manifest": str(manifest),
                "data": str(Path(data_yaml).resolve()),
                "output": str(target),
                "device": str(device),
            }
            request_path = target / "request.json"
            request_path.write_text(json.dumps(request), encoding="utf-8")
            command = [profile["python"], str(Path(__file__).with_name("export.py")), str(request_path)]
            with (target / "export.log").open("wb") as log:
                process = subprocess.Popen(
                    command, cwd=profile["source"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT
                )
                for chunk in iter(lambda stream=process.stdout: stream.read1(4096), b""):
                    sys.stdout.buffer.write(chunk)
                    sys.stdout.buffer.flush()
                    log.write(chunk)
                    log.flush()
                if process.wait():
                    raise RuntimeError(f"{key} export failed; see {target / 'export.log'}")
            predictions, source = read_cache(target, key, profile, index)
            source.update(action="exported", reason=reason)
        else:
            print(f"{profile['name']}: reused {found}", flush=True)
            source.update(action="reused", rejected_caches=reasons)
        for image_id, prediction in predictions.items():
            index[image_id]["predictions"][key] = prediction
        sources[key] = {**profile, **source}
    return {
        "images": images,
        "sources": sources,
        "dataset": {"data": str(data_yaml), "root": root, "images": len(images), "boxes": 1477},
        "confidence": 0.25,
        "model_order": list(sources),
    }
