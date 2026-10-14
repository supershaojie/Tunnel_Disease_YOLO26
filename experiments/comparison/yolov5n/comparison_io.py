"""Atomic provenance files and explicit loading for trusted private v5 checkpoints."""

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def checksum(path, provenance):
    path = Path(path)
    value = {**provenance, "sha256": digest(path), "bytes": path.stat().st_size}
    write_json(str(path) + ".source.json", value)
    return value


def verify_weights(path):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Weights missing: {path}; run prepare.py first")
    manifest = read_json(str(path) + ".source.json")
    if manifest.get("model") != "yolov5n" or manifest.get("kind") not in (
        "official_pretrained",
        "comparison_checkpoint",
    ):
        raise ValueError(f"Wrong or unverified checkpoint identity: {path}")
    actual = digest(path)
    if actual != manifest.get("sha256") or path.stat().st_size != manifest.get("bytes"):
        raise ValueError(f"Weight integrity mismatch: {path}; expected {manifest.get('sha256')}, actual {actual}")
    return manifest


def trusted_load(path, map_location="cpu", **kwargs):
    import torch

    verify_weights(path)
    return torch.load(path, map_location=map_location, weights_only=False, **kwargs)


def now():
    return datetime.now(timezone.utc).isoformat()


def status(directory, phase, state, **details):
    write_json(
        Path(directory) / f"{phase}_status.json",
        {"phase": phase, "state": state, "pid": os.getpid(), "time": now(), **details},
    )
