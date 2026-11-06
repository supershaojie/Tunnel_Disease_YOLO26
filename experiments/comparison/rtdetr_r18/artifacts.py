"""Reuse the existing result whitelist, adding the C2 source and complexity record."""

import json
from pathlib import Path

import common  # noqa: F401 - Bind this experiment before the shared artifact module.
from common import HERE
from shared import artifacts as _shared
from shared.artifacts import comparisons, figures, freeze_training  # noqa: F401

_base_members = _shared._members


def _members(directory):
    members, metrics, run, benchmark_status = _base_members(directory)
    members["source/backend.lock.json"] = HERE / "backend.lock.json"
    members["source/README.md"] = HERE / "README.md"
    members["source/initialization_source.json"] = run / "initialization_source.json"
    location = run / "complexity_location.json"
    if location.is_file():
        complexity = Path(json.loads(location.read_text(encoding="utf-8"))["directory"]) / "complexity.json"
        record = json.loads(complexity.read_text(encoding="utf-8"))
        if record["weight_sha256"] != metrics["weight_sha256"]:
            raise ValueError("Complexity result belongs to a different checkpoint")
        members["complexity/complexity.json"] = complexity
        members["complexity/complexity_status.json"] = complexity.parent / "complexity_status.json"
    return members, metrics, run, benchmark_status


_shared._members = _members
package = _shared.package
