"""Reuse the shared comparison runtime implementation."""

import sys

import common  # noqa: F401 - Initialize the experiment before importing shared modules.
from shared import runtime as _implementation

sys.modules[__name__] = _implementation
