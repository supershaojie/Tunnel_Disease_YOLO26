"""Reuse the shared comparison measure implementation."""

import sys

import common  # noqa: F401 - Initialize the experiment before importing shared modules.
from shared import measure as _implementation

sys.modules[__name__] = _implementation
