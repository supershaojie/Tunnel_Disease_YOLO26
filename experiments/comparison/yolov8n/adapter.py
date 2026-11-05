"""Reuse the shared comparison adapter implementation."""

import sys

import common  # noqa: F401 - Initialize the experiment before importing shared modules.
from shared import adapter as _implementation

sys.modules[__name__] = _implementation
