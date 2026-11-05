"""Select this experiment's locked backend for the shared comparison workflow."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared import common as _implementation  # noqa: E402

_implementation.configure(Path(__file__).resolve().parent)
sys.modules[__name__] = _implementation
