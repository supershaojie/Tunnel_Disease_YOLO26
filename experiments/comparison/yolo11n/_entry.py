"""Select YOLO11n configuration before importing the reused comparison runtime."""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def setup():
    """Reuse the locked v8.3.0 implementation in this independent process."""
    sys.path.insert(0, str(HERE.parent / "yolov8n"))
    import common

    common.configure(HERE)
    return common


def main(mode):
    """Dispatch an experiment entry using the existing production implementation."""
    setup()
    from runtime import main as dispatch

    return dispatch(mode)
