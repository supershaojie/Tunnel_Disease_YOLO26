"""Fixed NDP identity consumed by the reused b19 lifecycle."""

from types import SimpleNamespace

from tools.experiments import b19_common as common
from ultralytics.nn.modules import SPPF_NDP

V1 = SimpleNamespace(version=1, name="yolo26n_b19_ndp_sppf_v1", model=common.MODEL, block_type=SPPF_NDP)
