"""Fixed BDI identity consumed by the reused b19 lifecycle."""

from types import SimpleNamespace

from tools.experiments import b19_common as common
from ultralytics.nn.modules import Concat_BDI_P3

V1 = SimpleNamespace(version=1, name="yolo26n_b19_bdi_p3_v1", model=common.MODEL, block_type=Concat_BDI_P3)
