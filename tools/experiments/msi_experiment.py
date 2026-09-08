"""Fixed MSI identity consumed by the reused b19 lifecycle."""

from types import SimpleNamespace

from tools.experiments import b19_common as common
from ultralytics.nn.modules import C2PSA_MSI

V1 = SimpleNamespace(version=1, name="yolo26n_b19_msi_c2psa_v1", model=common.MODEL, block_type=C2PSA_MSI)
