"""Single fixed experiment identity with an explicit n-scale YAML filename."""

from types import SimpleNamespace

from tools.experiments.b19_common import MODEL
from ultralytics.nn.modules.mpdf_p3 import MPDFP3

V1 = SimpleNamespace(version=1, name="yolo26n_b19_mpdf_p3_v1", model=MODEL, block_type=MPDFP3)
