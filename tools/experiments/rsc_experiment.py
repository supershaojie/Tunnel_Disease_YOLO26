"""Explicit immutable experiment identity for the shared RSC training and delivery lifecycle."""

from dataclasses import dataclass

from tools.experiments import b19_common as common
from ultralytics.nn.modules import C2PSA_RSC, C2PSA_RSC_V2


@dataclass(frozen=True)
class RSCExperiment:
    """Bind output names, model YAML and required checkpoint class without global patching."""

    version: int
    block_type: type

    @property
    def name(self):
        """Return the isolated run identity."""
        return f"yolo26n_b19_rsc_c2psa_v{self.version}"

    @property
    def model(self):
        """Return the version-specific architecture."""
        return common.ROOT / f"ultralytics/cfg/models/26/yolo26n-rsc-c2psa-v{self.version}.yaml"


V1 = RSCExperiment(1, C2PSA_RSC)
V2 = RSCExperiment(2, C2PSA_RSC_V2)
EXPERIMENTS = {1: V1, 2: V2}
