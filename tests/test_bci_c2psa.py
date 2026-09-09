"""Independent BCI mathematics and experiment failure boundaries."""

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from tools.experiments.finish_b19_bci_c2psa import recall_curve
from tools.experiments.verify_b19_bci_c2psa import branch_checks
from ultralytics.nn.modules import C2PSA_BCI
from ultralytics.nn.modules.bci_c2psa import BCI


def test_independent_small_formula():
    """Exercise an asymmetric scalar float64 oracle and constant low-precision inputs."""
    assert branch_checks()["independent_float64_reference"]


def test_projection_sources_and_rectangle():
    """Compare the full residual with separate float64 algebra and check the bypass conditioning."""
    torch.manual_seed(7)
    module = BCI()
    torch.nn.init.normal_(module.po.weight, std=0.01)
    a, b = torch.randn(2, 128, 7, 11), torch.randn(2, 128, 7, 11)

    def reference(a):
        q = module.dwq(module.pq(a)).flatten(2).double()
        k = module.dwk(module.pk(b)).flatten(2).double()
        v = module.pv(b).flatten(2).double()
        q -= q.mean(-1, keepdim=True)
        k -= k.mean(-1, keepdim=True)
        q /= q.square().sum(-1, keepdim=True).sqrt().clamp_min(1e-6)
        k /= k.square().sum(-1, keepdim=True).sqrt().clamp_min(1e-6)
        logits = 4 * torch.einsum("bcn,bdn->bcd", q, k)
        attention = (logits - logits.amax(-1, keepdim=True)).exp()
        attention /= attention.sum(-1, keepdim=True)
        delta = torch.einsum("bcd,bdn->bcn", attention, v) - v
        return F.conv2d(delta.reshape(2, 32, 7, 11).float(), module.po.weight), attention

    original = a.clone()
    expected, attention = reference(a)
    torch.testing.assert_close(module(a, b), expected, atol=2e-7, rtol=2e-5)
    changed = a + torch.randn_like(a)
    _, changed_attention = reference(changed)
    assert not torch.equal(attention, changed_attention)
    assert torch.equal(a, original)
    assert not list(module.buffers())


@pytest.mark.parametrize("args", [(128, 128), (512, 512), (256, 256, 2), (256, 256, 1, 0.25)])
def test_reject_unverified_shapes(args):
    """Unverified channel or repeat settings must not silently select a different design."""
    with pytest.raises(ValueError, match="requires nano"):
        C2PSA_BCI(*args)


def test_recall_ties_and_frozen_test_threshold():
    """A tied false positive cannot be dropped to reach the target precision; test cannot select a threshold."""
    raw = {"conf": np.array([0.9, 0.8, 0.8]), "tp": np.array([True, True, False]), "targets": 2}
    result = recall_curve(raw, 2)
    assert len(result["curve"]) == 2
    assert result["val_selected"]["threshold"] == 0.9
    test = recall_curve(raw, 2, threshold=0.9, select=False)
    assert test["val_selected"] is None
    assert test["frozen_threshold_result"]["recall"] == 0.5
    raw["tp"][:] = False
    assert recall_curve(raw, 2)["val_selected"] is None
