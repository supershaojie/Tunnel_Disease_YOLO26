"""Focused numerical invariants for bounded reciprocal-imbalance logit correction."""

import math

import torch

from ultralytics.nn.modules.rsc_c2psa_v2 import reciprocal_logit_correction


def probability_checks(device="cpu"):
    """Check direction, head broadcasting, finite extremes, row normalization, and connected gradients."""
    torch.manual_seed(42)
    scores = torch.randn(3, 2, 17, 17, device=device, requires_grad=True)
    theta = torch.tensor([math.log(0.05 / 0.95), 4.0], device=device, requires_grad=True)
    a, d, delta = reciprocal_logit_correction(scores, theta)
    beta = 0.2 * theta.sigmoid()
    # Independent double-precision row-normalizer calculation, without relying on the implementation helper.
    logits = scores.double()
    log_a = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    expected_d = torch.empty_like(logits)
    for batch in range(3):
        for head in range(2):
            for i in range(17):
                for j in range(17):
                    expected_d[batch, head, i, j] = torch.tanh(
                        0.5 * (log_a[batch, head, j, i] - log_a[batch, head, i, j])
                    )
            separate = reciprocal_logit_correction(scores[batch : batch + 1, head : head + 1], theta[head : head + 1])
            torch.testing.assert_close(a[batch, head], separate[0][0, 0], atol=0, rtol=0)
    torch.testing.assert_close(d, expected_d.float())
    torch.testing.assert_close(delta, (expected_d * beta.double().view(1, 2, 1, 1)).float())
    torch.testing.assert_close(a, (logits + expected_d * beta.double().view(1, 2, 1, 1)).softmax(-1).float())
    torch.testing.assert_close(d, -d.transpose(-2, -1), atol=0, rtol=0)
    assert d.diagonal(dim1=-2, dim2=-1).count_nonzero() == 0
    assert (delta.abs() <= beta.view(1, 2, 1, 1)).all()
    assert torch.equal(d.sign(), (log_a.transpose(-2, -1) - log_a).sign().float())
    assert (a >= 0).all() and torch.isfinite(a).all() and not torch.allclose(a, a.transpose(-2, -1))
    torch.testing.assert_close(a.sum(-1), torch.ones_like(a.sum(-1)))
    (a * torch.randn_like(a)).sum().backward()
    for gradient in (scores.grad, theta.grad):
        assert gradient is not None and torch.isfinite(gradient).all() and gradient.count_nonzero()
    uniform_theta = torch.full((2,), math.log(0.05 / 0.95), device=device, requires_grad=True)
    uniform = torch.zeros(3, 2, 17, 17, device=device, requires_grad=True)
    flat, flat_d, flat_delta = reciprocal_logit_correction(uniform, uniform_theta)
    torch.testing.assert_close(flat, uniform.softmax(-1), atol=0, rtol=0)
    assert flat_d.count_nonzero() == flat_delta.count_nonzero() == 0
    (flat * torch.randn_like(flat)).sum().backward()
    assert uniform_theta.grad is not None and uniform_theta.grad.count_nonzero() == 0
    extreme = (
        torch.tensor(
            [[65504.0, -65504.0, -1000.0], [-2000.0, 65504.0, -65504.0], [0.0, 60000.0, -60000.0]], device=device
        )
        .half()
        .repeat(3, 2, 1, 1)
        .requires_grad_()
    )
    extreme_theta = theta.detach().clone().requires_grad_()
    stable = reciprocal_logit_correction(extreme, extreme_theta)
    for value in stable:
        assert value.dtype == torch.float32 and torch.isfinite(value).all()
    torch.testing.assert_close(stable[0].sum(-1), torch.ones(3, 2, 3, device=device))
    assert (stable[2].abs() <= beta.view(1, 2, 1, 1)).all()
    (stable[0] * torch.randn_like(stable[0])).sum().backward()
    assert torch.isfinite(extreme.grad).all() and torch.isfinite(extreme_theta.grad).all()
    return dict(
        formula_direction=True,
        batch_head_isolation=True,
        nonnegative=True,
        unit_rows=True,
        D_antisymmetric=True,
        D_zero_diagonal=True,
        logit_bound=True,
        uniform_zero_gradient_allowed=True,
        finite_half_extremes=True,
        fp32_correction=True,
        calibration_gradient_connected=True,
    )
