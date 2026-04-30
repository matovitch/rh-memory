from __future__ import annotations

import torch
import torch.nn as nn

from rh_memory.flow_integration import EulerFlowIntegrator, integrate_euler_midpoint_time


class ScaledVelocity(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.scale * x + t.view(t.shape[0], 1, 1)


def test_euler_integration_preserves_gradients() -> None:
    vector_field = ScaledVelocity()
    start = torch.ones(2, 1, 4, requires_grad=True)

    end = integrate_euler_midpoint_time(vector_field, start, steps=3)
    loss = end.square().mean()
    loss.backward()

    assert end.shape == start.shape
    assert start.grad is not None
    assert vector_field.scale.grad is not None


def test_euler_integrator_module_matches_function() -> None:
    vector_field = ScaledVelocity()
    start = torch.ones(2, 1, 4)

    module_end = EulerFlowIntegrator(vector_field, steps=4)(start)
    function_end = integrate_euler_midpoint_time(vector_field, start, steps=4)

    assert torch.allclose(module_end, function_end)


def test_euler_integration_requires_positive_steps() -> None:
    vector_field = ScaledVelocity()
    start = torch.ones(2, 1, 4)

    try:
        integrate_euler_midpoint_time(vector_field, start, steps=0)
    except ValueError as error:
        assert "steps must be positive" in str(error)
    else:
        raise AssertionError("expected non-positive steps to raise")