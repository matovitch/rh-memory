from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rh_memory.flow_models import DilatedConvFlow1d


def test_dilated_conv_flow_shape_and_gradients() -> None:
    model = DilatedConvFlow1d(
        sequence_length=64,
        width=16,
        time_dim=16,
        dilation_cycles=1,
        dilations=(1, 2, 4),
    )
    x = torch.randn(3, 1, 64, requires_grad=True)
    t = torch.tensor([0.1, 0.5, 0.9])

    velocity = model(x, t)
    loss = velocity.square().mean()
    loss.backward()

    assert velocity.shape == x.shape
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert all(param.grad is not None for param in model.parameters() if param.requires_grad)


def test_symmetric_flow_objective_shapes() -> None:
    model = DilatedConvFlow1d(
        sequence_length=32,
        width=8,
        time_dim=8,
        dilation_cycles=1,
        dilations=(1, 2),
    )
    start = torch.randn(2, 1, 32)
    end = torch.randn(2, 1, 32)
    t = torch.tensor([0.25, 0.75])
    x_t = (1.0 - t.view(2, 1, 1)) * start + t.view(2, 1, 1) * end
    target_velocity = end - start

    pred_velocity = model(x_t, t)
    loss = F.mse_loss(pred_velocity, target_velocity)

    assert pred_velocity.shape == target_velocity.shape
    assert loss.dim() == 0
    assert torch.isfinite(loss)