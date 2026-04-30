from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from flow_checkpoints import checkpoint_direction, state_dict_for_direction
from train_flow_reflow import sample_time

from rh_memory.flow_matching import flow_matching_loss
from rh_memory.flow_models import DilatedConvFlow1d


def test_sample_time_uniform_range() -> None:
    args = SimpleNamespace(time_distribution="uniform", eps=0.1, time_beta_alpha=0.1, time_beta_beta=0.1)

    t = sample_time(32, args, torch.device("cpu"))

    assert t.shape == (32,)
    assert torch.all(t >= 0.1)
    assert torch.all(t <= 0.9)


def test_sample_time_beta_range() -> None:
    args = SimpleNamespace(time_distribution="beta", eps=0.01, time_beta_alpha=0.5, time_beta_beta=0.5)

    t = sample_time(32, args, torch.device("cpu"))

    assert t.shape == (32,)
    assert torch.all(t >= 0.01)
    assert torch.all(t <= 0.99)


def test_reflow_loss_shapes_and_gradients() -> None:
    model = DilatedConvFlow1d(
        sequence_length=32,
        width=8,
        time_dim=8,
        dilation_cycles=1,
        dilations=(1, 2),
    )
    start = torch.randn(2, 1, 32)
    teacher_end = torch.randn(2, 1, 32)
    t = torch.tensor([0.25, 0.75])

    loss, pred_velocity, target_velocity = flow_matching_loss(model, start, teacher_end, t)
    loss.backward()

    assert pred_velocity.shape == target_velocity.shape == start.shape
    assert loss.dim() == 0
    assert torch.isfinite(loss)
    assert all(param.grad is not None for param in model.parameters() if param.requires_grad)


def test_directional_checkpoint_state_dict_lookup() -> None:
    state_dict = {"weight": torch.ones(1)}
    ckpt = {
        "version": "v1_directional_flow_matching",
        "direction": "image-to-energy",
        "state_dict": state_dict,
    }

    assert checkpoint_direction(ckpt) == "image-to-energy"
    assert state_dict_for_direction(ckpt, "image-to-energy") is state_dict


def test_directional_checkpoint_direction_mismatch_raises() -> None:
    ckpt = {
        "version": "v1_directional_flow_reflow",
        "direction": "image-to-energy",
        "state_dict": {"weight": torch.ones(1)},
    }

    try:
        state_dict_for_direction(ckpt, "energy-to-image")
    except ValueError as error:
        assert "does not match" in str(error)
    else:
        raise AssertionError("expected direction mismatch to raise")