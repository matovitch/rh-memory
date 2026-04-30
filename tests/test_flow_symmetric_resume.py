from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from train_flow_symmetric import validate_resume_checkpoint_pair


def make_checkpoint(*, direction: str, step: int = 7, version: str = "v1_directional_flow_matching") -> dict:
    return {
        "version": version,
        "direction": direction,
        "step": step,
        "flow_model_config": {
            "sequence_length": 1024,
            "width": 128,
            "time_dim": 128,
            "dilation_cycles": 2,
            "dilations": [1, 2, 4, 8],
            "kernel_size": 3,
        },
    }


def test_validate_resume_checkpoint_pair_returns_matching_step() -> None:
    step = validate_resume_checkpoint_pair(
        make_checkpoint(direction="image-to-energy"),
        make_checkpoint(direction="energy-to-image"),
    )

    assert step == 7


def test_validate_resume_checkpoint_pair_rejects_direction_mismatch() -> None:
    try:
        validate_resume_checkpoint_pair(
            make_checkpoint(direction="energy-to-image"),
            make_checkpoint(direction="energy-to-image"),
        )
    except ValueError as error:
        assert "image-to-energy checkpoint has direction" in str(error)
    else:
        raise AssertionError("expected direction mismatch to raise")


def test_validate_resume_checkpoint_pair_rejects_step_mismatch() -> None:
    try:
        validate_resume_checkpoint_pair(
            make_checkpoint(direction="image-to-energy", step=7),
            make_checkpoint(direction="energy-to-image", step=8),
        )
    except ValueError as error:
        assert "matching steps" in str(error)
    else:
        raise AssertionError("expected step mismatch to raise")


def test_validate_resume_checkpoint_pair_rejects_reflow_version() -> None:
    try:
        validate_resume_checkpoint_pair(
            make_checkpoint(direction="image-to-energy", version="v1_directional_flow_reflow"),
            make_checkpoint(direction="energy-to-image"),
        )
    except ValueError as error:
        assert "Cannot resume image-to-energy base flow" in str(error)
    else:
        raise AssertionError("expected version mismatch to raise")