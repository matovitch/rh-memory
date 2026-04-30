"""Flow checkpoint helpers for training and evaluation scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import torch
from path_utils import resolve_project_path

from rh_memory.flow_models import DilatedConvFlow1d


Direction = Literal["image-to-energy", "energy-to-image"]

SUPPORTED_FLOW_CHECKPOINT_VERSIONS = {"v1_directional_flow_matching", "v1_directional_flow_reflow"}


def require_flow_checkpoint_keys(flow_ckpt: dict[str, Any], checkpoint_path: Path) -> None:
    required_keys = [
        "config",
        "flow_model_config",
        "surrogate_checkpoint",
        "soft_scatter_checkpoint",
        "direction",
        "state_dict",
    ]
    missing = [key for key in required_keys if key not in flow_ckpt]
    if missing:
        raise KeyError(f"checkpoint {checkpoint_path} is missing required keys: {', '.join(missing)}")


def load_flow_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    path = resolve_project_path(path)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    version = ckpt.get("version")
    if version not in SUPPORTED_FLOW_CHECKPOINT_VERSIONS:
        raise ValueError(f"Unsupported flow checkpoint version {version!r}")
    require_flow_checkpoint_keys(ckpt, path)
    ckpt["_resolved_path"] = path
    return ckpt


def checkpoint_direction(ckpt: dict[str, Any]) -> Direction | None:
    direction = ckpt.get("direction")
    if direction is None:
        return None
    if direction not in ("image-to-energy", "energy-to-image"):
        raise ValueError(f"Unsupported checkpoint direction {direction!r}")
    return direction


def state_dict_for_direction(ckpt: dict[str, Any], direction: Direction) -> dict[str, torch.Tensor]:
    ckpt_direction = checkpoint_direction(ckpt)
    if ckpt_direction != direction:
        raise ValueError(f"checkpoint direction {ckpt_direction!r} does not match requested {direction!r}")
    return ckpt["state_dict"]


def build_flow_model(
    config: dict[str, Any], state_dict: dict[str, torch.Tensor], device: torch.device
) -> DilatedConvFlow1d:
    model = DilatedConvFlow1d(
        sequence_length=int(config["sequence_length"]),
        width=int(config["width"]),
        time_dim=int(config["time_dim"]),
        dilation_cycles=int(config["dilation_cycles"]),
        dilations=tuple(int(dilation) for dilation in config["dilations"]),
        kernel_size=int(config.get("kernel_size", 3)),
    ).to(device)
    model.load_state_dict(state_dict)
    return model