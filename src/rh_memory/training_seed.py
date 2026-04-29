"""Shared seed handling for training entrypoints."""

from __future__ import annotations

import secrets
from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

import torch


@dataclass(frozen=True)
class TrainingSeed:
    seed: int
    source: str


def resolve_training_seed(
    argument_seed: int | None,
    checkpoint: Mapping[str, object] | None = None,
) -> TrainingSeed:
    if argument_seed is not None:
        return TrainingSeed(int(argument_seed), "argument")
    if checkpoint is not None:
        if "seed" in checkpoint:
            return TrainingSeed(_seed_to_int(checkpoint["seed"]), "checkpoint")
        config = checkpoint.get("config")
        if isinstance(config, Mapping) and "seed" in config:
            config_mapping = cast(Mapping[str, object], config)
            return TrainingSeed(_seed_to_int(config_mapping["seed"]), "checkpoint config")
    return TrainingSeed(secrets.randbits(63), "system randomness")


def apply_training_seed(
    argument_seed: int | None,
    checkpoint: Mapping[str, object] | None = None,
) -> TrainingSeed:
    training_seed = resolve_training_seed(argument_seed, checkpoint)
    torch.manual_seed(training_seed.seed)
    return training_seed


def _seed_to_int(value: object) -> int:
    if isinstance(value, int | str):
        return int(value)
    raise TypeError(f"seed must be an int or stringified int, got {type(value).__name__}")
