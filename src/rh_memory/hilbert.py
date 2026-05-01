"""Hilbert-order utilities for square image tensors."""

from __future__ import annotations

import math
from functools import lru_cache

import torch
from torch import Tensor


def _rot(side: int, x: int, y: int, rx: int, ry: int) -> tuple[int, int]:
    if ry == 0:
        if rx == 1:
            x = side - 1 - x
            y = side - 1 - y
        x, y = y, x
    return x, y


def _hilbert_index_to_xy(side: int, index: int) -> tuple[int, int]:
    x = 0
    y = 0
    t = index
    step = 1
    while step < side:
        rx = 1 & (t // 2)
        ry = 1 & (t ^ rx)
        x, y = _rot(step, x, y, rx, ry)
        x += step * rx
        y += step * ry
        t //= 4
        step *= 2
    return x, y


def _validate_side(side: int) -> None:
    if side <= 0:
        raise ValueError(f"side must be positive, got {side}")
    if side & (side - 1):
        raise ValueError(f"side must be a power of two, got {side}")


@lru_cache(maxsize=None)
def _hilbert_order_tuple(side: int) -> tuple[int, ...]:
    _validate_side(side)
    order: list[int] = []
    for index in range(side * side):
        x, y = _hilbert_index_to_xy(side, index)
        order.append(y * side + x)
    if sorted(order) != list(range(side * side)):
        raise RuntimeError("Hilbert order generation did not produce a valid permutation")
    return tuple(order)


def hilbert_permutation(
    side: int = 32,
    *,
    device: torch.device | str | None = None,
) -> Tensor:
    """Return raster-flat indices in Hilbert sequence order."""

    return torch.tensor(_hilbert_order_tuple(side), dtype=torch.long, device=device)


def inverse_permutation(perm: Tensor) -> Tensor:
    """Return inverse permutation such that ``values[..., perm][..., inverse] == values``."""

    inverse = torch.empty_like(perm)
    inverse[perm] = torch.arange(perm.numel(), dtype=perm.dtype, device=perm.device)
    return inverse


def inverse_hilbert_permutation(
    side: int = 32,
    *,
    device: torch.device | str | None = None,
) -> Tensor:
    """Return sequence indices in raster-flat order for inverse Hilbert mapping."""

    return inverse_permutation(hilbert_permutation(side, device=device))


def hilbert_flatten_images(images: Tensor, *, side: int = 32) -> Tensor:
    """Flatten ``[B, 1, side, side]`` images to Hilbert-ordered ``[B, 1, side*side]``."""

    expected = (1, side, side)
    if tuple(images.shape[1:]) != expected:
        raise ValueError(
            f"images must have shape [B, {expected[0]}, {expected[1]}, {expected[2]}], got {tuple(images.shape)}"
        )
    flat = images.reshape(images.shape[0], 1, side * side)
    return flat.index_select(-1, hilbert_permutation(side, device=images.device))


def hilbert_unflatten_images(sequence: Tensor, *, side: int = 32) -> Tensor:
    """Invert ``hilbert_flatten_images`` from ``[B, 1, side*side]`` to image tensors."""

    expected = (1, side * side)
    if tuple(sequence.shape[1:]) != expected:
        raise ValueError(f"sequence must have shape [B, {expected[0]}, {expected[1]}], got {tuple(sequence.shape)}")
    raster = sequence.index_select(-1, inverse_hilbert_permutation(side, device=sequence.device))
    return raster.reshape(sequence.shape[0], 1, side, side)


def hilbert_metadata(side: int = 32) -> dict[str, int | str]:
    _validate_side(side)
    return {
        "order": "hilbert",
        "side": side,
        "sequence_length": side * side,
        "curve_order": int(math.log2(side)),
    }
