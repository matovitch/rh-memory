from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rh_memory.pipeline.primitives_permutation import (
    build_grouped_permutation,
    gather_permuted_stream,
    unpermute_from_permuted,
)


def grouped_bucket_collision_counts(perm_1d: torch.Tensor, C: int) -> torch.Tensor:
    bucket_ids = perm_1d.long() % C
    return torch.bincount(bucket_ids, minlength=C)


def test_build_grouped_permutation_balance_and_determinism():
    n, C = 24, 6
    p1 = build_grouped_permutation(n, C, seed=7, device="cpu")
    p2 = build_grouped_permutation(n, C, seed=7, device="cpu")
    assert torch.equal(p1, p2)
    assert p1.shape == (n,)
    counts = grouped_bucket_collision_counts(p1, C)
    assert torch.all(counts == n // C)


def test_build_grouped_permutation_requires_divisibility():
    with pytest.raises(ValueError):
        _ = build_grouped_permutation(10, 4, seed=1, device="cpu")


def test_gather_and_unpermute_roundtrip():
    B, n, C = 2, 12, 3
    x = torch.arange(B * n, dtype=torch.float32).view(B, n)
    perm = build_grouped_permutation(n, C, seed=11, device="cpu")
    x_perm = gather_permuted_stream(x, perm)
    x_back = unpermute_from_permuted(x_perm, perm)
    assert torch.equal(x_back, x)


def test_unpermute_length_mismatch_raises():
    with pytest.raises(ValueError):
        _ = unpermute_from_permuted(torch.randn(2, 8), torch.arange(7))
