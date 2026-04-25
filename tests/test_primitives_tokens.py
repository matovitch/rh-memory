from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from pipeline.primitives_tokens import (
    reconstructor_tokens_from_decoder_logits,
    reshape_permuted_to_bucket_tokens,
    surrogate_bucket_tokens_from_logits,
)


def test_reshape_permuted_to_bucket_tokens_shape():
    B, n, C = 2, 12, 3
    x_perm = torch.arange(B * n, dtype=torch.float32).view(B, n)
    tokens = reshape_permuted_to_bucket_tokens(x_perm, C)
    assert tokens.shape == (B, C, n // C)


def test_reshape_permuted_requires_divisibility():
    with pytest.raises(ValueError):
        _ = reshape_permuted_to_bucket_tokens(torch.randn(2, 10), C=4)


def test_surrogate_bucket_tokens_argmax_j_and_dib_norm():
    B, n, C = 1, 6, 3
    x_perm = torch.arange(n, dtype=torch.float32).unsqueeze(0).expand(B, -1)
    logits = torch.zeros(B, C, n)
    logits[0, 1, 4] = 10.0
    logits[0, 0, 2] = 10.0
    logits[0, 2, 5] = 10.0
    k_eff = 3
    bt, j_star = surrogate_bucket_tokens_from_logits(x_perm, logits, C, k_eff)
    assert bt.shape == (B, C, 2)
    assert torch.equal(j_star, torch.tensor([[2, 4, 5]], dtype=torch.long))
    for c in range(C):
        j = j_star[0, c].item()
        assert bt[0, c, 0].item() == x_perm[0, j].item()
        d = (c - (j % C)) % C
        assert abs(bt[0, c, 1].item() - (float(d) / float(k_eff))) < 1e-6


def test_surrogate_bucket_tokens_requires_positive_k_eff():
    with pytest.raises(ValueError):
        _ = surrogate_bucket_tokens_from_logits(torch.randn(1, 8), torch.randn(1, 4, 8), 4, 0)


def test_reconstructor_tokens_from_decoder_logits_alignment():
    B, C, n = 1, 3, 6
    x_perm = torch.tensor([[10.0, 11.0, 12.0, 13.0, 14.0, 15.0]])
    perm = torch.tensor([2, 4, 0, 5, 1, 3], dtype=torch.long)
    logits = torch.zeros(B, C, n)
    logits[0, 0, 3] = 2.5
    logits[0, 1, 0] = 1.5
    logits[0, 2, 4] = 0.5

    tokens, j_star, src_idx = reconstructor_tokens_from_decoder_logits(x_perm, logits, perm)
    assert tokens.shape == (B, C, 3)
    assert torch.equal(j_star, torch.tensor([[3, 0, 4]], dtype=torch.long))

    expected_src_idx_norm = torch.tensor(
        [[2.0 * (5.0 / 5.0) - 1.0, 2.0 * (2.0 / 5.0) - 1.0, 2.0 * (1.0 / 5.0) - 1.0]]
    )
    assert torch.allclose(src_idx, expected_src_idx_norm, atol=1e-6)
    assert tokens[0, 0, 0].item() == 13.0
    assert tokens[0, 1, 0].item() == 10.0
    assert tokens[0, 2, 0].item() == 14.0
    assert tokens[0, 0, 2].item() == 2.5
    assert tokens[0, 1, 2].item() == 1.5
    assert tokens[0, 2, 2].item() == 0.5


def test_reconstructor_tokens_shape_mismatch_raises():
    with pytest.raises(ValueError):
        _ = reconstructor_tokens_from_decoder_logits(
            torch.randn(2, 8),
            torch.randn(2, 4, 7),
            torch.arange(7),
        )
