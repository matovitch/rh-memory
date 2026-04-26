from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rh_memory.pipeline.primitives_tokens import (
    decoder_tokens_from_surrogate_logits_soft,
    normalized_entropy,
    reconstructor_tokens_from_decoder_logits_hard,
    reconstructor_tokens_from_decoder_logits_soft,
    reshape_permuted_to_bucket_tokens,
    scalar_dib_table,
)


def test_reshape_permuted_to_bucket_tokens_shape():
    B, n, C = 2, 12, 3
    x_perm = torch.arange(B * n, dtype=torch.float32).view(B, n)
    tokens = reshape_permuted_to_bucket_tokens(x_perm, C)
    assert tokens.shape == (B, C, n // C)


def test_reshape_permuted_requires_divisibility():
    with pytest.raises(ValueError):
        _ = reshape_permuted_to_bucket_tokens(torch.randn(2, 10), C=4)


def test_scalar_dib_table_is_zero_one_normalized():
    table = scalar_dib_table(3, 6, device=torch.device("cpu"), dtype=torch.float32)
    expected = torch.tensor(
        [
            [0.0, 1.0, 0.5, 0.0, 1.0, 0.5],
            [0.5, 0.0, 1.0, 0.5, 0.0, 1.0],
            [1.0, 0.5, 0.0, 1.0, 0.5, 0.0],
        ]
    )
    assert torch.allclose(table, expected)
    assert table.min().item() >= 0.0
    assert table.max().item() <= 1.0


def test_normalized_entropy_uniform_and_certain():
    uniform = torch.full((2, 4), 0.25)
    certain = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    assert torch.allclose(normalized_entropy(uniform), torch.ones(2), atol=1e-6)
    assert torch.allclose(normalized_entropy(certain), torch.zeros(1), atol=1e-6)


def test_decoder_tokens_from_surrogate_logits_soft_uniform():
    B, C, n = 1, 2, 4
    x_perm = torch.tensor([[1.0, 3.0, 5.0, 7.0]])
    logits = torch.zeros(B, C, n)

    tokens = decoder_tokens_from_surrogate_logits_soft(x_perm, logits)
    assert tokens.shape == (B, C, 3)
    assert torch.allclose(tokens[..., 0], torch.full((B, C), 4.0))
    assert torch.allclose(tokens[..., 1], torch.full((B, C), 0.5))
    assert torch.allclose(tokens[..., 2], torch.ones(B, C), atol=1e-6)


def test_reconstructor_tokens_from_decoder_logits_soft_uniform():
    B, C, n = 1, 2, 4
    x_perm = torch.tensor([[1.0, 3.0, 5.0, 7.0]])
    perm_1d = torch.tensor([2, 0, 3, 1])
    logits = torch.zeros(B, C, n)

    tokens = reconstructor_tokens_from_decoder_logits_soft(x_perm, perm_1d, logits)
    assert tokens.shape == (B, C, 3)
    assert torch.allclose(tokens[..., 0], torch.full((B, C), 4.0))
    assert torch.allclose(tokens[..., 1], torch.full((B, C), 0.5))
    assert torch.allclose(tokens[..., 2], torch.ones(B, C), atol=1e-6)


def test_reconstructor_tokens_from_decoder_logits_hard_uses_argmax_source_and_bucket_amplitude():
    B, C, n = 1, 2, 4
    x_perm = torch.tensor([[1.0, 3.0, 5.0, 7.0]])
    perm_1d = torch.tensor([2, 0, 3, 1])
    logits = torch.zeros(B, C, n)
    logits[0, 0, 3] = 10.0
    logits[0, 1, 1] = 10.0
    bucket_amplitude = torch.tensor([[11.0, -13.0]])

    tokens = reconstructor_tokens_from_decoder_logits_hard(
        x_perm,
        perm_1d,
        logits,
        bucket_amplitude=bucket_amplitude,
    )

    assert tokens.shape == (B, C, 3)
    assert torch.allclose(tokens[..., 0], bucket_amplitude)
    assert torch.allclose(tokens[..., 1], torch.tensor([[1.0 / 3.0, 0.0]]))
    assert torch.all(tokens[..., 2] < 0.01)


def test_soft_token_helpers_preserve_gradient_flow():
    B, C, n = 1, 2, 4
    x_perm = torch.tensor([[1.0, 3.0, 5.0, 7.0]])
    surrogate_logits = torch.randn(B, C, n, requires_grad=True)
    decoder_logits = torch.randn(B, C, n, requires_grad=True)
    perm_1d = torch.tensor([2, 0, 3, 1])

    decoder_tokens = decoder_tokens_from_surrogate_logits_soft(x_perm, surrogate_logits)
    reconstructor_tokens = reconstructor_tokens_from_decoder_logits_soft(x_perm, perm_1d, decoder_logits)
    loss = decoder_tokens.sum() + reconstructor_tokens.sum()
    loss.backward()

    assert surrogate_logits.grad is not None
    assert decoder_logits.grad is not None


def test_decoder_tokens_shape_mismatch_raises():
    with pytest.raises(ValueError):
        _ = decoder_tokens_from_surrogate_logits_soft(
            torch.randn(2, 8),
            torch.randn(2, 4, 7),
        )


def test_reconstructor_tokens_shape_mismatch_raises():
    with pytest.raises(ValueError):
        _ = reconstructor_tokens_from_decoder_logits_soft(
            torch.randn(1, 8),
            torch.arange(7),
            torch.randn(1, 4, 8),
        )


def test_reconstructor_tokens_hard_shape_mismatch_raises():
    with pytest.raises(ValueError):
        _ = reconstructor_tokens_from_decoder_logits_hard(
            torch.randn(1, 8),
            torch.arange(7),
            torch.randn(1, 4, 8),
        )
