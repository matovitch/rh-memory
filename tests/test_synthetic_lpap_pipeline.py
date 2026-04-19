"""Tests for synthetic LPAP / surrogate pipeline helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from synthetic_lpap_pipeline import (
    decoder_targets_from_j_star,
    max_padded_length,
    surrogate_bucket_tokens_from_logits,
    surrogate_teacher_one_hot,
)
from synthetic_lpap_pipeline import inv_perm_from_perm


def test_max_padded_length():
    assert max_padded_length(1024, 128) == 1024
    assert max_padded_length(1000, 128) == 1024


def test_inv_perm_roundtrip():
    perm = torch.tensor([2, 0, 3, 1], dtype=torch.long)
    inv = inv_perm_from_perm(perm)
    n = perm.numel()
    for j in range(n):
        assert perm[inv[perm[j]]].item() == perm[j].item()
    for s in range(n):
        assert perm[inv[s]].item() == s


def test_surrogate_teacher_one_hot_one_hot_rows():
    B, C = 2, 4
    n_pad = 8
    perm = torch.arange(n_pad, dtype=torch.long)
    out_carry_id = torch.tensor([[3, 1, 5, 7], [0, 2, 4, 6]], dtype=torch.int32)
    t = surrogate_teacher_one_hot(out_carry_id, perm)
    assert t.shape == (B, n_pad, C)
    for b in range(B):
        for j in range(n_pad):
            assert t[b, j].sum().item() in (0.0, 1.0)


def test_decoder_targets_from_j_star():
    j_star = torch.tensor([[3, 1, 7, 2], [0, 5, 4, 6]], dtype=torch.long)
    n_seq = 16
    tgt = decoder_targets_from_j_star(j_star, n_seq)
    assert tgt.shape == (2, 4, n_seq)
    B, Cb, _ = tgt.shape
    for b in range(B):
        for c in range(Cb):
            j = int(j_star[b, c])
            assert tgt[b, c].sum().item() == 1.0
            assert tgt[b, c, j].item() == 1.0


def test_surrogate_bucket_tokens_argmax_j_and_dib():
    B, n, C = 1, 6, 3
    x_perm = torch.arange(n, dtype=torch.float32).unsqueeze(0).expand(B, -1)
    logits = torch.zeros(B, n, C)
    logits[0, 4, 1] = 10.0
    logits[0, 2, 0] = 10.0
    logits[0, 5, 2] = 10.0
    bt, j_star = surrogate_bucket_tokens_from_logits(x_perm, logits, C)
    assert bt.shape == (B, C, 2)
    assert j_star[0, 0].item() == 2
    assert j_star[0, 1].item() == 4
    assert j_star[0, 2].item() == 5
    for c in range(C):
        j = j_star[0, c].item()
        assert bt[0, c, 0].item() == x_perm[0, j].item()
        d = (c - (j % C)) % C
        assert bt[0, c, 1].item() == float(d)
