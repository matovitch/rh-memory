from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from pipeline.primitives_targets import decoder_targets_from_j_star, surrogate_teacher_bucket_slot_targets


def test_surrogate_teacher_bucket_slot_targets_basic():
    B, C = 2, 4
    n_seq = 8
    out_slot_id = torch.tensor([[3, 1, 5, 7], [0, 2, 4, 6]], dtype=torch.int32)
    t = surrogate_teacher_bucket_slot_targets(out_slot_id, n_seq)
    assert t.shape == (B, C, n_seq)
    for b in range(B):
        for c in range(C):
            assert t[b, c].sum().item() in (0.0, 1.0)


def test_surrogate_teacher_bucket_slot_targets_respects_mask():
    out_slot_id = torch.tensor([[1, 2]], dtype=torch.int32)
    valid = torch.tensor([[True, False]])
    t = surrogate_teacher_bucket_slot_targets(out_slot_id, n_seq=4, valid_bucket=valid)
    assert t[0, 0, 1].item() == 1.0
    assert t[0, 1].sum().item() == 0.0


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
