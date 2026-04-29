from __future__ import annotations

import torch

from rh_memory.pipeline.primitives_targets import surrogate_teacher_bucket_slot_indices


def test_surrogate_teacher_bucket_slot_indices_basic():
    B, C = 2, 4
    n_seq = 8
    out_slot_id = torch.tensor([[3, 1, 5, 7], [0, 2, 4, 6]], dtype=torch.int32)
    idx, valid = surrogate_teacher_bucket_slot_indices(out_slot_id, n_seq)
    assert idx.shape == (B, C)
    assert valid.shape == (B, C)
    assert torch.equal(idx, out_slot_id.long())
    assert valid.all()


def test_surrogate_teacher_bucket_slot_indices_respects_mask():
    out_slot_id = torch.tensor([[1, 2]], dtype=torch.int32)
    valid = torch.tensor([[True, False]])
    idx, mask = surrogate_teacher_bucket_slot_indices(out_slot_id, n_seq=4, valid_bucket=valid)
    assert torch.equal(idx, out_slot_id.long())
    assert torch.equal(mask, valid)


def test_surrogate_teacher_bucket_slot_indices_clamps_invalid_targets():
    out_slot_id = torch.tensor([[-1, 9, 3]], dtype=torch.int32)
    idx, valid = surrogate_teacher_bucket_slot_indices(out_slot_id, n_seq=8)
    assert torch.equal(idx, torch.tensor([[0, 7, 3]], dtype=torch.long))
    assert torch.equal(valid, torch.tensor([[False, False, True]]))
