from __future__ import annotations

import torch
from torch import Tensor


def surrogate_teacher_bucket_slot_targets(
    out_slot_id: Tensor,
    n_seq: int,
    valid_bucket: Tensor | None = None,
) -> Tensor:
    """
    LPAP teacher targets in bucket-major layout: ``[B, C, n_seq]``.

    For each bucket ``c``, mark one permuted slot index ``j`` from ``out_slot_id[b, c]`` when valid.
    """
    B, C = out_slot_id.shape
    device = out_slot_id.device
    if valid_bucket is None:
        valid_bucket = (out_slot_id >= 0) & (out_slot_id < n_seq)
    targets = torch.zeros(B, C, n_seq, dtype=torch.float32, device=device)
    idx = out_slot_id.long().clamp(0, n_seq - 1)
    targets.scatter_(2, idx.unsqueeze(2), valid_bucket.unsqueeze(2).float())
    return targets


def decoder_targets_from_j_star(
    j_star: Tensor,
    n_seq: int,
    valid_bucket: Tensor | None = None,
) -> Tensor:
    """
    Sparse ``[B, C, n_seq]`` one-hot at ``j_star[b, c]``.
    """
    B, Cb = j_star.shape
    device = j_star.device
    if valid_bucket is None:
        valid_bucket = torch.ones(B, Cb, dtype=torch.bool, device=device)
    targets = torch.zeros(B, Cb, n_seq, dtype=torch.float32, device=device)
    j_clamped = j_star.long().clamp(0, n_seq - 1)
    targets.scatter_(2, j_clamped.unsqueeze(2).long(), valid_bucket.unsqueeze(2).float())
    return targets
