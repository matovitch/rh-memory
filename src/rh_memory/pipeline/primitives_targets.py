from __future__ import annotations

import torch
from jaxtyping import Bool, Int
from torch import Tensor


def surrogate_teacher_bucket_slot_indices(
    out_slot_id: Int[Tensor, "B C"],
    n_seq: int,
    valid_bucket: Bool[Tensor, "B C"] | None = None,
) -> tuple[Int[Tensor, "B C"], Bool[Tensor, "B C"]]:
    """
    LPAP teacher targets in bucket-major index layout.

    For each bucket ``c``, return the permuted slot index ``j`` from ``out_slot_id[b, c]``
    plus a validity mask. Invalid targets are clamped only to satisfy CE APIs.
    """
    if valid_bucket is None:
        valid_bucket = (out_slot_id >= 0) & (out_slot_id < n_seq)
    target_idx = out_slot_id.long().clamp(0, n_seq - 1)
    return target_idx, valid_bucket.bool()