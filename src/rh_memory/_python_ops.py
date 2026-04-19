"""Pure Python RH write helpers."""

from __future__ import annotations

import torch
from jaxtyping import Float, Int
from torch import Tensor

_permutation_cache = {}


def _get_permutation(n: int, seed: int, device: torch.device | str) -> Int[Tensor, "n"]:
    key = (n, seed, str(device))
    if key not in _permutation_cache:
        g = torch.Generator(device=device)
        g.manual_seed(seed)
        _permutation_cache[key] = torch.randperm(n, generator=g, device=device)
    return _permutation_cache[key]


def _validate_lpap_dtypes(
    table_values: Tensor,
    table_dib: Tensor,
    table_carry_id: Tensor,
    incoming_values: Tensor,
    incoming_carry_id: Tensor,
) -> None:
    if table_values.dtype != torch.float32:
        raise TypeError("table_values must be torch.float32")
    if incoming_values.dtype != torch.float32:
        raise TypeError("incoming_values must be torch.float32")
    if table_dib.dtype != torch.int32:
        raise TypeError("table_dib must be torch.int32")
    if table_carry_id.dtype != torch.int32:
        raise TypeError("table_carry_id must be torch.int32")
    if incoming_carry_id.dtype != torch.int32:
        raise TypeError("incoming_carry_id must be torch.int32")


def _gather_perm_incoming(
    incoming_values: Tensor,
    incoming_carry_id: Tensor,
    batch_size: int,
    stride: int,
    C: int,
    perm: Tensor,
) -> tuple[Tensor, Tensor]:
    """Permute-gather into owned buffers; does **not** modify the caller's ``incoming_*`` tensors."""
    # Dense row-major (B, n) so ``view(B, stride, C)`` is contiguous; avoid propagating alien strides from ``incoming_*``.
    gather_vals = torch.empty(
        incoming_values.shape,
        dtype=incoming_values.dtype,
        device=incoming_values.device,
    )
    gather_carry = torch.empty(
        incoming_carry_id.shape,
        dtype=incoming_carry_id.dtype,
        device=incoming_carry_id.device,
    )
    torch.gather(incoming_values, 1, perm, out=gather_vals)
    torch.gather(incoming_carry_id, 1, perm, out=gather_carry)
    inc_vals = gather_vals.view(batch_size, stride, C)
    inc_carry = gather_carry.view(batch_size, stride, C)
    return inc_vals, inc_carry


def python_linear_probing_amplitude_pooling(
    table_values      : Float [Tensor, "batch capacity"],
    table_dib         : Int   [Tensor, "batch capacity"],
    table_carry_id    : Int   [Tensor, "batch capacity"],
    incoming_values   : Float [Tensor, "batch n"],
    incoming_carry_id : Int   [Tensor, "batch n"],
    k: int,
    seed: int = 42,
) -> tuple[Float[Tensor, "batch capacity"], Int[Tensor, "batch capacity"], Int[Tensor, "batch capacity"]]:
    """
    Reference **linear-probing-based amplitude pooling** (Python): scatter-swap with virtual DIB.

    All tensors use **``torch.float32``** (``table_values``, ``incoming_values``) and **``torch.int32``**
    (``table_dib``, ``table_carry_id``, ``incoming_carry_id``). **Table** state is updated in place;
    **incoming** is read only; the permuted pipeline is held in internal gather buffers.

    Ordering: **absolute amplitude** with strict ``>`` vs the table; **max over stride** for pipeline
    winners. Zeros participate like any magnitude.

    Allocates gather buffers and ``disp_*`` swap buffers once per call (outside the ``k`` loop).
    """
    if table_values.dim() != 2:
        raise ValueError("table_values must be shaped (B, C)")
    if incoming_values.dim() != 2:
        raise ValueError("incoming_values must be shaped (B, n)")

    batch_size, n = incoming_values.shape
    C = table_values.size(1)
    if n % C != 0:
        raise ValueError("n must be divisible by C")
    stride = n // C

    _validate_lpap_dtypes(table_values, table_dib, table_carry_id, incoming_values, incoming_carry_id)

    perm = _get_permutation(n, seed, incoming_values.device).unsqueeze(0).expand(batch_size, -1)

    inc_vals, inc_carry = _gather_perm_incoming(
        incoming_values, incoming_carry_id, batch_size, stride, C, perm
    )

    inc_base_dib = torch.zeros((batch_size, stride, C), dtype=torch.int32, device=inc_vals.device)

    disp_vals  = torch.empty_like(table_values)
    disp_carry = torch.empty_like(table_carry_id)
    disp_dib   = torch.empty_like(table_dib)

    for step in range(k):
        p_eff_dib = step + inc_base_dib

        inc_vals_abs = inc_vals.abs()
        _, max_indices = inc_vals_abs.max(dim=1)

        win_vals = torch.gather(inc_vals, dim=1, index=max_indices.unsqueeze(1)).squeeze(1)
        win_carry = torch.gather(inc_carry, dim=1, index=max_indices.unsqueeze(1)).squeeze(1)
        win_eff_dib = torch.gather(p_eff_dib, dim=1, index=max_indices.unsqueeze(1)).squeeze(1)

        win_abs = win_vals.abs()
        t_abs = table_values.abs()

        update_mask = win_abs > t_abs

        disp_vals.copy_(table_values)
        disp_carry.copy_(table_carry_id)
        disp_dib.copy_(table_dib)

        table_values[update_mask] = win_vals[update_mask]
        table_carry_id[update_mask] = win_carry[update_mask]
        table_dib[update_mask] = win_eff_dib[update_mask]

        b_idx = torch.arange(batch_size, device=inc_vals.device).unsqueeze(1).expand(-1, C)[update_mask]
        c_idx = torch.arange(C, device=inc_vals.device).unsqueeze(0).expand(batch_size, -1)[update_mask]
        r_win = max_indices[update_mask]

        inc_vals[b_idx, r_win, c_idx] = disp_vals[update_mask]
        inc_carry[b_idx, r_win, c_idx] = disp_carry[update_mask]
        inc_base_dib[b_idx, r_win, c_idx] = disp_dib[update_mask] - step

        if step < k - 1:
            inc_vals.copy_(torch.roll(inc_vals, shifts=1, dims=2))
            inc_carry.copy_(torch.roll(inc_carry, shifts=1, dims=2))
            inc_base_dib.copy_(torch.roll(inc_base_dib, shifts=1, dims=2))

    return table_values, table_dib, table_carry_id
