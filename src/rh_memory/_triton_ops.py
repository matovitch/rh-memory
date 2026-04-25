import torch
import triton
import triton.language as tl
from jaxtyping import Float, Int
from torch import Tensor

from ._python_ops import _reshape_incoming_to_pipeline, _validate_lpap_dtypes


@triton.jit
def _linear_probing_amplitude_pool_kernel(
    table_values_ptr,
    table_dib_ptr,
    table_carry_id_ptr,
    inc_vals_ptr,
    inc_carry_id_ptr,
    inc_dib_ptr,
    stride,
    C,
    k: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
    BLOCK_SIZE_STRIDE: tl.constexpr,
):
    batch_idx = tl.program_id(0)

    c_offsets = tl.arange(0, BLOCK_SIZE_C)
    c_mask = c_offsets < C

    table_offset = batch_idx * C + c_offsets

    current_out_vals = tl.load(table_values_ptr + table_offset, mask=c_mask, other=0.0)
    current_out_dib = tl.load(table_dib_ptr + table_offset, mask=c_mask, other=0)
    current_out_carry = tl.load(table_carry_id_ptr + table_offset, mask=c_mask, other=-1)

    inc_base_offset = batch_idx * stride * C

    r_offsets = tl.arange(0, BLOCK_SIZE_STRIDE)
    r_mask = r_offsets < stride

    for step in range(k):
        in_c_offsets = (c_offsets[None, :] - step % C + C) % C

        inc_offsets = inc_base_offset + r_offsets[:, None] * C + in_c_offsets
        inc_mask = r_mask[:, None] & c_mask[None, :]

        inc_vals = tl.load(inc_vals_ptr + inc_offsets, mask=inc_mask, other=0.0)
        inc_carry = tl.load(inc_carry_id_ptr + inc_offsets, mask=inc_mask, other=-1)
        inc_base_dib = tl.load(inc_dib_ptr + inc_offsets, mask=inc_mask, other=0)

        inc_vals_abs = tl.abs(inc_vals)
        # padded stride lanes: -1.0 so they never win tl.max over axis=0
        inc_for_max = tl.where(inc_mask, inc_vals_abs, -1.0)
        _, max_r_indices = tl.max(inc_for_max, axis=0, return_indices=True)  # type: ignore

        winner_in_c_offsets = (c_offsets - step % C + C) % C
        winner_inc_offsets = inc_base_offset + max_r_indices * C + winner_in_c_offsets

        winner_vals = tl.load(inc_vals_ptr + winner_inc_offsets, mask=c_mask, other=0.0)
        winner_carry = tl.load(inc_carry_id_ptr + winner_inc_offsets, mask=c_mask, other=-1)
        winner_dib = tl.load(inc_dib_ptr + winner_inc_offsets, mask=c_mask, other=0)

        win_eff_dib = step + winner_dib

        winner_abs = tl.abs(winner_vals)
        t_abs_cmp = tl.abs(current_out_vals)

        update_mask = (winner_abs >= t_abs_cmp) & c_mask

        disp_vals = current_out_vals
        disp_carry = current_out_carry
        disp_dib = current_out_dib

        current_out_vals = tl.where(update_mask, winner_vals, current_out_vals)
        current_out_carry = tl.where(update_mask, winner_carry, current_out_carry)
        current_out_dib = tl.where(update_mask, win_eff_dib, current_out_dib)

        tl.store(inc_vals_ptr + winner_inc_offsets, disp_vals, mask=update_mask)
        tl.store(inc_carry_id_ptr + winner_inc_offsets, disp_carry, mask=update_mask)
        tl.store(inc_dib_ptr + winner_inc_offsets, disp_dib - step, mask=update_mask)

    tl.store(table_values_ptr + table_offset, current_out_vals, mask=c_mask)
    tl.store(table_dib_ptr + table_offset, current_out_dib, mask=c_mask)
    tl.store(table_carry_id_ptr + table_offset, current_out_carry, mask=c_mask)


def triton_linear_probing_amplitude_pooling(
    table_values: Float[Tensor, "batch capacity"],
    table_dib: Int[Tensor, "batch capacity"],
    table_carry_id: Int[Tensor, "batch capacity"],
    incoming_values: Float[Tensor, "batch n"],
    incoming_carry_id: Int[Tensor, "batch n"],
    k: int,
) -> tuple[Float[Tensor, "batch capacity"], Int[Tensor, "batch capacity"], Int[Tensor, "batch capacity"]]:
    """
    Triton LPAP: same semantics as :func:`python_linear_probing_amplitude_pooling`.

    Same dtype contract: **``torch.float32``** values, **``torch.int32``** integer tables and
    ``incoming_carry_id``. **Incoming** is not modified. ``incoming_values`` is expected to already be
    shuffled; ``incoming_carry_id`` may be any position-aligned payload (e.g. shuffled source ids or
    contiguous slot ids ``0..n-1``).
    The kernel writes
    contiguous staging buffers when needed and copies back—**ownership is an optimization hint**, not
    a mandate to mutate every buffer in place.
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

    inc_vals, inc_carry = _reshape_incoming_to_pipeline(
        incoming_values, incoming_carry_id, batch_size, stride
    )

    inc_dib = torch.zeros((batch_size, stride, C), dtype=torch.int32, device=inc_vals.device)

    vals_arg = table_values.contiguous()
    dib_arg = table_dib.contiguous()
    carry_arg = table_carry_id.contiguous()

    need_vals_copy = vals_arg.data_ptr() != table_values.data_ptr()
    need_dib_copy = dib_arg.data_ptr() != table_dib.data_ptr()
    need_carry_copy = carry_arg.data_ptr() != table_carry_id.data_ptr()

    BLOCK_SIZE_STRIDE = triton.next_power_of_2(stride)
    BLOCK_SIZE_C = triton.next_power_of_2(C)

    grid = (batch_size,)

    _linear_probing_amplitude_pool_kernel[grid](
        vals_arg,
        dib_arg,
        carry_arg,
        inc_vals,
        inc_carry,
        inc_dib,
        stride,
        C,
        k,
        BLOCK_SIZE_C=BLOCK_SIZE_C,
        BLOCK_SIZE_STRIDE=BLOCK_SIZE_STRIDE,
        num_warps=4,
    )

    if need_vals_copy:
        table_values.copy_(vals_arg)
    if need_dib_copy:
        table_dib.copy_(dib_arg)
    if need_carry_copy:
        table_carry_id.copy_(carry_arg)

    return table_values, table_dib, table_carry_id
