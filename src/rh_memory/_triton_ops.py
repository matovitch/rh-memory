import torch
import triton
import triton.language as tl

@triton.jit
def _fast_rh_write_kernel(
    table_values_ptr,
    table_dib_ptr,
    table_gamma_ptr,
    inc_vals_ptr,
    inc_gams_ptr,
    stride,
    C,
    a,
    k,
    BLOCK_SIZE_C: tl.constexpr,
    BLOCK_SIZE_STRIDE: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    c_block_idx = tl.program_id(1)

    c_offsets = c_block_idx * BLOCK_SIZE_C + tl.arange(0, BLOCK_SIZE_C)
    c_mask = c_offsets < C

    table_offset = batch_idx * C + c_offsets
    
    current_out_vals = tl.load(table_values_ptr + table_offset, mask=c_mask, other=0.0)
    current_out_dib  = tl.load(table_dib_ptr + table_offset, mask=c_mask, other=0)
    current_out_gams = tl.load(table_gamma_ptr + table_offset, mask=c_mask, other=0.0)

    # Calculate base array pointers for this batch
    inc_base_offset = batch_idx * stride * C

    r_offsets = tl.arange(0, BLOCK_SIZE_STRIDE)
    r_mask = r_offsets < stride

    # The simulation essentially computes for each step `s` of k
    for step in range(k):
        # We need to find the maximum among elements in `inc_vals` that are targeting `c`.
        # At step `s`, an incoming element `(r, in_c)` targets `(in_c + (r*a)%C + s) % C`.
        # If it targets `out_c` (which is `c_offsets`), then `out_c = (in_c + (r*a)%C + s) % C`.
        # Therefore, the element to load is `in_c = (out_c - (r*a)%C - s) % C`.
        
        shift = (r_offsets * a) % C
        
        in_c_offsets = (c_offsets[None, :] - shift[:, None] - step + 2 * C) % C
        
        # Calculate linear index into the incoming arrays
        # row major: r * C + in_c
        inc_offsets = inc_base_offset + r_offsets[:, None] * C + in_c_offsets
        
        inc_mask = r_mask[:, None] & c_mask[None, :]
        
        # Load the elements
        inc_vals = tl.load(inc_vals_ptr + inc_offsets, mask=inc_mask, other=0.0)
        inc_gams = tl.load(inc_gams_ptr + inc_offsets, mask=inc_mask, other=0.0)
        
        inc_vals_abs = tl.abs(inc_vals)
        
        # Now find the maximum across the stride dimension `dim=0`.
        inc_vals_abs_masked = tl.where(r_mask[:, None], inc_vals_abs, -1.0)
        
        # We need both the max magnitude and its index `r` to load the actual values and gammas
        max_abs_vals, max_r_indices = tl.max(inc_vals_abs_masked, axis=0, return_indices=True) # type: ignore
        
        # Reconstruct the winner offsets using the max_r_indices
        winner_in_c_offsets = (c_offsets - ((max_r_indices * a) % C) - step + 2 * C) % C
        winner_inc_offsets = inc_base_offset + max_r_indices * C + winner_in_c_offsets
        
        winner_vals = tl.load(inc_vals_ptr + winner_inc_offsets, mask=c_mask, other=0.0)
        winner_gams = tl.load(inc_gams_ptr + winner_inc_offsets, mask=c_mask, other=0.0)
        
        # Compare with incumbents
        current_abs = tl.abs(current_out_vals)
        mask = (max_abs_vals > current_abs) & c_mask

        # If winner is selected, we must update the output table
        current_out_vals = tl.where(mask, winner_vals, current_out_vals)
        current_out_gams = tl.where(mask, winner_gams, current_out_gams)
        current_out_dib = tl.where(mask, tl.full((BLOCK_SIZE_C,), step, dtype=tl.int32), current_out_dib)

        # In Python implementation:
        # inc_vals.masked_fill_(is_winning_and_updating, 0.0)
        # inc_gams.masked_fill_(is_winning_and_updating, 0.0)
        # We need to zero out the taken element in memory so it doesn't win again
        tl.store(inc_vals_ptr + winner_inc_offsets, tl.zeros((BLOCK_SIZE_C,), dtype=tl.float32), mask=mask)
        tl.store(inc_gams_ptr + winner_inc_offsets, tl.zeros((BLOCK_SIZE_C,), dtype=tl.float32), mask=mask)
        # However, tl.store to global memory might affect other blocks in the SAME step if they load it.
        # But each incoming element belongs to exactly ONE bucket at any given step.
        # It's targeting exactly ONE `out_c` at step `s`.
        # So the `max` computation over different `out_c` are finding maximums of entirely disjoint sets of elements!
        # Because for a given `r` and `s`, `(in_c + (r*a)%C + s) % C` maps `in_c` one-to-one to `out_c`.
        # Thus, writing 0.0 back to global memory is perfectly safe AND correct because no other `c_block_idx` will need that element THIS step,
        # and subsequent steps WILL need to see that it was zeroed.

    tl.store(table_values_ptr + table_offset, current_out_vals, mask=c_mask)
    tl.store(table_dib_ptr + table_offset, current_out_dib, mask=c_mask)
    tl.store(table_gamma_ptr + table_offset, current_out_gams, mask=c_mask)


def triton_fast_rh_write_batched(
    table_values: torch.Tensor,
    table_dib: torch.Tensor,
    table_gamma: torch.Tensor,
    incoming_values: torch.Tensor,
    incoming_gammas: torch.Tensor,
    a: int,
    k: int,
):
    if table_values.dim() != 2:
        raise ValueError("table_values must be shaped (B, C)")
    if incoming_values.dim() != 2:
        raise ValueError("incoming_values must be shaped (B, n)")

    batch_size, n = incoming_values.shape
    C = table_values.size(1)
    if n % C != 0:
        raise ValueError("n must be divisible by C")
    stride = n // C

    # Since the kernel modifies `incoming_values` and `gammas` (writes 0.0 to winning items),
    # and the original python function modifies the local copied view `inc_vals`, we need to clone them.
    # The python version reshapes it to (B, stride, C).
    inc_vals = incoming_values.view(batch_size, stride, C).clone()
    inc_gams = incoming_gammas.view(batch_size, stride, C).clone()

    out_values = table_values.clone()
    out_dib = table_dib.clone()
    out_gamma = table_gamma.clone()
    
    # Needs to be a power of 2 for Triton 
    BLOCK_SIZE_STRIDE = triton.next_power_of_2(stride)
    BLOCK_SIZE_C = 64 # Let's keep it sane to avoid shared memory limits if stride is large
    
    grid = (batch_size, triton.cdiv(C, BLOCK_SIZE_C))

    _fast_rh_write_kernel[grid](
        out_values,
        out_dib,
        out_gamma,
        inc_vals,
        inc_gams,
        batch_size,
        stride,
        C,
        a,
        k,
        BLOCK_SIZE_C=BLOCK_SIZE_C, # type: ignore
        BLOCK_SIZE_STRIDE=BLOCK_SIZE_STRIDE,
        num_warps=4, # type: ignore
    )

    return out_values, out_dib, out_gamma
