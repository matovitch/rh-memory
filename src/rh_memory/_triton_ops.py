import torch
import triton
import triton.language as tl
from jaxtyping import Float, Int
from torch import Tensor
from ._python_ops import _get_permutation

@triton.jit
def _exact_rh_write_kernel(
    table_values_ptr,
    table_dib_ptr,
    table_gamma_ptr,
    inc_vals_ptr,
    inc_gams_ptr,
    inc_dib_ptr,
    stride,
    C,
    n,
    k,
    BLOCK_SIZE_C: tl.constexpr,
    BLOCK_SIZE_STRIDE: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    
    # We map the entire C array into a single block to safely use tl.debug_barrier
    c_offsets = tl.arange(0, BLOCK_SIZE_C)
    c_mask = c_offsets < C

    table_offset = batch_idx * C + c_offsets
    
    current_out_vals = tl.load(table_values_ptr + table_offset, mask=c_mask, other=0.0)
    current_out_dib  = tl.load(table_dib_ptr + table_offset, mask=c_mask, other=0)
    current_out_gams = tl.load(table_gamma_ptr + table_offset, mask=c_mask, other=0.0)

    inc_base_offset = batch_idx * stride * C

    r_offsets = tl.arange(0, BLOCK_SIZE_STRIDE)
    r_mask = r_offsets < stride

    for step in range(k):
        # Calculate targeting alignment for this step
        in_c_offsets = (c_offsets[None, :] - step % C + C) % C
        
        inc_offsets = inc_base_offset + r_offsets[:, None] * C + in_c_offsets
        inc_mask = r_mask[:, None] & c_mask[None, :]
        
        # Load active pipeline elements
        inc_vals = tl.load(inc_vals_ptr + inc_offsets, mask=inc_mask, other=0.0)
        inc_gams = tl.load(inc_gams_ptr + inc_offsets, mask=inc_mask, other=0.0)
        inc_base_dib = tl.load(inc_dib_ptr + inc_offsets, mask=inc_mask, other=0)
        
        # 1. Virtual DIB tracker eval
        p_eff_dib = step + inc_base_dib
        
        inc_vals_abs = tl.abs(inc_vals)
        active_mask_pipe = (inc_vals != 0.0) & inc_mask
        
        # 32-bit Mantissa Truncation: 
        # Erase bottom 5 bits of the float32 int representation and inject 5-bit DIB.
        val_int32 = inc_vals_abs.to(tl.int32, bitcast=True)
        val_int32_clean = val_int32 & -32 # 0xFFFFFFE0 is -32 in signed 32-bit
        dib_int32 = p_eff_dib.to(tl.int32)
        score = val_int32_clean | (dib_int32 & 0x1F)
        
        # Inactive elements get -1 (negative), strictly less than any valid positive bit-packed score
        score = tl.where(active_mask_pipe, score, tl.full([1], -1, dtype=tl.int32))
        
        # 2. Max reduction across stride mapping
        max_score, max_r_indices = tl.max(score, axis=0, return_indices=True) # type: ignore
        
        # Find winning pipeline slot offsets
        winner_in_c_offsets = (c_offsets - step % C + C) % C
        winner_inc_offsets = inc_base_offset + max_r_indices * C + winner_in_c_offsets
        
        winner_vals = tl.load(inc_vals_ptr + winner_inc_offsets, mask=c_mask, other=0.0)
        winner_gams = tl.load(inc_gams_ptr + winner_inc_offsets, mask=c_mask, other=0.0)
        winner_dib  = tl.load(inc_dib_ptr + winner_inc_offsets, mask=c_mask, other=0)
        
        win_eff_dib = step + winner_dib # DIB is effectively steps elapsed + offset
        
        # Compare vs incumbents
        t_active = (current_out_vals != 0.0) | (current_out_dib >= 0)
        
        current_out_vals_abs = tl.abs(current_out_vals)
        t_val_int32 = current_out_vals_abs.to(tl.int32, bitcast=True)
        t_val_int32_clean = t_val_int32 & -32 # 0xFFFFFFE0 is -32
        t_dib_int32 = current_out_dib.to(tl.int32)
        
        t_score = t_val_int32_clean | (t_dib_int32 & 0x1F)
        t_score = tl.where(t_active, t_score, tl.full([1], -1, dtype=tl.int32))
        
        # Ensure we don't attempt to swap if the winner was actually an empty padded element
        win_active = (winner_vals != 0.0)
        update_mask = (max_score > t_score) & c_mask & win_active
        
        # 3. The Swap
        disp_vals = current_out_vals
        disp_gams = current_out_gams
        disp_dib = current_out_dib
        
        current_out_vals = tl.where(update_mask, winner_vals, current_out_vals)
        current_out_gams = tl.where(update_mask, winner_gams, current_out_gams)
        current_out_dib = tl.where(update_mask, win_eff_dib, current_out_dib)
        
        # Scatter displaced incumbents back to the exact pipeline slot that won
        tl.store(inc_vals_ptr + winner_inc_offsets, disp_vals, mask=update_mask)
        tl.store(inc_gams_ptr + winner_inc_offsets, disp_gams, mask=update_mask)
        tl.store(inc_dib_ptr + winner_inc_offsets, disp_dib - step, mask=update_mask)
        
        # MUST sync warps because the displaced items might be read by other warps in step+1
        tl.debug_barrier()

    tl.store(table_values_ptr + table_offset, current_out_vals, mask=c_mask)
    tl.store(table_dib_ptr + table_offset, current_out_dib, mask=c_mask)
    tl.store(table_gamma_ptr + table_offset, current_out_gams, mask=c_mask)


def triton_exact_parallel_rh(
    table_values: Float[Tensor, "batch capacity"],
    table_dib: Int[Tensor, "batch capacity"],
    table_gamma: Float[Tensor, "batch capacity"],
    incoming_values: Float[Tensor, "batch n"],
    incoming_gammas: Float[Tensor, "batch n"],
    k: int,
    seed: int = 42,
) -> tuple[Float[Tensor, "batch capacity"], Int[Tensor, "batch capacity"], Float[Tensor, "batch capacity"]]:
    if table_values.dim() != 2:
        raise ValueError("table_values must be shaped (B, C)")
    if incoming_values.dim() != 2:
        raise ValueError("incoming_values must be shaped (B, n)")

    batch_size, n = incoming_values.shape
    C = table_values.size(1)
    if n % C != 0:
        raise ValueError("n must be divisible by C")
    stride = n // C

    # Apply permutation to simulate hashing
    perm = _get_permutation(n, seed, incoming_values.device).unsqueeze(0).expand(batch_size, -1)
    incoming_values = torch.gather(incoming_values, 1, perm)
    incoming_gammas = torch.gather(incoming_gammas, 1, perm)

    # Clone explicitly for scattered mutations
    inc_vals = incoming_values.view(batch_size, stride, C).clone()
    inc_gams = incoming_gammas.view(batch_size, stride, C).clone()
    # Explicit zero-initialized DIB offset tensor for exact RH tracking
    inc_dib = torch.zeros((batch_size, stride, C), dtype=torch.int32, device=inc_vals.device)

    out_values = table_values.clone().contiguous()
    out_dib = table_dib.clone().contiguous()
    if out_dib.dtype != torch.int32:
        out_dib = out_dib.to(torch.int32)
    out_gamma = table_gamma.clone().contiguous()
    
    BLOCK_SIZE_STRIDE = triton.next_power_of_2(stride)
    BLOCK_SIZE_C = triton.next_power_of_2(C)
    
    grid = (batch_size,)

    _exact_rh_write_kernel[grid](
        out_values,
        out_dib,
        out_gamma,
        inc_vals,
        inc_gams,
        inc_dib,
        stride,
        C,
        n,
        k,
        BLOCK_SIZE_C=BLOCK_SIZE_C,
        BLOCK_SIZE_STRIDE=BLOCK_SIZE_STRIDE,
        num_warps=4,
    )

    return out_values, out_dib, out_gamma
