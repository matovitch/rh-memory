"""Pure Python RH write helpers."""

from __future__ import annotations

import torch

def python_fast_rh_write_batched(
    table_values: torch.Tensor,
    table_dib: torch.Tensor,
    table_gamma: torch.Tensor,
    incoming_values: torch.Tensor,
    incoming_gammas: torch.Tensor,
    a: int,
    k: int,
):
    """
    Exact Parallel Robin Hood GPU Memory simulator. 
    Implements the Scatter-Swap and Virtual DIB tracking algorithm.
    Used as the PyTorch baseline before full Triton kernel integration.
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

    # Shape: [B, stride, C]
    inc_vals = incoming_values.view(batch_size, stride, C).clone()
    inc_gams = incoming_gammas.view(batch_size, stride, C).clone()
    
    # Virtual DIB Tracker (base offsets)
    inc_base_dib = torch.zeros((batch_size, stride, C), dtype=torch.long, device=inc_vals.device)

    # Initial alignment: row r is shifted right by (r * a) % C
    for r in range(stride):
        shift = (r * a) % C
        if shift != 0:
            inc_vals[:, r, :] = torch.roll(inc_vals[:, r, :], shifts=shift, dims=-1)
            inc_gams[:, r, :] = torch.roll(inc_gams[:, r, :], shifts=shift, dims=-1)

    out_values = table_values.clone()
    out_dib = table_dib.clone()
    out_gamma = table_gamma.clone()

    for step in range(k):
        # 1. Compute effective DIB for pipeline
        p_eff_dib = step + inc_base_dib
        
        # We only consider non-zero pipeline slots
        active_mask = (inc_vals != 0.0)
        
        # 2. Create composite score for max reduction (magnitudes primary, DIB secondary)
        score = inc_vals.abs() * 10000.0 + p_eff_dib.float()
        score[~active_mask] = -1.0 # Ensure empty slots never win
        
        max_score, max_indices = torch.max(score, dim=1) # [B, C]
        
        win_vals = torch.gather(inc_vals, 1, max_indices.unsqueeze(1)).squeeze(1) # [B, C]
        win_gams = torch.gather(inc_gams, 1, max_indices.unsqueeze(1)).squeeze(1)
        win_eff_dib = torch.gather(p_eff_dib, 1, max_indices.unsqueeze(1)).squeeze(1)
        win_active = torch.gather(active_mask, 1, max_indices.unsqueeze(1)).squeeze(1)
        
        # 3. Compare with incumbents
        t_active = (out_values != 0.0) | (out_dib >= 0)
        t_score = out_values.abs() * 10000.0 + out_dib.float()
        t_score[~t_active] = -0.5 
        
        update_mask = (max_score > t_score) & win_active
        
        disp_vals = out_values.clone()
        disp_gams = out_gamma.clone()
        disp_dib = out_dib.clone()
        
        # 4. Write winners to table
        out_values = torch.where(update_mask, win_vals, out_values)
        out_gamma = torch.where(update_mask, win_gams, out_gamma)
        out_dib = torch.where(update_mask, win_eff_dib, out_dib)
        
        # 5. Handle scatter swapping
        b_idx = torch.arange(batch_size, device=inc_vals.device).unsqueeze(1).expand(-1, C)[update_mask]
        c_idx = torch.arange(C, device=inc_vals.device).unsqueeze(0).expand(batch_size, -1)[update_mask]
        r_win = max_indices[update_mask]
        
        # Determine elements to zero out (winners that updated the table but where the table was EMPTY)
        # Actually, if we just ALWAYS scatter `disp_vals`, and `disp_vals` was 0.0 for an empty table,
        # it intrinsically puts 0.0 into the pipeline, making it effectively empty!
        # There is no need for a special zeroing logic for successful non-displacing updates.
        inc_vals[b_idx, r_win, c_idx] = disp_vals[update_mask]
        inc_gams[b_idx, r_win, c_idx] = disp_gams[update_mask]
        inc_base_dib[b_idx, r_win, c_idx] = disp_dib[update_mask] - step
        
        # For elements that won their column but LOST to the table (~update_mask),
        # we do NOTHING: they remain in the pipeline untouched and will naturally roll.
        
        # 6. Roll the pipeline circularly
        if step < k - 1:
            inc_vals = torch.roll(inc_vals, shifts=1, dims=2)
            inc_gams = torch.roll(inc_gams, shifts=1, dims=2)
            inc_base_dib = torch.roll(inc_base_dib, shifts=1, dims=2)
            
    return out_values, out_dib, out_gamma
