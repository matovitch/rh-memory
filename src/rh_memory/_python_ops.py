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

def python_exact_parallel_rh(
    table_values: Float[Tensor, "batch capacity"],
    table_dib: Int[Tensor, "batch capacity"],
    table_gamma: Float[Tensor, "batch capacity"],
    incoming_values: Float[Tensor, "batch n"],
    incoming_gammas: Float[Tensor, "batch n"],
    k: int,
    seed: int = 42,
) -> tuple[Float[Tensor, "batch capacity"], Int[Tensor, "batch capacity"], Float[Tensor, "batch capacity"]]:
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

    # Apply permutation to simulate hashing
    perm = _get_permutation(n, seed, incoming_values.device).unsqueeze(0).expand(batch_size, -1)
    incoming_values = torch.gather(incoming_values, dim=1, index=perm)
    incoming_gammas = torch.gather(incoming_gammas, dim=1, index=perm)

    # Shape: [B, stride, C]
    inc_vals = incoming_values.view(batch_size, stride, C).clone()
    inc_gams = incoming_gammas.view(batch_size, stride, C).clone()
    
    # Virtual DIB Tracker (base offsets)
    inc_base_dib = torch.zeros((batch_size, stride, C), dtype=torch.long, device=inc_vals.device)

    out_values = table_values .clone()
    out_dib    = table_dib    .clone()
    out_gamma  = table_gamma  .clone()

    for step in range(k):
        # 1. Compute effective DIB for pipeline
        p_eff_dib = step + inc_base_dib
        
        # We only consider non-zero pipeline slots
        active_mask = (inc_vals != 0.0)
        
        # 2. Lexicographic max reduction over stride (magnitudes primary, DIB secondary)
        inc_vals_abs = inc_vals.abs()
        best_val_abs = torch.full  ((batch_size, C), -1.0                  , device=inc_vals.device)
        best_dib     = torch.full  ((batch_size, C), -1 , dtype=torch.long , device=inc_vals.device)
        max_indices  = torch.zeros ((batch_size, C)     , dtype=torch.long , device=inc_vals.device)
        
        for r in range(stride):
            curr_val_abs = inc_vals_abs [:, r, :]
            curr_dib     = p_eff_dib    [:, r, :]
            curr_active  = active_mask  [:, r, :]
            
            # Empty slots are treated as -1 magnitude so they never win against active ones
            curr_eff_abs = torch.where(curr_active, curr_val_abs, torch.tensor(-1.0, device=inc_vals.device))
            
            update = (curr_eff_abs > best_val_abs) | ((curr_eff_abs == best_val_abs) & (curr_dib > best_dib))
            
            max_indices  = torch.where(update, r, max_indices)
            best_val_abs = torch.where(update, curr_eff_abs, best_val_abs)
            best_dib     = torch.where(update, curr_dib, best_dib)
            
        win_vals    = torch.gather(inc_vals    , dim=1, index=max_indices.unsqueeze(1)).squeeze(1) # [B, C]
        win_gams    = torch.gather(inc_gams    , dim=1, index=max_indices.unsqueeze(1)).squeeze(1)
        win_eff_dib = torch.gather(p_eff_dib   , dim=1, index=max_indices.unsqueeze(1)).squeeze(1)
        win_active  = torch.gather(active_mask , dim=1, index=max_indices.unsqueeze(1)).squeeze(1)
        
        # 3. Compare with incumbents using lexicographic ordering
        t_active = (out_values != 0.0) | (out_dib >= 0)
        
        win_abs = win_vals.abs()
        t_abs = torch.where(t_active, out_values.abs(), torch.tensor(-1.0, device=out_values.device))
        
        update_mask = win_active & ((win_abs > t_abs) | ((win_abs == t_abs) & (win_eff_dib > out_dib)))
        
        disp_vals = out_values .clone()
        disp_gams = out_gamma  .clone()
        disp_dib  = out_dib    .clone()
        
        # 4. Write winners to table
        out_values = torch.where(update_mask, win_vals, out_values)
        out_gamma  = torch.where(update_mask, win_gams, out_gamma)
        out_dib    = torch.where(update_mask, win_eff_dib, out_dib)
        
        # 5. Handle scatter swapping
        b_idx = torch.arange(batch_size , device=inc_vals.device).unsqueeze(1).expand(-1, C          )[update_mask]
        c_idx = torch.arange(C          , device=inc_vals.device).unsqueeze(0).expand(batch_size, -1 )[update_mask]
        r_win = max_indices[update_mask]
        
        # Determine elements to zero out (winners that updated the table but where the table was EMPTY)
        # Actually, if we just ALWAYS scatter `disp_vals`, and `disp_vals` was 0.0 for an empty table,
        # it intrinsically puts 0.0 into the pipeline, making it effectively empty!
        # There is no need for a special zeroing logic for successful non-displacing updates.
        inc_vals     [b_idx, r_win, c_idx] = disp_vals [update_mask]
        inc_gams     [b_idx, r_win, c_idx] = disp_gams [update_mask]
        inc_base_dib [b_idx, r_win, c_idx] = disp_dib  [update_mask] - step
        
        # For elements that won their column but LOST to the table (~update_mask),
        # we do NOTHING: they remain in the pipeline untouched and will naturally roll.
        
        # 6. Roll the pipeline circularly
        if step < k - 1:
            inc_vals     = torch.roll(inc_vals     , shifts=1, dims=2)
            inc_gams     = torch.roll(inc_gams     , shifts=1, dims=2)
            inc_base_dib = torch.roll(inc_base_dib , shifts=1, dims=2)
            
    return out_values, out_dib, out_gamma
