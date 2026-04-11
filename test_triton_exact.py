import torch

def exact_python_sim(table_vals, table_dib, table_gams, inc_vals, inc_gams, stride, C, a, k=24):
    B = table_vals.shape[0]
    
    p_vals = inc_vals.clone().view(B, stride, C)
    p_gams = inc_gams.clone().view(B, stride, C)
    p_base_dib = torch.zeros((B, stride, C), dtype=torch.long)
    
    # Initial pre-alignment: row r is shifted right by (r * a) % C
    for b in range(B):
        for r in range(stride):
            shift = (r * a) % C
            p_vals[b, r] = torch.roll(p_vals[b, r], shifts=shift, dims=0)
            p_gams[b, r] = torch.roll(p_gams[b, r], shifts=shift, dims=0)
    
    t_vals = table_vals.clone()
    t_dib = table_dib.clone()
    t_gams = table_gams.clone()
    
    for step in range(k):
        # 1. Compute effective DIB
        p_eff_dib = step + p_base_dib
        
        # Mask out empty pipeline slots (values == 0)
        active_mask = (p_vals != 0.0)
        
        # 2. Extract max across stride per bucket
        # Priority: Absolute magnitude, then Effective DIB
        # Since we want to select indices argmax, we need a composite score.
        score = p_vals.abs() * 10000.0 + p_eff_dib.float()
        score[~active_mask] = -1.0
        
        max_score, max_indices = torch.max(score, dim=1) # [B, C]
        
        # Gather winning pipeline attributes
        win_vals = torch.gather(p_vals, 1, max_indices.unsqueeze(1)).squeeze(1) # [B, C]
        win_gams = torch.gather(p_gams, 1, max_indices.unsqueeze(1)).squeeze(1)
        win_eff_dib = torch.gather(p_eff_dib, 1, max_indices.unsqueeze(1)).squeeze(1)
        win_active = torch.gather(active_mask, 1, max_indices.unsqueeze(1)).squeeze(1)
        
        # 3. Compare with table incumbents
        t_active = (t_vals != 0.0) & (t_dib >= 0)
        t_score = t_vals.abs() * 10000.0 + t_dib.float()
        t_score[~t_active] = -1.0
        
        win_score = max_score
        
        # Mask where pipeline beats table
        update_mask = (win_score > t_score) & win_active
        
        for b in range(B):
            for c in range(C):
                if update_mask[b, c]:
                    # Displaced incumbent
                    disp_vals = t_vals[b, c].clone()
                    disp_gams = t_gams[b, c].clone()
                    disp_dib = t_dib[b, c].clone()
                    
                    # Update table
                    t_vals[b, c] = win_vals[b, c]
                    t_gams[b, c] = win_gams[b, c]
                    t_dib[b, c] = win_eff_dib[b, c]
                    
                    # Scatter-swap: Put displaced into pipeline slot
                    r_win = max_indices[b, c]
                    p_vals[b, r_win, c] = disp_vals
                    p_gams[b, r_win, c] = disp_gams
                    p_base_dib[b, r_win, c] = disp_dib - step
                else:
                    # If the pipeline slot didn't win (either lost to table or another pipeline element),
                    # we must ONLY clear it if it was the max element that just lost AND we want to drop it?
                    # Wait: if it lost to the table natively, it means it's smaller than incumbent.
                    # Should it continue probing? Yes! All elements that didn't win should continue probing!
                    # Wait, no. If it DID win the pipeline max, but lost to table, it stays in pipeline.
                    # If it successfully wrote to table and didn't displace anything, it leaves pipeline (zeroed).
                    r_win = max_indices[b, c]
                    if win_active[b, c] and win_score[b, c] > t_score[b, c] and t_vals[b,c] == 0:
                         p_vals[b, r_win, c] = 0.0
                         p_gams[b, r_win, c] = 0.0
                         p_base_dib[b, r_win, c] = 0
                         
        # 4. Roll the pipeline by 1 step
        p_vals = torch.roll(p_vals, shifts=1, dims=2)
        p_gams = torch.roll(p_gams, shifts=1, dims=2)
        p_base_dib = torch.roll(p_base_dib, shifts=1, dims=2)
        
    return t_vals, t_dib, t_gams

