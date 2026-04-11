import torch
import math
import sys
from pathlib import Path

# Ensure rh_memory is accessible
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from rh_memory._python_ops import python_fast_rh_write_batched
from rh_memory._triton_ops import triton_exact_parallel_rh # type: ignore

def main():
    torch.manual_seed(42)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    n = 1024
    C = 128
    chunk_size = 8192
    exact_k = C
    
    # Create base sine waves (B, n) with random frequencies and phases
    t = torch.linspace(0, 1, n).unsqueeze(0).expand(chunk_size, n).to(device)
    freqs1 = torch.empty(chunk_size, 1, device=device).uniform_(2.0, 15.0)
    phases1 = torch.empty(chunk_size, 1, device=device).uniform_(0.0, 2 * math.pi)
    sine_waves = 1 - torch.sin(2 * math.pi * freqs1 * t + phases1).abs()
    
    # Add high frequency noise and magnitude envelope
    mag_env = torch.empty(chunk_size, 1, device=device).uniform_(0.5, 6.0)
    sign = torch.where(torch.rand(chunk_size, n, device=device) < 0.5, -1.0, 1.0)
    noise = torch.randn(chunk_size, n, device=device) * 0.0
    signal = (sine_waves + noise) * mag_env * sign
    
    # Create block-sparse mask
    blocks = n // 4
    mask = torch.zeros(chunk_size, n, device=device)
    rand_idx_in_block = torch.randint(0, 4, (chunk_size, blocks), device=device)
    col_offsets = torch.arange(blocks, device=device).unsqueeze(0) * 4
    abs_keep_idx = col_offsets + rand_idx_in_block
    mask.scatter_(1, abs_keep_idx, 1.0)
    
    raw_inputs = signal * mask
    
    # Start state bounds
    table_values = torch.zeros(chunk_size, C, dtype=torch.float32, device=device)
    table_dib = torch.zeros(chunk_size, C, dtype=torch.long if device == "cpu" else torch.int32, device=device)
    table_gamma = torch.zeros(chunk_size, C, dtype=torch.float32, device=device)
    
    incoming_gammas = torch.arange(n, dtype=torch.float32, device=device).unsqueeze(0).expand(chunk_size, n)
    
    print(f"Running exact parallel pooling with n={n}, C={C}, batch_size={chunk_size}, exact_k={exact_k}...")
    
    if device == "cuda":
        out_values, out_dib, out_gamma = triton_exact_parallel_rh(
            table_values,
            table_dib,
            table_gamma,
            raw_inputs,
            incoming_gammas,
            a=17,
            k=exact_k,
        )
    else:
        out_values, out_dib, out_gamma = python_fast_rh_write_batched(
            table_values,
            table_dib,
            table_gamma,
            raw_inputs,
            incoming_gammas,
            a=17,
            k=exact_k,
        )
    
    active_mask = out_values.abs() > 1e-7
    valid_dibs = out_dib[active_mask].float()
    
    print(f"Total active slots filled: {active_mask.sum().item()} out of {chunk_size * C}")
    print("DIB Distribution Quantiles:")
    
    quantiles = torch.tensor([0.5, 0.75, 0.9, 0.95, 0.99, 1.0], device=device)
    computed = torch.quantile(valid_dibs, quantiles)
    
    for q, v in zip(quantiles, computed):
        print(f"  {int(q.item()*100):>3}% DIB <= {v.item():.0f}")

if __name__ == "__main__":
    main()
