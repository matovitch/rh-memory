import torch
import sys
import math
from pathlib import Path

# Ensure rh_memory is accessible
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rh_memory._cpu_ops import cpu_rh_write_batched

def generate_split(num_samples: int, chunk_size: int, n: int, C: int, out_path: Path):
    print(f"Generating {num_samples} synthetic samples (n={n}, C={C}) for {out_path.name}...")
    
    all_tokens = []
    all_targets = []
    all_magnitudes = []
    all_indices = []
    
    for i in range(0, num_samples, chunk_size):
        if i % (chunk_size * 4) == 0:
            print(f"Generated {i}/{num_samples} samples...")
            
        # Create base sine waves (B, n) with random frequencies and phases
        t = torch.linspace(0, 1, n).unsqueeze(0).expand(chunk_size, n)
        freqs1 = torch.empty(chunk_size, 1).uniform_(2.0, 15.0)
        phases1 = torch.empty(chunk_size, 1).uniform_(0.0, 2 * math.pi)
        sine_waves = torch.sin(2 * math.pi * freqs1 * t + phases1)
        
        # Add high frequency noise and magnitude envelope
        mag_env = torch.empty(chunk_size, 1).uniform_(0.5, 3.0)
        noise = torch.randn(chunk_size, n) * 0.1
        signal = (sine_waves + noise) * mag_env
        
        # Create block-sparse mask (Keep 1 active index uniformly per block of 8)
        blocks = n // 8  # 128
        mask = torch.zeros(chunk_size, n)
        # Choose 1 random index per 8-element block for each batch
        rand_idx_in_block = torch.randint(0, 8, (chunk_size, blocks))
        
        # Calculate absolute indices (B, blocks)
        col_offsets = torch.arange(blocks).unsqueeze(0) * 8
        abs_keep_idx = col_offsets + rand_idx_in_block
        
        # Scatter 1.0 mask over selected indices
        mask.scatter_(1, abs_keep_idx, 1.0)
        
        # Result: Smooth wave with exactly 1 non-zero value per 8 samples
        raw_inputs = signal * mask
        
        # Start state bounds
        table_values = torch.zeros(chunk_size, C, dtype=torch.float32)
        table_dib = torch.zeros(chunk_size, C, dtype=torch.long)
        table_gamma = torch.ones(chunk_size, C, dtype=torch.float32)
        
        abs_vals = raw_inputs.abs()
        _, sort_permutation = torch.sort(abs_vals, dim=-1, descending=True)
        sorted_values = torch.gather(raw_inputs, dim=-1, index=sort_permutation)
        
        # We don't use gammas for this setup, so set 1.0.
        sorted_gammas = torch.ones_like(sorted_values)
        
        out_values, out_dib, out_gamma = cpu_rh_write_batched(
            table_values,
            table_dib,
            table_gamma,
            sorted_values,
            sort_permutation,
            sorted_gammas,
            capacity=C,
            a=1
        )
        
        # Safely Reconstruct Indices
        matches = (out_values.unsqueeze(2) == raw_inputs.unsqueeze(1))
        # Mask out 0.0 values from accidentally matching random original zeros or empty initialization
        is_nonzero = out_values.abs().unsqueeze(2) > 1e-7
        valid_matches = matches & is_nonzero
        out_indices = valid_matches.float().argmax(dim=2)
        
        memory_type = torch.full((chunk_size, C), -1.0, dtype=torch.float32)
        
        tokens_4d = torch.stack([
            out_values, out_gamma, out_dib.float(), memory_type
        ], dim=-1)
        
        targets = torch.zeros(chunk_size, C, n, dtype=torch.float32)
        targets.scatter_(2, out_indices.unsqueeze(2), 1.0)
        
        magnitudes = out_values.abs()
        
        all_tokens.append(tokens_4d)
        all_targets.append(targets)
        all_magnitudes.append(magnitudes)
        all_indices.append(out_indices)
        
    dataset = {
        'tokens_4d': torch.cat(all_tokens, dim=0),
        'targets': torch.cat(all_targets, dim=0),
        'magnitudes': torch.cat(all_magnitudes, dim=0),
        'indices': torch.cat(all_indices, dim=0),
        'meta': {'n': n, 'C': C}
    }
    
    torch.save(dataset, out_path)
    print(f"\nDataset saved to {out_path}")
    print(f"Shape of tokens_4d: {dataset['tokens_4d'].shape}\n")


def main():
    torch.manual_seed(42)
    
    chunk_size = 128
    n = 1024
    C = 128
    
    out_dir = Path(__file__).parent
    
    # Generate Training Split
    generate_split(
        num_samples=10112, 
        chunk_size=chunk_size, 
        n=n, 
        C=C, 
        out_path=out_dir / "synthetic_dataset_train.pt"
    )
    
    # Generate Test Split
    generate_split(
        num_samples=1024, 
        chunk_size=chunk_size, 
        n=n, 
        C=C, 
        out_path=out_dir / "synthetic_dataset_test.pt"
    )

if __name__ == "__main__":
    main()