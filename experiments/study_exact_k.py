import torch
import math
import sys
from pathlib import Path

# Ensure rh_memory is accessible
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_decoder import SyntheticRHDataset

def main():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    n = 1024
    chunk_size = 1024
    C_options = [32, 128, 256]
    fast_ks = [5, 10, 32]
    
    for fast_k in fast_ks:
        for C in C_options:
            print(f"\n--- Running study with n={n}, C={C}, batch_size={chunk_size}, fast_k={fast_k} ---")
            
            # Use exactly the same generator as in training
            dataset = SyntheticRHDataset(n=n, C=C, chunk_size=chunk_size, num_samples=chunk_size, fast_k=fast_k, seed=42, device=device)
            loader = torch.utils.data.DataLoader(dataset, batch_size=None, num_workers=0)
            
            # Since num_samples=chunk_size, it will yield exactly 1 batch
            try:
                batch = next(iter(loader))
                tokens_3d, targets, magnitudes, out_indices = batch
            except StopIteration:
                print("Failed to get batch.")
                continue
                
            # tokens_3d is [B, C, 3] -> (out_values, decoder_gamma, out_dib)
            out_dib = tokens_3d[..., 2]
            
            # Active slots are those with non-zero magnitudes
            active_mask = magnitudes > 1e-7
            valid_dibs = out_dib[active_mask].float()
            
            print(f"Total active slots filled: {active_mask.sum().item()} out of {chunk_size * C}")
            
            if len(valid_dibs) > 0:
                print("DIB Distribution Quantiles:")
                quantiles = torch.tensor([0.5, 0.75, 0.9, 0.95, 0.99, 1.0], device=device)
                computed = torch.quantile(valid_dibs, quantiles)
                
                for q, v in zip(quantiles, computed):
                    print(f"  {int(q.item()*100):>3}% DIB <= {v.item():.0f}")
            else:
                print("All slots empty.")

if __name__ == "__main__":
    main()
