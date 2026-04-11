import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader

import sys
import math
from pathlib import Path

# Ensure rh_memory is accessible
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rh_memory.decoder import RHDecoder, RHLoss
from rh_memory._cpu_ops import cpu_rh_write_batched
from rh_memory._python_ops import python_fast_rh_write_batched
from rh_memory._triton_ops import triton_exact_parallel_rh # type: ignore

class SyntheticRHDataset(IterableDataset):
    def __init__(self, n, C, chunk_size=128, num_samples=None, fast_k=32, device="cpu"):
        super().__init__()
        self.n = n
        self.C = C
        self.chunk_size = chunk_size
        self.num_samples = num_samples  # If None, infinite dataset
        self.fast_k = fast_k
        self.device = device
        
    def __iter__(self):
        n = self.n
        C = self.C
        chunk_size = self.chunk_size
        samples_generated = 0
        device = self.device
        
        while self.num_samples is None or samples_generated < self.num_samples:
            # Create base sine waves (B, n) with random frequencies and phases
            t = torch.linspace(0, 1, n, device=device).unsqueeze(0).expand(chunk_size, n)
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
            table_dib = torch.zeros(chunk_size, C, dtype=torch.long if device == "cpu" or str(device) == "cpu" else torch.int32, device=device)
            # Use table_gamma (initialized to 0) to carry the source indices 
            table_gamma = torch.zeros(chunk_size, C, dtype=torch.float32, device=device)
            
            # Fast pooling path uses triton kernel if on gpu
            incoming_gammas = torch.arange(n, dtype=torch.float32, device=device).unsqueeze(0).expand(chunk_size, n)
            
            if "cuda" in str(device):
                out_values, out_dib, out_gamma = triton_exact_parallel_rh(
                    table_values,
                    table_dib,
                    table_gamma,
                    raw_inputs,
                    incoming_gammas,
                    a=16,
                    k=self.fast_k,
                )
            else:
                out_values, out_dib, out_gamma = python_fast_rh_write_batched(
                    table_values,
                    table_dib,
                    table_gamma,
                    raw_inputs,
                    incoming_gammas,
                    a=16,
                    k=self.fast_k,
                )
            
            out_indices = out_gamma.long()
            
            # Restore the all-ones gamma for the neural network tokens to match original behavior
            decoder_gamma = torch.ones_like(out_gamma)
            tokens_3d = torch.stack([out_values, decoder_gamma, out_dib.float()], dim=-1)
            
            targets = torch.zeros(chunk_size, C, n, dtype=torch.float32, device=device)
            targets.scatter_(2, out_indices.unsqueeze(2), 1.0)
            magnitudes = out_values.abs()
            
            for i in range(chunk_size):
                if self.num_samples is not None and samples_generated >= self.num_samples:
                    break
                yield tokens_3d[i], targets[i], magnitudes[i], out_indices[i]
                samples_generated += 1


def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    torch.manual_seed(worker_seed)

def main():
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Metadata
    n = 1024
    C = 128
    
    print(f"Dataset meta - n={n}, C={C}")
    
    B = 128          # mini-batch size
    train_dataset = SyntheticRHDataset(n=n, C=C, chunk_size=128, device=device) # type: ignore
    test_dataset = SyntheticRHDataset(n=n, C=C, chunk_size=128, num_samples=1024, device=device) # type: ignore
    
    # If device is CUDA, use num_workers=0 to avoid CUDA multiprocessing issues
    workers = 0 if "cuda" in str(device) else 2
    train_loader = DataLoader(train_dataset, batch_size=B, num_workers=workers, prefetch_factor=None, worker_init_fn=worker_init_fn)
    test_loader = DataLoader(test_dataset, batch_size=B, num_workers=workers, prefetch_factor=None, worker_init_fn=worker_init_fn)
    
    # Hyperparameters
    d_model = 256
    total_steps = 17_000_000 // B
    eval_every_steps = 50_000 // B
    learning_rate = 2e-4
    
    print(f"Initializing RHDecoder [n={n}, C={C}, d_model={d_model}] on {device}")
    decoder = RHDecoder(
        sequence_length=n,
        bucket_count=C,
        d_model=d_model,
        n_heads=8,
        num_layers=6,
        dim_feedforward=d_model * 8,
        dropout=0.0,
    ).to(device)
    
    criterion = RHLoss()
    optimizer = optim.AdamW(decoder.parameters(), lr=learning_rate)
    
    start_step = 0
    checkpoint_path = Path("experiments/checkpoint.pt")
    if checkpoint_path.exists():
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        decoder.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_step = checkpoint['step']
        print(f"Resumed training from step {start_step}")
    
    decoder.train()
    print("Starting Training Loop...")
    
    running_loss = 0.0
    running_accuracy = 0.0
    batches_since_eval = 0
    
    for step_idx, batch in enumerate(train_loader, 1):
        step = start_step + step_idx
        if step > total_steps:
            break
            
        batch_tokens = batch[0].to(device)
        batch_targets = batch[1].to(device)
        batch_magnitudes = batch[2].to(device)
        batch_indices = batch[3].to(device)
        
        optimizer.zero_grad()
        
        # Forward & Backward
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            logits = decoder(batch_tokens)
            loss = criterion(logits, batch_targets, batch_magnitudes)
        
        loss.backward()
        optimizer.step()
        
        # Accuracy metric
        _, predicted_indices = torch.max(logits, dim=2)
        active_mask = batch_magnitudes > 0.01
        correct = (predicted_indices == batch_indices) & active_mask
        
        total_active = active_mask.sum().item()
        accuracy = (correct.sum().item() / total_active * 100) if total_active > 0 else 0.0
        
        running_loss += loss.item()
        running_accuracy += accuracy
        batches_since_eval += 1
        
        if step % eval_every_steps == 0 or step == 1:
            avg_loss = running_loss / batches_since_eval
            avg_acc = running_accuracy / batches_since_eval
            
            # Run Evaluation
            decoder.eval()
            test_loss = 0.0
            test_acc = 0.0
            test_batches = 0
            
            with torch.no_grad():
                for test_batch in test_loader:
                    b_toks = test_batch[0].to(device)
                    b_targs = test_batch[1].to(device)
                    b_mags = test_batch[2].to(device)
                    b_idx = test_batch[3].to(device)
                    
                    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                        logits = decoder(b_toks)
                        loss = criterion(logits, b_targs, b_mags)
                    
                    _, pred_idx = torch.max(logits, dim=2)
                    act_mask = b_mags > 0.01
                    correct = (pred_idx == b_idx) & act_mask
                    
                    tot_active = act_mask.sum().item()
                    acc = (correct.sum().item() / tot_active * 100) if tot_active > 0 else 0.0
                    
                    test_loss += loss.item()
                    test_acc += acc
                    test_batches += 1
                    
            avg_test_loss = test_loss / test_batches
            avg_test_acc = test_acc / test_batches
            
            samples_processed = step * B
            print(f"Step {step} | Samples: {samples_processed} | "
                  f"Train Loss: {avg_loss:.6f} | Train Acc: {avg_acc:.2f}% | "
                  f"Test Loss: {avg_test_loss:.6f} | Test Acc: {avg_test_acc:.2f}%")
            
            print(f"Saving checkpoint to {checkpoint_path}...")
            torch.save({
                'step': step,
                'model_state_dict': decoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_path)
            
            # Reset running metrics
            running_loss = 0.0
            running_accuracy = 0.0
            batches_since_eval = 0
            
            decoder.train()
            
    print("Training finished.")
    
    # Final evaluation pass to gather average logits
    print("Gathering final test logits for visualization...")
    decoder.eval()
    import torchvision.utils as vutils
    
    all_test_logits = []
    all_test_indices = []
    all_test_mags = []
    
    with torch.no_grad():
        for test_batch in test_loader:
            b_toks = test_batch[0].to(device)
            b_mags = test_batch[2].to(device)
            b_idx = test_batch[3].to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                logits = decoder(b_toks) # [B, C, n]
            all_test_logits.append(logits.float().cpu())
            all_test_indices.append(b_idx.cpu())
            all_test_mags.append(b_mags.cpu())
            
    all_logits_tensor = torch.cat(all_test_logits, dim=0) # [Total_B, C, n]
    all_indices_tensor = torch.cat(all_test_indices, dim=0) # [Total_B, C]
    all_mags_tensor = torch.cat(all_test_mags, dim=0) # [Total_B, C]
    
    # Flatten outputs for filtering
    n_seq = all_logits_tensor.shape[-1]
    all_logits_flat = all_logits_tensor.view(-1, n_seq)       # [Total_B * C, n]
    all_indices_flat = all_indices_tensor.view(-1)            # [Total_B * C]
    all_mags_flat = all_mags_tensor.view(-1)                  # [Total_B * C]
    
    # Filter out empty buckets
    active_mask = all_mags_flat > 0.01
    active_logits = all_logits_flat[active_mask]              # [Num_Active, n]
    active_indices = all_indices_flat[active_mask].long()     # [Num_Active]
    
    # Create an [n, n] image (True Position vs Predicted Position)
    conditional_logits = torch.zeros(n_seq, n_seq)
    
    for t_true in range(n_seq):
        mask_t = (active_indices == t_true)
        if mask_t.sum() > 0:
            conditional_logits[t_true] = active_logits[mask_t].mean(dim=0)
    
    # Normalize logits into image range [0, 1] for visualization
    logits_min = conditional_logits.min()
    logits_max = conditional_logits.max()
    logits_img = (conditional_logits - logits_min) / (logits_max - logits_min + 1e-8)
    
    # Add channel and batch dims for save_image: [1, 1, n, n]
    logits_img = logits_img.unsqueeze(0).unsqueeze(0)
    
    print(f"Saving conditional logits grid image to 'logits_grid.png'...")
    vutils.save_image(logits_img, 'logits_grid.png', normalize=False)
    print("Done!")

if __name__ == "__main__":
    main()
