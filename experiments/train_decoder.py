import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

import sys
from pathlib import Path

# Ensure rh_memory is accessible
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rh_memory.decoder import RHDecoder, RHLoss


def main():
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load Pre-generated dataset
    train_path = Path(__file__).parent / "synthetic_dataset_train.pt"
    test_path = Path(__file__).parent / "synthetic_dataset_test.pt"
    
    if not train_path.exists() or not test_path.exists():
        print(f"Datasets not found. Please run generate_dataset.py first.")
        return
        
    print(f"Loading datasets...")
    train_dict = torch.load(train_path)
    test_dict = torch.load(test_path)
    
    # Metadata
    n = train_dict['meta']['n']
    C = train_dict['meta']['C']
    
    train_dataset = TensorDataset(
        train_dict['tokens_4d'], train_dict['targets'], 
        train_dict['magnitudes'], train_dict['indices']
    )
    test_dataset = TensorDataset(
        test_dict['tokens_4d'], test_dict['targets'], 
        test_dict['magnitudes'], test_dict['indices']
    )
    
    B = 32          # mini-batch size
    train_loader = DataLoader(train_dataset, batch_size=B, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=B, shuffle=False, drop_last=False)
    
    # Hyperparameters
    d_model = 128
    epochs = 1500
    learning_rate = 5e-4
    
    print(f"Initializing RHDecoder [n={n}, C={C}, d_model={d_model}] on {device}")
    decoder = RHDecoder(
        sequence_length=n,
        bucket_count=C,
        d_model=d_model,
        n_heads=4,
        num_layers=4
    ).to(device)
    
    criterion = RHLoss()
    optimizer = optim.AdamW(decoder.parameters(), lr=learning_rate)
    
    decoder.train()
    print("Starting Training Loop...")
    
    for epoch in range(1, epochs + 1):
        decoder.train()
        running_loss = 0.0
        running_accuracy = 0.0
        batches = 0
        
        for batch in train_loader:
            batch_tokens = batch[0].to(device)
            batch_targets = batch[1].to(device)
            batch_magnitudes = batch[2].to(device)
            batch_indices = batch[3].to(device)
            
            optimizer.zero_grad()
            
            # Forward & Backward
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
            batches += 1
            
        if epoch % 5 == 0 or epoch == 1:
            avg_loss = running_loss / batches
            avg_acc = running_accuracy / batches
            
            # Run Evaluation
            decoder.eval()
            test_loss = 0.0
            test_acc = 0.0
            test_batches = 0
            
            with torch.no_grad():
                for batch in test_loader:
                    b_toks = batch[0].to(device)
                    b_targs = batch[1].to(device)
                    b_mags = batch[2].to(device)
                    b_idx = batch[3].to(device)
                    
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
            
            print(f"Epoch {epoch:03d}/{epochs} | "
                  f"Train Loss: {avg_loss:.6f} | Train Acc: {avg_acc:.2f}% | "
                  f"Test Loss: {avg_test_loss:.6f} | Test Acc: {avg_test_acc:.2f}%")
            
    print("Training finished.")

if __name__ == "__main__":
    main()
