import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest
import torch
import torch.nn as nn
from rh_memory.decoder import RHDecoder, RHLoss

@pytest.fixture
def dummy_data():
    batch_size = 2
    C = 16
    n = 64
    d_model = 32
    n_heads = 4
    num_layers = 2
    
    tokens = torch.randn(batch_size, C, 2)
    # Per-bucket BCE weights (here: random positive, like |amplitude|)
    abs_amplitude = torch.abs(torch.randn(batch_size, C))
    
    # Ground truth positions (random sequence indices)
    gt_indices = torch.randint(0, n, (batch_size, C))
    targets = torch.zeros(batch_size, C, n)
    targets.scatter_(2, gt_indices.unsqueeze(2), 1.0)
    
    return {
        'batch_size': batch_size,
        'C': C,
        'n': n,
        'd_model': d_model,
        'n_heads': n_heads,
        'num_layers': num_layers,
        'tokens': tokens,
        'targets': targets,
        'abs_amplitude': abs_amplitude,
        'gt_indices': gt_indices
    }

def test_decoder_forward(dummy_data):
    decoder = RHDecoder(
        sequence_length=dummy_data['n'],
        bucket_count=dummy_data['C'],
        d_model=dummy_data['d_model'],
        n_heads=dummy_data['n_heads'],
        num_layers=dummy_data['num_layers']
    )
    
    tokens = dummy_data['tokens']
    logits = decoder(tokens)
    
    assert logits.shape == (dummy_data['batch_size'], dummy_data['C'], dummy_data['n']), \
        f"Expected output shape {(dummy_data['batch_size'], dummy_data['C'], dummy_data['n'])}, but got {logits.shape}"

def test_decoder_reconstruction(dummy_data):
    decoder = RHDecoder(
        sequence_length=dummy_data['n'],
        bucket_count=dummy_data['C'],
        d_model=dummy_data['d_model'],
        n_heads=dummy_data['n_heads'],
        num_layers=dummy_data['num_layers']
    )
    
    logits = torch.randn(dummy_data['batch_size'], dummy_data['C'], dummy_data['n'])
    reconstructed = decoder.reconstruct(logits)
    
    assert reconstructed.shape == (dummy_data['batch_size'], dummy_data['n']), \
        f"Expected reconstructed shape {(dummy_data['batch_size'], dummy_data['n'])}, but got {reconstructed.shape}"

def test_rh_loss(dummy_data):
    loss_fn = RHLoss()
    logits = torch.randn(dummy_data['batch_size'], dummy_data['C'], dummy_data['n'])
    
    # Calculate loss
    loss = loss_fn(logits, dummy_data['targets'], dummy_data['abs_amplitude'])
    
    assert loss.dim() == 0, "Loss should be a scalar"
    assert not torch.isnan(loss), "Loss should not be NaN"
    assert loss.item() > 0, "Loss should be positive"
