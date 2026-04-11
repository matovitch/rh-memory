import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest
import torch

from rh_memory import BatchedMemoryState, compute_write_gammas

def test_compute_write_gammas():
    values = torch.tensor([1.0, 2.0, 3.0])
    gammas = compute_write_gammas(values, alpha=0.5, epsilon=0.1)

    assert gammas.shape == values.shape
    assert torch.all(gammas >= 0.1)
    assert torch.all(gammas < 1.0)

def test_batched_memory_initialization():
    state = BatchedMemoryState.empty(batch_size=2, capacity=8)
    assert state.values.shape == (2, 8)
    assert state.dib.shape == (2, 8)
    assert state.gamma.shape == (2, 8)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton tests require CUDA")
def test_triton_exact_parallel_match_sequential():
    from rh_memory import python_fast_rh_write_batched, triton_exact_parallel_rh # type: ignore
    
    torch.manual_seed(0)
    B = 2
    C = 128
    a = 16
    stride = 8
    n = C * stride
    k = 24
    
    t_vals = torch.zeros(B, C, dtype=torch.float32, device="cuda")
    t_dib = torch.zeros(B, C, dtype=torch.long, device="cuda")
    t_gams = torch.zeros(B, C, dtype=torch.float32, device="cuda")
    
    inc_vals = torch.randn(B, n, dtype=torch.float32, device="cuda")
    inc_gams = torch.ones(B, n, dtype=torch.float32, device="cuda") * 0.9
    
    # Create mask to make it sparse
    mask = (torch.rand(B, n, device="cuda") > 0.5).float()
    inc_vals *= mask
    
    p_vals, p_dib, p_gams = python_fast_rh_write_batched(t_vals, t_dib, t_gams, inc_vals, inc_gams, a, k)
    tr_vals, tr_dib, tr_gams = triton_exact_parallel_rh(t_vals, t_dib, t_gams, inc_vals, inc_gams, a, k)
    
    assert torch.allclose(p_vals, tr_vals, atol=1e-5), "Values mismatch"
    assert torch.equal(p_dib, tr_dib), "DIB mismatch"
    assert torch.allclose(p_gams, tr_gams, atol=1e-5), "Gamma mismatch"

