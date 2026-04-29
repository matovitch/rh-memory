import pytest
import torch

from rh_memory import (
    python_linear_probing_amplitude_pooling,
    triton_linear_probing_amplitude_pooling,
)


def test_triton_linear_probing_amplitude_pooling_matches_python():
    """Compare Python vs Triton; without CUDA, ``tests/conftest.py`` enables Triton interpreter."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    B = 2
    C = 128
    stride = 8
    n = C * stride
    k = 24

    t_vals = torch.zeros(B, C, dtype=torch.float32, device=device)
    t_dib = torch.zeros(B, C, dtype=torch.int32, device=device)
    t_carry = torch.full((B, C), -1, dtype=torch.int32, device=device)

    inc_vals = torch.randn(B, n, dtype=torch.float32, device=device)
    inc_carry = torch.arange(n, dtype=torch.int32, device=device).unsqueeze(0).expand(B, n)

    mask = (torch.rand(B, n, device=device) > 0.5).float()
    inc_vals *= mask

    # Table tensors are updated in place; compare with identical fresh tensor trees per call.
    def _clone_args():
        return (
            t_vals.clone(),
            t_dib.clone(),
            t_carry.clone(),
            inc_vals.clone(),
            inc_carry.clone(),
        )

    p_vals, p_dib, p_carry = python_linear_probing_amplitude_pooling(*_clone_args(), k)
    tr_vals, tr_dib, tr_carry = triton_linear_probing_amplitude_pooling(*_clone_args(), k)

    assert torch.allclose(p_vals, tr_vals, atol=1e-5), "Values mismatch"
    assert torch.equal(p_dib, tr_dib), "DIB mismatch"
    assert torch.equal(p_carry, tr_carry), "carry_id mismatch"


@pytest.mark.parametrize(
    "pool_fn",
    [python_linear_probing_amplitude_pooling, triton_linear_probing_amplitude_pooling],
)
def test_lpap_may_mutate_contiguous_incoming_tensors(pool_fn):
    """Direct LPAP callers should treat incoming tensors as scratch unless they pass clones."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(1)
    B = 2
    C = 16
    stride = 4
    n = C * stride
    k = 5

    table_values = torch.zeros(B, C, dtype=torch.float32, device=device)
    table_dib = torch.zeros(B, C, dtype=torch.int32, device=device)
    table_carry_id = torch.full((B, C), -1, dtype=torch.int32, device=device)
    incoming_values = torch.randn(B, n, dtype=torch.float32, device=device).contiguous()
    incoming_carry_id = torch.arange(n, dtype=torch.int32, device=device).unsqueeze(0).expand(B, n).contiguous()
    values_before = incoming_values.clone()
    carry_before = incoming_carry_id.clone()

    pool_fn(
        table_values,
        table_dib,
        table_carry_id,
        incoming_values,
        incoming_carry_id,
        k,
    )

    values_changed = not torch.equal(incoming_values, values_before)
    carry_changed = not torch.equal(incoming_carry_id, carry_before)
    assert values_changed or carry_changed
