import pytest
import torch

from rh_memory.surrogate import RHSurrogate, RHSurrogateLoss


@pytest.fixture
def surrogate_dims():
    batch_size = 2
    n = 64
    C = 16
    stride = n // C
    d_model = 32
    n_heads = 4
    num_layers = 2
    return {
        "batch_size": batch_size,
        "n": n,
        "C": C,
        "stride": stride,
        "d_model": d_model,
        "n_heads": n_heads,
        "num_layers": num_layers,
    }


def test_surrogate_forward_shape(surrogate_dims):
    model = RHSurrogate(
        sequence_length=surrogate_dims["n"],
        bucket_count=surrogate_dims["C"],
        stride=surrogate_dims["stride"],
        fast_k=5.0,
        d_model=surrogate_dims["d_model"],
        n_heads=surrogate_dims["n_heads"],
        num_layers=surrogate_dims["num_layers"],
    )
    x = torch.randn(surrogate_dims["batch_size"], surrogate_dims["C"], surrogate_dims["stride"])
    logits = model(x)

    expected = (
        surrogate_dims["batch_size"],
        surrogate_dims["C"],
        surrogate_dims["n"],
    )
    assert logits.shape == expected, f"Expected {expected}, got {logits.shape}"
    assert torch.isfinite(logits).all()


def test_surrogate_forward_shape_with_ring_causal_mask(surrogate_dims):
    model = RHSurrogate(
        sequence_length=surrogate_dims["n"],
        bucket_count=surrogate_dims["C"],
        stride=surrogate_dims["stride"],
        fast_k=1.0,
        d_model=surrogate_dims["d_model"],
        n_heads=surrogate_dims["n_heads"],
        num_layers=surrogate_dims["num_layers"],
    )
    x = torch.randn(surrogate_dims["batch_size"], surrogate_dims["C"], surrogate_dims["stride"])
    logits = model(x)
    expected = (
        surrogate_dims["batch_size"],
        surrogate_dims["C"],
        surrogate_dims["n"],
    )
    assert logits.shape == expected, f"Expected {expected}, got {logits.shape}"
    assert torch.isfinite(logits).all()


def test_rhsurrogate_loss(surrogate_dims):
    B, n, C = surrogate_dims["batch_size"], surrogate_dims["n"], surrogate_dims["C"]
    target_idx = torch.randint(0, n, (B, C))
    valid_bucket = torch.ones(B, C, dtype=torch.bool)

    abs_amp = torch.abs(torch.randn(B, C))

    loss_fn = RHSurrogateLoss()
    logits = torch.randn(B, C, n, requires_grad=True)
    loss = loss_fn(logits, target_idx, abs_amp, valid_bucket)

    assert loss.dim() == 0
    assert not torch.isnan(loss)
    assert loss.item() > 0
    loss.backward()
    assert logits.grad is not None


def test_rhsurrogate_loss_ignores_invalid_buckets(surrogate_dims):
    B, n, C = surrogate_dims["batch_size"], surrogate_dims["n"], surrogate_dims["C"]
    target_idx = torch.randint(0, n, (B, C))
    valid_bucket = torch.zeros(B, C, dtype=torch.bool)
    abs_amp = torch.ones(B, C)
    logits = torch.randn(B, C, n, requires_grad=True)
    loss = RHSurrogateLoss()(logits, target_idx, abs_amp, valid_bucket)
    assert torch.isfinite(loss)
    assert loss.item() == 0.0
