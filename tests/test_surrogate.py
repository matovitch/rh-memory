import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest
import torch

from rh_memory.surrogate import RHSurrogate, RHSurrogateLoss


@pytest.fixture
def surrogate_dims():
    batch_size = 2
    n = 64
    C = 16
    d_model = 32
    n_heads = 4
    num_layers = 2
    return {
        "batch_size": batch_size,
        "n": n,
        "C": C,
        "d_model": d_model,
        "n_heads": n_heads,
        "num_layers": num_layers,
    }


def test_surrogate_forward_shape(surrogate_dims):
    model = RHSurrogate(
        sequence_length=surrogate_dims["n"],
        bucket_count=surrogate_dims["C"],
        d_model=surrogate_dims["d_model"],
        n_heads=surrogate_dims["n_heads"],
        num_layers=surrogate_dims["num_layers"],
    )
    x = torch.randn(surrogate_dims["batch_size"], surrogate_dims["n"], 1)
    logits = model(x)

    expected = (
        surrogate_dims["batch_size"],
        surrogate_dims["n"],
        surrogate_dims["C"],
    )
    assert logits.shape == expected, f"Expected {expected}, got {logits.shape}"
    assert torch.isfinite(logits).all()


def test_rhsurrogate_loss(surrogate_dims):
    B, n, C = surrogate_dims["batch_size"], surrogate_dims["n"], surrogate_dims["C"]
    gt_bucket = torch.randint(0, C, (B, n))
    targets = torch.zeros(B, n, C)
    targets.scatter_(2, gt_bucket.unsqueeze(2), 1.0)

    abs_amp = torch.abs(torch.randn(B, n))

    loss_fn = RHSurrogateLoss()
    logits = torch.randn(B, n, C)
    loss = loss_fn(logits, targets, abs_amp)

    assert loss.dim() == 0
    assert not torch.isnan(loss)
    assert loss.item() > 0

