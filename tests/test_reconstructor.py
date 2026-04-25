import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rh_memory.reconstructor import RHReconstructor, RHReconstructorLoss


@pytest.fixture
def recon_dims():
    return {
        "B": 2,
        "C": 8,
        "n": 32,
        "d_model": 64,
        "n_heads": 8,
    }


def test_reconstructor_forward_shape(recon_dims):
    model = RHReconstructor(
        sequence_length=recon_dims["n"],
        bucket_count=recon_dims["C"],
        d_model=recon_dims["d_model"],
        n_heads=recon_dims["n_heads"],
        num_token_layers=2,
        num_query_layers=2,
        dim_feedforward=4 * recon_dims["d_model"],
        dropout=0.0,
    )
    tokens = torch.randn(recon_dims["B"], recon_dims["C"], 3)
    out = model(tokens)
    assert out.shape == (recon_dims["B"], recon_dims["n"])
    assert torch.isfinite(out).all()


def test_reconstructor_loss_scalar(recon_dims):
    model = RHReconstructor(
        sequence_length=recon_dims["n"],
        bucket_count=recon_dims["C"],
        d_model=recon_dims["d_model"],
        n_heads=recon_dims["n_heads"],
        num_token_layers=1,
        num_query_layers=1,
        dim_feedforward=4 * recon_dims["d_model"],
        dropout=0.0,
    )
    loss_fn = RHReconstructorLoss()
    tokens = torch.randn(recon_dims["B"], recon_dims["C"], 3)
    target = torch.randn(recon_dims["B"], recon_dims["n"])
    pred = model(tokens)
    loss = loss_fn(pred, target)
    assert loss.dim() == 0
    assert torch.isfinite(loss)
    assert loss.item() >= 0.0


def test_token_encoder_permutation_equivariance(recon_dims):
    model = RHReconstructor(
        sequence_length=recon_dims["n"],
        bucket_count=recon_dims["C"],
        d_model=recon_dims["d_model"],
        n_heads=recon_dims["n_heads"],
        num_token_layers=2,
        num_query_layers=1,
        dim_feedforward=4 * recon_dims["d_model"],
        dropout=0.0,
    )
    model.eval()

    tokens = torch.randn(recon_dims["B"], recon_dims["C"], 3)
    perm = torch.randperm(recon_dims["C"])
    encoded = model.encode_tokens(tokens)
    encoded_perm = model.encode_tokens(tokens[:, perm, :])

    # Encoder is permutation-equivariant over set tokens.
    assert torch.allclose(encoded[:, perm, :], encoded_perm, atol=1e-6, rtol=1e-5)
