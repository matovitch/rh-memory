import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rh_memory.decoder import RHDecoder, RHDecoderDistillationLoss


def test_decoder_forward_shape():
    B, C, n = 2, 8, 32
    model = RHDecoder(
        sequence_length=n,
        bucket_count=C,
        d_model=64,
        n_heads=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.0,
    )
    tokens = torch.randn(B, C, 3)
    logits = model(tokens)
    assert logits.shape == (B, C, n)
    assert torch.isfinite(logits).all()


def test_decoder_allows_gradient_flow_to_tokens():
    B, C, n = 2, 8, 32
    model = RHDecoder(
        sequence_length=n,
        bucket_count=C,
        d_model=64,
        n_heads=4,
        num_layers=1,
        dim_feedforward=128,
        dropout=0.0,
    )
    tokens = torch.randn(B, C, 3, requires_grad=True)
    logits = model(tokens)
    loss = logits.square().mean()
    loss.backward()
    assert tokens.grad is not None
    assert torch.isfinite(tokens.grad).all()


def test_decoder_distillation_loss_scalar_and_grad():
    B, C, n = 2, 4, 16
    decoder_logits = torch.randn(B, C, n, requires_grad=True)
    teacher_logits = torch.randn(B, C, n)
    weights = torch.rand(B, C).abs()

    loss = RHDecoderDistillationLoss(temperature=2.0)(decoder_logits, teacher_logits, weights)
    assert loss.dim() == 0
    assert torch.isfinite(loss)
    assert loss.item() >= 0.0
    loss.backward()
    assert decoder_logits.grad is not None
    assert torch.isfinite(decoder_logits.grad).all()


def test_decoder_distillation_loss_handles_zero_weights():
    B, C, n = 2, 4, 16
    decoder_logits = torch.randn(B, C, n, requires_grad=True)
    teacher_logits = torch.randn(B, C, n)
    weights = torch.zeros(B, C)

    loss = RHDecoderDistillationLoss()(decoder_logits, teacher_logits, weights)
    assert torch.isfinite(loss)
    assert loss.item() == 0.0
