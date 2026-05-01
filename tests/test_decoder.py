import pytest
import torch
import torch.nn.functional as F

from rh_memory.decoder import RHDecoder, RHDecoderDistillationLoss
from rh_memory.decoder_scatter import SoftGatherTokenizationHead, SoftScatterReconstructionHead, decoder_soft_scatter
from rh_memory.pipeline.primitives_tokens import decoder_tokens_from_surrogate_logits_soft


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


def test_decoder_soft_scatter_uniform_identity_permutation():
    B, C, n = 2, 3, 6
    logits = torch.zeros(B, C, n)
    amplitudes = torch.ones(B, C)
    perm_1d = torch.arange(n)

    reconstruction, probs = decoder_soft_scatter(logits, amplitudes, perm_1d, temperature=1.0)

    assert probs.shape == (B, C, n)
    assert reconstruction.shape == (B, n)
    assert torch.allclose(reconstruction, torch.full((B, n), C / n))


def test_decoder_soft_scatter_uses_permutation():
    B, C, n = 1, 2, 4
    logits = torch.full((B, C, n), -20.0)
    logits[0, 0, 1] = 20.0
    logits[0, 1, 3] = 20.0
    amplitudes = torch.tensor([[2.0, 3.0]])
    perm_1d = torch.tensor([2, 0, 3, 1])

    reconstruction, _probs = decoder_soft_scatter(logits, amplitudes, perm_1d, temperature=1.0)

    assert torch.allclose(reconstruction, torch.tensor([[2.0, 3.0, 0.0, 0.0]]), atol=1e-5)


def test_soft_scatter_head_allows_gradient_flow_to_logits_and_temperature():
    B, C, n = 2, 3, 8
    head = SoftScatterReconstructionHead(init_temperature=1.0, min_temperature=0.05)
    logits = torch.randn(B, C, n, requires_grad=True)
    amplitudes = torch.randn(B, C)
    target = torch.randn(B, n)
    perm_1d = torch.arange(n)

    reconstruction, _probs, doubt, support, temperature = head(logits, amplitudes, perm_1d)
    loss = F.l1_loss(reconstruction, target) + doubt.mean() + support.mean() * 1e-3 + temperature * 1e-3
    loss.backward()

    assert reconstruction.shape == (B, n)
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert head.raw_temperature.grad is not None
    assert torch.isfinite(head.raw_temperature.grad).all()


def test_soft_gather_head_matches_existing_soft_tokenization_at_temperature_one():
    B, C, n = 2, 3, 8
    x_perm = torch.randn(B, n)
    surrogate_logits = torch.randn(B, C, n)
    head = SoftGatherTokenizationHead(init_temperature=1.0, min_temperature=0.05)

    tokens = head(x_perm, surrogate_logits)
    expected = decoder_tokens_from_surrogate_logits_soft(x_perm, surrogate_logits, temperature=1.0)

    assert tokens.shape == (B, C, 3)
    assert torch.allclose(tokens, expected, atol=1e-6)


def test_soft_gather_head_learns_temperature_for_amplitude_and_dib_only():
    B, C, n = 2, 3, 8
    x_perm = torch.randn(B, n)
    surrogate_logits = torch.randn(B, C, n, requires_grad=True)
    head = SoftGatherTokenizationHead(init_temperature=0.7, min_temperature=0.05)

    tokens = head(x_perm, surrogate_logits)
    loss = tokens[..., :2].sum()
    loss.backward()

    assert tokens.shape == (B, C, 3)
    assert surrogate_logits.grad is not None
    assert torch.isfinite(surrogate_logits.grad).all()
    assert head.raw_temperature.grad is not None
    assert torch.isfinite(head.raw_temperature.grad).all()


def test_soft_gather_head_uses_fixed_entropy_temperature():
    B, C, n = 1, 2, 4
    x_perm = torch.randn(B, n)
    surrogate_logits = torch.randn(B, C, n)
    head = SoftGatherTokenizationHead(init_temperature=0.35, min_temperature=0.05)

    tokens = head(x_perm, surrogate_logits)
    fixed_entropy_tokens = decoder_tokens_from_surrogate_logits_soft(x_perm, surrogate_logits, temperature=1.0)

    assert torch.allclose(tokens[..., 2], fixed_entropy_tokens[..., 2], atol=1e-6)
    assert not torch.allclose(tokens[..., :2], fixed_entropy_tokens[..., :2], atol=1e-6)


def test_soft_gather_head_shape_mismatch_raises():
    head = SoftGatherTokenizationHead()
    with pytest.raises(ValueError):
        _ = head(torch.randn(2, 8), torch.randn(2, 3, 7))
