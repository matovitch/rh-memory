import torch
import torch.nn.functional as F

from rh_memory.decoder import RHDecoder, RHDecoderDistillationLoss
from rh_memory.decoder_scatter import SoftScatterReconstructionHead, decoder_soft_scatter


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
