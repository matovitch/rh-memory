import sys
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def test_decoder_forward_and_loss():
    from rh_memory import DecoderConfig, RHMemoryDecoder, build_support_target, weighted_bce_with_logits

    torch.manual_seed(0)

    config = DecoderConfig(n_slots=16, n_buckets=4, model_dim=32, num_heads=4, num_layers=2)
    decoder = RHMemoryDecoder(config)

    values = torch.tensor([[1.5, -2.0, 0.0, 3.0]], dtype=torch.float32)
    dib = torch.tensor([[0.0, 1.0, 2.0, 1.0]], dtype=torch.float32)
    gamma = torch.tensor([[0.9, 0.8, 0.7, 0.6]], dtype=torch.float32)
    memory_type = False

    logits = decoder(values, dib, gamma, memory_type)
    assert logits.shape == (1, 16)

    original = torch.tensor([[0.0, 1.5, 0.0, -2.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    target, pos_weight = build_support_target(original)
    assert target.shape == original.shape
    assert pos_weight.shape == original.shape
    assert torch.equal(target, (original != 0).to(dtype=original.dtype))

    loss = weighted_bce_with_logits(logits, target, positive_weight=pos_weight)
    assert torch.isfinite(loss)
    assert loss.ndim == 0
