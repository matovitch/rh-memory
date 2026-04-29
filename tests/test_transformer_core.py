import pytest
import torch

from rh_memory.transformer_core import MultiheadAttentionCore, TransformerCrossBlock


def test_multihead_attention_core_rope_self_mask_shape():
    B, L, D, H = 2, 8, 32, 4
    x = torch.randn(B, L, D)
    mask = torch.ones(L, L, dtype=torch.bool)
    attn = MultiheadAttentionCore(embed_dim=D, num_heads=H, use_rope=True, mode="self")
    y = attn(x_q=x, attn_mask=mask)
    assert y.shape == (B, L, D)
    assert torch.isfinite(y).all()


def test_multihead_attention_core_rope_self_mask_shape_error():
    B, L, D, H = 1, 6, 24, 4
    x = torch.randn(B, L, D)
    bad_mask = torch.ones(L, L - 1, dtype=torch.bool)
    attn = MultiheadAttentionCore(embed_dim=D, num_heads=H, use_rope=True, mode="self")
    with pytest.raises(ValueError):
        _ = attn(x_q=x, attn_mask=bad_mask)


def test_transformer_cross_block_shape():
    B, N, C, D, H = 2, 16, 8, 32, 4
    q = torch.randn(B, N, D)
    memory = torch.randn(B, C, D)
    block = TransformerCrossBlock(d_model=D, n_heads=H, enable_query_self_attn=True)
    out = block(q, memory)
    assert out.shape == (B, N, D)
    assert torch.isfinite(out).all()
