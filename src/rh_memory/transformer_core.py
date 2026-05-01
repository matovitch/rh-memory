"""Shared transformer primitives with optional RoPE and self/cross attention."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Bool, Float
from torch import Tensor
from typing import cast


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: int = 10000) -> None:
        super().__init__()
        self.dim = dim
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(
        self, seq_len: int, device: torch.device
    ) -> tuple[Float[Tensor, "seq_len head_dim"], Float[Tensor, "seq_len head_dim"]]:
        inv_freq = cast(Tensor, self.inv_freq)
        t = torch.arange(seq_len, device=device).to(dtype=inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return torch.cos(emb), torch.sin(emb)


def apply_rotary_pos_emb(
    x: Float[Tensor, "B n_heads seq_len head_dim"],
    cos: Float[Tensor, "seq_len head_dim"],
    sin: Float[Tensor, "seq_len head_dim"],
) -> Float[Tensor, "B n_heads seq_len head_dim"]:
    # x shape: [B, n_heads, seq_len, head_dim]
    d = x.shape[-1] // 2
    x1 = x[..., :d]
    x2 = x[..., d:]
    rotated = torch.cat((-x2, x1), dim=-1)
    return (x * cos.unsqueeze(0).unsqueeze(0)) + (rotated * sin.unsqueeze(0).unsqueeze(0))


class MultiheadAttentionCore(nn.Module):
    """
    Attention core with optional RoPE and self/cross mode.

    - mode='self': q, k, v from x_q
    - mode='cross': q from x_q, k/v from x_kv (required)
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        use_rope: bool = False,
        mode: str = "self",
    ) -> None:
        super().__init__()
        if mode not in {"self", "cross"}:
            raise ValueError(f"Unsupported mode={mode}, expected 'self' or 'cross'")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        if self.head_dim * num_heads != embed_dim:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.mode = mode
        self.use_rope = use_rope

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.rotary_emb = RotaryEmbedding(self.head_dim) if use_rope else None

    def _validate_mask(self, attn_mask: Bool[Tensor, "q_len kv_len"], q_len: int, kv_len: int) -> None:
        if attn_mask.dtype != torch.bool:
            raise TypeError("attn_mask must be a boolean tensor")
        expected = (q_len, kv_len)
        if attn_mask.shape != expected:
            raise ValueError(f"attn_mask must have shape {expected}, got {tuple(attn_mask.shape)}")

    def _reshape_heads(self, x: Float[Tensor, "B seq_len d_model"]) -> Float[Tensor, "B n_heads seq_len head_dim"]:
        B, seq_len, _ = x.shape
        return x.view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        x_q: Float[Tensor, "B q_len d_model"],
        x_kv: Float[Tensor, "B kv_len d_model"] | None = None,
        attn_mask: Bool[Tensor, "q_len kv_len"] | None = None,
    ) -> Float[Tensor, "B q_len d_model"]:
        if self.mode == "cross":
            if x_kv is None:
                raise ValueError("x_kv is required for cross-attention mode")
            kv = x_kv
        else:
            kv = x_q if x_kv is None else x_kv

        q = self._reshape_heads(self.q_proj(x_q))
        k = self._reshape_heads(self.k_proj(kv))
        v = self._reshape_heads(self.v_proj(kv))

        q_len = q.shape[2]
        kv_len = k.shape[2]

        if self.rotary_emb is not None:
            if q_len != kv_len:
                raise ValueError("RoPE requires q_len == kv_len")
            cos, sin = self.rotary_emb(q_len, x_q.device)
            q = apply_rotary_pos_emb(q, cos, sin)
            k = apply_rotary_pos_emb(k, cos, sin)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)
        if attn_mask is not None:
            self._validate_mask(attn_mask, q_len=q_len, kv_len=kv_len)
            scores = scores.masked_fill(~attn_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).contiguous().view(x_q.shape[0], q_len, self.embed_dim)
        return self.out_proj(context)


class TransformerFFN(nn.Module):
    def __init__(self, d_model: int, dim_feedforward: int = 2048, dropout: float = 0.0) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

    def forward(self, x: Float[Tensor, "B seq_len d_model"]) -> Float[Tensor, "B seq_len d_model"]:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class TransformerBlock(nn.Module):
    """Pre-norm residual transformer block with configurable self/cross attention."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.0,
        use_rope: bool = False,
        mode: str = "self",
    ) -> None:
        super().__init__()
        self.attn = MultiheadAttentionCore(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            use_rope=use_rope,
            mode=mode,
        )
        self.ffn = TransformerFFN(d_model=d_model, dim_feedforward=dim_feedforward, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.mode = mode

    def forward(
        self,
        x: Float[Tensor, "B q_len d_model"],
        x_kv: Float[Tensor, "B kv_len d_model"] | None = None,
        attn_mask: Bool[Tensor, "q_len kv_len"] | None = None,
    ) -> Float[Tensor, "B q_len d_model"]:
        x_norm = self.norm1(x)
        if self.mode == "cross":
            attn_out = self.attn(x_norm, x_kv=x_kv, attn_mask=attn_mask)
        else:
            attn_out = self.attn(x_norm, attn_mask=attn_mask)
        x = x + self.dropout1(attn_out)
        x = x + self.dropout2(self.ffn(self.norm2(x)))
        return x


class TransformerCrossBlock(nn.Module):
    """Query block with self-attn (optional) + cross-attn + FFN."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.0,
        self_use_rope: bool = False,
        cross_use_rope: bool = False,
        enable_query_self_attn: bool = True,
    ) -> None:
        super().__init__()
        self.enable_query_self_attn = enable_query_self_attn
        if enable_query_self_attn:
            self.self_block = TransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                use_rope=self_use_rope,
                mode="self",
            )
        else:
            self.self_block = None
        self.cross_block = TransformerBlock(
            d_model=d_model,
            n_heads=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            use_rope=cross_use_rope,
            mode="cross",
        )

    def forward(
        self,
        q: Float[Tensor, "B q_len d_model"],
        memory: Float[Tensor, "B C d_model"],
        self_attn_mask: Bool[Tensor, "q_len q_len"] | None = None,
    ) -> Float[Tensor, "B q_len d_model"]:
        if self.self_block is not None:
            q = self.self_block(q, attn_mask=self_attn_mask)
        q = self.cross_block(q, x_kv=memory)
        return q
