"""Shared RoPE-based Transformer encoder blocks over a 1D sequence axis (bucket or position)."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        self.dim = dim
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len, device):
        inv_freq = self.inv_freq
        assert isinstance(inv_freq, torch.Tensor)
        t = torch.arange(seq_len, device=device).to(dtype=inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return torch.cos(emb), torch.sin(emb)


def apply_rotary_pos_emb(x, cos, sin):
    # x shape: [B, n_heads, seq_len, head_dim]
    d = x.shape[-1] // 2
    x1 = x[..., :d]
    x2 = x[..., d:]

    rotated = torch.cat((-x2, x1), dim=-1)

    return (x * cos.unsqueeze(0).unsqueeze(0)) + (rotated * sin.unsqueeze(0).unsqueeze(0))


class RoPEMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.rotary_emb = RotaryEmbedding(self.head_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x is [B, seq_len, embed_dim] (decoder: seq_len=C; surrogate: seq_len=n)
        B, seq_len, _ = x.shape

        q = self.q_proj(x).view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(seq_len, x.device)
        q = apply_rotary_pos_emb(q, cos, sin)
        k = apply_rotary_pos_emb(k, cos, sin)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).contiguous().view(B, seq_len, self.embed_dim)
        return self.out_proj(context)


class RoPETransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = RoPEMultiheadAttention(d_model, nhead, dropout=dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src):
        src2 = self.norm1(src)
        src2 = self.self_attn(src2)
        src = src + self.dropout1(src2)

        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src
