"""Shared decoder components."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class DecoderConfig:
	n_slots: int
	n_buckets: int
	model_dim: int = 128
	num_heads: int = 4
	num_layers: int = 2
	ff_dim: int | None = None
	dropout: float = 0.0
	rope_theta: float = 10000.0


def encode_memory_type(is_slow: torch.Tensor | bool | int | float, *, device=None, dtype=None) -> torch.Tensor:
	"""Encode fast/slow memory as a signed scalar.

	The default convention is fast = -1 and slow = +1.
	"""
	raw = torch.as_tensor(is_slow, device=device)
	if raw.dtype == torch.bool:
		encoded = raw.to(dtype if dtype is not None else torch.float32)
		return encoded.mul(2.0).sub(1.0)
	tensor = raw.to(dtype if dtype is not None else torch.float32)
	if tensor.min() >= 0 and tensor.max() <= 1:
		return tensor.mul(2.0).sub(1.0)
	return tensor


def build_support_target(original_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
	"""Build a BCE target mask and positive-class weights from dense original values."""
	target = (original_values != 0).to(dtype=original_values.dtype)
	positive_weight = original_values.abs()
	return target, positive_weight


def weighted_bce_with_logits(
	logits: torch.Tensor,
	target: torch.Tensor,
	positive_weight: torch.Tensor | None = None,
	reduction: str = "mean",
) -> torch.Tensor:
	"""Compute BCEWithLogitsLoss with optional magnitude weighting on the positive class."""
	if positive_weight is None:
		return F.binary_cross_entropy_with_logits(logits, target, reduction=reduction)

	loss = F.binary_cross_entropy_with_logits(
		logits,
		target,
		pos_weight=positive_weight,
		reduction="none",
	)
	if reduction == "none":
		return loss
	if reduction == "sum":
		return loss.sum()
	if reduction == "mean":
		return loss.mean()
	raise ValueError("reduction must be one of 'none', 'sum', or 'mean'")


def _maybe_unsqueeze_batch(tensor: torch.Tensor) -> tuple[torch.Tensor, bool]:
	if tensor.dim() == 1:
		return tensor.unsqueeze(0), True
	return tensor, False


def _broadcast_feature(feature: torch.Tensor | bool | int | float, reference: torch.Tensor) -> torch.Tensor:
	tensor = torch.as_tensor(feature, device=reference.device, dtype=reference.dtype)
	while tensor.dim() < reference.dim():
		tensor = tensor.unsqueeze(-1)
	return torch.broadcast_to(tensor, reference.shape)


class RotaryEmbedding(nn.Module):
	def __init__(self, dim: int, base: float = 10000.0) -> None:
		super().__init__()
		if dim % 2 != 0:
			raise ValueError("rotary dimension must be even")
		self.dim = dim
		self.base = base
		inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
		self.register_buffer("inv_freq", inv_freq, persistent=False)

	def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
		positions = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
		frequencies = torch.einsum("i,j->ij", positions, self.inv_freq)
		emb = torch.cat((frequencies, frequencies), dim=-1)
		return emb.cos().to(dtype), emb.sin().to(dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
	x1, x2 = x.chunk(2, dim=-1)
	return torch.cat((-x2, x1), dim=-1)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
	return (x * cos) + (_rotate_half(x) * sin)


class RoPEMultiHeadSelfAttention(nn.Module):
	def __init__(self, model_dim: int, num_heads: int, dropout: float = 0.0, rope_theta: float = 10000.0) -> None:
		super().__init__()
		if model_dim % num_heads != 0:
			raise ValueError("model_dim must be divisible by num_heads")
		self.model_dim = model_dim
		self.num_heads = num_heads
		self.head_dim = model_dim // num_heads
		if self.head_dim % 2 != 0:
			raise ValueError("head dimension must be even for RoPE")
		self.q_proj = nn.Linear(model_dim, model_dim)
		self.k_proj = nn.Linear(model_dim, model_dim)
		self.v_proj = nn.Linear(model_dim, model_dim)
		self.out_proj = nn.Linear(model_dim, model_dim)
		self.dropout = nn.Dropout(dropout)
		self.rotary = RotaryEmbedding(self.head_dim, base=rope_theta)

	def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
		batch_size, seq_len, _ = x.shape
		q = self.q_proj(x)
		k = self.k_proj(x)
		v = self.v_proj(x)

		q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
		k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
		v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

		cos, sin = self.rotary(seq_len, device=x.device, dtype=x.dtype)
		cos = cos.unsqueeze(0).unsqueeze(0)
		sin = sin.unsqueeze(0).unsqueeze(0)
		q = _apply_rope(q, cos, sin)
		k = _apply_rope(k, cos, sin)

		scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
		if key_padding_mask is not None:
			if key_padding_mask.shape != (batch_size, seq_len):
				raise ValueError("key_padding_mask must have shape (batch_size, seq_len)")
			mask = key_padding_mask[:, None, None, :].to(dtype=torch.bool)
			scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)

		attn = torch.softmax(scores, dim=-1)
		attn = self.dropout(attn)
		context = torch.matmul(attn, v)
		context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.model_dim)
		output = self.out_proj(context)
		if key_padding_mask is not None:
			output = output * key_padding_mask.unsqueeze(-1).to(dtype=output.dtype)
		return output


class TransformerBlock(nn.Module):
	def __init__(self, model_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.0, rope_theta: float = 10000.0) -> None:
		super().__init__()
		self.norm_1 = nn.LayerNorm(model_dim)
		self.attention = RoPEMultiHeadSelfAttention(model_dim, num_heads, dropout=dropout, rope_theta=rope_theta)
		self.norm_2 = nn.LayerNorm(model_dim)
		self.ff = nn.Sequential(
			nn.Linear(model_dim, ff_dim),
			nn.GELU(),
			nn.Dropout(dropout),
			nn.Linear(ff_dim, model_dim),
			nn.Dropout(dropout),
		)

	def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
		x = x + self.attention(self.norm_1(x), key_padding_mask=key_padding_mask)
		x = x + self.ff(self.norm_2(x))
		return x


class RHMemoryDecoder(nn.Module):
	"""Shared decoder for fast and slow RH-Memory buckets."""

	def __init__(self, config: DecoderConfig) -> None:
		super().__init__()
		ff_dim = config.ff_dim if config.ff_dim is not None else 4 * config.model_dim
		self.config = config
		self.input_dim = 5
		self.input_proj = nn.Linear(self.input_dim, config.model_dim)
		self.blocks = nn.ModuleList(
			TransformerBlock(
				model_dim=config.model_dim,
				num_heads=config.num_heads,
				ff_dim=ff_dim,
				dropout=config.dropout,
				rope_theta=config.rope_theta,
			)
			for _ in range(config.num_layers)
		)
		self.final_norm = nn.LayerNorm(config.model_dim)
		self.output_head = nn.Linear(config.n_buckets * config.model_dim, config.n_slots)

	def build_token_features(
		self,
		values: torch.Tensor,
		dib: torch.Tensor,
		gamma: torch.Tensor,
		memory_type: torch.Tensor | bool | int | float,
	) -> torch.Tensor:
		values, squeezed = _maybe_unsqueeze_batch(values)
		dib = torch.as_tensor(dib, device=values.device, dtype=values.dtype)
		gamma = torch.as_tensor(gamma, device=values.device, dtype=values.dtype)
		memory_type = torch.as_tensor(memory_type, device=values.device)

		values = values.to(dtype=self.input_proj.weight.dtype)
		dib = torch.broadcast_to(dib.to(dtype=values.dtype), values.shape)
		gamma = torch.broadcast_to(gamma.to(dtype=values.dtype), values.shape)
		memory_type = encode_memory_type(memory_type, device=values.device, dtype=values.dtype)
		memory_type = torch.broadcast_to(memory_type, values.shape)

		sign = torch.where(values >= 0, torch.ones_like(values), -torch.ones_like(values))
		magnitude = values.abs()
		features = torch.stack((sign, magnitude, gamma, dib, memory_type), dim=-1)
		if squeezed:
			features = features.squeeze(0)
		return features

	def forward(
		self,
		values: torch.Tensor,
		dib: torch.Tensor,
		gamma: torch.Tensor,
		memory_type: torch.Tensor | bool | int | float,
		key_padding_mask: torch.Tensor | None = None,
	) -> torch.Tensor:
		features = self.build_token_features(values, dib, gamma, memory_type)
		features, squeezed = _maybe_unsqueeze_batch(features)
		if features.shape[1] != self.config.n_buckets:
			raise ValueError(f"expected {self.config.n_buckets} bucket tokens, got {features.shape[1]}")
		x = self.input_proj(features)
		for block in self.blocks:
			x = block(x, key_padding_mask=key_padding_mask)
		x = self.final_norm(x)
		logits = self.output_head(x.reshape(x.shape[0], -1))
		if squeezed:
			logits = logits.squeeze(0)
		return logits

	def loss(
		self,
		logits: torch.Tensor,
		target_support: torch.Tensor,
		target_magnitude: torch.Tensor | None = None,
		reduction: str = "mean",
	) -> torch.Tensor:
		return weighted_bce_with_logits(logits, target_support, positive_weight=target_magnitude, reduction=reduction)
