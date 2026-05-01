from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor


def reshape_permuted_to_bucket_tokens(x_perm: Float[Tensor, "B N"], C: int) -> Float[Tensor, "B C stride"]:
    """
    Convert permuted stream ``[B, N]`` to surrogate input ``[B, C, stride]``.
    """
    B, n = x_perm.shape
    if n % C != 0:
        raise ValueError(f"reshape requires N % C == 0, got N={n}, C={C}")
    stride = n // C
    return x_perm.view(B, stride, C).transpose(1, 2).contiguous()


def normalized_entropy(probs: Float[Tensor, "... N"], eps: float = 1e-12) -> Float[Tensor, "..."]:
    """Return entropy normalized to ``[0, 1]`` along the last dimension."""
    n = probs.size(-1)
    if n <= 1:
        return torch.zeros_like(probs[..., 0])
    safe_probs = probs.clamp_min(eps)
    entropy = -(probs * safe_probs.log()).sum(dim=-1)
    return entropy / math.log(n)


def scalar_dib_table(
    C: int,
    n: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Float[Tensor, "C N"]:
    """Build scalar ring-displacement table normalized to ``[0, 1]``."""
    if C <= 0:
        raise ValueError(f"C must be positive, got {C}")
    if n <= 0:
        raise ValueError(f"N must be positive, got {n}")
    bucket_id = torch.arange(C, device=device, dtype=torch.long).view(C, 1)
    slot_id = torch.arange(n, device=device, dtype=torch.long).view(1, n)
    dib = (bucket_id - (slot_id % C)) % C
    return dib.to(dtype=dtype) / float(max(1, C - 1))


def decoder_tokens_from_surrogate_logits_soft(
    x_perm: Float[Tensor, "B N"],
    surrogate_logits: Float[Tensor, "B C N"],
    *,
    temperature: float = 1.0,
) -> Float[Tensor, "B C 3"]:
    """
    Build differentiable decoder tokens from surrogate logits.

    Output channels are ``[soft_amplitude, soft_normalized_dib, surrogate_doubt]``.
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    B, C, n = surrogate_logits.shape
    if x_perm.shape != (B, n):
        raise ValueError(f"x_perm must have shape (B={B}, N={n}), got {tuple(x_perm.shape)}")

    probs = F.softmax(surrogate_logits / temperature, dim=-1)
    soft_amplitude = torch.einsum("bcn,bn->bc", probs, x_perm)
    dib = scalar_dib_table(C, n, device=surrogate_logits.device, dtype=surrogate_logits.dtype)
    soft_normalized_dib = torch.einsum("bcn,cn->bc", probs, dib)
    surrogate_doubt = normalized_entropy(probs)
    return torch.stack([soft_amplitude, soft_normalized_dib, surrogate_doubt], dim=-1)
