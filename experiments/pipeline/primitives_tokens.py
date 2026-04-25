from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor


def reshape_permuted_to_bucket_tokens(x_perm: Tensor, C: int) -> Tensor:
    """
    Convert permuted stream ``[B, n]`` to surrogate input ``[B, C, stride]``.
    """
    B, n = x_perm.shape
    if n % C != 0:
        raise ValueError(f"reshape requires n % C == 0, got n={n}, C={C}")
    stride = n // C
    return x_perm.view(B, stride, C).transpose(1, 2).contiguous()


def surrogate_bucket_tokens_from_logits(
    x_flat: Tensor,
    logits: Tensor,
    C: int,
    k_eff: int,
) -> Tuple[Tensor, Tensor]:
    """
    Surrogate-only [B, C, 2] from bucket-major logits ``[B, C, n]``.

    Per bucket column c, j*(c) = argmax_j logits[b, c, j].
    Value V[c] = x_flat[b, j*]; D[c] = (c - (j* % C)) mod C normalized by ``k_eff``.
    Returns (bucket_tokens [B,C,2], j_star [B,C]).
    """
    if k_eff <= 0:
        raise ValueError(f"k_eff must be >= 1, got {k_eff}")
    logits_c = logits[:, :C, :]
    j_star = logits_c.argmax(dim=2)
    V = torch.gather(x_flat, 1, j_star)
    B, _ = x_flat.shape
    device = x_flat.device
    c_idx = torch.arange(C, device=device, dtype=torch.long).view(1, C).expand(B, -1)
    D = (c_idx - (j_star % C).long()) % C
    dib_norm = D.float() / float(k_eff)
    bucket_tokens = torch.stack([V, dib_norm], dim=-1)
    return bucket_tokens, j_star


def reconstructor_tokens_from_decoder_logits(
    x_perm: Tensor,
    decoder_logits: Tensor,
    perm_1d: Tensor,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Build reconstructor tokens ``[B, C, 3]`` from decoder logits ``[B, C, n]``:
    ``[selected_value, normalized_source_index, selected_logit]``.
    """
    B, C, n = decoder_logits.shape
    if x_perm.shape != (B, n):
        raise ValueError(f"x_perm must have shape ({B}, {n}), got {tuple(x_perm.shape)}")
    if perm_1d.numel() != n:
        raise ValueError(f"perm length mismatch: perm has {perm_1d.numel()} entries, logits n={n}")

    j_star = decoder_logits.argmax(dim=2)
    selected_value = torch.gather(x_perm, 1, j_star)
    selected_logit = torch.gather(decoder_logits, 2, j_star.unsqueeze(2)).squeeze(2)
    perm = perm_1d.long().unsqueeze(0).expand(B, -1)
    source_index = torch.gather(perm, 1, j_star).float()
    denom = max(1, n - 1)
    source_index_norm = 2.0 * (source_index / float(denom)) - 1.0

    tokens = torch.stack([selected_value, source_index_norm, selected_logit], dim=-1)
    return tokens, j_star, source_index_norm
