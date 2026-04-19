"""
Shared harmonic synthesis + LPAP permutation helpers for surrogate / decoder experiments.

See doc_llm/pooling.md (LPAP routing): perm[j] is the source index read at permuted slot j.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
from torch import Tensor

from rh_memory._python_ops import python_linear_probing_amplitude_pooling
from rh_memory._triton_ops import triton_linear_probing_amplitude_pooling


def max_padded_length(n: int, C: int) -> int:
    """Padded sequence length for routing when length must be a multiple of ``C`` (stride grid)."""
    return n + (C - (n % C)) % C


def harmonic_raw_batch(
    chunk_size: int,
    n: int,
    device: torch.device | str,
    harmonic_decay: float,
    harmonic_amp_threshold: float,
    max_harmonics: int,
) -> Tensor:
    """Pseudo-harmonic peaks (shared by train_surrogate / train_decoder_surrogate)."""
    t = torch.linspace(0.0, 1.0, n, device=device, dtype=torch.float32).unsqueeze(0).expand(
        chunk_size, n
    )
    gamma = harmonic_decay
    tau = harmonic_amp_threshold
    max_h = max_harmonics

    sum_peaks = torch.zeros(chunk_size, n, device=device, dtype=torch.float32)
    k_h = 1
    while k_h <= max_h:
        sigma_k = gamma**k_h
        if sigma_k < tau:
            break
        z = torch.randn(chunk_size, 1, device=device, dtype=torch.float32)
        alpha_k = z * sigma_k

        a_k = torch.empty(chunk_size, 1, device=device, dtype=torch.float32).uniform_(1.0, 5.0)
        phi_k = torch.empty(chunk_size, 1, device=device, dtype=torch.float32).uniform_(
            -math.pi, math.pi
        )

        angle = k_h * math.pi * t + phi_k
        envelope_inner = (1.0 - torch.sin(angle).abs()).clamp(min=1e-8)
        envelope = envelope_inner.pow(torch.exp(a_k))

        sum_peaks += alpha_k * envelope
        k_h += 1

    signs = torch.empty(chunk_size, n, device=device, dtype=torch.float32).uniform_(0.0, 1.0).round() * 2 - 1
    return sum_peaks * signs


def pad_for_lpap(raw_inputs: Tensor, incoming_carry_id: Tensor, C: int) -> Tuple[Tensor, Tensor, int]:
    """Pad sequence length to a multiple of C for LPAP."""
    n = raw_inputs.shape[1]
    pad_len = (C - (n % C)) % C
    if pad_len > 0:
        raw_inputs_padded = torch.nn.functional.pad(raw_inputs, (0, pad_len), value=0.0)
        incoming_carry_id_padded = torch.nn.functional.pad(incoming_carry_id, (0, pad_len), value=-1)
    else:
        raw_inputs_padded = raw_inputs
        incoming_carry_id_padded = incoming_carry_id
    n_pad = raw_inputs_padded.shape[1]
    return raw_inputs_padded, incoming_carry_id_padded, n_pad


def lpap_pool(
    table_values: Tensor,
    table_dib: Tensor,
    table_carry_id: Tensor,
    raw_inputs_padded: Tensor,
    incoming_carry_id_padded: Tensor,
    k_eff: int,
    seed: int,
    device: torch.device | str,
) -> Tuple[Tensor, Tensor, Tensor]:
    if "cuda" in str(device):
        return triton_linear_probing_amplitude_pooling(
            table_values,
            table_dib,
            table_carry_id,
            raw_inputs_padded,
            incoming_carry_id_padded,
            k=k_eff,
            seed=seed,
        )
    return python_linear_probing_amplitude_pooling(
        table_values,
        table_dib,
        table_carry_id,
        raw_inputs_padded,
        incoming_carry_id_padded,
        k=k_eff,
        seed=seed,
    )


def inv_perm_from_perm(perm: Tensor) -> Tensor:
    """perm[j] = source index at permuted slot j. Returns inv_perm[s] = j of shape [n_pad]."""
    n_pad = perm.numel()
    inv = torch.empty(n_pad, dtype=torch.long, device=perm.device)
    inv[perm.long()] = torch.arange(n_pad, device=perm.device, dtype=torch.long)
    return inv


def gather_permuted_stream(
    raw_inputs_padded: Tensor,
    incoming_carry_id_padded: Tensor,
    perm_1d: Tensor,
) -> Tuple[Tensor, Tensor]:
    """Gather along source axis with LPAP permutation indices."""
    perm = perm_1d.long().unsqueeze(0).expand(raw_inputs_padded.shape[0], -1)
    x_perm = torch.gather(raw_inputs_padded, 1, perm)
    carry_perm = torch.gather(incoming_carry_id_padded, 1, perm)
    return x_perm, carry_perm


def surrogate_teacher_one_hot(
    out_carry_id: Tensor,
    perm_1d: Tensor,
) -> Tensor:
    """
    Teacher [B, n_pad, C]: slot j hits bucket c iff out_carry_id[b, c] == perm_1d[j].
    """
    B, _ = out_carry_id.shape
    n_pad = perm_1d.numel()
    s_bj = perm_1d.unsqueeze(0).expand(B, n_pad)
    eq = out_carry_id.unsqueeze(2).long() == s_bj.unsqueeze(1).long()
    return eq.transpose(1, 2).float()


def surrogate_bucket_tokens_from_logits(
    x_perm: Tensor,
    logits: Tensor,
    C: int,
) -> Tuple[Tensor, Tensor]:
    """
    Surrogate-only [B, C, 2]: per bucket column c, j*(c) = argmax_j logits[b, j, c].

    Value V[c] = x_perm[b, j*]; D[c] = (c - (j* % C)) mod C as float.
    Returns (bucket_tokens [B,C,2], j_star [B,C]).
    """
    logits_c = logits[:, :, :C]
    j_star = logits_c.argmax(dim=1)
    V = torch.gather(x_perm, 1, j_star)
    B, _ = x_perm.shape
    device = x_perm.device
    c_idx = torch.arange(C, device=device, dtype=torch.long).view(1, C).expand(B, -1)
    D = (c_idx - (j_star % C).long()) % C
    bucket_tokens = torch.stack([V, D.float()], dim=-1)
    return bucket_tokens, j_star


def decoder_targets_from_j_star(
    j_star: Tensor,
    n_seq: int,
    valid_bucket: Tensor | None = None,
) -> Tensor:
    """
    Sparse ``[B, C, n_seq]`` one-hot at ``j_star[b, c]`` — permuted-slot targets aligned with
    ``surrogate_bucket_tokens_from_logits`` (same forward pass).

    If ``valid_bucket`` is omitted, every ``(b, c)`` is supervised.
    """
    B, Cb = j_star.shape
    device = j_star.device
    if valid_bucket is None:
        valid_bucket = torch.ones(B, Cb, dtype=torch.bool, device=device)
    targets = torch.zeros(B, Cb, n_seq, dtype=torch.float32, device=device)
    j_clamped = j_star.long().clamp(0, n_seq - 1)
    targets.scatter_(2, j_clamped.unsqueeze(2).long(), valid_bucket.unsqueeze(2).float())
    return targets
