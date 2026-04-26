from __future__ import annotations

import torch
from jaxtyping import Float, Int
from torch import Tensor


def build_grouped_permutation(n: int, C: int, seed: int, device: torch.device | str) -> Int[Tensor, "N"]:
    """
    Build a grouped permutation of length ``N`` (requires ``N % C == 0``).

    The range ``0..N-1`` is partitioned into ``stride=N//C`` contiguous chunks of size ``C``.
    Each chunk is independently permuted, then offset by ``i*C``.
    """
    if n % C != 0:
        raise ValueError(f"grouped permutation requires n % C == 0, got n={n}, C={C}")
    stride = n // C
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    out = torch.empty(n, dtype=torch.long, device=device)
    for i in range(stride):
        base = i * C
        out[base : base + C] = torch.randperm(C, generator=g, device=device) + base
    return out


def gather_permuted_stream(raw_inputs: Float[Tensor, "B N"], perm_1d: Int[Tensor, "N"]) -> Float[Tensor, "B N"]:
    """Gather along source axis with LPAP permutation indices."""
    perm = perm_1d.long().unsqueeze(0).expand(raw_inputs.shape[0], -1)
    x_perm = torch.gather(raw_inputs, 1, perm)
    return x_perm


def unpermute_from_permuted(x_perm: Float[Tensor, "B N"], perm_1d: Int[Tensor, "N"]) -> Float[Tensor, "B N"]:
    """Map permuted stream ``x_perm[b, j]`` back to source order using ``perm[j]``."""
    B, n = x_perm.shape
    if perm_1d.numel() != n:
        raise ValueError(f"perm length mismatch: perm has {perm_1d.numel()} entries, x has N={n}")
    out = torch.zeros_like(x_perm)
    index = perm_1d.long().unsqueeze(0).expand(B, -1)
    out.scatter_(1, index, x_perm)
    return out