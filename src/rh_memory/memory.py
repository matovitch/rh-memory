"""Unified GPU exact parallel memory state utilities."""

from __future__ import annotations

from dataclasses import dataclass

import torch

# We'll use the triton operations once implemented, falling back to python ops for tests
from ._python_ops import python_fast_rh_write_batched

def compute_write_gammas(
    values: torch.Tensor,
    alpha: float,
    epsilon: float = 0.05,
) -> torch.Tensor:
    if not 0.0 <= epsilon < 1.0:
        raise ValueError("epsilon must be in [0, 1)")

    return epsilon + (1.0 - epsilon) * (1.0 - torch.exp(-alpha * values.abs()))

@dataclass
class BatchedMemoryState:
    values: torch.Tensor
    dib: torch.Tensor
    gamma: torch.Tensor
    alpha: float = 1.0
    epsilon: float = 0.05
    a: int = 1
    k: int = 24  # Max probe steps

    @classmethod
    def empty(
        cls,
        batch_size: int,
        capacity: int,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cuda",
        alpha: float = 1.0,
        epsilon: float = 0.05,
        a: int = 1,
        k: int = 24,
    ) -> "BatchedMemoryState":
        values = torch.zeros(batch_size, capacity, dtype=dtype, device=device)
        dib = torch.zeros(batch_size, capacity, dtype=torch.long, device=device)
        gamma = torch.zeros(batch_size, capacity, dtype=dtype, device=device)
        return cls(values=values, dib=dib, gamma=gamma, alpha=alpha, epsilon=epsilon, a=a, k=k)

    def write(
        self,
        incoming_values: torch.Tensor,
    ) -> "BatchedMemoryState":
        self.advance_time()
        
        incoming_gammas = compute_write_gammas(incoming_values, self.alpha, self.epsilon)
        
        # Will be replaced/updated with literal exact triton op
        self.values, self.dib, self.gamma = python_fast_rh_write_batched(
            self.values,
            self.dib,
            self.gamma,
            incoming_values,
            incoming_gammas,
            self.a,
            self.k,
        )
        return self

    def advance_time(self) -> "BatchedMemoryState":
        self.values *= self.gamma
        return self
