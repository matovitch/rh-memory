from __future__ import annotations

import torch
from jaxtyping import Float, Int
from torch import Tensor

from ._python_ops import python_linear_probing_amplitude_pooling
from ._triton_ops import triton_linear_probing_amplitude_pooling


def lpap_pool(
    table_values: Float[Tensor, "B C"],
    table_dib: Int[Tensor, "B C"],
    table_carry_id: Int[Tensor, "B C"],
    shuffled_inputs: Float[Tensor, "B N"],
    incoming_carry_id: Int[Tensor, "B N"],
    k_eff: int,
    device: torch.device | str,
) -> tuple[Float[Tensor, "B C"], Int[Tensor, "B C"], Int[Tensor, "B C"]]:
    if "cuda" in str(device):
        return triton_linear_probing_amplitude_pooling(
            table_values,
            table_dib,
            table_carry_id,
            shuffled_inputs,
            incoming_carry_id,
            k=k_eff,
        )
    return python_linear_probing_amplitude_pooling(
        table_values,
        table_dib,
        table_carry_id,
        shuffled_inputs,
        incoming_carry_id,
        k=k_eff,
    )
