from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor

from ._python_ops import python_linear_probing_amplitude_pooling
from ._triton_ops import triton_linear_probing_amplitude_pooling


def lpap_pool(
    table_values: Tensor,
    table_dib: Tensor,
    table_carry_id: Tensor,
    shuffled_inputs: Tensor,
    incoming_carry_id: Tensor,
    k_eff: int,
    device: torch.device | str,
) -> Tuple[Tensor, Tensor, Tensor]:
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
