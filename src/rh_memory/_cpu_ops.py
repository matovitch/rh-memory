"""Compiled CPU RH Torch extension bindings."""

from __future__ import annotations

from pathlib import Path

import torch
from jaxtyping import Float, Int
from torch import Tensor
from torch.utils.cpp_extension import load


def extension_available() -> bool:
    return hasattr(torch.ops.rh_memory, "cpu_rh_write")


def cpu_rh_advance_time(
    table_values: Float[Tensor, "batch capacity"],
    table_gamma: Float[Tensor, "batch capacity"],
    cutoff_bound_slow_mag: Float[Tensor, "batch"],
    cutoff_bound_slow_gamma: Float[Tensor, "batch"],
    delta_steps: int,
):
    source_path = Path(__file__).resolve().parents[2] / "csrc" / "rh_cpu.cpp"
    load(
        name="rh_memory_cpu",
        sources=[str(source_path)],
        extra_cflags=["-O3"],
        is_python_module=False,
        verbose=False,
    )
    return torch.ops.rh_memory.cpu_rh_advance_time(
        table_values,
        table_gamma,
        cutoff_bound_slow_mag,
        cutoff_bound_slow_gamma,
        delta_steps,
    )


def cpu_rh_write(
    table_values: Float[Tensor, "capacity"],
    table_dib: Int[Tensor, "capacity"],
    table_gamma: Float[Tensor, "capacity"],
    incoming_values: Float[Tensor, "n"],
    incoming_indices: Int[Tensor, "n"],
    incoming_gammas: Float[Tensor, "n"],
    capacity: int,
    n: int,
    r: int = 0,
):
    source_path = Path(__file__).resolve().parents[2] / "csrc" / "rh_cpu.cpp"
    load(
        name="rh_memory_cpu",
        sources=[str(source_path)],
        extra_cflags=["-O3"],
        is_python_module=False,
        verbose=False,
    )
    return torch.ops.rh_memory.cpu_rh_write(
        table_values,
        table_dib,
        table_gamma,
        incoming_values,
        incoming_indices,
        incoming_gammas,
        capacity,
        n,
        r,
    )


def cpu_rh_write_batched(
    table_values: Float[Tensor, "batch capacity"],
    table_dib: Int[Tensor, "batch capacity"],
    table_gamma: Float[Tensor, "batch capacity"],
    incoming_values: Float[Tensor, "batch n"],
    incoming_indices: Int[Tensor, "batch n"],
    incoming_gammas: Float[Tensor, "batch n"],
    capacity: int,
    n: int,
    r: int = 0,
):
    source_path = Path(__file__).resolve().parents[2] / "csrc" / "rh_cpu.cpp"
    load(
        name="rh_memory_cpu",
        sources=[str(source_path)],
        extra_cflags=["-O3"],
        is_python_module=False,
        verbose=False,
    )
    return torch.ops.rh_memory.cpu_rh_write_batched(
        table_values,
        table_dib,
        table_gamma,
        incoming_values,
        incoming_indices,
        incoming_gammas,
        capacity,
        n,
        r,
    )