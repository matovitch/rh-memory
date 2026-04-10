"""Compiled CPU RH Torch extension bindings."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.cpp_extension import load


def extension_available() -> bool:
    return hasattr(torch.ops.rh_memory, "cpu_rh_write")


def cpu_rh_advance_time(
    table_values: torch.Tensor,
    table_gamma: torch.Tensor,
    cutoff_bound_slow_mag: torch.Tensor,
    cutoff_bound_slow_gamma: torch.Tensor,
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
    table_values: torch.Tensor,
    table_dib: torch.Tensor,
    table_gamma: torch.Tensor,
    incoming_values: torch.Tensor,
    incoming_indices: torch.Tensor,
    incoming_gammas: torch.Tensor,
    capacity: int,
    a: int = 1,
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
        a,
    )


def cpu_rh_write_batched(
    table_values: torch.Tensor,
    table_dib: torch.Tensor,
    table_gamma: torch.Tensor,
    incoming_values: torch.Tensor,
    incoming_indices: torch.Tensor,
    incoming_gammas: torch.Tensor,
    capacity: int,
    a: int = 1,
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
        a,
    )