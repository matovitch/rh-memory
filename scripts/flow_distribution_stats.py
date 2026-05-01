"""Distribution statistics for flow evaluation scripts."""

from __future__ import annotations

import torch
from torch import Tensor


DEFAULT_QUANTILES = (0.01, 0.05, 0.50, 0.95, 0.99)


def parse_step_counts(value: str) -> tuple[int, ...]:
    try:
        steps = tuple(int(part) for part in value.split(",") if part)
    except ValueError as error:
        raise ValueError("step counts must be comma-separated integers") from error
    if not steps or any(step <= 0 for step in steps):
        raise ValueError("step counts must contain positive integers")
    return steps


def distribution_stats(
    x: Tensor,
    *,
    quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
) -> dict[str, float]:
    flat = x.detach().float().flatten()
    if flat.numel() == 0:
        raise ValueError("cannot compute distribution stats for an empty tensor")
    stats = {
        "mean": flat.mean().item(),
        "std": flat.std(unbiased=False).item(),
        "rms": flat.square().mean().sqrt().item(),
        "l1_per_sample": x.detach().float().flatten(1).abs().sum(dim=1).mean().item(),
        "min": flat.min().item(),
        "max": flat.max().item(),
        "frac_lt_0": (flat < 0.0).float().mean().item(),
        "frac_gt_1": (flat > 1.0).float().mean().item(),
    }
    q_values = torch.quantile(flat, torch.tensor(quantiles, device=flat.device, dtype=flat.dtype))
    for quantile, q_value in zip(quantiles, q_values, strict=True):
        stats[f"q{int(round(quantile * 100)):02d}"] = q_value.item()
    return stats


def concatenate_batches(batches: list[Tensor]) -> Tensor:
    if not batches:
        raise ValueError("at least one batch is required")
    return torch.cat([batch.detach().cpu() for batch in batches], dim=0)


def distribution_delta(generated: dict[str, float], reference: dict[str, float]) -> dict[str, float]:
    reference_l1 = max(abs(reference["l1_per_sample"]), 1e-12)
    return {
        "abs_mean": abs(generated["mean"] - reference["mean"]),
        "abs_std": abs(generated["std"] - reference["std"]),
        "abs_rms": abs(generated["rms"] - reference["rms"]),
        "rel_l1_mass": abs(generated["l1_per_sample"] - reference["l1_per_sample"]) / reference_l1,
    }
