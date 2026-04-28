from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rh_memory.flow_eval import distribution_delta, distribution_stats, parse_step_counts


def test_parse_step_counts() -> None:
    assert parse_step_counts("1,2,4,8") == (1, 2, 4, 8)


def test_distribution_stats_are_finite_and_ordered() -> None:
    x = torch.linspace(-1.0, 2.0, 24).reshape(2, 1, 12)

    stats = distribution_stats(x)

    assert stats["q01"] <= stats["q05"] <= stats["q50"] <= stats["q95"] <= stats["q99"]
    assert stats["frac_lt_0"] > 0.0
    assert stats["frac_gt_1"] > 0.0
    assert all(torch.isfinite(torch.tensor(value)) for value in stats.values())


def test_distribution_delta_uses_unpaired_stats() -> None:
    reference = distribution_stats(torch.ones(2, 1, 4))
    generated = distribution_stats(torch.full((2, 1, 4), 2.0))

    delta = distribution_delta(generated, reference)

    assert delta["abs_mean"] == 1.0
    assert delta["abs_rms"] == 1.0
    assert delta["rel_l1_mass"] == 1.0