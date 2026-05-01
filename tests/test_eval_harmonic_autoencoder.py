from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from eval_harmonic_autoencoder import build_energy_pair_grid, make_projected_energy_batch

from rh_memory.decoder import RHDecoder
from rh_memory.decoder_scatter import SoftScatterReconstructionHead
from rh_memory.pipeline import PipelineConfig


def test_make_projected_energy_batch_returns_expected_shapes() -> None:
    config = PipelineConfig(n=16, C=4, batch_size=2, seed=123, fast_k=1.0)
    decoder = RHDecoder(sequence_length=16, bucket_count=4, d_model=32, n_heads=4, num_layers=1, dim_feedforward=64)
    scatter_head = SoftScatterReconstructionHead(init_temperature=1.0, min_temperature=0.05)
    sample = SimpleNamespace(
        raw_inputs=torch.randn(2, 16),
        decoder_tokens=torch.randn(2, 4, 3),
        perm_1d=torch.arange(16),
    )

    raw_energy, projected_energy = make_projected_energy_batch(
        sample,
        decoder,
        scatter_head,
        config,
        device=torch.device("cpu"),
    )

    assert raw_energy.shape == (2, 1, 16)
    assert projected_energy.shape == (2, 1, 16)


def test_build_energy_pair_grid_layout() -> None:
    panels = torch.zeros(2, 1, 2, 2)

    grid = build_energy_pair_grid(panels, panels, separator=1)

    assert grid.shape == (1, 5, 5)
