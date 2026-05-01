from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from eval_harmonic_autoencoder import (
    build_energy_panel_grid,
    make_decoder_hard_scatter_energy_batch,
    make_lpap_bucket_energy_batch,
    make_projected_energy_batch,
    make_surrogate_hard_scatter_energy_batch,
    normalize_energy_panels,
)

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


def test_make_lpap_bucket_energy_batch_scatters_selected_values_to_source_order() -> None:
    config = PipelineConfig(n=16, C=4, batch_size=2, seed=123, fast_k=1.0)
    sample = SimpleNamespace(
        x_perm=torch.randn(2, 16),
        perm_1d=torch.arange(16),
    )

    lpap_energy = make_lpap_bucket_energy_batch(sample, config, device=torch.device("cpu"))

    assert lpap_energy.shape == (2, 1, 16)
    assert (lpap_energy >= 0).all()
    assert (lpap_energy > 0).sum(dim=2).max() <= config.C


def test_make_surrogate_hard_scatter_energy_batch_uses_argmax_logits() -> None:
    config = PipelineConfig(n=4, C=2, batch_size=1, seed=123, fast_k=1.0)
    surrogate_logits = torch.zeros(1, 2, 4)
    surrogate_logits[0, 0, 1] = 10.0
    surrogate_logits[0, 1, 3] = 10.0
    sample = SimpleNamespace(
        surrogate_logits=surrogate_logits,
        decoder_tokens=torch.tensor([[[2.0, 0.0, 0.0], [-3.0, 0.0, 0.0]]]),
        perm_1d=torch.tensor([2, 0, 3, 1]),
    )

    surrogate_energy = make_surrogate_hard_scatter_energy_batch(sample, config, device=torch.device("cpu"))

    assert torch.equal(surrogate_energy, torch.tensor([[[2.0, -3.0, 0.0, 0.0]]]))


class FixedDecoder(torch.nn.Module):
    def forward(self, decoder_tokens: torch.Tensor) -> torch.Tensor:
        logits = torch.zeros(decoder_tokens.shape[0], decoder_tokens.shape[1], 4)
        logits[0, 0, 2] = 10.0
        logits[0, 1, 0] = 10.0
        return logits


def test_make_decoder_hard_scatter_energy_batch_uses_decoder_argmax_logits() -> None:
    config = PipelineConfig(n=4, C=2, batch_size=1, seed=123, fast_k=1.0)
    sample = SimpleNamespace(
        decoder_tokens=torch.tensor([[[2.0, 0.0, 0.0], [-3.0, 0.0, 0.0]]]),
        perm_1d=torch.tensor([2, 0, 3, 1]),
    )

    decoder_energy = make_decoder_hard_scatter_energy_batch(
        sample,
        FixedDecoder(),
        config,
        device=torch.device("cpu"),
    )

    assert torch.equal(decoder_energy, torch.tensor([[[0.0, 0.0, -3.0, 2.0]]]))


def test_normalize_energy_panels_uses_absolute_shared_scale() -> None:
    raw = torch.tensor([[[-1.0, 1.0, -2.0, 3.0]]], dtype=torch.float32)
    lpap = torch.tensor([[[0.0, -2.0, 0.0, 0.0]]], dtype=torch.float32)
    surrogate = torch.tensor([[[0.0, 0.0, -3.0, 0.0]]], dtype=torch.float32)
    decoder = torch.tensor([[[0.0, -2.0, 0.0, 0.0]]], dtype=torch.float32)
    projected = torch.tensor([[[1.0, -1.0, 1.0, -1.0]]], dtype=torch.float32)

    raw_image, lpap_image, surrogate_image, decoder_image, projected_image = normalize_energy_panels(
        raw, lpap, surrogate, decoder, projected, side=2
    )

    assert raw_image.shape == (1, 1, 2, 2)
    assert lpap_image.shape == (1, 1, 2, 2)
    assert surrogate_image.shape == (1, 1, 2, 2)
    assert decoder_image.shape == (1, 1, 2, 2)
    assert projected_image.shape == (1, 1, 2, 2)
    assert torch.isclose(raw_image[0, 0, 0, 0], torch.tensor(1.0 / 3.0))
    assert torch.isclose(raw_image[0, 0, 1, 1], torch.tensor(1.0))
    assert torch.isclose(lpap_image[0, 0, 0, 1], torch.tensor(2.0 / 3.0))
    assert torch.isclose(surrogate_image[0, 0, 1, 0], torch.tensor(1.0))
    assert torch.isclose(decoder_image[0, 0, 0, 1], torch.tensor(2.0 / 3.0))
    assert torch.isclose(projected_image[0, 0, 0, 0], torch.tensor(1.0 / 3.0))


def test_build_energy_panel_grid_layout() -> None:
    panels = torch.zeros(2, 1, 2, 2)

    grid = build_energy_panel_grid(panels, panels, panels, panels, panels, separator=1)

    assert grid.shape == (1, 5, 14)
