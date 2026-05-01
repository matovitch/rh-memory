from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from eval_lpap_autoencoder import build_autoencoder_from_checkpoint, build_comparison_grid, normalize_energy_pair

from rh_memory.decoder import RHDecoder
from rh_memory.decoder_scatter import SoftScatterReconstructionHead
from rh_memory.flow_models import DilatedConvFlow1d
from rh_memory.pipeline import PipelineConfig
from rh_memory.surrogate import RHSurrogate


def _make_checkpoint() -> dict:
    config = PipelineConfig(n=16, C=4, batch_size=2, seed=123, fast_k=1.0)
    image_to_energy = DilatedConvFlow1d(
        sequence_length=16,
        width=8,
        time_dim=8,
        dilation_cycles=1,
        dilations=(1, 2),
        kernel_size=3,
    )
    energy_to_image = DilatedConvFlow1d(
        sequence_length=16,
        width=8,
        time_dim=8,
        dilation_cycles=1,
        dilations=(1, 2),
        kernel_size=3,
    )
    surrogate = RHSurrogate(
        sequence_length=16,
        bucket_count=4,
        stride=4,
        fast_k=1.0,
        d_model=32,
        n_heads=4,
        num_layers=1,
        dim_feedforward=64,
    )
    decoder = RHDecoder(
        sequence_length=16,
        bucket_count=4,
        d_model=32,
        n_heads=4,
        num_layers=1,
        dim_feedforward=64,
    )
    scatter_head = SoftScatterReconstructionHead(init_temperature=1.0, min_temperature=0.05)
    return {
        "version": "v1_lpap_autoencoder_e2e",
        "config": config.to_dict(),
        "flow_model_config": image_to_energy.config_dict(),
        "surrogate_model_config": {
            "d_model": 32,
            "n_heads": 4,
            "num_layers": 1,
            "dim_feedforward": 64,
        },
        "decoder_model_config": {
            "d_model": 32,
            "n_heads": 4,
            "num_layers": 1,
            "dim_feedforward": 64,
        },
        "branch": {
            "name": "C4",
            "C": 4,
            "fast_k": 1.0,
            "surrogate_temperature": 1.0,
        },
        "image_to_energy_flow_state_dict": image_to_energy.state_dict(),
        "energy_to_image_flow_state_dict": energy_to_image.state_dict(),
        "surrogate_state_dict": surrogate.state_dict(),
        "decoder_state_dict": decoder.state_dict(),
        "scatter_head_state_dict": scatter_head.state_dict(),
        "image_to_energy_steps": 2,
        "energy_to_image_steps": 3,
        "image_scale": 1.0,
    }


def test_build_autoencoder_from_checkpoint_reconstructs_model() -> None:
    model, side, image_scale = build_autoencoder_from_checkpoint(_make_checkpoint(), torch.device("cpu"))

    output = model(torch.randn(2, 1, 16))

    assert side == 4
    assert image_scale == 1.0
    assert output.reconstructed_image.shape == (2, 1, 16)
    assert output.learned_energy.shape == (2, 1, 16)
    assert output.projected_energy.shape == (2, 1, 16)


def test_normalize_energy_pair_uses_absolute_shared_scale() -> None:
    learned = torch.tensor([[[-1.0, 1.0, -2.0, 3.0]]], dtype=torch.float32)
    projected = torch.tensor([[[1.0, -1.0, 1.0, -1.0]]], dtype=torch.float32)

    learned_image, projected_image = normalize_energy_pair(learned, projected, side=2)

    assert learned_image.shape == (1, 1, 2, 2)
    assert projected_image.shape == (1, 1, 2, 2)
    assert torch.isclose(learned_image[0, 0, 0, 0], torch.tensor(0.0))
    assert torch.isclose(learned_image[0, 0, 1, 1], torch.tensor(1.0))
    assert torch.isclose(projected_image[0, 0, 0, 0], torch.tensor(0.0))


def test_build_comparison_grid_layout() -> None:
    panels = torch.zeros(2, 1, 2, 2)

    grid = build_comparison_grid(panels, panels, panels, panels, separator=1)

    assert grid.shape == (1, 5, 11)
