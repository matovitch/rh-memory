from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from train_lpap_autoencoder import CHECKPOINT_VERSION, make_e2e_checkpoint, validate_directional_flow_pair

from rh_memory.decoder import RHDecoder
from rh_memory.decoder_scatter import SoftScatterReconstructionHead
from rh_memory.lpap_autoencoder import LPAPAutoencoder, LPAPBottleneckBranch
from rh_memory.pipeline import PipelineConfig
from rh_memory.surrogate import RHSurrogate
from rh_memory.training_seed import TrainingSeed


class TinyFlow(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return x * self.weight + t.view(t.shape[0], 1, 1) * 0.0


def flow_checkpoint(direction: str) -> dict:
    return {
        "version": "v1_directional_flow_matching",
        "direction": direction,
        "flow_model_config": {
            "sequence_length": 16,
            "width": 8,
            "time_dim": 8,
            "dilation_cycles": 1,
            "dilations": [1, 2],
            "kernel_size": 3,
        },
    }


def test_validate_directional_flow_pair_accepts_matching_configs() -> None:
    validate_directional_flow_pair(flow_checkpoint("image-to-energy"), flow_checkpoint("energy-to-image"))


def test_validate_directional_flow_pair_rejects_direction_mismatch() -> None:
    try:
        validate_directional_flow_pair(flow_checkpoint("energy-to-image"), flow_checkpoint("energy-to-image"))
    except ValueError as error:
        assert "wrong direction" in str(error)
    else:
        raise AssertionError("expected direction mismatch to raise")


def test_make_e2e_checkpoint_schema_contains_component_state_dicts() -> None:
    config = PipelineConfig(n=16, C=4, batch_size=2, seed=123, fast_k=1.0)
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
    decoder = RHDecoder(sequence_length=16, bucket_count=4, d_model=32, n_heads=4, num_layers=1, dim_feedforward=64)
    branch = LPAPBottleneckBranch(
        name="C4",
        surrogate=surrogate,
        decoder=decoder,
        scatter_head=SoftScatterReconstructionHead(init_temperature=1.0, min_temperature=0.05),
        perm_1d=torch.arange(16),
        fast_k=1.0,
    )
    model = LPAPAutoencoder(
        image_to_energy_flow=TinyFlow(),
        energy_to_image_flow=TinyFlow(),
        bottleneck_branches=[branch],
        image_to_energy_steps=2,
        energy_to_image_steps=3,
        default_branch="C4",
    )
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    args = Namespace(
        branch_name="C4",
        image_to_energy_steps=2,
        energy_to_image_steps=3,
        lpap_weight=0.25,
        lr=1e-4,
        image_scale=1.0,
        image_manifest=Path("data/train/manifest.json"),
        eval_image_manifest=Path("data/eval/manifest.json"),
        surrogate_checkpoint=Path("scripts/checkpoints/surrogate.pt"),
        soft_scatter_checkpoint=Path("scripts/checkpoints/scatter.pt"),
        image_to_energy_checkpoint=Path("scripts/checkpoints/i2e.pt"),
        energy_to_image_checkpoint=Path("scripts/checkpoints/e2i.pt"),
        surrogate_temperature=1.0,
        seed=123,
    )

    ckpt = make_e2e_checkpoint(
        model=model,
        optimizer=optimizer,
        step=5,
        config=config,
        args=args,
        training_seed=TrainingSeed(seed=123, source="argument"),
        surrogate_ckpt={"version": "v2_ce_no_decoder", "model_config": {"d_model": 32}},
        soft_scatter_ckpt={"version": "v1_decoder_soft_scatter_l1", "model_config": {"d_model": 32}},
        image_to_energy_ckpt=flow_checkpoint("image-to-energy"),
        energy_to_image_ckpt=flow_checkpoint("energy-to-image"),
        metrics={"train_total": 1.0},
    )

    assert ckpt["version"] == CHECKPOINT_VERSION
    assert ckpt["step"] == 5
    assert ckpt["branch"]["name"] == "C4"
    assert ckpt["image_to_energy_steps"] == 2
    assert ckpt["energy_to_image_steps"] == 3
    assert "image_to_energy_flow_state_dict" in ckpt
    assert "energy_to_image_flow_state_dict" in ckpt
    assert "surrogate_state_dict" in ckpt
    assert "decoder_state_dict" in ckpt
    assert "scatter_head_state_dict" in ckpt
    assert ckpt["metrics"] == {"train_total": 1.0}
