from __future__ import annotations

import torch
import torch.nn as nn

from rh_memory.decoder import RHDecoder
from rh_memory.decoder_scatter import SoftScatterReconstructionHead
from rh_memory.lpap_autoencoder import (
    LPAPAutoencoder,
    LPAPAutoencoderLoss,
    LPAPBottleneckBranch,
    lpap_surrogate_targets,
)
from rh_memory.surrogate import RHSurrogate


class TinyFlow(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.1))
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.scale * x + self.bias + t.view(t.shape[0], 1, 1) * 0.01


def _branch(name: str, *, n: int = 16, C: int = 4) -> LPAPBottleneckBranch:
    surrogate = RHSurrogate(
        sequence_length=n,
        bucket_count=C,
        stride=n // C,
        fast_k=1.0,
        d_model=32,
        n_heads=4,
        num_layers=1,
        dim_feedforward=64,
    )
    decoder = RHDecoder(
        sequence_length=n,
        bucket_count=C,
        d_model=32,
        n_heads=4,
        num_layers=1,
        dim_feedforward=64,
    )
    return LPAPBottleneckBranch(
        name=name,
        surrogate=surrogate,
        decoder=decoder,
        scatter_head=SoftScatterReconstructionHead(init_temperature=1.0, min_temperature=0.05),
        perm_1d=torch.arange(n),
        fast_k=1.0,
    )


def test_lpap_surrogate_targets_shapes_and_zero_invalid_weight_handling() -> None:
    x_perm = torch.randn(2, 16)

    targets = lpap_surrogate_targets(x_perm, bucket_count=4, k_eff=1)

    assert targets.target_idx.shape == (2, 4)
    assert targets.valid_bucket.shape == (2, 4)
    assert targets.weights.shape == (2, 4)
    assert targets.valid_bucket.dtype == torch.bool
    assert torch.isfinite(targets.weights).all()


def test_lpap_surrogate_targets_do_not_backpropagate_to_energy() -> None:
    x_perm = torch.randn(2, 16, requires_grad=True)

    targets = lpap_surrogate_targets(x_perm, bucket_count=4, k_eff=1)

    assert not targets.weights.requires_grad
    assert not targets.target_idx.requires_grad


def test_lpap_autoencoder_forward_shapes() -> None:
    n, B = 16, 2
    model = LPAPAutoencoder(
        image_to_energy_flow=TinyFlow(),
        energy_to_image_flow=TinyFlow(),
        bottleneck_branches=[_branch("C4", n=n, C=4)],
        image_to_energy_steps=2,
        energy_to_image_steps=2,
    )
    image_seq = torch.randn(B, 1, n)

    output = model(image_seq)

    assert output.branch_name == "C4"
    assert output.reconstructed_image.shape == (B, 1, n)
    assert output.learned_energy.shape == (B, 1, n)
    assert output.energy_perm.shape == (B, n)
    assert output.surrogate_logits.shape == (B, 4, n)
    assert output.decoder_tokens.shape == (B, 4, 3)
    assert output.decoder_logits.shape == (B, 4, n)
    assert output.projected_energy.shape == (B, 1, n)
    assert output.scatter_probs.shape == (B, 4, n)


def test_lpap_autoencoder_selects_named_branch() -> None:
    n = 16
    model = LPAPAutoencoder(
        image_to_energy_flow=TinyFlow(),
        energy_to_image_flow=TinyFlow(),
        bottleneck_branches=[_branch("C4", n=n, C=4), _branch("C8", n=n, C=8)],
        image_to_energy_steps=1,
        energy_to_image_steps=1,
        default_branch="C4",
    )

    output = model(torch.randn(2, 1, n), branch_name="C8")

    assert output.branch_name == "C8"
    assert output.surrogate_logits.shape == (2, 8, n)
    assert output.decoder_tokens.shape == (2, 8, 3)


def test_lpap_autoencoder_loss_backpropagates_through_trainable_path() -> None:
    n = 16
    i2e_flow = TinyFlow()
    e2i_flow = TinyFlow()
    branch = _branch("C4", n=n, C=4)
    model = LPAPAutoencoder(
        image_to_energy_flow=i2e_flow,
        energy_to_image_flow=e2i_flow,
        bottleneck_branches=[branch],
        image_to_energy_steps=2,
        energy_to_image_steps=2,
    )
    criterion = LPAPAutoencoderLoss(lpap_weight=0.25)
    image_seq = torch.randn(2, 1, n)

    output = model(image_seq)
    loss = criterion(output, image_seq)
    loss.total.backward()

    assert loss.total.dim() == 0
    assert torch.isfinite(loss.total)
    assert torch.isfinite(loss.reconstruction)
    assert torch.isfinite(loss.lpap_surrogate)
    assert i2e_flow.scale.grad is not None
    assert e2i_flow.scale.grad is not None
    assert branch.surrogate.input_proj.weight.grad is not None
    assert branch.decoder.input_proj.weight.grad is not None
    assert branch.scatter_head.raw_temperature.grad is not None
