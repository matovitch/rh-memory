"""Run synthetic harmonic inner-autoencoder inference and save energy comparison panels."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from eval_lpap_autoencoder import normalize_energy_pair, resolve_device, save_grayscale_png, square_side
from path_utils import resolve_project_path
from train_flow_symmetric import load_soft_scatter_autoencoder, load_surrogate

from rh_memory.pipeline import PipelineConfig, harmonic_stage, surrogate_stage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--surrogate-checkpoint",
        type=Path,
        default=Path("scripts/checkpoints/surrogate_ce_checkpoint.pt"),
    )
    parser.add_argument(
        "--soft-scatter-checkpoint",
        type=Path,
        default=Path("scripts/checkpoints/decoder_soft_scatter_l1_checkpoint.pt"),
    )
    parser.add_argument("--output", type=Path, default=Path("scripts/outputs/harmonic_autoencoder_eval.png"))
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--raw-energy-scale", type=float, default=1.0)
    parser.add_argument("--projected-energy-scale", type=float, default=1.0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


@torch.no_grad()
def make_projected_energy_batch(
    sample,
    decoder,
    scatter_head,
    config: PipelineConfig,
    *,
    device: torch.device,
    raw_energy_scale: float = 1.0,
    projected_energy_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    raw_energy = sample.raw_inputs.to(device).unsqueeze(1).mul(raw_energy_scale)
    decoder_tokens = sample.decoder_tokens.to(device)
    decoder_logits = decoder(decoder_tokens)[:, : config.C, : config.n]
    projected, _probs, _doubt, _support, _temperature = scatter_head(
        decoder_logits,
        decoder_tokens[..., 0],
        sample.perm_1d.to(device),
    )
    projected_energy = projected.unsqueeze(1).mul(projected_energy_scale)
    return raw_energy, projected_energy


def build_energy_pair_grid(
    raw_energy_images: torch.Tensor,
    projected_energy_images: torch.Tensor,
    *,
    separator: int = 2,
) -> torch.Tensor:
    batch_size, _channels, height, width = raw_energy_images.shape
    vertical = torch.ones((batch_size, 1, height, separator), dtype=torch.float32, device=raw_energy_images.device)
    rows = torch.cat((raw_energy_images, vertical, projected_energy_images), dim=3)
    horizontal = torch.ones((1, 1, separator, rows.shape[3]), dtype=torch.float32, device=rows.device)
    stacked_rows: list[torch.Tensor] = []
    for row_index in range(batch_size):
        stacked_rows.append(rows[row_index : row_index + 1])
        if row_index + 1 != batch_size:
            stacked_rows.append(horizontal)
    return torch.cat(stacked_rows, dim=2).squeeze(0).clamp(0.0, 1.0)


def main() -> None:
    args = parse_args()
    if args.num_samples <= 0:
        raise ValueError(f"num-samples must be positive, got {args.num_samples}")
    batch_size = args.batch_size if args.batch_size is not None else args.num_samples
    if batch_size < args.num_samples:
        raise ValueError(
            f"batch-size must be >= num-samples, got batch-size={batch_size}, num-samples={args.num_samples}"
        )

    device = resolve_device(args.device)
    args.surrogate_checkpoint = resolve_project_path(args.surrogate_checkpoint)
    args.soft_scatter_checkpoint = resolve_project_path(args.soft_scatter_checkpoint)

    surrogate, surrogate_config = load_surrogate(args.surrogate_checkpoint, device)
    decoder, scatter_head, scatter_config, scatter_ckpt = load_soft_scatter_autoencoder(
        args.soft_scatter_checkpoint, device
    )
    if surrogate_config.n != scatter_config.n or surrogate_config.C != scatter_config.C:
        raise ValueError("surrogate and soft-scatter checkpoints must use matching n/C")

    surrogate_temperature = float(scatter_ckpt.get("surrogate_temperature", 1.0))
    seed = args.seed if args.seed is not None else surrogate_config.seed
    torch.manual_seed(seed)
    config = PipelineConfig(
        n=surrogate_config.n,
        C=surrogate_config.C,
        batch_size=batch_size,
        seed=seed,
        fast_k=surrogate_config.fast_k,
        harmonic_decay=surrogate_config.harmonic_decay,
        harmonic_amp_threshold=surrogate_config.harmonic_amp_threshold,
        max_harmonics=surrogate_config.max_harmonics,
    )
    side = square_side(config.n)

    harmonic_stream = harmonic_stage(config=config, device=device)
    surrogate_stream = surrogate_stage(
        harmonic_stream,
        config=config,
        surrogate=surrogate,
        temperature=surrogate_temperature,
    )
    sample = next(surrogate_stream)
    raw_energy, projected_energy = make_projected_energy_batch(
        sample,
        decoder,
        scatter_head,
        config,
        device=device,
        raw_energy_scale=args.raw_energy_scale,
        projected_energy_scale=args.projected_energy_scale,
    )
    raw_energy = raw_energy[: args.num_samples]
    projected_energy = projected_energy[: args.num_samples]

    raw_energy_display, projected_energy_display = normalize_energy_pair(raw_energy, projected_energy, side=side)
    grid = build_energy_pair_grid(raw_energy_display, projected_energy_display)

    output_path = resolve_project_path(args.output)
    save_grayscale_png(grid, output_path)

    reconstruction_l1 = (projected_energy - raw_energy).abs().flatten(1).mean(dim=1)
    cosine_num = (projected_energy.flatten(1) * raw_energy.flatten(1)).sum(dim=1)
    cosine_den = torch.linalg.vector_norm(projected_energy.flatten(1), ord=2, dim=1) * torch.linalg.vector_norm(
        raw_energy.flatten(1), ord=2, dim=1
    )
    cosine = cosine_num / cosine_den.clamp_min(1e-8)

    print(f"Using device: {device}")
    print(f"Surrogate checkpoint: {args.surrogate_checkpoint}")
    print(f"Soft-scatter checkpoint: {args.soft_scatter_checkpoint}")
    print(f"Surrogate temperature: {surrogate_temperature}")
    print("Columns: raw_harmonic_energy | projected_energy")
    print(f"Saved comparison grid to {output_path}")
    print("")
    for sample_index in range(args.num_samples):
        print(
            f"sample {sample_index:>2} | "
            f"energy_l1 {reconstruction_l1[sample_index].item():.6f} | "
            f"cosine {cosine[sample_index].item():.6f}"
        )


if __name__ == "__main__":
    main()
