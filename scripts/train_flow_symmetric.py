"""Train symmetric image/energy flow-matching models."""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import torch
import torch.optim as optim
from path_utils import project_relative_path, resolve_project_path
from torch.utils.data import DataLoader

from rh_memory.decoder import RHDecoder
from rh_memory.decoder_scatter import SoftScatterReconstructionHead
from rh_memory.flow_matching import flow_matching_loss as flow_loss
from rh_memory.flow_models import DilatedConvFlow1d
from rh_memory.hilbert import hilbert_flatten_images, hilbert_metadata
from rh_memory.image_shards import GrayscaleImageShardDataset, InMemoryGrayscaleImageShardDataset
from rh_memory.pipeline import PipelineConfig, harmonic_stage, surrogate_stage
from rh_memory.surrogate import RHSurrogate
from rh_memory.training_seed import apply_training_seed


def relative_l2_percent(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    err = torch.linalg.vector_norm((pred - target).flatten(1), ord=2, dim=1)
    den = torch.linalg.vector_norm(target.flatten(1), ord=2, dim=1).clamp_min(eps)
    return ((err / den).mean().item()) * 100.0


def cosine_similarity(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    pred_flat = pred.flatten(1)
    target_flat = target.flatten(1)
    num = (pred_flat * target_flat).sum(dim=1)
    den = torch.linalg.vector_norm(pred_flat, ord=2, dim=1) * torch.linalg.vector_norm(target_flat, ord=2, dim=1)
    return (num / den.clamp_min(eps)).mean().item()


def tensor_stats(x: torch.Tensor) -> dict[str, float]:
    flat = x.detach().flatten(1)
    return {
        "mean": flat.mean().item(),
        "std": flat.std(unbiased=False).item(),
        "l1": flat.abs().sum(dim=1).mean().item(),
        "rms": flat.square().mean(dim=1).sqrt().mean().item(),
        "min": flat.min().item(),
        "max": flat.max().item(),
    }


def load_surrogate(checkpoint_path: Path, device: torch.device):
    checkpoint_path = resolve_project_path(checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if ckpt.get("version") != "v2_ce_no_decoder":
        raise ValueError(f"Unsupported surrogate checkpoint version {ckpt.get('version')!r}")
    config = PipelineConfig.from_dict(ckpt["config"])
    model_config = ckpt["model_config"]
    model = RHSurrogate(
        sequence_length=config.sequence_length,
        bucket_count=config.C,
        stride=config.stride,
        fast_k=config.fast_k,
        d_model=model_config["d_model"],
        n_heads=model_config["n_heads"],
        num_layers=model_config["num_layers"],
        dim_feedforward=model_config["dim_feedforward"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model, config


def load_soft_scatter_autoencoder(checkpoint_path: Path, device: torch.device):
    checkpoint_path = resolve_project_path(checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if ckpt.get("version") != "v1_decoder_soft_scatter_l1":
        raise ValueError(f"Unsupported soft-scatter checkpoint version {ckpt.get('version')!r}")
    config = PipelineConfig.from_dict(ckpt["config"])
    model_config = ckpt["model_config"]
    decoder = RHDecoder(
        sequence_length=config.sequence_length,
        bucket_count=config.C,
        d_model=model_config["d_model"],
        n_heads=model_config["n_heads"],
        num_layers=model_config["num_layers"],
        dim_feedforward=model_config["dim_feedforward"],
    ).to(device)
    decoder.load_state_dict(ckpt["model_state_dict"])
    scatter_head = SoftScatterReconstructionHead(
        init_temperature=float(ckpt.get("scatter_temperature", 1.0)),
        min_temperature=float(ckpt.get("min_scatter_temperature", 0.05)),
    ).to(device)
    scatter_head.load_state_dict(ckpt["scatter_head_state_dict"])
    decoder.eval()
    scatter_head.eval()
    for param in itertools.chain(decoder.parameters(), scatter_head.parameters()):
        param.requires_grad_(False)
    return decoder, scatter_head, config, ckpt


def parse_dilations(value: str) -> tuple[int, ...]:
    try:
        dilations = tuple(int(part) for part in value.split(",") if part)
    except ValueError as error:
        raise argparse.ArgumentTypeError("dilations must be comma-separated integers") from error
    if not dilations or any(dilation <= 0 for dilation in dilations):
        raise argparse.ArgumentTypeError("dilations must contain positive integers")
    return dilations


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-manifest", type=Path, default=Path("data/grayscale_32x32_torch/manifest.json"))
    parser.add_argument("--surrogate-checkpoint", type=Path, default=Path("scripts/checkpoints/surrogate_ce_checkpoint.pt"))
    parser.add_argument("--soft-scatter-checkpoint", type=Path, default=Path("scripts/checkpoints/decoder_soft_scatter_l1_checkpoint.pt"))
    parser.add_argument(
        "--image-to-energy-checkpoint",
        type=Path,
        default=Path("scripts/checkpoints/symmetric_flow_image_to_energy_checkpoint.pt"),
    )
    parser.add_argument(
        "--energy-to-image-checkpoint",
        type=Path,
        default=Path("scripts/checkpoints/symmetric_flow_energy_to_image_checkpoint.pt"),
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--total-steps", type=int, default=40_000)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--eps", type=float, default=1e-4)
    parser.add_argument("--time-beta-alpha", type=float, default=0.1)
    parser.add_argument("--time-beta-beta", type=float, default=0.1)
    parser.add_argument("--surrogate-temperature", type=float, default=1.0)
    parser.add_argument("--flow-width", type=int, default=128)
    parser.add_argument("--time-dim", type=int, default=128)
    parser.add_argument("--dilation-cycles", type=int, default=2)
    parser.add_argument("--dilations", type=parse_dilations, default=(1, 2, 4, 8, 16, 32, 64, 128))
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--preload-images", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--image-scale", type=float, default=1.0)
    parser.add_argument("--raw-energy-scale", type=float, default=1.0)
    parser.add_argument("--projected-energy-scale", type=float, default=1.0)
    return parser.parse_args()


def make_directional_checkpoint(
    *,
    direction: str,
    state_dict: dict[str, torch.Tensor],
    step: int,
    config: PipelineConfig,
    flow_model_config: dict[str, int | bool | list[int]],
    args,
    scatter_ckpt: dict,
    training_seed,
    metrics: dict,
) -> dict:
    return {
        "version": "v1_directional_flow_matching",
        "checkpoint_role": "directional_flow_matching",
        "source_script": "scripts/train_flow_symmetric.py",
        "direction": direction,
        "step": step,
        "state_dict": state_dict,
        "config": config.to_dict(),
        "flow_model_config": flow_model_config,
        "hilbert": hilbert_metadata(32),
        "normalization": {
            "image_scale": args.image_scale,
            "raw_energy_scale": args.raw_energy_scale,
            "projected_energy_scale": args.projected_energy_scale,
        },
        "surrogate_checkpoint": project_relative_path(args.surrogate_checkpoint),
        "soft_scatter_checkpoint": project_relative_path(args.soft_scatter_checkpoint),
        "soft_scatter_checkpoint_version": scatter_ckpt.get("version"),
        "surrogate_temperature": args.surrogate_temperature,
        "time_sampling": {
            "distribution": "beta",
            "alpha": args.time_beta_alpha,
            "beta": args.time_beta_beta,
        },
        "eps": args.eps,
        "lr": args.lr,
        "seed": args.seed,
        "seed_source": training_seed.source,
        "metrics": metrics,
    }


def make_image_iterator(args, device: torch.device):
    generator = torch.Generator().manual_seed(args.seed)
    if args.preload_images:
        if args.num_workers != 0:
            raise ValueError("--preload-images should use --num-workers 0 to avoid duplicating the in-memory dataset")
        print(f"Preloading image shards into RAM from {args.image_manifest}...")
        dataset = InMemoryGrayscaleImageShardDataset(args.image_manifest, as_float=True)
        print(f"Preloaded {len(dataset)} images as uint8 ({dataset.images.numel() / 1024**3:.2f} GiB)")
    else:
        dataset = GrayscaleImageShardDataset(args.image_manifest, as_float=True)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        generator=generator,
    )
    while True:
        for images in loader:
            yield hilbert_flatten_images(images.to(device)).mul(args.image_scale)


@torch.no_grad()
def make_energy_batch(sample, decoder, scatter_head, config: PipelineConfig, args, device: torch.device):
    raw_energy = sample.raw_inputs.to(device).unsqueeze(1).mul(args.raw_energy_scale)
    decoder_tokens = sample.decoder_tokens.to(device)
    decoder_logits = decoder(decoder_tokens)[:, : config.C, : config.n]
    projected, _probs, _doubt, _support, _temperature = scatter_head(
        decoder_logits,
        decoder_tokens[..., 0],
        sample.perm_1d.to(device),
    )
    projected_energy = projected.unsqueeze(1).mul(args.projected_energy_scale)
    return raw_energy, projected_energy


def main():
    args = parse_args()
    if not (0.0 <= args.eps < 0.5):
        raise ValueError(f"eps must be in [0, 0.5), got {args.eps}")
    if args.time_beta_alpha <= 0.0 or args.time_beta_beta <= 0.0:
        raise ValueError(
            "time-beta-alpha and time-beta-beta must be positive, got "
            f"{args.time_beta_alpha} and {args.time_beta_beta}"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    args.image_to_energy_checkpoint = resolve_project_path(args.image_to_energy_checkpoint)
    args.energy_to_image_checkpoint = resolve_project_path(args.energy_to_image_checkpoint)
    args.surrogate_checkpoint = resolve_project_path(args.surrogate_checkpoint)
    args.soft_scatter_checkpoint = resolve_project_path(args.soft_scatter_checkpoint)

    training_seed = apply_training_seed(args.seed, None)
    args.seed = training_seed.seed
    print(f"Using seed: {training_seed.seed} ({training_seed.source})")

    surrogate, surrogate_config = load_surrogate(args.surrogate_checkpoint, device)
    decoder, scatter_head, scatter_config, scatter_ckpt = load_soft_scatter_autoencoder(args.soft_scatter_checkpoint, device)
    if surrogate_config.n != scatter_config.n or surrogate_config.C != scatter_config.C:
        raise ValueError("surrogate and soft-scatter checkpoints must use matching n/C")

    config = PipelineConfig(
        n=surrogate_config.n,
        C=surrogate_config.C,
        batch_size=args.batch_size,
        seed=args.seed,
        fast_k=surrogate_config.fast_k,
        harmonic_decay=surrogate_config.harmonic_decay,
        harmonic_amp_threshold=surrogate_config.harmonic_amp_threshold,
        max_harmonics=surrogate_config.max_harmonics,
    )
    if config.n != 1024:
        raise ValueError(f"this first flow script expects n=1024 for 32x32 Hilbert images, got {config.n}")

    image_to_energy_flow = DilatedConvFlow1d(
        sequence_length=config.n,
        width=args.flow_width,
        time_dim=args.time_dim,
        dilation_cycles=args.dilation_cycles,
        dilations=args.dilations,
    ).to(device)
    energy_to_image_flow = DilatedConvFlow1d(
        sequence_length=config.n,
        width=args.flow_width,
        time_dim=args.time_dim,
        dilation_cycles=args.dilation_cycles,
        dilations=args.dilations,
    ).to(device)
    optimizer = optim.AdamW(
        itertools.chain(image_to_energy_flow.parameters(), energy_to_image_flow.parameters()),
        lr=args.lr,
    )

    image_iter = make_image_iterator(args, device)
    harmonic_stream = harmonic_stage(config=config, device=device)
    surrogate_stream = surrogate_stage(harmonic_stream, config=config, surrogate=surrogate, temperature=args.surrogate_temperature)

    running_i2e = 0.0
    running_e2i = 0.0
    running_batches = 0
    time_dist = torch.distributions.Beta(args.time_beta_alpha, args.time_beta_beta)

    for step in range(1, args.total_steps + 1):
        image_seq = next(image_iter)
        sample = next(surrogate_stream)
        raw_energy_seq, projected_energy_seq = make_energy_batch(sample, decoder, scatter_head, config, args, device)
        t_unit = time_dist.sample((config.batch_size,)).to(device=device)
        t = args.eps + (1.0 - 2.0 * args.eps) * t_unit
        s = 1.0 - t

        image_to_energy_flow.train()
        energy_to_image_flow.train()
        optimizer.zero_grad()
        i2e_loss, i2e_pred_velocity, i2e_target_velocity = flow_loss(
            image_to_energy_flow,
            image_seq,
            raw_energy_seq,
            t,
        )
        e2i_loss, e2i_pred_velocity, e2i_target_velocity = flow_loss(
            energy_to_image_flow,
            projected_energy_seq,
            image_seq,
            s,
        )
        loss = i2e_loss + e2i_loss
        loss.backward()
        optimizer.step()

        running_i2e += i2e_loss.item()
        running_e2i += e2i_loss.item()
        running_batches += 1

        if step == 1 or step % args.eval_every == 0:
            image_to_energy_flow.eval()
            energy_to_image_flow.eval()
            with torch.no_grad():
                avg_i2e = running_i2e / max(running_batches, 1)
                avg_e2i = running_e2i / max(running_batches, 1)
                i2e_cos = cosine_similarity(i2e_pred_velocity, i2e_target_velocity)
                e2i_cos = cosine_similarity(e2i_pred_velocity, e2i_target_velocity)
                i2e_rel = relative_l2_percent(i2e_pred_velocity, i2e_target_velocity)
                e2i_rel = relative_l2_percent(e2i_pred_velocity, e2i_target_velocity)
                endpoint_stats = {
                    "image": tensor_stats(image_seq),
                    "raw_energy": tensor_stats(raw_energy_seq),
                    "projected_energy": tensor_stats(projected_energy_seq),
                }

            print(
                f"Step {step} | I2E MSE: {avg_i2e:.6f} | E2I MSE: {avg_e2i:.6f} | "
                f"I2E VelCos: {i2e_cos:.4f} | E2I VelCos: {e2i_cos:.4f} | "
                f"I2E VelRelL2: {i2e_rel:.2f}% | E2I VelRelL2: {e2i_rel:.2f}%"
            )
            print(
                "Endpoint RMS | "
                f"image: {endpoint_stats['image']['rms']:.4f} | "
                f"raw_energy: {endpoint_stats['raw_energy']['rms']:.4f} | "
                f"projected_energy: {endpoint_stats['projected_energy']['rms']:.4f}"
            )

            args.image_to_energy_checkpoint.parent.mkdir(parents=True, exist_ok=True)
            args.energy_to_image_checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                make_directional_checkpoint(
                    direction="image-to-energy",
                    state_dict=image_to_energy_flow.state_dict(),
                    step=step,
                    config=config,
                    flow_model_config=image_to_energy_flow.config_dict(),
                    args=args,
                    scatter_ckpt=scatter_ckpt,
                    training_seed=training_seed,
                    metrics={
                        "mse": avg_i2e,
                        "velocity_cosine": i2e_cos,
                        "velocity_rel_l2_percent": i2e_rel,
                        "endpoint_stats": endpoint_stats,
                    },
                ),
                args.image_to_energy_checkpoint,
            )
            torch.save(
                make_directional_checkpoint(
                    direction="energy-to-image",
                    state_dict=energy_to_image_flow.state_dict(),
                    step=step,
                    config=config,
                    flow_model_config=energy_to_image_flow.config_dict(),
                    args=args,
                    scatter_ckpt=scatter_ckpt,
                    training_seed=training_seed,
                    metrics={
                        "mse": avg_e2i,
                        "velocity_cosine": e2i_cos,
                        "velocity_rel_l2_percent": e2i_rel,
                        "endpoint_stats": endpoint_stats,
                    },
                ),
                args.energy_to_image_checkpoint,
            )
            print(
                "Saved directional checkpoints to "
                f"{args.image_to_energy_checkpoint} and {args.energy_to_image_checkpoint}"
                )
            running_i2e = 0.0
            running_e2i = 0.0
            running_batches = 0

    print("Training finished.")


if __name__ == "__main__":
    main()
