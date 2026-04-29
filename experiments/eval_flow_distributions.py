"""Compare symmetric flow endpoint distributions across Euler step counts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from rh_memory.flow_eval import concatenate_batches, distribution_delta, distribution_stats, parse_step_counts
from rh_memory.flow_models import DilatedConvFlow1d
from rh_memory.hilbert import hilbert_flatten_images
from rh_memory.image_shards import GrayscaleImageShardDataset, InMemoryGrayscaleImageShardDataset
from rh_memory.pipeline import PipelineConfig, harmonic_stage, surrogate_stage

from train_flow_symmetric import load_soft_scatter_autoencoder, load_surrogate, make_energy_batch


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("experiments/checkpoints/symmetric_flow_checkpoint.pt"))
    parser.add_argument("--image-manifest", type=Path, default=Path("data/grayscale_32x32_torch/manifest.json"))
    parser.add_argument("--surrogate-checkpoint", type=Path, default=None)
    parser.add_argument("--soft-scatter-checkpoint", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-batches", type=int, default=16)
    parser.add_argument("--steps", type=parse_step_counts, default=(1, 4, 16, 64, 128))
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--surrogate-temperature", type=float, default=None)
    parser.add_argument("--preload-images", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def build_flow_model(config: dict, state_dict: dict[str, torch.Tensor], device: torch.device) -> DilatedConvFlow1d:
    model = DilatedConvFlow1d(
        sequence_length=int(config["sequence_length"]),
        width=int(config["width"]),
        time_dim=int(config["time_dim"]),
        dilation_cycles=int(config["dilation_cycles"]),
        dilations=tuple(int(dilation) for dilation in config["dilations"]),
        kernel_size=int(config.get("kernel_size", 3)),
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def make_image_iterator(args, device: torch.device, image_scale: float, seed: int):
    generator = torch.Generator().manual_seed(seed)
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
            yield hilbert_flatten_images(images.to(device)).mul(image_scale)


@torch.no_grad()
def integrate_euler(model, start: torch.Tensor, *, steps: int) -> torch.Tensor:
    x = start
    dt = 1.0 / steps
    for step in range(steps):
        t = torch.full((start.shape[0],), (step + 0.5) * dt, device=start.device, dtype=start.dtype)
        x = x + dt * model(x, t)
    return x


def format_stats(stats: dict[str, float]) -> str:
    return (
        f"mean {stats['mean']:+.4f} | std {stats['std']:.4f} | rms {stats['rms']:.4f} | "
        f"l1/sample {stats['l1_per_sample']:.2f} | min {stats['min']:+.4f} | max {stats['max']:+.4f} | "
        f"q01 {stats['q01']:+.4f} | q50 {stats['q50']:+.4f} | q99 {stats['q99']:+.4f} | "
        f"<0 {100.0 * stats['frac_lt_0']:.2f}% | >1 {100.0 * stats['frac_gt_1']:.2f}%"
    )


def format_delta(delta: dict[str, float]) -> str:
    return (
        f"d_mean {delta['abs_mean']:.4f} | d_std {delta['abs_std']:.4f} | "
        f"d_rms {delta['abs_rms']:.4f} | rel_l1_mass {100.0 * delta['rel_l1_mass']:.2f}%"
    )


def main():
    args = parse_args()
    if args.batch_size <= 0 or args.num_batches <= 0:
        raise ValueError("batch-size and num-batches must be positive")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    flow_ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if flow_ckpt.get("version") != "v1_symmetric_flow_matching":
        raise ValueError(f"Unsupported flow checkpoint version {flow_ckpt.get('version')!r}")

    seed = args.seed if args.seed is not None else int(flow_ckpt.get("seed", 42))
    torch.manual_seed(seed)
    normalization = flow_ckpt.get("normalization", {})
    image_scale = float(normalization.get("image_scale", 1.0))
    raw_energy_scale = float(normalization.get("raw_energy_scale", 1.0))
    projected_energy_scale = float(normalization.get("projected_energy_scale", 1.0))
    surrogate_checkpoint = args.surrogate_checkpoint or Path(flow_ckpt["surrogate_checkpoint"])
    soft_scatter_checkpoint = args.soft_scatter_checkpoint or Path(flow_ckpt["soft_scatter_checkpoint"])
    surrogate_temperature = (
        args.surrogate_temperature
        if args.surrogate_temperature is not None
        else float(flow_ckpt.get("surrogate_temperature", 1.0))
    )

    print(f"Using device: {device}")
    print(f"Flow checkpoint: {args.checkpoint} (step {flow_ckpt.get('step', 'unknown')})")
    print(f"Euler steps: {','.join(str(step) for step in args.steps)}")
    print(
        f"Scales | image {image_scale:.4g} | raw_energy {raw_energy_scale:.4g} | "
        f"projected_energy {projected_energy_scale:.4g}"
    )

    image_to_energy_flow = build_flow_model(
        flow_ckpt["flow_model_config"],
        flow_ckpt["image_to_energy_state_dict"],
        device,
    )
    energy_to_image_flow = build_flow_model(
        flow_ckpt["flow_model_config"],
        flow_ckpt["energy_to_image_state_dict"],
        device,
    )
    surrogate, surrogate_config = load_surrogate(Path(surrogate_checkpoint), device)
    decoder, scatter_head, scatter_config, _scatter_ckpt = load_soft_scatter_autoencoder(Path(soft_scatter_checkpoint), device)
    if surrogate_config.n != scatter_config.n or surrogate_config.C != scatter_config.C:
        raise ValueError("surrogate and soft-scatter checkpoints must use matching n/C")

    config_data = dict(flow_ckpt["config"])
    config_data["batch_size"] = args.batch_size
    config = PipelineConfig.from_dict(config_data)

    energy_args = SimpleNamespace(
        raw_energy_scale=raw_energy_scale,
        projected_energy_scale=projected_energy_scale,
    )

    image_iter = make_image_iterator(args, device, image_scale=image_scale, seed=seed)
    harmonic_stream = harmonic_stage(config=config, device=device)
    surrogate_stream = surrogate_stage(harmonic_stream, config=config, surrogate=surrogate, temperature=surrogate_temperature)

    references: dict[str, list[torch.Tensor]] = {
        "image": [],
        "raw_energy": [],
        "projected_energy": [],
    }
    generated_i2e: dict[int, list[torch.Tensor]] = {step: [] for step in args.steps}
    generated_e2i: dict[int, list[torch.Tensor]] = {step: [] for step in args.steps}

    with torch.no_grad():
        for _batch_idx in range(args.num_batches):
            image_seq = next(image_iter)
            sample = next(surrogate_stream)
            raw_energy_seq, projected_energy_seq = make_energy_batch(
                sample,
                decoder,
                scatter_head,
                config,
                energy_args,
                device,
            )
            references["image"].append(image_seq)
            references["raw_energy"].append(raw_energy_seq)
            references["projected_energy"].append(projected_energy_seq)
            for steps in args.steps:
                generated_i2e[steps].append(integrate_euler(image_to_energy_flow, image_seq, steps=steps))
                generated_e2i[steps].append(integrate_euler(energy_to_image_flow, projected_energy_seq, steps=steps))

    reference_stats = {
        name: distribution_stats(concatenate_batches(batches))
        for name, batches in references.items()
    }

    print("\nReference distributions")
    for name in ("image", "raw_energy", "projected_energy"):
        print(f"{name:>18} | {format_stats(reference_stats[name])}")

    print("\nGenerated distributions: image_to_energy_flow(image) vs raw_energy reference")
    for steps in args.steps:
        stats = distribution_stats(concatenate_batches(generated_i2e[steps]))
        delta = distribution_delta(stats, reference_stats["raw_energy"])
        print(f"steps {steps:>3} | {format_stats(stats)} | {format_delta(delta)}")

    print("\nGenerated distributions: energy_to_image_flow(projected_energy) vs image reference")
    for steps in args.steps:
        stats = distribution_stats(concatenate_batches(generated_e2i[steps]))
        delta = distribution_delta(stats, reference_stats["image"])
        print(f"steps {steps:>3} | {format_stats(stats)} | {format_delta(delta)}")


if __name__ == "__main__":
    main()
