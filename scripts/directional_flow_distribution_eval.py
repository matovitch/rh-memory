"""Shared directional flow distribution evaluation helpers."""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

import torch
from flow_checkpoints import (
    SUPPORTED_FLOW_CHECKPOINT_VERSIONS,
    build_flow_model,
    checkpoint_direction,
    require_flow_checkpoint_keys,
    state_dict_for_direction,
)
from flow_distribution_stats import (
    concatenate_batches,
    distribution_delta,
    distribution_stats,
    parse_step_counts,
)
from path_utils import resolve_project_path
from torch.utils.data import DataLoader
from train_flow_symmetric import (
    load_soft_scatter_autoencoder,
    load_surrogate,
    make_energy_batch,
)

from rh_memory.flow_integration import integrate_euler_midpoint_time as integrate_euler
from rh_memory.hilbert import hilbert_flatten_images
from rh_memory.image_shards import (
    GrayscaleImageShardDataset,
    InMemoryGrayscaleImageShardDataset,
)
from rh_memory.pipeline import PipelineConfig, harmonic_stage, surrogate_stage


Direction = Literal["image-to-energy", "energy-to-image"]

def parse_args(*, default_checkpoint: Path, fixed_direction: Direction):
    parser = argparse.ArgumentParser(description="Evaluate directional flow endpoint distributions.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=default_checkpoint,
    )
    parser.add_argument(
        "--image-manifest",
        type=Path,
        default=Path("data/grayscale_32x32_torch/manifest.json"),
    )
    parser.add_argument("--surrogate-checkpoint", type=Path, default=None)
    parser.add_argument("--soft-scatter-checkpoint", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-batches", type=int, default=16)
    parser.add_argument("--steps", type=parse_step_counts, default=(1, 4, 16, 64, 128))
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--surrogate-temperature", type=float, default=None)
    parser.add_argument("--preload-images", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()
    args.direction = fixed_direction
    return args


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


def print_checkpoint_header(args, flow_ckpt: dict[str, Any], device: torch.device) -> None:
    print(f"Using device: {device}")
    print(f"Flow checkpoint: {args.checkpoint} (step {flow_ckpt.get('step', 'unknown')})")
    print(f"Checkpoint version: {flow_ckpt.get('version', 'unknown')}")
    print(f"Checkpoint role: {flow_ckpt.get('checkpoint_role', 'unknown')}")
    ckpt_direction = checkpoint_direction(flow_ckpt)
    if ckpt_direction is not None:
        print(f"Checkpoint direction: {ckpt_direction}")
    if flow_ckpt.get("version") == "v1_directional_flow_reflow":
        print(f"Teacher checkpoint: {flow_ckpt.get('teacher_checkpoint', 'unknown')}")
        print(f"Teacher steps: {flow_ckpt.get('teacher_steps', 'unknown')}")
        if "time_sampling" in flow_ckpt:
            print(f"Time sampling: {flow_ckpt['time_sampling']}")
    print(f"Evaluated direction: {args.direction}")
    print(f"Euler steps: {','.join(str(step) for step in args.steps)}")


def resolve_requested_direction(flow_ckpt: dict[str, Any], requested: Direction) -> Direction:
    ckpt_direction = checkpoint_direction(flow_ckpt)
    if ckpt_direction is None:
        raise ValueError("directional checkpoint does not declare a direction")
    if requested != ckpt_direction:
        raise ValueError(f"requested direction {requested!r} conflicts with checkpoint direction {ckpt_direction!r}")
    return requested


def main(*, default_checkpoint: Path, fixed_direction: Direction):
    args = parse_args(default_checkpoint=default_checkpoint, fixed_direction=fixed_direction)
    if args.batch_size <= 0 or args.num_batches <= 0:
        raise ValueError("batch-size and num-batches must be positive")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.checkpoint = resolve_project_path(args.checkpoint)
    flow_ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    version = flow_ckpt.get("version")
    if version not in SUPPORTED_FLOW_CHECKPOINT_VERSIONS:
        raise ValueError(f"Unsupported flow checkpoint version {version!r}")
    require_flow_checkpoint_keys(flow_ckpt, args.checkpoint)
    args.direction = resolve_requested_direction(flow_ckpt, args.direction)

    seed = args.seed if args.seed is not None else int(flow_ckpt.get("seed", 42))
    torch.manual_seed(seed)
    normalization = flow_ckpt.get("normalization", {})
    image_scale = float(normalization.get("image_scale", 1.0))
    raw_energy_scale = float(normalization.get("raw_energy_scale", 1.0))
    projected_energy_scale = float(normalization.get("projected_energy_scale", 1.0))
    surrogate_checkpoint = resolve_project_path(
        args.surrogate_checkpoint or Path(flow_ckpt["surrogate_checkpoint"])
    )
    soft_scatter_checkpoint = resolve_project_path(
        args.soft_scatter_checkpoint or Path(flow_ckpt["soft_scatter_checkpoint"])
    )
    surrogate_temperature = (
        args.surrogate_temperature
        if args.surrogate_temperature is not None
        else float(flow_ckpt.get("surrogate_temperature", 1.0))
    )

    print_checkpoint_header(args, flow_ckpt, device)
    print(
        f"Scales | image {image_scale:.4g} | raw_energy {raw_energy_scale:.4g} | "
        f"projected_energy {projected_energy_scale:.4g}"
    )

    flow_model = build_flow_model(
        flow_ckpt["flow_model_config"],
        state_dict_for_direction(flow_ckpt, args.direction),
        device,
    )
    flow_model.eval()
    surrogate, surrogate_config = load_surrogate(surrogate_checkpoint, device)
    decoder, scatter_head, scatter_config, _scatter_ckpt = load_soft_scatter_autoencoder(
        soft_scatter_checkpoint, device
    )
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
    surrogate_stream = surrogate_stage(
        harmonic_stream,
        config=config,
        surrogate=surrogate,
        temperature=surrogate_temperature,
    )

    references: dict[str, list[torch.Tensor]] = {
        "image": [],
        "raw_energy": [],
        "projected_energy": [],
    }
    generated: dict[int, list[torch.Tensor]] = {step: [] for step in args.steps}

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
            if args.direction == "image-to-energy":
                source_seq = image_seq
            else:
                source_seq = projected_energy_seq
            for steps in args.steps:
                generated[steps].append(integrate_euler(flow_model, source_seq, steps=steps))

    reference_stats = {
        name: distribution_stats(concatenate_batches(batches))
        for name, batches in references.items()
    }

    print("\nReference distributions")
    for name in ("image", "raw_energy", "projected_energy"):
        print(f"{name:>18} | {format_stats(reference_stats[name])}")

    if args.direction == "image-to-energy":
        print("\nGenerated distributions: image_to_energy_flow(image) vs raw_energy reference")
        for steps in args.steps:
            stats = distribution_stats(concatenate_batches(generated[steps]))
            delta = distribution_delta(stats, reference_stats["raw_energy"])
            print(f"steps {steps:>3} | {format_stats(stats)} | {format_delta(delta)}")
    else:
        print("\nGenerated distributions: energy_to_image_flow(projected_energy) vs image reference")
        for steps in args.steps:
            stats = distribution_stats(concatenate_batches(generated[steps]))
            delta = distribution_delta(stats, reference_stats["image"])
            print(f"steps {steps:>3} | {format_stats(stats)} | {format_delta(delta)}")