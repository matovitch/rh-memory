"""Train one flow direction with on-the-fly reflow teacher endpoints."""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.optim as optim
from flow_checkpoints import (
    Direction,
    build_flow_model,
    checkpoint_direction,
    load_flow_checkpoint,
    state_dict_for_direction,
)
from flow_distribution_stats import distribution_delta, distribution_stats, parse_step_counts
from path_utils import project_relative_path, resolve_project_path
from train_flow_symmetric import (
    cosine_similarity,
    load_soft_scatter_autoencoder,
    load_surrogate,
    make_energy_batch,
    make_image_iterator,
    relative_l2_percent,
    tensor_stats,
)

from rh_memory.flow_integration import integrate_euler_midpoint_time as integrate_euler
from rh_memory.flow_models import flow_matching_loss as reflow_loss
from rh_memory.hilbert import hilbert_metadata
from rh_memory.pipeline import PipelineConfig, harmonic_stage, surrogate_stage
from rh_memory.training_seed import apply_training_seed


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direction", choices=("image-to-energy", "energy-to-image"), required=True)
    parser.add_argument("--teacher-checkpoint", type=Path, default=None)
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--image-manifest", type=Path, default=Path("data/grayscale_32x32_torch/manifest.json"))
    parser.add_argument("--surrogate-checkpoint", type=Path, default=None)
    parser.add_argument("--soft-scatter-checkpoint", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--total-steps", type=int, default=40_000)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--teacher-steps", type=int, default=None)
    parser.add_argument("--eval-steps", type=parse_step_counts, default=(1, 4, 16))
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--eps", type=float, default=1e-4)
    parser.add_argument("--time-distribution", choices=("uniform", "beta"), default="beta")
    parser.add_argument("--time-beta-alpha", type=float, default=0.1)
    parser.add_argument("--time-beta-beta", type=float, default=0.1)
    parser.add_argument("--surrogate-temperature", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--preload-images", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--image-scale", type=float, default=None)
    parser.add_argument("--raw-energy-scale", type=float, default=None)
    parser.add_argument("--projected-energy-scale", type=float, default=None)
    parser.add_argument("--max-grad-norm", type=float, default=None)
    return parser.parse_args()


def default_checkpoint_path(direction: Direction) -> Path:
    if direction == "image-to-energy":
        return Path("scripts/checkpoints/reflow_image_to_energy_checkpoint.pt")
    return Path("scripts/checkpoints/reflow_energy_to_image_checkpoint.pt")


def default_teacher_checkpoint_path(direction: Direction) -> Path:
    if direction == "image-to-energy":
        return Path("scripts/checkpoints/symmetric_flow_image_to_energy_checkpoint.pt")
    return Path("scripts/checkpoints/symmetric_flow_energy_to_image_checkpoint.pt")


def default_teacher_steps(direction: Direction) -> int:
    if direction == "image-to-energy":
        return 64
    return 128


def sample_time(batch_size: int, args, device: torch.device) -> torch.Tensor:
    if args.time_distribution == "uniform":
        t_unit = torch.rand(batch_size, device=device)
    elif args.time_distribution == "beta":
        distribution = torch.distributions.Beta(args.time_beta_alpha, args.time_beta_beta)
        t_unit = distribution.sample((batch_size,)).to(device=device)
    else:
        raise ValueError(f"Unsupported time distribution {args.time_distribution!r}")
    return args.eps + (1.0 - 2.0 * args.eps) * t_unit


def format_delta(delta: dict[str, float]) -> str:
    return (
        f"d_mean {delta['abs_mean']:.4f} | d_std {delta['abs_std']:.4f} | "
        f"d_rms {delta['abs_rms']:.4f} | rel_l1_mass {100.0 * delta['rel_l1_mass']:.2f}%"
    )


def resolve_scale(value: float | None, normalization: dict[str, Any], key: str) -> float:
    if value is not None:
        return float(value)
    return float(normalization.get(key, 1.0))


def checkpoint_config(ckpt: dict[str, Any], batch_size: int, seed: int) -> PipelineConfig:
    config_data = dict(ckpt["config"])
    config_data["batch_size"] = batch_size
    config_data["seed"] = seed
    return PipelineConfig.from_dict(config_data)


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.total_steps <= 0 or args.eval_every <= 0:
        raise ValueError("batch-size, total-steps, and eval-every must be positive")
    if args.teacher_steps is None:
        args.teacher_steps = default_teacher_steps(args.direction)
    if args.teacher_steps <= 0:
        raise ValueError("teacher-steps must be positive")
    if not args.eval_steps or any(step <= 0 for step in args.eval_steps):
        raise ValueError("eval-steps must contain positive integers")
    if not (0.0 <= args.eps < 0.5):
        raise ValueError(f"eps must be in [0, 0.5), got {args.eps}")
    if args.time_beta_alpha <= 0.0 or args.time_beta_beta <= 0.0:
        raise ValueError("time-beta-alpha and time-beta-beta must be positive")
    if args.max_grad_norm is not None and args.max_grad_norm <= 0.0:
        raise ValueError("max-grad-norm must be positive when provided")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    direction: Direction = args.direction
    args.teacher_checkpoint = resolve_project_path(
        args.teacher_checkpoint or default_teacher_checkpoint_path(direction)
    )
    args.init_checkpoint = resolve_project_path(args.init_checkpoint or args.teacher_checkpoint)
    args.checkpoint = resolve_project_path(args.checkpoint or default_checkpoint_path(direction))

    teacher_ckpt = load_flow_checkpoint(args.teacher_checkpoint, device)
    init_ckpt = load_flow_checkpoint(args.init_checkpoint, device)
    resume_ckpt = None
    if args.checkpoint.exists():
        resume_ckpt = load_flow_checkpoint(args.checkpoint, device)
        if resume_ckpt.get("version") != "v1_directional_flow_reflow":
            raise ValueError(f"Cannot resume reflow from checkpoint version {resume_ckpt.get('version')!r}")
        if checkpoint_direction(resume_ckpt) != direction:
            raise ValueError(
                f"checkpoint direction {checkpoint_direction(resume_ckpt)!r} does not match requested {direction!r}"
            )

    training_seed = apply_training_seed(args.seed, resume_ckpt or init_ckpt)
    args.seed = training_seed.seed
    print(f"Using seed: {training_seed.seed} ({training_seed.source})")
    print(f"Reflow direction: {direction}")
    print(f"Teacher checkpoint: {args.teacher_checkpoint} (step {teacher_ckpt.get('step', 'unknown')})")
    print(f"Init checkpoint: {args.init_checkpoint} (step {init_ckpt.get('step', 'unknown')})")
    print(f"Output checkpoint: {args.checkpoint}")
    print(f"Teacher Euler steps: {args.teacher_steps}")

    flow_model_config = dict(init_ckpt["flow_model_config"])
    if dict(teacher_ckpt["flow_model_config"]) != flow_model_config:
        raise ValueError("teacher and init checkpoints must use matching flow_model_config")

    teacher_model = build_flow_model(flow_model_config, state_dict_for_direction(teacher_ckpt, direction), device)
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad_(False)

    student_model = build_flow_model(flow_model_config, state_dict_for_direction(init_ckpt, direction), device)
    optimizer = optim.AdamW(student_model.parameters(), lr=args.lr)

    start_step = 0
    if resume_ckpt is not None:
        student_model.load_state_dict(state_dict_for_direction(resume_ckpt, direction))
        if "optimizer_state_dict" in resume_ckpt:
            optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
        start_step = int(resume_ckpt.get("step", 0))
        print(f"Resumed reflow checkpoint from step {start_step}")

    normalization = teacher_ckpt.get("normalization", {})
    args.image_scale = resolve_scale(args.image_scale, normalization, "image_scale")
    args.raw_energy_scale = resolve_scale(args.raw_energy_scale, normalization, "raw_energy_scale")
    args.projected_energy_scale = resolve_scale(args.projected_energy_scale, normalization, "projected_energy_scale")
    args.surrogate_checkpoint = resolve_project_path(
        args.surrogate_checkpoint or Path(teacher_ckpt["surrogate_checkpoint"])
    )
    args.soft_scatter_checkpoint = resolve_project_path(
        args.soft_scatter_checkpoint or Path(teacher_ckpt["soft_scatter_checkpoint"])
    )
    args.surrogate_temperature = (
        args.surrogate_temperature
        if args.surrogate_temperature is not None
        else float(teacher_ckpt.get("surrogate_temperature", 1.0))
    )

    surrogate, surrogate_config = load_surrogate(args.surrogate_checkpoint, device)
    decoder, scatter_head, scatter_config, scatter_ckpt = load_soft_scatter_autoencoder(
        args.soft_scatter_checkpoint, device
    )
    if surrogate_config.n != scatter_config.n or surrogate_config.C != scatter_config.C:
        raise ValueError("surrogate and soft-scatter checkpoints must use matching n/C")

    config = checkpoint_config(teacher_ckpt, args.batch_size, args.seed)
    if config.n != int(flow_model_config["sequence_length"]):
        raise ValueError(f"pipeline n={config.n} does not match flow length {flow_model_config['sequence_length']}")

    energy_args = SimpleNamespace(
        raw_energy_scale=args.raw_energy_scale,
        projected_energy_scale=args.projected_energy_scale,
    )
    image_iter = make_image_iterator(args, device)
    harmonic_stream = harmonic_stage(config=config, device=device)
    surrogate_stream = surrogate_stage(
        harmonic_stream,
        config=config,
        surrogate=surrogate,
        temperature=args.surrogate_temperature,
    )

    running_loss = 0.0
    running_batches = 0

    for step_idx in range(1, args.total_steps - start_step + 1):
        step = start_step + step_idx
        image_seq = next(image_iter)
        sample = next(surrogate_stream)
        raw_energy_seq, projected_energy_seq = make_energy_batch(
            sample, decoder, scatter_head, config, energy_args, device
        )
        if direction == "image-to-energy":
            source_seq = image_seq
            reference_seq = raw_energy_seq
            reference_name = "raw_energy"
        else:
            source_seq = projected_energy_seq
            reference_seq = image_seq
            reference_name = "image"

        with torch.no_grad():
            teacher_end_seq = integrate_euler(teacher_model, source_seq, steps=args.teacher_steps)
        t = sample_time(config.batch_size, args, device)

        student_model.train()
        optimizer.zero_grad()
        loss, pred_velocity, target_velocity = reflow_loss(student_model, source_seq, teacher_end_seq, t)
        loss.backward()
        if args.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), args.max_grad_norm)
        optimizer.step()

        running_loss += loss.item()
        running_batches += 1

        if step == 1 or step % args.eval_every == 0:
            student_model.eval()
            with torch.no_grad():
                avg_loss = running_loss / max(running_batches, 1)
                velocity_cos = cosine_similarity(pred_velocity, target_velocity)
                velocity_rel = relative_l2_percent(pred_velocity, target_velocity)
                teacher_stats = distribution_stats(teacher_end_seq)
                reference_stats = distribution_stats(reference_seq)
                teacher_delta = distribution_delta(teacher_stats, reference_stats)
                student_endpoint_stats: dict[int, dict[str, float]] = {}
                student_endpoint_delta: dict[int, dict[str, float]] = {}
                for eval_steps in args.eval_steps:
                    student_endpoint = integrate_euler(student_model, source_seq, steps=eval_steps)
                    endpoint_stats = distribution_stats(student_endpoint)
                    student_endpoint_stats[eval_steps] = endpoint_stats
                    student_endpoint_delta[eval_steps] = distribution_delta(endpoint_stats, reference_stats)
                endpoint_stats = {
                    "source": tensor_stats(source_seq),
                    "teacher_end": tensor_stats(teacher_end_seq),
                    reference_name: tensor_stats(reference_seq),
                }

            print(f"Step {step} | MSE: {avg_loss:.6f} | VelCos: {velocity_cos:.4f} | VelRelL2: {velocity_rel:.2f}%")
            print(f"Teacher endpoint vs {reference_name} | {format_delta(teacher_delta)}")
            for eval_steps in args.eval_steps:
                print(
                    f"Student steps {eval_steps:>3} vs {reference_name} | "
                    f"{format_delta(student_endpoint_delta[eval_steps])}"
                )

            args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
            direction_state_dict = student_model.state_dict()
            torch.save(
                {
                    "version": "v1_directional_flow_reflow",
                    "checkpoint_role": "directional_flow_reflow",
                    "source_script": "scripts/train_flow_reflow.py",
                    "step": step,
                    "direction": direction,
                    "state_dict": direction_state_dict,
                    "teacher_checkpoint": project_relative_path(args.teacher_checkpoint),
                    "teacher_step": teacher_ckpt.get("step"),
                    "teacher_steps": args.teacher_steps,
                    "teacher_integrator": "euler_midpoint_time",
                    "init_checkpoint": project_relative_path(args.init_checkpoint),
                    "init_step": init_ckpt.get("step"),
                    "optimizer_state_dict": optimizer.state_dict(),
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
                        "distribution": args.time_distribution,
                        "alpha": args.time_beta_alpha if args.time_distribution == "beta" else None,
                        "beta": args.time_beta_beta if args.time_distribution == "beta" else None,
                    },
                    "eps": args.eps,
                    "lr": args.lr,
                    "seed": args.seed,
                    "seed_source": training_seed.source,
                    "metrics": {
                        "reflow_mse": avg_loss,
                        "velocity_cosine": velocity_cos,
                        "velocity_rel_l2_percent": velocity_rel,
                        "teacher_distribution_delta": teacher_delta,
                        "student_endpoint_delta": student_endpoint_delta,
                        "endpoint_stats": endpoint_stats,
                    },
                },
                args.checkpoint,
            )
            print(f"Saved checkpoint to {args.checkpoint}")
            running_loss = 0.0
            running_batches = 0

    print("Reflow training finished.")


if __name__ == "__main__":
    main()
