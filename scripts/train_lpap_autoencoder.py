"""Train the end-to-end LPAP autoencoder on grayscale image shards."""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import torch
import torch.optim as optim
from flow_checkpoints import build_flow_model, checkpoint_direction, load_flow_checkpoint, state_dict_for_direction
from path_utils import project_relative_path, resolve_project_path
from torch.utils.data import DataLoader

from rh_memory.decoder import RHDecoder
from rh_memory.decoder_scatter import SoftScatterReconstructionHead
from rh_memory.hilbert import hilbert_flatten_images, hilbert_metadata
from rh_memory.image_shards import GrayscaleImageShardDataset, InMemoryGrayscaleImageShardDataset
from rh_memory.lpap_autoencoder import LPAPAutoencoder, LPAPAutoencoderLoss, LPAPBottleneckBranch
from rh_memory.pipeline import PipelineConfig, build_grouped_permutation
from rh_memory.surrogate import RHSurrogate
from rh_memory.training_seed import apply_training_seed

CHECKPOINT_VERSION = "v1_lpap_autoencoder_e2e"


def tensor_stats(x: torch.Tensor) -> dict[str, float]:
    flat = x.detach().flatten(1)
    return {
        "mean": flat.mean().item(),
        "std": flat.std(unbiased=False).item(),
        "rms": flat.square().mean(dim=1).sqrt().mean().item(),
        "min": flat.min().item(),
        "max": flat.max().item(),
    }


def grad_norm(parameters) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        total += parameter.grad.detach().float().square().sum().item()
    return total**0.5


def load_surrogate(checkpoint_path: Path, device: torch.device) -> tuple[RHSurrogate, PipelineConfig, dict[str, Any]]:
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
        d_model=int(model_config["d_model"]),
        n_heads=int(model_config["n_heads"]),
        num_layers=int(model_config["num_layers"]),
        dim_feedforward=int(model_config["dim_feedforward"]),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    return model, config, ckpt


def load_soft_scatter_autoencoder(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[RHDecoder, SoftScatterReconstructionHead, PipelineConfig, dict[str, Any]]:
    checkpoint_path = resolve_project_path(checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if ckpt.get("version") != "v1_decoder_soft_scatter_l1":
        raise ValueError(f"Unsupported soft-scatter checkpoint version {ckpt.get('version')!r}")
    config = PipelineConfig.from_dict(ckpt["config"])
    model_config = ckpt["model_config"]
    decoder = RHDecoder(
        sequence_length=config.sequence_length,
        bucket_count=config.C,
        d_model=int(model_config["d_model"]),
        n_heads=int(model_config["n_heads"]),
        num_layers=int(model_config["num_layers"]),
        dim_feedforward=int(model_config["dim_feedforward"]),
    ).to(device)
    decoder.load_state_dict(ckpt["model_state_dict"])
    scatter_head = SoftScatterReconstructionHead(
        init_temperature=float(ckpt.get("scatter_temperature", 1.0)),
        min_temperature=float(ckpt.get("min_scatter_temperature", 0.05)),
    ).to(device)
    scatter_head.load_state_dict(ckpt["scatter_head_state_dict"])
    return decoder, scatter_head, config, ckpt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-manifest", type=Path, default=Path("data/grayscale_32x32_torch/manifest.json"))
    parser.add_argument("--eval-image-manifest", type=Path, required=True)
    parser.add_argument(
        "--surrogate-checkpoint", type=Path, default=Path("scripts/checkpoints/surrogate_ce_checkpoint.pt")
    )
    parser.add_argument(
        "--soft-scatter-checkpoint",
        type=Path,
        default=Path("scripts/checkpoints/decoder_soft_scatter_l1_checkpoint.pt"),
    )
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
    parser.add_argument("--checkpoint", type=Path, default=Path("scripts/checkpoints/lpap_autoencoder_checkpoint.pt"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--total-steps", type=int, default=100_000)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--preload-images", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--image-scale", type=float, default=1.0)
    parser.add_argument("--lpap-weight", type=float, default=0.1)
    parser.add_argument("--image-to-energy-steps", type=int, default=16)
    parser.add_argument("--energy-to-image-steps", type=int, default=16)
    parser.add_argument("--surrogate-temperature", type=float, default=1.0)
    parser.add_argument("--branch-name", default="C128")
    return parser.parse_args()


def make_image_iterator(
    manifest_path: Path,
    *,
    batch_size: int,
    seed: int,
    image_scale: float,
    preload_images: bool,
    num_workers: int,
    device: torch.device,
    shuffle: bool,
) -> Iterator[torch.Tensor]:
    manifest_path = resolve_project_path(manifest_path)
    if preload_images:
        if num_workers != 0:
            raise ValueError("--preload-images should use --num-workers 0 to avoid duplicating the in-memory dataset")
        dataset = InMemoryGrayscaleImageShardDataset(manifest_path, as_float=True)
    else:
        dataset = GrayscaleImageShardDataset(manifest_path, as_float=True)
    if len(dataset) < batch_size:
        raise ValueError(
            f"image manifest {manifest_path} has {len(dataset)} images, fewer than batch_size={batch_size}"
        )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
        num_workers=num_workers,
        generator=generator,
    )
    while True:
        for images in loader:
            yield hilbert_flatten_images(images.to(device)).mul(image_scale)


def make_e2e_checkpoint(
    *,
    model: LPAPAutoencoder,
    optimizer: optim.Optimizer,
    step: int,
    config: PipelineConfig,
    args: argparse.Namespace,
    training_seed,
    surrogate_ckpt: dict[str, Any],
    soft_scatter_ckpt: dict[str, Any],
    image_to_energy_ckpt: dict[str, Any],
    energy_to_image_ckpt: dict[str, Any],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    branch = cast(LPAPBottleneckBranch, model.branches[args.branch_name])
    return {
        "version": CHECKPOINT_VERSION,
        "checkpoint_role": "lpap_autoencoder_e2e",
        "source_script": "scripts/train_lpap_autoencoder.py",
        "step": step,
        "config": config.to_dict(),
        "hilbert": hilbert_metadata(32),
        "image_to_energy_flow_state_dict": model.image_to_energy.vector_field.state_dict(),
        "energy_to_image_flow_state_dict": model.energy_to_image.vector_field.state_dict(),
        "surrogate_state_dict": branch.surrogate.state_dict(),
        "decoder_state_dict": branch.decoder.state_dict(),
        "scatter_head_state_dict": branch.scatter_head.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "flow_model_config": dict(image_to_energy_ckpt["flow_model_config"]),
        "surrogate_model_config": surrogate_ckpt["model_config"],
        "decoder_model_config": soft_scatter_ckpt["model_config"],
        "branch": {
            "name": args.branch_name,
            "C": config.C,
            "fast_k": config.fast_k,
            "k_eff": branch.k_eff,
            "surrogate_temperature": args.surrogate_temperature,
        },
        "image_to_energy_steps": args.image_to_energy_steps,
        "energy_to_image_steps": args.energy_to_image_steps,
        "lpap_weight": args.lpap_weight,
        "lr": args.lr,
        "image_scale": args.image_scale,
        "image_manifest": project_relative_path(args.image_manifest),
        "eval_image_manifest": project_relative_path(args.eval_image_manifest),
        "source_checkpoints": {
            "surrogate": project_relative_path(args.surrogate_checkpoint),
            "soft_scatter": project_relative_path(args.soft_scatter_checkpoint),
            "image_to_energy": project_relative_path(args.image_to_energy_checkpoint),
            "energy_to_image": project_relative_path(args.energy_to_image_checkpoint),
        },
        "source_checkpoint_versions": {
            "surrogate": surrogate_ckpt.get("version"),
            "soft_scatter": soft_scatter_ckpt.get("version"),
            "image_to_energy": image_to_energy_ckpt.get("version"),
            "energy_to_image": energy_to_image_ckpt.get("version"),
        },
        "seed": args.seed,
        "seed_source": training_seed.source,
        "metrics": metrics,
    }


def load_e2e_checkpoint(path: Path, device: torch.device) -> dict[str, Any] | None:
    path = resolve_project_path(path)
    if not path.exists():
        return None
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if ckpt.get("version") != CHECKPOINT_VERSION:
        raise ValueError(f"Unsupported LPAP autoencoder checkpoint version {ckpt.get('version')!r}")
    return ckpt


def validate_directional_flow_pair(image_to_energy_ckpt: dict[str, Any], energy_to_image_ckpt: dict[str, Any]) -> None:
    if checkpoint_direction(image_to_energy_ckpt) != "image-to-energy":
        raise ValueError("image-to-energy checkpoint has the wrong direction")
    if checkpoint_direction(energy_to_image_ckpt) != "energy-to-image":
        raise ValueError("energy-to-image checkpoint has the wrong direction")
    if dict(image_to_energy_ckpt["flow_model_config"]) != dict(energy_to_image_ckpt["flow_model_config"]):
        raise ValueError("directional flow checkpoints must use matching flow_model_config")


def main() -> None:
    args = parse_args()
    if args.total_steps <= 0:
        raise ValueError(f"total_steps must be positive, got {args.total_steps}")
    if args.eval_every <= 0:
        raise ValueError(f"eval_every must be positive, got {args.eval_every}")
    if args.eval_batches <= 0:
        raise ValueError(f"eval_batches must be positive, got {args.eval_batches}")
    if args.lpap_weight < 0:
        raise ValueError(f"lpap_weight must be non-negative, got {args.lpap_weight}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    args.checkpoint = resolve_project_path(args.checkpoint)
    args.image_manifest = resolve_project_path(args.image_manifest)
    args.eval_image_manifest = resolve_project_path(args.eval_image_manifest)
    args.surrogate_checkpoint = resolve_project_path(args.surrogate_checkpoint)
    args.soft_scatter_checkpoint = resolve_project_path(args.soft_scatter_checkpoint)
    args.image_to_energy_checkpoint = resolve_project_path(args.image_to_energy_checkpoint)
    args.energy_to_image_checkpoint = resolve_project_path(args.energy_to_image_checkpoint)

    resume_ckpt = load_e2e_checkpoint(args.checkpoint, device)
    training_seed = apply_training_seed(args.seed, resume_ckpt)
    args.seed = training_seed.seed
    print(f"Using seed: {training_seed.seed} ({training_seed.source})")

    surrogate, surrogate_config, surrogate_ckpt = load_surrogate(args.surrogate_checkpoint, device)
    decoder, scatter_head, scatter_config, soft_scatter_ckpt = load_soft_scatter_autoencoder(
        args.soft_scatter_checkpoint, device
    )
    if surrogate_config.n != scatter_config.n or surrogate_config.C != scatter_config.C:
        raise ValueError("surrogate and soft-scatter checkpoints must use matching n/C")

    image_to_energy_ckpt = load_flow_checkpoint(args.image_to_energy_checkpoint, device)
    energy_to_image_ckpt = load_flow_checkpoint(args.energy_to_image_checkpoint, device)
    validate_directional_flow_pair(image_to_energy_ckpt, energy_to_image_ckpt)

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
    if config.n != int(image_to_energy_ckpt["flow_model_config"]["sequence_length"]):
        raise ValueError("surrogate sequence length does not match flow checkpoints")

    image_to_energy_flow = build_flow_model(
        image_to_energy_ckpt["flow_model_config"],
        state_dict_for_direction(image_to_energy_ckpt, "image-to-energy"),
        device,
    )
    energy_to_image_flow = build_flow_model(
        energy_to_image_ckpt["flow_model_config"],
        state_dict_for_direction(energy_to_image_ckpt, "energy-to-image"),
        device,
    )
    perm_1d = build_grouped_permutation(config.n, config.C, config.seed, device)
    branch = LPAPBottleneckBranch(
        name=args.branch_name,
        surrogate=surrogate,
        decoder=decoder,
        scatter_head=scatter_head,
        perm_1d=perm_1d,
        fast_k=config.fast_k,
        surrogate_temperature=args.surrogate_temperature,
    )
    model = LPAPAutoencoder(
        image_to_energy_flow=image_to_energy_flow,
        energy_to_image_flow=energy_to_image_flow,
        bottleneck_branches=[branch],
        image_to_energy_steps=args.image_to_energy_steps,
        energy_to_image_steps=args.energy_to_image_steps,
        default_branch=args.branch_name,
    ).to(device)
    criterion = LPAPAutoencoderLoss(lpap_weight=args.lpap_weight)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    start_step = 0
    if resume_ckpt is not None:
        model.image_to_energy.vector_field.load_state_dict(resume_ckpt["image_to_energy_flow_state_dict"])
        model.energy_to_image.vector_field.load_state_dict(resume_ckpt["energy_to_image_flow_state_dict"])
        branch.surrogate.load_state_dict(resume_ckpt["surrogate_state_dict"])
        branch.decoder.load_state_dict(resume_ckpt["decoder_state_dict"])
        branch.scatter_head.load_state_dict(resume_ckpt["scatter_head_state_dict"])
        optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
        start_step = int(resume_ckpt.get("step", 0))
        print(f"Resumed LPAP autoencoder checkpoint from step {start_step}")

    train_iter = make_image_iterator(
        args.image_manifest,
        batch_size=args.batch_size,
        seed=args.seed,
        image_scale=args.image_scale,
        preload_images=args.preload_images,
        num_workers=args.num_workers,
        device=device,
        shuffle=True,
    )
    eval_iter = make_image_iterator(
        args.eval_image_manifest,
        batch_size=args.batch_size,
        seed=args.seed + 1,
        image_scale=args.image_scale,
        preload_images=args.preload_images,
        num_workers=args.num_workers,
        device=device,
        shuffle=False,
    )

    running_total = 0.0
    running_reconstruction = 0.0
    running_lpap = 0.0
    running_batches = 0

    for step_idx in range(1, args.total_steps - start_step + 1):
        step = start_step + step_idx
        image_seq = next(train_iter)

        model.train()
        optimizer.zero_grad()
        output = model(image_seq, branch_name=args.branch_name)
        losses = criterion(output, image_seq)
        losses.total.backward()
        grad_stats = {
            "i2e": grad_norm(model.image_to_energy.vector_field.parameters()),
            "surrogate": grad_norm(branch.surrogate.parameters()),
            "decoder": grad_norm(branch.decoder.parameters()),
            "scatter": grad_norm(branch.scatter_head.parameters()),
            "e2i": grad_norm(model.energy_to_image.vector_field.parameters()),
        }
        optimizer.step()

        running_total += losses.total.item()
        running_reconstruction += losses.reconstruction.item()
        running_lpap += losses.lpap_surrogate.item()
        running_batches += 1

        if step == 1 or step % args.eval_every == 0:
            avg_total = running_total / max(running_batches, 1)
            avg_reconstruction = running_reconstruction / max(running_batches, 1)
            avg_lpap = running_lpap / max(running_batches, 1)

            model.eval()
            eval_total = 0.0
            eval_reconstruction = 0.0
            eval_lpap = 0.0
            eval_doubt = 0.0
            eval_support = 0.0
            eval_batches = 0
            eval_stats: dict[str, dict[str, float]] = {}
            with torch.no_grad():
                for _ in range(args.eval_batches):
                    eval_image = next(eval_iter)
                    eval_output = model(eval_image, branch_name=args.branch_name)
                    eval_losses = criterion(eval_output, eval_image)
                    eval_total += eval_losses.total.item()
                    eval_reconstruction += eval_losses.reconstruction.item()
                    eval_lpap += eval_losses.lpap_surrogate.item()
                    eval_doubt += eval_output.scatter_doubt.mean().item()
                    eval_support += eval_output.scatter_support.mean().item()
                    eval_batches += 1
                    eval_stats = {
                        "image": tensor_stats(eval_image),
                        "learned_energy": tensor_stats(eval_output.learned_energy),
                        "projected_energy": tensor_stats(eval_output.projected_energy),
                        "reconstructed_image": tensor_stats(eval_output.reconstructed_image),
                    }

            denom = max(eval_batches, 1)
            metrics = {
                "train_total": avg_total,
                "train_reconstruction": avg_reconstruction,
                "train_lpap_surrogate": avg_lpap,
                "eval_total": eval_total / denom,
                "eval_reconstruction": eval_reconstruction / denom,
                "eval_lpap_surrogate": eval_lpap / denom,
                "eval_scatter_doubt": eval_doubt / denom,
                "eval_scatter_support": eval_support / denom,
                "scatter_temperature": branch.scatter_head.temperature().item(),
                "grad_norms": grad_stats,
                "eval_stats": eval_stats,
            }
            print(
                f"Step {step} | Train: {avg_total:.6f} | TrainMSE: {avg_reconstruction:.6f} | "
                f"TrainLPAP: {avg_lpap:.6f} | Eval: {metrics['eval_total']:.6f} | "
                f"EvalMSE: {metrics['eval_reconstruction']:.6f} | EvalLPAP: {metrics['eval_lpap_surrogate']:.6f} | "
                f"Doubt: {metrics['eval_scatter_doubt']:.4f} | "
                f"Support: {metrics['eval_scatter_support']:.1f}/{config.n} | "
                f"Temp: {metrics['scatter_temperature']:.4f}"
            )
            print(
                "GradNorms | "
                f"I2E: {grad_stats['i2e']:.3e} | Surrogate: {grad_stats['surrogate']:.3e} | "
                f"Decoder: {grad_stats['decoder']:.3e} | Scatter: {grad_stats['scatter']:.3e} | "
                f"E2I: {grad_stats['e2i']:.3e}"
            )

            args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                make_e2e_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    step=step,
                    config=config,
                    args=args,
                    training_seed=training_seed,
                    surrogate_ckpt=surrogate_ckpt,
                    soft_scatter_ckpt=soft_scatter_ckpt,
                    image_to_energy_ckpt=image_to_energy_ckpt,
                    energy_to_image_ckpt=energy_to_image_ckpt,
                    metrics=metrics,
                ),
                args.checkpoint,
            )
            print(f"Saved checkpoint to {args.checkpoint}")

            running_total = 0.0
            running_reconstruction = 0.0
            running_lpap = 0.0
            running_batches = 0

    print("Training finished.")


if __name__ == "__main__":
    main()
