"""Run LPAP autoencoder inference and save image/energy comparison panels."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from flow_checkpoints import build_flow_model
from path_utils import resolve_project_path

from rh_memory.decoder import RHDecoder
from rh_memory.decoder_scatter import SoftScatterReconstructionHead
from rh_memory.hilbert import hilbert_flatten_images, hilbert_unflatten_images
from rh_memory.image_shards import GrayscaleImageShardDataset, InMemoryGrayscaleImageShardDataset
from rh_memory.lpap_autoencoder import LPAPAutoencoder, LPAPBottleneckBranch
from rh_memory.pipeline import PipelineConfig, build_grouped_permutation
from rh_memory.surrogate import RHSurrogate


CHECKPOINT_VERSION = "v1_lpap_autoencoder_e2e"
ImageDataset = GrayscaleImageShardDataset | InMemoryGrayscaleImageShardDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("scripts/checkpoints/lpap_autoencoder_reflow2_e2i_4step_checkpoint.pt"),
    )
    parser.add_argument("--image-manifest", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("scripts/outputs/lpap_autoencoder_eval.png"))
    parser.add_argument("--num-images", type=int, default=8)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--preload-images", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    return torch.device(requested)


def load_e2e_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    resolved = resolve_project_path(path)
    checkpoint = torch.load(resolved, map_location=device, weights_only=False)
    if checkpoint.get("version") != CHECKPOINT_VERSION:
        raise ValueError(f"Unsupported LPAP autoencoder checkpoint version {checkpoint.get('version')!r}")
    checkpoint["_resolved_path"] = resolved
    return checkpoint


def resolve_manifest_path(args: argparse.Namespace, checkpoint: dict[str, Any]) -> Path:
    if args.image_manifest is not None:
        return resolve_project_path(args.image_manifest)
    manifest_value = checkpoint.get("eval_image_manifest") or checkpoint.get("image_manifest")
    if manifest_value is None:
        raise KeyError("checkpoint does not contain image_manifest or eval_image_manifest")
    return resolve_project_path(Path(manifest_value))


def square_side(sequence_length: int) -> int:
    side = math.isqrt(sequence_length)
    if side * side != sequence_length:
        raise ValueError(f"sequence length must be a perfect square, got {sequence_length}")
    return side


def build_autoencoder_from_checkpoint(
    checkpoint: dict[str, Any], device: torch.device
) -> tuple[LPAPAutoencoder, int, float]:
    config = PipelineConfig.from_dict(dict(checkpoint["config"]))
    flow_model_config = dict(checkpoint["flow_model_config"])
    surrogate_model_config = dict(checkpoint["surrogate_model_config"])
    decoder_model_config = dict(checkpoint["decoder_model_config"])
    branch_meta = dict(checkpoint["branch"])
    if int(flow_model_config["sequence_length"]) != config.n:
        raise ValueError("flow_model_config sequence length does not match checkpoint config")
    if int(branch_meta.get("C", config.C)) != config.C:
        raise ValueError("branch metadata C does not match checkpoint config")
    side = square_side(config.n)

    image_to_energy_flow = build_flow_model(flow_model_config, checkpoint["image_to_energy_flow_state_dict"], device)
    energy_to_image_flow = build_flow_model(flow_model_config, checkpoint["energy_to_image_flow_state_dict"], device)

    surrogate = RHSurrogate(
        sequence_length=config.sequence_length,
        bucket_count=config.C,
        stride=config.stride,
        fast_k=config.fast_k,
        d_model=int(surrogate_model_config["d_model"]),
        n_heads=int(surrogate_model_config["n_heads"]),
        num_layers=int(surrogate_model_config["num_layers"]),
        dim_feedforward=int(surrogate_model_config["dim_feedforward"]),
    ).to(device)
    surrogate.load_state_dict(checkpoint["surrogate_state_dict"])

    decoder = RHDecoder(
        sequence_length=config.sequence_length,
        bucket_count=config.C,
        d_model=int(decoder_model_config["d_model"]),
        n_heads=int(decoder_model_config["n_heads"]),
        num_layers=int(decoder_model_config["num_layers"]),
        dim_feedforward=int(decoder_model_config["dim_feedforward"]),
    ).to(device)
    decoder.load_state_dict(checkpoint["decoder_state_dict"])

    scatter_head = SoftScatterReconstructionHead(init_temperature=1.0, min_temperature=0.05).to(device)
    scatter_head.load_state_dict(checkpoint["scatter_head_state_dict"])

    perm_1d = build_grouped_permutation(config.n, config.C, config.seed, device)
    branch_name = str(branch_meta["name"])
    branch = LPAPBottleneckBranch(
        name=branch_name,
        surrogate=surrogate,
        decoder=decoder,
        scatter_head=scatter_head,
        perm_1d=perm_1d,
        fast_k=float(branch_meta.get("fast_k", config.fast_k)),
        surrogate_temperature=float(branch_meta.get("surrogate_temperature", 1.0)),
    )

    model = LPAPAutoencoder(
        image_to_energy_flow=image_to_energy_flow,
        energy_to_image_flow=energy_to_image_flow,
        bottleneck_branches=[branch],
        image_to_energy_steps=int(checkpoint["image_to_energy_steps"]),
        energy_to_image_steps=int(checkpoint["energy_to_image_steps"]),
        default_branch=branch_name,
    ).to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, side, float(checkpoint.get("image_scale", 1.0))


def load_dataset(manifest_path: Path, *, preload_images: bool) -> ImageDataset:
    if preload_images:
        return InMemoryGrayscaleImageShardDataset(manifest_path, as_float=True)
    return GrayscaleImageShardDataset(manifest_path, as_float=True)


def select_images(dataset: ImageDataset, *, start_index: int, num_images: int) -> tuple[torch.Tensor, list[int]]:
    if num_images <= 0:
        raise ValueError(f"num_images must be positive, got {num_images}")
    if start_index < 0:
        raise ValueError(f"start_index must be non-negative, got {start_index}")
    end_index = start_index + num_images
    if end_index > len(dataset):
        raise ValueError(f"requested images [{start_index}, {end_index}) exceed dataset size {len(dataset)}")
    indices = list(range(start_index, end_index))
    images = torch.stack([dataset[index] for index in indices], dim=0)
    return images, indices


def normalize_energy_pair(
    learned_energy: torch.Tensor, projected_energy: torch.Tensor, *, side: int
) -> tuple[torch.Tensor, torch.Tensor]:
    pair = torch.cat((learned_energy.abs(), projected_energy.abs()), dim=1)
    flat = pair.flatten(1)
    pair_min = flat.amin(dim=1, keepdim=True).view(pair.shape[0], 1, 1)
    pair_max = flat.amax(dim=1, keepdim=True).view(pair.shape[0], 1, 1)
    normalized = (pair - pair_min) / (pair_max - pair_min).clamp_min(1e-8)
    return normalized[:, 0:1].reshape(-1, 1, side, side), normalized[:, 1:2].reshape(-1, 1, side, side)


def make_spacer(*, batch_size: int, height: int, width: int, device: torch.device) -> torch.Tensor:
    return torch.ones((batch_size, 1, height, width), dtype=torch.float32, device=device)


def build_comparison_grid(
    input_images: torch.Tensor,
    reconstructed_images: torch.Tensor,
    learned_energy_images: torch.Tensor,
    projected_energy_images: torch.Tensor,
    *,
    separator: int = 2,
) -> torch.Tensor:
    batch_size, _channels, height, width = input_images.shape
    vertical = make_spacer(batch_size=batch_size, height=height, width=separator, device=input_images.device)
    rows = torch.cat(
        (
            input_images,
            vertical,
            reconstructed_images,
            vertical,
            learned_energy_images,
            vertical,
            projected_energy_images,
        ),
        dim=3,
    )
    horizontal = make_spacer(batch_size=1, height=separator, width=rows.shape[3], device=rows.device)
    stacked_rows: list[torch.Tensor] = []
    for row_index in range(batch_size):
        stacked_rows.append(rows[row_index : row_index + 1])
        if row_index + 1 != batch_size:
            stacked_rows.append(horizontal)
    return torch.cat(stacked_rows, dim=2).squeeze(0).clamp(0.0, 1.0)


def save_grayscale_png(image: torch.Tensor, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image_u8 = image.squeeze(0).mul(255.0).round().to(dtype=torch.uint8).cpu().numpy()
    Image.fromarray(image_u8, mode="L").save(output_path)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    checkpoint = load_e2e_checkpoint(args.checkpoint, device)
    manifest_path = resolve_manifest_path(args, checkpoint)
    model, side, image_scale = build_autoencoder_from_checkpoint(checkpoint, device)
    dataset = load_dataset(manifest_path, preload_images=args.preload_images)
    input_images, indices = select_images(dataset, start_index=args.start_index, num_images=args.num_images)

    input_images = input_images.to(device)
    image_sequence = hilbert_flatten_images(input_images, side=side).mul(image_scale)
    with torch.no_grad():
        output = model(image_sequence)

    input_display = input_images.clamp(0.0, 1.0)
    reconstruction_display = hilbert_unflatten_images(output.reconstructed_image / max(image_scale, 1e-8), side=side)
    reconstruction_display = reconstruction_display.clamp(0.0, 1.0)
    learned_energy_display, projected_energy_display = normalize_energy_pair(
        output.learned_energy,
        output.projected_energy,
        side=side,
    )
    grid = build_comparison_grid(
        input_display,
        reconstruction_display,
        learned_energy_display,
        projected_energy_display,
    )

    output_path = resolve_project_path(args.output)
    save_grayscale_png(grid, output_path)

    reconstruction_mse = ((output.reconstructed_image - image_sequence) ** 2).flatten(1).mean(dim=1)
    energy_l1 = (output.learned_energy - output.projected_energy).abs().flatten(1).mean(dim=1)

    print(f"Using device: {device}")
    print(f"Checkpoint: {checkpoint['_resolved_path']}")
    print(f"Image manifest: {manifest_path}")
    print("Columns: input | reconstruction | learned_energy | projected_energy")
    print(f"Saved comparison grid to {output_path}")
    print("")
    for sample_index, dataset_index in enumerate(indices):
        print(
            f"sample {sample_index:>2} (dataset index {dataset_index:>6}) | "
            f"image_mse {reconstruction_mse[sample_index].item():.6f} | "
            f"energy_l1 {energy_l1[sample_index].item():.6f}"
        )


if __name__ == "__main__":
    main()
