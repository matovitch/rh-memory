"""Evaluate a hard surrogate scatter baseline on synthetic harmonic signals.

This script leaves the training pipeline untouched. It loads a frozen surrogate,
builds soft decoder tokens from surrogate logits, takes a hard argmax over the
surrogate logits for each bucket, then scatters each bucket's soft amplitude back
to the predicted source coordinate.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from rh_memory.pipeline import PipelineConfig, harmonic_stage, surrogate_stage
from rh_memory.pipeline.primitives_tokens import normalized_entropy
from rh_memory.surrogate import RHSurrogate


def relative_l2_percent(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    err = torch.linalg.vector_norm(pred - target, ord=2, dim=1)
    den = torch.linalg.vector_norm(target, ord=2, dim=1).clamp_min(eps)
    return (err / den) * 100.0


def retained_energy_ratio(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    pred_energy = pred.square().sum(dim=1)
    target_energy = target.square().sum(dim=1).clamp_min(eps)
    return pred_energy / target_energy


def cosine_similarity(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    num = (pred * target).sum(dim=1)
    den = torch.linalg.vector_norm(pred, ord=2, dim=1) * torch.linalg.vector_norm(target, ord=2, dim=1)
    return num / den.clamp_min(eps)


def load_surrogate(checkpoint_path: Path, device: torch.device) -> tuple[RHSurrogate, PipelineConfig]:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if ckpt.get("version") != "v2_ce_no_decoder":
        raise ValueError(
            f"Unsupported surrogate checkpoint version {ckpt.get('version')!r}; "
            "expected v2_ce_no_decoder."
        )
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


def surrogate_hard_scatter_baseline(
    surrogate_logits: torch.Tensor,
    decoder_tokens: torch.Tensor,
    perm_1d: torch.Tensor,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Scatter surrogate-token soft amplitudes at argmax surrogate source positions."""
    B, _C, n = surrogate_logits.shape
    predicted_slot = surrogate_logits.argmax(dim=-1)
    source_index_by_slot = perm_1d.to(device=surrogate_logits.device, dtype=torch.long)
    predicted_source = source_index_by_slot[predicted_slot]

    soft_amplitude = decoder_tokens[..., 0]
    baseline = torch.zeros(B, n, dtype=decoder_tokens.dtype, device=decoder_tokens.device)
    baseline.scatter_add_(1, predicted_source, soft_amplitude)

    probs = torch.softmax(surrogate_logits / temperature, dim=-1)
    doubt = normalized_entropy(probs)
    return baseline, predicted_slot, predicted_source, doubt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate hard surrogate scatter RelL2 on harmonic batches.")
    p.add_argument("--surrogate-checkpoint", type=Path, default=Path("experiments/surrogate_ce_checkpoint.pt"))
    p.add_argument("--n", type=int, default=None, help="Override sequence length; must match checkpoint.")
    p.add_argument("--C", type=int, default=None, help="Override bucket count; must match checkpoint.")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--batches", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--print-every", type=int, default=0, help="Print per-batch metrics every N batches; 0 disables.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is not available")

    torch.manual_seed(args.seed)
    surrogate, surrogate_config = load_surrogate(args.surrogate_checkpoint, device)

    n = args.n if args.n is not None else surrogate_config.n
    C = args.C if args.C is not None else surrogate_config.C
    if n != surrogate_config.n or C != surrogate_config.C:
        raise ValueError(
            f"n/C must match surrogate checkpoint; got n={n}, C={C}, "
            f"checkpoint has n={surrogate_config.n}, C={surrogate_config.C}."
        )
    config = PipelineConfig(
        n=n,
        C=C,
        batch_size=args.batch_size,
        seed=args.seed,
        fast_k=surrogate_config.fast_k,
        harmonic_decay=surrogate_config.harmonic_decay,
        harmonic_amp_threshold=surrogate_config.harmonic_amp_threshold,
        max_harmonics=surrogate_config.max_harmonics,
    )

    print(
        f"Surrogate hard scatter baseline | device={device} | N={config.n} C={config.C} "
        f"stride={config.stride} batch_size={config.batch_size} batches={args.batches} temperature={args.temperature}"
    )
    print(f"Loaded surrogate: {args.surrogate_checkpoint}")

    base_stream = harmonic_stage(config=config, device=device)
    stream = surrogate_stage(base_stream, config=config, surrogate=surrogate, temperature=args.temperature)

    all_rel_l2: list[torch.Tensor] = []
    all_zero_rel_l2: list[torch.Tensor] = []
    all_energy: list[torch.Tensor] = []
    all_cos: list[torch.Tensor] = []
    all_doubt: list[torch.Tensor] = []
    all_unique_sources: list[float] = []

    with torch.no_grad():
        for batch_idx in range(1, args.batches + 1):
            sample = next(stream)
            baseline, _predicted_slot, predicted_source, doubt = surrogate_hard_scatter_baseline(
                sample.surrogate_logits,
                sample.decoder_tokens,
                sample.perm_1d,
                args.temperature,
            )
            target = sample.raw_inputs
            zero = torch.zeros_like(target)

            rel_l2 = relative_l2_percent(baseline, target)
            zero_rel_l2 = relative_l2_percent(zero, target)
            energy = retained_energy_ratio(baseline, target)
            cos = cosine_similarity(baseline, target)
            unique_sources = torch.tensor(
                [torch.unique(predicted_source[row]).numel() for row in range(predicted_source.size(0))],
                dtype=torch.float32,
            ).mean().item()

            all_rel_l2.append(rel_l2.detach().cpu())
            all_zero_rel_l2.append(zero_rel_l2.detach().cpu())
            all_energy.append(energy.detach().cpu())
            all_cos.append(cos.detach().cpu())
            all_doubt.append(doubt.detach().cpu().reshape(-1))
            all_unique_sources.append(unique_sources)

            if args.print_every > 0 and batch_idx % args.print_every == 0:
                print(
                    f"Batch {batch_idx} | "
                    f"Surrogate RelL2={rel_l2.mean().item():.2f}% | "
                    f"Zero RelL2={zero_rel_l2.mean().item():.2f}% | "
                    f"RetainedEnergy={energy.mean().item():.4f} | "
                    f"Cos={cos.mean().item():.4f} | "
                    f"Doubt={doubt.mean().item():.4f} | "
                    f"UniqueSources={unique_sources:.1f}/{config.C}"
                )

    rel_l2_all = torch.cat(all_rel_l2)
    zero_rel_l2_all = torch.cat(all_zero_rel_l2)
    energy_all = torch.cat(all_energy)
    cos_all = torch.cat(all_cos)
    doubt_all = torch.cat(all_doubt)
    oracle_from_energy = torch.sqrt((1.0 - energy_all).clamp_min(0.0)) * 100.0

    print("Summary:")
    print(f"  Surrogate hard scatter RelL2 mean: {rel_l2_all.mean().item():.2f}%")
    print(f"  Surrogate hard scatter RelL2 p50 : {rel_l2_all.median().item():.2f}%")
    print(f"  Surrogate hard scatter RelL2 p90 : {rel_l2_all.quantile(0.90).item():.2f}%")
    print(f"  Zero baseline RelL2 mean         : {zero_rel_l2_all.mean().item():.2f}%")
    print(f"  Retained energy mean             : {energy_all.mean().item():.4f}")
    print(f"  Retained energy p50              : {energy_all.median().item():.4f}")
    print(f"  Scatter cosine mean              : {cos_all.mean().item():.4f}")
    print(f"  sqrt(1-energy) RelL2             : {oracle_from_energy.mean().item():.2f}%")
    print(f"  Surrogate doubt mean             : {doubt_all.mean().item():.4f}")
    print(f"  Unique predicted sources mean    : {sum(all_unique_sources) / max(1, len(all_unique_sources)):.1f}/{config.C}")


if __name__ == "__main__":
    main()