"""Evaluate a soft decoder scatter baseline on synthetic harmonic signals.

This script leaves the training pipeline untouched. It loads a frozen surrogate and
soft-distilled decoder, builds decoder input tokens from the surrogate, converts
decoder logits to probabilities, then scatters each bucket's soft amplitude over
all source coordinates according to the decoder probability mass.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from rh_memory.decoder import RHDecoder
from rh_memory.decoder_scatter import decoder_soft_scatter, effective_support
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


def load_decoder(checkpoint_path: Path, device: torch.device) -> tuple[RHDecoder, PipelineConfig, float]:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if ckpt.get("version") != "v1_soft_decoder_distill":
        raise ValueError(
            f"Unsupported decoder checkpoint version {ckpt.get('version')!r}; "
            "expected v1_soft_decoder_distill."
        )
    config = PipelineConfig.from_dict(ckpt["config"])
    model_config = ckpt["model_config"]
    model = RHDecoder(
        sequence_length=config.sequence_length,
        bucket_count=config.C,
        d_model=model_config["d_model"],
        n_heads=model_config["n_heads"],
        num_layers=model_config["num_layers"],
        dim_feedforward=model_config["dim_feedforward"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model, config, float(ckpt.get("temperature", 1.0))


def decoder_soft_scatter_baseline(
    decoder: RHDecoder,
    decoder_tokens: torch.Tensor,
    perm_1d: torch.Tensor,
    n: int,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Scatter decoder-token soft amplitudes over every source position by decoder probabilities."""
    logits = decoder(decoder_tokens)[:, : decoder_tokens.size(1), :n]
    baseline, probs = decoder_soft_scatter(logits, decoder_tokens[..., 0], perm_1d, temperature)
    doubt = normalized_entropy(probs)
    support = effective_support(probs)
    return baseline, doubt, support


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate soft decoder scatter RelL2 on harmonic batches.")
    p.add_argument("--surrogate-checkpoint", type=Path, default=Path("experiments/surrogate_ce_checkpoint.pt"))
    p.add_argument("--decoder-checkpoint", type=Path, default=Path("experiments/decoder_soft_distill_checkpoint.pt"))
    p.add_argument("--n", type=int, default=None, help="Override sequence length; must match checkpoints.")
    p.add_argument("--C", type=int, default=None, help="Override bucket count; must match checkpoints.")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--batches", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--temperature", type=float, default=None, help="Override checkpoint distillation temperature.")
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
    decoder, decoder_config, checkpoint_temperature = load_decoder(args.decoder_checkpoint, device)

    n = args.n if args.n is not None else surrogate_config.n
    C = args.C if args.C is not None else surrogate_config.C
    if n != surrogate_config.n or C != surrogate_config.C:
        raise ValueError(
            f"n/C must match surrogate checkpoint; got n={n}, C={C}, "
            f"checkpoint has n={surrogate_config.n}, C={surrogate_config.C}."
        )
    if n != decoder_config.n or C != decoder_config.C:
        raise ValueError(
            f"n/C must match decoder checkpoint; got n={n}, C={C}, "
            f"checkpoint has n={decoder_config.n}, C={decoder_config.C}."
        )
    temperature = checkpoint_temperature if args.temperature is None else args.temperature
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
        f"Decoder soft scatter baseline | device={device} | N={config.n} C={config.C} "
        f"stride={config.stride} batch_size={config.batch_size} batches={args.batches} temperature={temperature}"
    )
    print(f"Loaded surrogate: {args.surrogate_checkpoint}")
    print(f"Loaded decoder  : {args.decoder_checkpoint}")

    base_stream = harmonic_stage(config=config, device=device)
    stream = surrogate_stage(base_stream, config=config, surrogate=surrogate, temperature=temperature)

    all_rel_l2: list[torch.Tensor] = []
    all_zero_rel_l2: list[torch.Tensor] = []
    all_energy: list[torch.Tensor] = []
    all_cos: list[torch.Tensor] = []
    all_doubt: list[torch.Tensor] = []
    all_support: list[torch.Tensor] = []

    with torch.no_grad():
        for batch_idx in range(1, args.batches + 1):
            sample = next(stream)
            baseline, doubt, support = decoder_soft_scatter_baseline(
                decoder,
                sample.decoder_tokens,
                sample.perm_1d,
                config.n,
                temperature,
            )
            target = sample.raw_inputs
            zero = torch.zeros_like(target)

            rel_l2 = relative_l2_percent(baseline, target)
            zero_rel_l2 = relative_l2_percent(zero, target)
            energy = retained_energy_ratio(baseline, target)
            cos = cosine_similarity(baseline, target)

            all_rel_l2.append(rel_l2.detach().cpu())
            all_zero_rel_l2.append(zero_rel_l2.detach().cpu())
            all_energy.append(energy.detach().cpu())
            all_cos.append(cos.detach().cpu())
            all_doubt.append(doubt.detach().cpu().reshape(-1))
            all_support.append(support.detach().cpu().reshape(-1))

            if args.print_every > 0 and batch_idx % args.print_every == 0:
                print(
                    f"Batch {batch_idx} | "
                    f"Decoder RelL2={rel_l2.mean().item():.2f}% | "
                    f"Zero RelL2={zero_rel_l2.mean().item():.2f}% | "
                    f"RetainedEnergy={energy.mean().item():.4f} | "
                    f"Cos={cos.mean().item():.4f} | "
                    f"Doubt={doubt.mean().item():.4f} | "
                    f"EffSupport={support.mean().item():.1f}/{config.n}"
                )

    rel_l2_all = torch.cat(all_rel_l2)
    zero_rel_l2_all = torch.cat(all_zero_rel_l2)
    energy_all = torch.cat(all_energy)
    cos_all = torch.cat(all_cos)
    doubt_all = torch.cat(all_doubt)
    support_all = torch.cat(all_support)
    oracle_from_energy = torch.sqrt((1.0 - energy_all).clamp_min(0.0)) * 100.0

    print("Summary:")
    print(f"  Decoder soft scatter RelL2 mean: {rel_l2_all.mean().item():.2f}%")
    print(f"  Decoder soft scatter RelL2 p50 : {rel_l2_all.median().item():.2f}%")
    print(f"  Decoder soft scatter RelL2 p90 : {rel_l2_all.quantile(0.90).item():.2f}%")
    print(f"  Zero baseline RelL2 mean       : {zero_rel_l2_all.mean().item():.2f}%")
    print(f"  Retained energy mean           : {energy_all.mean().item():.4f}")
    print(f"  Retained energy p50            : {energy_all.median().item():.4f}")
    print(f"  Scatter cosine mean            : {cos_all.mean().item():.4f}")
    print(f"  sqrt(1-energy) RelL2           : {oracle_from_energy.mean().item():.2f}%")
    print(f"  Decoder doubt mean             : {doubt_all.mean().item():.4f}")
    print(f"  Effective support mean         : {support_all.mean().item():.1f}/{config.n}")


if __name__ == "__main__":
    main()
