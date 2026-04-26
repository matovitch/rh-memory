"""Evaluate an oracle LPAP scatter baseline on synthetic harmonic signals.

This script leaves the training pipeline untouched. It generates harmonic batches,
runs LPAP with source-position carry ids, scatters the retained bucket values back
to source coordinates, and reports reconstruction RelL2 / retained energy.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from rh_memory.pipeline import PipelineConfig, harmonic_stage
from rh_memory.pooling_utils import lpap_pool


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


def lpap_scatter_baseline(sample, config: PipelineConfig, device: torch.device) -> torch.Tensor:
    """Run LPAP and scatter retained signed amplitudes back to unpermuted source positions."""
    B = sample.x_perm.size(0)
    table_values = torch.zeros(B, config.C, dtype=torch.float32, device=device)
    table_dib = torch.zeros(B, config.C, dtype=torch.int32, device=device)
    table_carry_id = torch.full((B, config.C), -1, dtype=torch.int32, device=device)

    # Carry source ids, not permuted slot ids, so table_carry_id is directly scatter-ready.
    source_ids = sample.perm_1d.to(device=device, dtype=torch.int32).unsqueeze(0).expand(B, config.n)
    out_values, _out_dib, out_source_id = lpap_pool(
        table_values,
        table_dib,
        table_carry_id,
        sample.x_perm.clone(),
        source_ids.clone(),
        config.k_eff,
        device,
    )

    valid = (out_source_id >= 0) & (out_source_id < config.n)
    baseline = torch.zeros(B, config.n, dtype=sample.raw_inputs.dtype, device=device)
    scatter_idx = out_source_id.long().clamp(0, config.n - 1)
    baseline.scatter_add_(1, scatter_idx, out_values * valid.to(dtype=out_values.dtype))
    return baseline


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate oracle LPAP scatter RelL2 on harmonic batches.")
    p.add_argument("--n", type=int, default=1024)
    p.add_argument("--C", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--batches", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fast-k", type=float, default=5.0)
    p.add_argument("--harmonic-decay", type=float, default=0.65)
    p.add_argument("--harmonic-amp-threshold", type=float, default=0.1)
    p.add_argument("--max-harmonics", type=int, default=64)
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
    config = PipelineConfig(
        n=args.n,
        C=args.C,
        batch_size=args.batch_size,
        seed=args.seed,
        fast_k=args.fast_k,
        harmonic_decay=args.harmonic_decay,
        harmonic_amp_threshold=args.harmonic_amp_threshold,
        max_harmonics=args.max_harmonics,
    )

    print(
        f"LPAP scatter baseline | device={device} | N={config.n} C={config.C} "
        f"stride={config.stride} k_eff={config.k_eff} batch_size={config.batch_size} batches={args.batches}"
    )

    stream = harmonic_stage(config=config, device=device)
    all_rel_l2: list[torch.Tensor] = []
    all_zero_rel_l2: list[torch.Tensor] = []
    all_energy: list[torch.Tensor] = []
    all_cos: list[torch.Tensor] = []

    with torch.no_grad():
        for batch_idx in range(1, args.batches + 1):
            sample = next(stream)
            baseline = lpap_scatter_baseline(sample, config, device)
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

            if args.print_every > 0 and batch_idx % args.print_every == 0:
                print(
                    f"Batch {batch_idx} | "
                    f"LPAP RelL2={rel_l2.mean().item():.2f}% | "
                    f"Zero RelL2={zero_rel_l2.mean().item():.2f}% | "
                    f"RetainedEnergy={energy.mean().item():.4f} | "
                    f"Cos={cos.mean().item():.4f}"
                )

    rel_l2_all = torch.cat(all_rel_l2)
    zero_rel_l2_all = torch.cat(all_zero_rel_l2)
    energy_all = torch.cat(all_energy)
    cos_all = torch.cat(all_cos)
    oracle_from_energy = torch.sqrt((1.0 - energy_all).clamp_min(0.0)) * 100.0

    print("Summary:")
    print(f"  LPAP scatter RelL2 mean: {rel_l2_all.mean().item():.2f}%")
    print(f"  LPAP scatter RelL2 p50 : {rel_l2_all.median().item():.2f}%")
    print(f"  LPAP scatter RelL2 p90 : {rel_l2_all.quantile(0.90).item():.2f}%")
    print(f"  Zero baseline RelL2 mean: {zero_rel_l2_all.mean().item():.2f}%")
    print(f"  Retained energy mean    : {energy_all.mean().item():.4f}")
    print(f"  Retained energy p50     : {energy_all.median().item():.4f}")
    print(f"  Scatter cosine mean     : {cos_all.mean().item():.4f}")
    print(f"  sqrt(1-energy) RelL2    : {oracle_from_energy.mean().item():.2f}%")


if __name__ == "__main__":
    main()