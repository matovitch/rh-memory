"""
Compare Triton vs pure-PyTorch linear-probing-based amplitude pooling (torch.utils.benchmark).

Run from the repo root (with CUDA), e.g.:
  pixi run bench-lp-amplitude-pool
  # or
  python benchmarks/bench_linear_probing_amplitude_pooling.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.benchmark import Compare, Timer

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from rh_memory import (  # noqa: E402
    python_linear_probing_amplitude_pooling,
    triton_linear_probing_amplitude_pooling,
)


def _make_inputs(
    B: int,
    C: int,
    stride: int,
    k: int,
    device: torch.device,
    seed: int = 0,
) -> tuple[torch.Tensor, ...]:
    n = C * stride
    g = torch.Generator(device=device)
    g.manual_seed(seed)

    t_vals = torch.zeros(B, C, dtype=torch.float32, device=device)
    t_dib = torch.zeros(B, C, dtype=torch.int32, device=device)
    t_carry = torch.full((B, C), -1, dtype=torch.int32, device=device)
    inc_vals = torch.randn(B, n, dtype=torch.float32, device=device, generator=g)
    inc_carry = torch.arange(n, dtype=torch.int32, device=device).unsqueeze(0).expand(B, n)
    mask = (torch.rand(B, n, device=device, generator=g) > 0.5).to(inc_vals.dtype)
    inc_vals = inc_vals * mask
    return t_vals, t_dib, t_carry, inc_vals, inc_carry, k


def _warmup(
    t_vals: torch.Tensor,
    t_dib: torch.Tensor,
    t_carry: torch.Tensor,
    inc_vals: torch.Tensor,
    inc_carry: torch.Tensor,
    k: int,
    iters: int = 4,
) -> None:
    for _ in range(iters):
        triton_linear_probing_amplitude_pooling(t_vals, t_dib, t_carry, inc_vals, inc_carry, k)
    for _ in range(iters):
        python_linear_probing_amplitude_pooling(t_vals, t_dib, t_carry, inc_vals, inc_carry, k)
    torch.cuda.synchronize()


def bench_config(
    B: int,
    C: int,
    stride: int,
    k: int,
    device: torch.device,
    seed: int,
) -> list:
    t_vals, t_dib, t_carry, inc_vals, inc_carry, k_ = _make_inputs(
        B, C, stride, k, device, seed=seed
    )
    assert k_ == k
    _warmup(t_vals, t_dib, t_carry, inc_vals, inc_carry, k)

    n = C * stride
    sub_label = f"B={B} n={n} C={C} stride={stride} k={k}"
    g: dict = {
        "t_vals": t_vals,
        "t_dib": t_dib,
        "t_carry": t_carry,
        "inc_vals": inc_vals,
        "inc_carry": inc_carry,
        "k": k,
        "torch": torch,
        "python_linear_probing_amplitude_pooling": python_linear_probing_amplitude_pooling,
        "triton_linear_probing_amplitude_pooling": triton_linear_probing_amplitude_pooling,
    }

    results = []
    for name, stmt in (
        ("triton", "triton_linear_probing_amplitude_pooling(t_vals, t_dib, t_carry, inc_vals, inc_carry, k)"),
        ("python", "python_linear_probing_amplitude_pooling(t_vals, t_dib, t_carry, inc_vals, inc_carry, k)"),
    ):
        timer = Timer(
            stmt=stmt,
            globals=g,
            label="linear_probing_amplitude_pooling",
            sub_label=sub_label,
            description=name,
        )
        results.append(timer.blocked_autorange())
        torch.cuda.synchronize()
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark triton vs python linear_probing_amplitude_pooling"
    )
    parser.add_argument("--B", type=int, default=8, help="batch size")
    parser.add_argument("--C", type=int, default=128, help="table capacity")
    parser.add_argument("--stride", type=int, default=8, help="pipeline depth (n = C * stride)")
    parser.add_argument("--k", type=int, default=24, help="pooling iterations")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--preset",
        choices=("default", "suite"),
        default="default",
        help="'suite' runs several (B,C,stride,k) combinations",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA is required for this benchmark (Triton).", file=sys.stderr)
        sys.exit(1)

    device = torch.device("cuda")
    all_results: list = []

    if args.preset == "default":
        all_results.extend(
            bench_config(args.B, args.C, args.stride, args.k, device, args.seed)
        )
    else:
        suite = [
            (2, 128, 8, 24),
            (8, 128, 8, 24),
            (8, 256, 16, 32),
            (16, 128, 8, 24),
        ]
        for B, C, stride, k in suite:
            all_results.extend(bench_config(B, C, stride, k, device, args.seed))

    Compare(all_results).print()


if __name__ == "__main__":
    main()
