"""Train RHReconstructor on synthetic harmonic series from decoder-derived tokens."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

_EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(_EXP))
sys.path.insert(0, str(_EXP.parent / "src"))

from rh_memory.decoder import RHDecoder
from rh_memory.reconstructor import RHReconstructor, RHReconstructorLoss
from rh_memory.surrogate import RHSurrogate

from pipeline import (
    decoder_stage,
    harmonic_stage,
    iter_take,
    reconstructor_training_adapter,
    surrogate_stage,
    worker_init_fn,
)
from pipeline.utils import IterableFactoryDataset


def relative_l2_percent(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    err = torch.linalg.vector_norm(pred - target, ord=2, dim=1)
    den = torch.linalg.vector_norm(target, ord=2, dim=1).clamp_min(eps)
    return ((err / den).mean().item()) * 100.0


def load_surrogate(checkpoint_path: Path, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    meta = ckpt["meta"]
    model = RHSurrogate(
        sequence_length=meta["seq_max"],
        bucket_count=meta["C"],
        stride=meta["stride"],
        fast_k=meta["fast_k"],
        d_model=meta["d_model"],
        n_heads=meta["n_heads"],
        num_layers=meta["num_layers"],
        dim_feedforward=meta["dim_feedforward"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, meta


def load_decoder(
    checkpoint_path: Path,
    device: torch.device,
    sequence_length: int,
    bucket_count: int,
    d_model: int,
    n_heads: int,
    num_layers: int,
    ff_mult: int,
):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = RHDecoder(
        sequence_length=sequence_length,
        bucket_count=bucket_count,
        d_model=d_model,
        n_heads=n_heads,
        num_layers=num_layers,
        dim_feedforward=d_model * ff_mult,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def parse_args():
    p = argparse.ArgumentParser(description="Train RHReconstructor from decoder-derived tokens.")
    p.add_argument("--surrogate-checkpoint", type=Path, default=Path("experiments/surrogate_checkpoint.pt"))
    p.add_argument("--decoder-checkpoint", type=Path, default=Path("experiments/decoder_surrogate_checkpoint.pt"))
    p.add_argument("--n", type=int, default=None, help="Override sequence length (default: from surrogate meta)")
    p.add_argument("--C", type=int, default=None, help="Override bucket count (default: from surrogate meta)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--decoder-d-model", type=int, default=512)
    p.add_argument("--decoder-n-heads", type=int, default=16)
    p.add_argument("--decoder-num-layers", type=int, default=6)
    p.add_argument("--decoder-ff-mult", type=int, default=4)
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--n-heads", type=int, default=16)
    p.add_argument("--num-token-layers", type=int, default=4)
    p.add_argument("--num-query-layers", type=int, default=4)
    p.add_argument("--ff-mult", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--total-steps", type=int, default=50_000_000 // 128)
    p.add_argument("--eval-every", type=int, default=50_000 // 128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--checkpoint", type=Path, default=Path("experiments/reconstructor_checkpoint.pt"))
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    surrogate, meta = load_surrogate(args.surrogate_checkpoint, device)
    print(f"Loaded surrogate from {args.surrogate_checkpoint}")

    n = args.n if args.n is not None else meta["n"]
    C = args.C if args.C is not None else meta["C"]
    if n % C != 0:
        raise ValueError(f"Grouped permutation mode requires n % C == 0, got n={n}, C={C}")

    decoder = load_decoder(
        args.decoder_checkpoint,
        device,
        sequence_length=n,
        bucket_count=C,
        d_model=args.decoder_d_model,
        n_heads=args.decoder_n_heads,
        num_layers=args.decoder_num_layers,
        ff_mult=args.decoder_ff_mult,
    )
    print(f"Loaded decoder from {args.decoder_checkpoint}")
    print(f"n={n}, C={C}")

    B = args.batch_size
    def make_train_stream():
        base = harmonic_stage(
            n=n,
            C=C,
            chunk_size=B,
            seed=meta["seed"],
            device=device,
            harmonic_decay=meta["harmonic_decay"],
            harmonic_amp_threshold=meta["harmonic_amp_threshold"],
            max_harmonics=meta["max_harmonics"],
        )
        sur = surrogate_stage(base, surrogate=surrogate, fast_k=meta["fast_k"])
        dec = decoder_stage(sur, decoder=decoder)
        return reconstructor_training_adapter(dec)

    test_num = min(1024, B * 8)
    test_chunks = max(1, (test_num + B - 1) // B)

    def make_test_stream():
        base = harmonic_stage(
            n=n,
            C=C,
            chunk_size=B,
            seed=meta["seed"],
            device=device,
            harmonic_decay=meta["harmonic_decay"],
            harmonic_amp_threshold=meta["harmonic_amp_threshold"],
            max_harmonics=meta["max_harmonics"],
        )
        sur = surrogate_stage(base, surrogate=surrogate, fast_k=meta["fast_k"])
        dec = decoder_stage(sur, decoder=decoder)
        return iter_take(reconstructor_training_adapter(dec), test_chunks)

    train_dataset = IterableFactoryDataset(make_train_stream)
    test_dataset = IterableFactoryDataset(make_test_stream)

    workers = 0 if "cuda" in str(device) else 0
    train_loader = DataLoader(
        train_dataset,
        batch_size=None,
        num_workers=workers,
        prefetch_factor=None,
        worker_init_fn=worker_init_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=None,
        num_workers=workers,
        prefetch_factor=None,
        worker_init_fn=worker_init_fn,
    )

    model = RHReconstructor(
        sequence_length=n,
        bucket_count=C,
        d_model=args.d_model,
        n_heads=args.n_heads,
        num_token_layers=args.num_token_layers,
        num_query_layers=args.num_query_layers,
        dim_feedforward=args.d_model * args.ff_mult,
    ).to(device)

    criterion = RHReconstructorLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    start_step = 0
    if args.checkpoint.exists():
        print(f"Loading checkpoint from {args.checkpoint}...")
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_step = ckpt.get("step", 0)
        print(f"Resumed from step {start_step}")

    model.train()
    running_loss = 0.0
    running_rel_l2 = 0.0
    batches_since_eval = 0

    for step_idx, batch in enumerate(train_loader, 1):
        step = start_step + step_idx
        if step > args.total_steps:
            break

        recon_tokens = batch[0].to(device)
        target_series = batch[1].to(device)

        optimizer.zero_grad()
        pred = model(recon_tokens)
        loss = criterion(pred, target_series)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_rel_l2 += relative_l2_percent(pred.detach(), target_series.detach())
        batches_since_eval += 1

        if step % args.eval_every == 0 or step == 1:
            avg_train = running_loss / batches_since_eval
            avg_train_rel_l2 = running_rel_l2 / batches_since_eval

            model.eval()
            test_loss = 0.0
            test_rel_l2 = 0.0
            test_batches = 0
            with torch.no_grad():
                for tb in test_loader:
                    tokens_t = tb[0].to(device)
                    target_t = tb[1].to(device)
                    pred_t = model(tokens_t)
                    tl = criterion(pred_t, target_t)
                    test_loss += tl.item()
                    test_rel_l2 += relative_l2_percent(pred_t, target_t)
                    test_batches += 1

            avg_test = test_loss / max(test_batches, 1)
            avg_test_rel_l2 = test_rel_l2 / max(test_batches, 1)
            print(
                f"Step {step} | Train MSE: {avg_train:.6f} | Train RelL2: {avg_train_rel_l2:.2f}% | "
                f"Test MSE: {avg_test:.6f} | Test RelL2: {avg_test_rel_l2:.2f}%"
            )

            torch.save(
                {
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "meta": {
                        "n": n,
                        "C": C,
                        "seed": args.seed,
                        "fast_k": meta["fast_k"],
                        "d_model": args.d_model,
                        "n_heads": args.n_heads,
                        "num_token_layers": args.num_token_layers,
                        "num_query_layers": args.num_query_layers,
                        "dim_feedforward": args.d_model * args.ff_mult,
                    },
                },
                args.checkpoint,
            )
            print(f"Saved checkpoint to {args.checkpoint}")

            running_loss = 0.0
            running_rel_l2 = 0.0
            batches_since_eval = 0
            model.train()

    print("Training finished.")


if __name__ == "__main__":
    main()
