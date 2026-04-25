"""Train RHSurrogate on harmonic synthetic data with LPAP bucket labels (permuted slot stream)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

_EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(_EXP))
sys.path.insert(0, str(_EXP.parent / "src"))

from rh_memory.surrogate import RHSurrogate, RHSurrogateLoss

from pipeline import (
    harmonic_stage,
    iter_take,
    surrogate_stage,
    surrogate_training_adapter,
    worker_init_fn,
)
from pipeline.utils import IterableFactoryDataset


def bucket_slot_accuracy_percent(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Percent of buckets where predicted slot index j matches the one-hot teacher target.

    Rows without a teacher assignment (all-zero targets) are ignored.
    """
    pred_slot = logits.argmax(dim=2)
    true_slot = targets.argmax(dim=2)
    has_teacher = targets.sum(dim=2) > 0
    denom = has_teacher.sum().item()
    if denom == 0:
        return 0.0
    correct = ((pred_slot == true_slot) & has_teacher).sum().item()
    return correct / denom * 100.0


def parse_args():
    p = argparse.ArgumentParser(description="Train RHSurrogate on synthetic harmonic + LPAP teacher.")
    p.add_argument("--n", type=int, default=1024)
    p.add_argument("--C", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--n-heads", type=int, default=16)
    p.add_argument("--num-layers", type=int, default=6)
    p.add_argument("--ff-mult", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--total-steps", type=int, default=50_000_000 // 128)
    p.add_argument("--eval-every", type=int, default=50_000 // 128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fast-k", type=float, default=5.0)
    p.add_argument("--checkpoint", type=Path, default=Path("experiments/surrogate_checkpoint.pt"))
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    n = args.n
    C = args.C
    if n % C != 0:
        raise ValueError(f"Grouped permutation mode requires n % C == 0, got n={n}, C={C}")
    seq_max = n
    print(f"Dataset meta — n={n}, C={C}, sequence_length={seq_max}")

    B = args.batch_size
    def make_train_stream():
        base = harmonic_stage(
            n=n,
            C=C,
            chunk_size=B,
            seed=args.seed,
            device=device,
            harmonic_decay=0.65,
            harmonic_amp_threshold=0.1,
            max_harmonics=64,
        )
        sur = surrogate_stage(base, surrogate=model, fast_k=args.fast_k)
        return surrogate_training_adapter(sur)

    test_num = min(1024, B * 8)
    test_chunks = max(1, (test_num + B - 1) // B)

    def make_test_stream():
        base = harmonic_stage(
            n=n,
            C=C,
            chunk_size=B,
            seed=args.seed,
            device=device,
            harmonic_decay=0.65,
            harmonic_amp_threshold=0.1,
            max_harmonics=64,
        )
        sur = surrogate_stage(base, surrogate=model, fast_k=args.fast_k)
        return iter_take(surrogate_training_adapter(sur), test_chunks)

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

    model = RHSurrogate(
        sequence_length=seq_max,
        bucket_count=C,
        stride=seq_max // C,
        fast_k=args.fast_k,
        d_model=args.d_model,
        n_heads=args.n_heads,
        num_layers=args.num_layers,
        dim_feedforward=args.d_model * args.ff_mult,
    ).to(device)

    criterion = RHSurrogateLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    start_step = 0
    if args.checkpoint.exists():
        print(f"Loading checkpoint from {args.checkpoint}...")
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_step = ckpt.get("step", 0)
        print(f"Resumed from step {start_step}")

    meta = {
        "n": n,
        "C": C,
        "seq_max": seq_max,
        "stride": seq_max // C,
        "seed": args.seed,
        "fast_k": args.fast_k,
        "harmonic_decay": 0.65,
        "harmonic_amp_threshold": 0.1,
        "max_harmonics": 64,
        "d_model": args.d_model,
        "n_heads": args.n_heads,
        "num_layers": args.num_layers,
        "dim_feedforward": args.d_model * args.ff_mult,
    }

    model.train()
    running_loss = 0.0
    running_acc = 0.0
    batches_since_eval = 0

    for step_idx, batch in enumerate(train_loader, 1):
        step = start_step + step_idx
        if step > args.total_steps:
            break

        position_tokens = batch[0].to(device)
        targets_bCn = batch[1].to(device)
        weights_bn = batch[2].to(device)

        optimizer.zero_grad()

        logits_full = model(position_tokens)
        logits = logits_full[:, :C, :n]
        targets = targets_bCn[:, :C, :n]

        loss = criterion(logits, targets, weights_bn)
        loss.backward()
        optimizer.step()

        train_acc = bucket_slot_accuracy_percent(logits, targets)
        running_loss += loss.item()
        running_acc += train_acc
        batches_since_eval += 1

        if step % args.eval_every == 0 or step == 1:
            avg_train = running_loss / batches_since_eval
            avg_train_acc = running_acc / batches_since_eval
            model.eval()
            test_loss = 0.0
            test_acc = 0.0
            test_batches = 0
            with torch.no_grad():
                for tb in test_loader:
                    pt = tb[0].to(device)
                    tg = tb[1].to(device)
                    w = tb[2].to(device)
                    lf = model(pt)
                    lt = lf[:, :C, :n]
                    tt = tg[:, :C, :n]
                    test_loss += criterion(lt, tt, w).item()
                    test_acc += bucket_slot_accuracy_percent(lt, tt)
                    test_batches += 1
            avg_test = test_loss / max(test_batches, 1)
            avg_test_acc = test_acc / max(test_batches, 1)
            print(
                f"Step {step} | Train Loss: {avg_train:.6f} | Train Acc: {avg_train_acc:.2f}% | "
                f"Test Loss: {avg_test:.6f} | Test Acc: {avg_test_acc:.2f}%"
            )

            torch.save(
                {
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "meta": meta,
                },
                args.checkpoint,
            )
            print(f"Saved checkpoint to {args.checkpoint}")

            running_loss = 0.0
            running_acc = 0.0
            batches_since_eval = 0
            model.train()

    print("Training finished.")


if __name__ == "__main__":
    main()
