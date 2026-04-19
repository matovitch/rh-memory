"""Train RHSurrogate on harmonic synthetic data with LPAP bucket labels (permuted slot stream)."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, IterableDataset

_EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(_EXP))
sys.path.insert(0, str(_EXP.parent / "src"))

from rh_memory.surrogate import RHSurrogate, RHSurrogateLoss

from synthetic_lpap_pipeline import (
    gather_permuted_stream,
    harmonic_raw_batch,
    lpap_pool,
    max_padded_length,
    pad_for_lpap,
    surrogate_teacher_one_hot,
)

from rh_memory._python_ops import _get_permutation


class SurrogateSyntheticDataset(IterableDataset):
    def __init__(
        self,
        n: int,
        C: int,
        chunk_size: int,
        seed: int,
        device: torch.device | str,
        num_samples: int | None = None,
        fast_k: float = 5.0,
        harmonic_decay: float = 0.65,
        harmonic_amp_threshold: float = 0.1,
        max_harmonics: int = 64,
    ):
        super().__init__()
        self.n = n
        self.C = C
        self.chunk_size = chunk_size
        self.seed = seed
        self.device = device
        self.num_samples = num_samples
        self.fast_k = fast_k
        self.harmonic_decay = harmonic_decay
        self.harmonic_amp_threshold = harmonic_amp_threshold
        self.max_harmonics = max_harmonics

    def __iter__(self):
        n = self.n
        C = self.C
        chunk_size = self.chunk_size
        samples_generated = 0
        device = self.device

        k_eff = int(self.fast_k * math.log(C))

        while self.num_samples is None or samples_generated < self.num_samples:
            raw_inputs = harmonic_raw_batch(
                chunk_size,
                n,
                device,
                self.harmonic_decay,
                self.harmonic_amp_threshold,
                self.max_harmonics,
            )

            incoming_carry_id = torch.arange(n, dtype=torch.int32, device=device).unsqueeze(0).expand(
                chunk_size, n
            )

            raw_inputs_padded, incoming_carry_id_padded, n_pad = pad_for_lpap(raw_inputs, incoming_carry_id, C)

            table_values = torch.zeros(chunk_size, C, dtype=torch.float32, device=device)
            table_dib = torch.zeros(chunk_size, C, dtype=torch.int32, device=device)
            table_carry_id = torch.full((chunk_size, C), -1, dtype=torch.int32, device=device)

            _ov, _od, out_carry_id = lpap_pool(
                table_values,
                table_dib,
                table_carry_id,
                raw_inputs_padded,
                incoming_carry_id_padded,
                k_eff,
                self.seed,
                device,
            )

            perm_1d = _get_permutation(n_pad, self.seed, device)
            x_perm, carry_perm = gather_permuted_stream(raw_inputs_padded, incoming_carry_id_padded, perm_1d)

            position_tokens = x_perm.unsqueeze(-1)
            targets_bnC = surrogate_teacher_one_hot(out_carry_id, perm_1d)

            abs_amp = x_perm.abs()
            valid_j = carry_perm >= 0
            has_teacher = targets_bnC.sum(dim=2) > 0
            weights = abs_amp * valid_j.float() * has_teacher.float()

            batch = (
                position_tokens,
                targets_bnC,
                weights,
            )

            if self.num_samples is None:
                yield batch
                samples_generated += chunk_size
            else:
                remaining = self.num_samples - samples_generated
                if remaining <= 0:
                    break
                if remaining < chunk_size:
                    take = remaining
                    yield (
                        batch[0][:take],
                        batch[1][:take],
                        batch[2][:take],
                    )
                    samples_generated += take
                else:
                    yield batch
                    samples_generated += chunk_size


def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    torch.manual_seed(worker_seed)


def parse_args():
    p = argparse.ArgumentParser(description="Train RHSurrogate on synthetic harmonic + LPAP teacher.")
    p.add_argument("--n", type=int, default=1024)
    p.add_argument("--C", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--ff-mult", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--total-steps", type=int, default=15_000_000 // 128)
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
    seq_max = max_padded_length(n, C)
    print(f"Dataset meta — n={n}, C={C}, max_padded_length={seq_max}")

    B = args.batch_size
    train_dataset = SurrogateSyntheticDataset(
        n=n,
        C=C,
        chunk_size=B,
        seed=args.seed,
        device=device,
        num_samples=None,
        fast_k=args.fast_k,
    )
    test_num = min(1024, B * 8)
    test_dataset = SurrogateSyntheticDataset(
        n=n,
        C=C,
        chunk_size=B,
        seed=args.seed,
        device=device,
        num_samples=test_num,
        fast_k=args.fast_k,
    )

    workers = 0 if "cuda" in str(device) else 2
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
        "max_C": C,
        "seq_max": seq_max,
        "seed": args.seed,
        "fast_k": args.fast_k,
        "harmonic_decay": train_dataset.harmonic_decay,
        "harmonic_amp_threshold": train_dataset.harmonic_amp_threshold,
        "max_harmonics": train_dataset.max_harmonics,
        "d_model": args.d_model,
        "n_heads": args.n_heads,
        "num_layers": args.num_layers,
        "dim_feedforward": args.d_model * args.ff_mult,
    }

    model.train()
    running_loss = 0.0
    batches_since_eval = 0

    for step_idx, batch in enumerate(train_loader, 1):
        step = start_step + step_idx
        if step > args.total_steps:
            break

        position_tokens = batch[0].to(device)
        targets_bnC = batch[1].to(device)
        weights_bn = batch[2].to(device)

        optimizer.zero_grad()

        logits_full = model(position_tokens)
        logits = logits_full[:, :, :C]
        targets = targets_bnC[:, :, :C]

        loss = criterion(logits, targets, weights_bn)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        batches_since_eval += 1

        if step % args.eval_every == 0 or step == 1:
            avg_train = running_loss / batches_since_eval
            model.eval()
            test_loss = 0.0
            test_batches = 0
            with torch.no_grad():
                for tb in test_loader:
                    pt = tb[0].to(device)
                    tg = tb[1].to(device)
                    w = tb[2].to(device)
                    lf = model(pt)
                    lt = lf[:, :, :C]
                    tt = tg[:, :, :C]
                    test_loss += criterion(lt, tt, w).item()
                    test_batches += 1
            avg_test = test_loss / max(test_batches, 1)
            print(f"Step {step} | Train Loss: {avg_train:.6f} | Test Loss: {avg_test:.6f}")

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
            batches_since_eval = 0
            model.train()

    print("Training finished.")


if __name__ == "__main__":
    main()
