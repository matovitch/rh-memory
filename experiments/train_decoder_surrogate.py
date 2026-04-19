"""Train RHDecoder on surrogate-derived bucket tokens with surrogate-aligned permuted-slot supervision."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, IterableDataset

_SUR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SUR))
sys.path.insert(0, str(_SUR.parent / "src"))

from rh_memory.decoder import RHDecoder, RHDecoderLoss
from rh_memory.surrogate import RHSurrogate

from synthetic_lpap_pipeline import (
    decoder_targets_from_j_star,
    gather_permuted_stream,
    harmonic_raw_batch,
    max_padded_length,
    pad_for_lpap,
    surrogate_bucket_tokens_from_logits,
)

from rh_memory._python_ops import _get_permutation


class DecoderSurrogateSyntheticDataset(IterableDataset):
    """
    Decoder inputs are surrogate-derived bucket rows; supervision is ``j_star`` from the same forward pass
    (invert surrogate column routing). LPAP is not run in this loop.
    """

    def __init__(
        self,
        n: int,
        C: int,
        chunk_size: int,
        seed: int,
        device: torch.device | str,
        surrogate: RHSurrogate,
        sequence_length: int,
        num_samples: int | None = None,
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
        self.surrogate = surrogate
        self.sequence_length = sequence_length
        self.num_samples = num_samples
        self.harmonic_decay = harmonic_decay
        self.harmonic_amp_threshold = harmonic_amp_threshold
        self.max_harmonics = max_harmonics

    def __iter__(self):
        chunk_size = self.chunk_size
        samples_generated = 0
        device = self.device
        n = self.n
        surrogate = self.surrogate
        seq_len = self.sequence_length

        C = self.C

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

            perm_1d = _get_permutation(n_pad, self.seed, device)
            x_perm, _carry_perm = gather_permuted_stream(raw_inputs_padded, incoming_carry_id_padded, perm_1d)

            position_tokens = x_perm.unsqueeze(-1)

            with torch.no_grad():
                logits_full = surrogate(position_tokens)

            logits_c = logits_full[:, :, :C]
            bucket_tokens, j_star = surrogate_bucket_tokens_from_logits(x_perm, logits_c, C)

            valid_bucket = torch.ones(chunk_size, C, dtype=torch.bool, device=device)

            targets = decoder_targets_from_j_star(j_star, seq_len, valid_bucket)

            abs_amplitude = bucket_tokens[..., 0].abs() * valid_bucket.float()

            j_target = j_star.long().clamp(0, seq_len - 1)

            batch = (
                bucket_tokens,
                targets,
                abs_amplitude,
                j_target,
                valid_bucket,
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

                    def slice_lead(t):
                        if torch.is_tensor(t) and t.dim() >= 1 and t.shape[0] == chunk_size:
                            return t[:take]
                        return t

                    batch = tuple(slice_lead(x) for x in batch)
                    yield batch
                    samples_generated += take
                else:
                    yield batch
                    samples_generated += chunk_size


def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    torch.manual_seed(worker_seed)


def bucket_count_from_surrogate_meta(meta: dict) -> int:
    """``C`` / ``max_C`` in new checkpoints; fall back to legacy ``C_options`` list."""
    if "C" in meta:
        return int(meta["C"])
    if "max_C" in meta:
        return int(meta["max_C"])
    if "C_options" in meta:
        return max(int(x) for x in meta["C_options"])
    raise KeyError("surrogate meta must include C, max_C, or C_options")


def load_surrogate(checkpoint_path: Path, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    meta = ckpt["meta"]
    max_C = bucket_count_from_surrogate_meta(meta)
    seq_max = meta.get("seq_max", meta["n"])
    model = RHSurrogate(
        sequence_length=seq_max,
        bucket_count=max_C,
        d_model=meta["d_model"],
        n_heads=meta["n_heads"],
        num_layers=meta["num_layers"],
        dim_feedforward=meta["dim_feedforward"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, meta


def parse_args():
    p = argparse.ArgumentParser(
        description="Train RHDecoder on surrogate bucket features; labels = j_star (same pass, no LPAP in loop)."
    )
    p.add_argument("--surrogate-checkpoint", type=Path, default=Path("experiments/surrogate_checkpoint.pt"))
    p.add_argument("--n", type=int, default=None, help="Override sequence length (default: from surrogate meta)")
    p.add_argument("--C", type=int, default=None, help="Override bucket count (default: from surrogate meta)")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--num-layers", type=int, default=6)
    p.add_argument("--ff-mult", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--total-steps", type=int, default=15_000_000 // 128)
    p.add_argument("--eval-every", type=int, default=50_000 // 128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--checkpoint", type=Path, default=Path("experiments/decoder_surrogate_checkpoint.pt"))
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    surrogate, meta = load_surrogate(args.surrogate_checkpoint, device)
    print(f"Loaded surrogate from {args.surrogate_checkpoint}")

    n = args.n if args.n is not None else meta["n"]
    C = args.C if args.C is not None else bucket_count_from_surrogate_meta(meta)
    seq_len = max_padded_length(n, C)

    print(f"n={n}, C={C}, decoder sequence_length={seq_len}")

    B = args.batch_size
    train_dataset = DecoderSurrogateSyntheticDataset(
        n=n,
        C=C,
        chunk_size=B,
        seed=meta["seed"],
        device=device,
        surrogate=surrogate,
        sequence_length=seq_len,
        num_samples=None,
        harmonic_decay=meta["harmonic_decay"],
        harmonic_amp_threshold=meta["harmonic_amp_threshold"],
        max_harmonics=meta["max_harmonics"],
    )
    test_num = min(1024, B * 8)
    test_dataset = DecoderSurrogateSyntheticDataset(
        n=n,
        C=C,
        chunk_size=B,
        seed=meta["seed"],
        device=device,
        surrogate=surrogate,
        sequence_length=seq_len,
        num_samples=test_num,
        harmonic_decay=meta["harmonic_decay"],
        harmonic_amp_threshold=meta["harmonic_amp_threshold"],
        max_harmonics=meta["max_harmonics"],
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

    decoder = RHDecoder(
        sequence_length=seq_len,
        bucket_count=C,
        d_model=args.d_model,
        n_heads=args.n_heads,
        num_layers=args.num_layers,
        dim_feedforward=args.d_model * args.ff_mult,
    ).to(device)

    criterion = RHDecoderLoss()
    optimizer = optim.AdamW(decoder.parameters(), lr=args.lr)

    start_step = 0
    if args.checkpoint.exists():
        print(f"Loading decoder checkpoint from {args.checkpoint}...")
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        decoder.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_step = ckpt.get("step", 0)
        print(f"Resumed from step {start_step}")

    decoder.train()
    running_loss = 0.0
    running_accuracy = 0.0
    batches_since_eval = 0

    for step_idx, batch in enumerate(train_loader, 1):
        step = start_step + step_idx
        if step > args.total_steps:
            break

        batch_tokens = batch[0].to(device)
        batch_targets = batch[1].to(device)
        batch_abs_amplitude = batch[2].to(device)
        j_target = batch[3].to(device)
        valid_bucket = batch[4].to(device)

        optimizer.zero_grad()

        logits = decoder(batch_tokens)
        loss = criterion(logits, batch_targets, batch_abs_amplitude)

        loss.backward()
        optimizer.step()

        _, predicted_indices = torch.max(logits, dim=2)
        active_mask = valid_bucket & (batch_abs_amplitude > 0.01)
        correct = (predicted_indices == j_target) & active_mask

        total_active = active_mask.sum().item()
        accuracy = (correct.sum().item() / total_active * 100) if total_active > 0 else 0.0

        running_loss += loss.item()
        running_accuracy += accuracy
        batches_since_eval += 1

        if step % args.eval_every == 0 or step == 1:
            avg_loss = running_loss / batches_since_eval
            avg_acc = running_accuracy / batches_since_eval

            decoder.eval()
            test_loss = 0.0
            test_acc = 0.0
            test_batches = 0
            with torch.no_grad():
                for tb in test_loader:
                    b_toks = tb[0].to(device)
                    b_targs = tb[1].to(device)
                    b_abs = tb[2].to(device)
                    j_tgt = tb[3].to(device)
                    vb = tb[4].to(device)

                    logits_t = decoder(b_toks)
                    tl = criterion(logits_t, b_targs, b_abs)

                    _, pred_idx = torch.max(logits_t, dim=2)
                    am = vb & (b_abs > 0.01)
                    corr = (pred_idx == j_tgt) & am
                    tot_a = am.sum().item()
                    acc_t = (corr.sum().item() / tot_a * 100) if tot_a > 0 else 0.0

                    test_loss += tl.item()
                    test_acc += acc_t
                    test_batches += 1

            avg_test_loss = test_loss / max(test_batches, 1)
            avg_test_acc = test_acc / max(test_batches, 1)

            print(
                f"Step {step} | Train Loss: {avg_loss:.6f} | Train Acc: {avg_acc:.2f}% | "
                f"Test Loss: {avg_test_loss:.6f} | Test Acc: {avg_test_acc:.2f}%"
            )

            torch.save(
                {
                    "step": step,
                    "model_state_dict": decoder.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "surrogate_checkpoint": str(args.surrogate_checkpoint),
                    "meta": meta,
                },
                args.checkpoint,
            )
            print(f"Saved checkpoint to {args.checkpoint}")

            running_loss = 0.0
            running_accuracy = 0.0
            batches_since_eval = 0
            decoder.train()

    print("Training finished.")


if __name__ == "__main__":
    main()
