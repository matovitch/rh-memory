"""Train RHSurrogate on harmonic synthetic data with LPAP bucket labels (permuted slot stream)."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.optim as optim
from path_utils import resolve_project_path

from rh_memory.surrogate import RHSurrogate, RHSurrogateLoss

from rh_memory.pipeline import (
    PipelineConfig,
    harmonic_stage,
    iter_take,
    surrogate_training_adapter,
)
from rh_memory.training_seed import apply_training_seed


def bucket_slot_accuracy_percent(
    logits: torch.Tensor,
    target_idx: torch.Tensor,
    valid_bucket: torch.Tensor,
) -> float:
    """
    Percent of valid buckets where predicted slot index j matches the sparse teacher target.
    """
    pred_slot = logits.argmax(dim=2)
    valid = valid_bucket.bool()
    denom = valid.sum().item()
    if denom == 0:
        return 0.0
    correct = ((pred_slot == target_idx.long()) & valid).sum().item()
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
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--fast-k", type=float, default=5.0)
    p.add_argument("--checkpoint", type=Path, default=Path("scripts/checkpoints/surrogate_ce_checkpoint.pt"))
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    args.checkpoint = resolve_project_path(args.checkpoint)

    ckpt = None
    if args.checkpoint.exists():
        print(f"Loading checkpoint from {args.checkpoint}...")
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    training_seed = apply_training_seed(args.seed, ckpt)
    args.seed = training_seed.seed
    print(f"Using seed: {training_seed.seed} ({training_seed.source})")

    config = PipelineConfig(
        n=args.n,
        C=args.C,
        batch_size=args.batch_size,
        seed=args.seed,
        fast_k=args.fast_k,
    )
    print(f"Dataset meta — n={config.n}, C={config.C}, sequence_length={config.sequence_length}")

    def make_train_stream():
        base = harmonic_stage(
            config=config,
            device=device,
        )
        return surrogate_training_adapter(base, config=config)

    test_num = min(1024, config.batch_size * 8)
    test_chunks = max(1, (test_num + config.batch_size - 1) // config.batch_size)

    def make_test_stream():
        base = harmonic_stage(
            config=config,
            device=device,
        )
        return iter_take(surrogate_training_adapter(base, config=config), test_chunks)

    model = RHSurrogate(
        sequence_length=config.sequence_length,
        bucket_count=config.C,
        stride=config.stride,
        fast_k=args.fast_k,
        d_model=args.d_model,
        n_heads=args.n_heads,
        num_layers=args.num_layers,
        dim_feedforward=args.d_model * args.ff_mult,
    ).to(device)

    criterion = RHSurrogateLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    start_step = 0
    if ckpt is not None:
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_step = ckpt.get("step", 0)
        print(f"Resumed from step {start_step}")

    model_config = {
        "d_model": args.d_model,
        "n_heads": args.n_heads,
        "num_layers": args.num_layers,
        "dim_feedforward": args.d_model * args.ff_mult,
    }

    model.train()
    running_loss = 0.0
    running_acc = 0.0
    batches_since_eval = 0

    train_stream = make_train_stream()

    for step_idx in range(1, args.total_steps - start_step + 1):
        batch = next(train_stream)
        step = start_step + step_idx

        position_tokens = batch[0].to(device)
        target_idx = batch[1].to(device)
        valid_bucket = batch[2].to(device)
        weights_bn = batch[3].to(device)

        optimizer.zero_grad()

        logits_full = model(position_tokens)
        logits = logits_full[:, : config.C, : config.n]

        loss = criterion(logits, target_idx, weights_bn, valid_bucket)
        loss.backward()
        optimizer.step()

        train_acc = bucket_slot_accuracy_percent(logits, target_idx, valid_bucket)
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
                for tb in make_test_stream():
                    pt = tb[0].to(device)
                    target_t = tb[1].to(device)
                    valid_t = tb[2].to(device)
                    w = tb[3].to(device)
                    lf = model(pt)
                    lt = lf[:, : config.C, : config.n]
                    test_loss += criterion(lt, target_t, w, valid_t).item()
                    test_acc += bucket_slot_accuracy_percent(lt, target_t, valid_t)
                    test_batches += 1
            avg_test = test_loss / max(test_batches, 1)
            avg_test_acc = test_acc / max(test_batches, 1)
            print(
                f"Step {step} | Train Loss: {avg_train:.6f} | Train Acc: {avg_train_acc:.2f}% | "
                f"Test Loss: {avg_test:.6f} | Test Acc: {avg_test_acc:.2f}%"
            )

            args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "version": "v2_ce_no_decoder",
                    "checkpoint_role": "surrogate",
                    "source_script": "scripts/train_surrogate.py",
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": config.to_dict(),
                    "model_config": model_config,
                    "seed": args.seed,
                    "seed_source": training_seed.source,
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
