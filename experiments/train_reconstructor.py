"""Train RHReconstructor from a frozen surrogate and frozen soft-distilled decoder."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.optim as optim

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from rh_memory.decoder import RHDecoder
from rh_memory.reconstructor import RHReconstructor, RHReconstructorLoss
from rh_memory.surrogate import RHSurrogate

from rh_memory.pipeline import (
    PipelineConfig,
    decoder_stage,
    harmonic_stage,
    iter_take,
    reconstructor_training_adapter,
    surrogate_stage,
)


def relative_l2_percent(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    err = torch.linalg.vector_norm(pred - target, ord=2, dim=1)
    den = torch.linalg.vector_norm(target, ord=2, dim=1).clamp_min(eps)
    return ((err / den).mean().item()) * 100.0


def load_surrogate(checkpoint_path: Path, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if ckpt.get("version") != "v2_ce_no_decoder":
        raise ValueError(
            f"Unsupported surrogate checkpoint version {ckpt.get('version')!r}; "
            "retrain the surrogate with the CE/no-decoder pipeline."
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
    for p in model.parameters():
        p.requires_grad_(False)
    return model, config


def load_decoder(checkpoint_path: Path, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if ckpt.get("version") != "v1_soft_decoder_distill":
        raise ValueError(
            f"Unsupported decoder checkpoint version {ckpt.get('version')!r}; "
            "train the decoder with the soft distillation pipeline."
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
    for p in model.parameters():
        p.requires_grad_(False)
    return model, config, ckpt


def parse_args():
    p = argparse.ArgumentParser(description="Train RHReconstructor from frozen soft surrogate/decoder tokens.")
    p.add_argument("--surrogate-checkpoint", type=Path, default=Path("experiments/surrogate_ce_checkpoint.pt"))
    p.add_argument("--decoder-checkpoint", type=Path, default=Path("experiments/decoder_soft_distill_checkpoint.pt"))
    p.add_argument("--n", type=int, default=None, help="Override sequence length (default: from surrogate meta)")
    p.add_argument("--C", type=int, default=None, help="Override bucket count (default: from surrogate meta)")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--num-token-layers", type=int, default=4)
    p.add_argument("--num-query-layers", type=int, default=4)
    p.add_argument("--ff-mult", type=int, default=4)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--reconstructor-token-mode", choices=("hard", "soft"), default="hard")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--total-steps", type=int, default=50_000_000 // 128)
    p.add_argument("--eval-every", type=int, default=50_000 // 128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--checkpoint", type=Path, default=Path("experiments/reconstructor_frozen_decoder_checkpoint.pt"))
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    surrogate, surrogate_config = load_surrogate(args.surrogate_checkpoint, device)
    print(f"Loaded surrogate from {args.surrogate_checkpoint}")
    decoder, decoder_config, decoder_ckpt = load_decoder(args.decoder_checkpoint, device)
    print(f"Loaded decoder from {args.decoder_checkpoint}")

    n = args.n if args.n is not None else surrogate_config.n
    C = args.C if args.C is not None else surrogate_config.C
    if n != surrogate_config.n or C != surrogate_config.C:
        raise ValueError(
            "Reconstructor n/C overrides must match the frozen surrogate checkpoint; "
            f"got n={n}, C={C}, checkpoint has n={surrogate_config.n}, C={surrogate_config.C}."
        )
    if n != decoder_config.n or C != decoder_config.C:
        raise ValueError(
            "Reconstructor n/C overrides must match the frozen decoder checkpoint; "
            f"got n={n}, C={C}, checkpoint has n={decoder_config.n}, C={decoder_config.C}."
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
    print(f"n={config.n}, C={config.C}")
    print(f"reconstructor_token_mode={args.reconstructor_token_mode}")

    def make_train_stream():
        base = harmonic_stage(
            config=config,
            device=device,
        )
        sur = surrogate_stage(base, config=config, surrogate=surrogate, temperature=args.temperature)
        dec = decoder_stage(
            sur,
            decoder=decoder,
            temperature=args.temperature,
            token_mode=args.reconstructor_token_mode,
        )
        return reconstructor_training_adapter(dec)

    test_num = min(1024, config.batch_size * 8)
    test_chunks = max(1, (test_num + config.batch_size - 1) // config.batch_size)

    def make_test_stream():
        base = harmonic_stage(
            config=config,
            device=device,
        )
        sur = surrogate_stage(base, config=config, surrogate=surrogate, temperature=args.temperature)
        dec = decoder_stage(
            sur,
            decoder=decoder,
            temperature=args.temperature,
            token_mode=args.reconstructor_token_mode,
        )
        return iter_take(reconstructor_training_adapter(dec), test_chunks)

    model = RHReconstructor(
        sequence_length=config.n,
        bucket_count=config.C,
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
        if ckpt.get("version") != "v4_frozen_soft_decoder_reconstructor":
            raise ValueError(
                f"Unsupported reconstructor checkpoint version {ckpt.get('version')!r}; "
                "use a v4 frozen soft decoder/reconstructor checkpoint or choose a new --checkpoint path."
            )
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_step = ckpt.get("step", 0)
        print(f"Resumed from step {start_step}")

    decoder.eval()
    model.train()
    running_loss = 0.0
    running_rel_l2 = 0.0
    batches_since_eval = 0

    train_stream = make_train_stream()

    for step_idx in range(1, args.total_steps - start_step + 1):
        batch = next(train_stream)
        step = start_step + step_idx

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

            decoder.eval()
            model.eval()
            test_loss = 0.0
            test_rel_l2 = 0.0
            test_batches = 0
            with torch.no_grad():
                for tb in make_test_stream():
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
                    "version": "v4_frozen_soft_decoder_reconstructor",
                    "checkpoint_role": "reconstructor",
                    "source_script": "experiments/train_reconstructor.py",
                    "meta": {
                        "config": config.to_dict(),
                        "seed": args.seed,
                        "temperature": args.temperature,
                        "reconstructor_token_mode": args.reconstructor_token_mode,
                        "surrogate_checkpoint": str(args.surrogate_checkpoint),
                        "decoder_checkpoint": str(args.decoder_checkpoint),
                        "decoder_version": decoder_ckpt.get("version"),
                        "decoder_model_config": decoder_ckpt.get("model_config"),
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
            decoder.eval()
            model.train()

    print("Training finished.")


if __name__ == "__main__":
    main()
