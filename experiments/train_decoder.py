"""Train RHDecoder by soft distillation from a frozen RHSurrogate."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.optim as optim

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from rh_memory.decoder import RHDecoder, RHDecoderDistillationLoss
from rh_memory.surrogate import RHSurrogate

from rh_memory.pipeline import (
    PipelineConfig,
    harmonic_stage,
    iter_take,
    surrogate_stage,
)
from rh_memory.pipeline.primitives_tokens import normalized_entropy


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


def expected_source_index_error(
    decoder_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    weights: torch.Tensor,
    temperature: float,
) -> float:
    n = decoder_logits.size(-1)
    slots = torch.arange(n, device=decoder_logits.device, dtype=decoder_logits.dtype)
    decoder_probs = torch.softmax(decoder_logits / temperature, dim=-1)
    teacher_probs = torch.softmax(teacher_logits / temperature, dim=-1)
    decoder_expected = torch.einsum("bcn,n->bc", decoder_probs, slots)
    teacher_expected = torch.einsum("bcn,n->bc", teacher_probs, slots)
    weights = weights.to(dtype=decoder_logits.dtype).clamp_min(0.0)
    denom = weights.sum().clamp_min(torch.finfo(decoder_logits.dtype).eps)
    err = (decoder_expected - teacher_expected).abs() / float(max(1, n - 1))
    return ((err * weights).sum() / denom).item()


def average_doubt(logits: torch.Tensor, temperature: float) -> float:
    probs = torch.softmax(logits / temperature, dim=-1)
    return normalized_entropy(probs).mean().item()


def parse_args():
    p = argparse.ArgumentParser(description="Train RHDecoder by soft distillation from a frozen RHSurrogate.")
    p.add_argument("--surrogate-checkpoint", type=Path, default=Path("experiments/checkpoints/surrogate_ce_checkpoint.pt"))
    p.add_argument("--n", type=int, default=None, help="Override sequence length (default: from surrogate meta)")
    p.add_argument("--C", type=int, default=None, help="Override bucket count (default: from surrogate meta)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--n-heads", type=int, default=16)
    p.add_argument("--num-layers", type=int, default=6)
    p.add_argument("--ff-mult", type=int, default=4)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--total-steps", type=int, default=50_000_000 // 128)
    p.add_argument("--eval-every", type=int, default=50_000 // 128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--checkpoint", type=Path, default=Path("experiments/checkpoints/decoder_soft_distill_checkpoint.pt"))
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    surrogate, surrogate_config = load_surrogate(args.surrogate_checkpoint, device)
    print(f"Loaded surrogate from {args.surrogate_checkpoint}")

    n = args.n if args.n is not None else surrogate_config.n
    C = args.C if args.C is not None else surrogate_config.C
    if n != surrogate_config.n or C != surrogate_config.C:
        raise ValueError(
            "Decoder n/C overrides must match the frozen surrogate checkpoint; "
            f"got n={n}, C={C}, checkpoint has n={surrogate_config.n}, C={surrogate_config.C}."
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

    def make_train_stream():
        base = harmonic_stage(config=config, device=device)
        return surrogate_stage(base, config=config, surrogate=surrogate, temperature=args.temperature)

    test_num = min(1024, config.batch_size * 8)
    test_chunks = max(1, (test_num + config.batch_size - 1) // config.batch_size)

    def make_test_stream():
        base = harmonic_stage(config=config, device=device)
        sur = surrogate_stage(base, config=config, surrogate=surrogate, temperature=args.temperature)
        return iter_take(sur, test_chunks)

    decoder = RHDecoder(
        sequence_length=config.n,
        bucket_count=config.C,
        d_model=args.d_model,
        n_heads=args.n_heads,
        num_layers=args.num_layers,
        dim_feedforward=args.d_model * args.ff_mult,
    ).to(device)

    criterion = RHDecoderDistillationLoss(temperature=args.temperature)
    optimizer = optim.AdamW(decoder.parameters(), lr=args.lr)

    start_step = 0
    if args.checkpoint.exists():
        print(f"Loading checkpoint from {args.checkpoint}...")
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        if ckpt.get("version") != "v1_soft_decoder_distill":
            raise ValueError(
                f"Unsupported decoder checkpoint version {ckpt.get('version')!r}; "
                "use a v1 soft decoder distillation checkpoint or choose a new --checkpoint path."
            )
        decoder.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_step = ckpt.get("step", 0)
        print(f"Resumed from step {start_step}")

    model_config = {
        "d_model": args.d_model,
        "n_heads": args.n_heads,
        "num_layers": args.num_layers,
        "dim_feedforward": args.d_model * args.ff_mult,
    }

    decoder.train()
    running_loss = 0.0
    running_expected_err = 0.0
    batches_since_eval = 0

    train_stream = make_train_stream()

    for step_idx in range(1, args.total_steps - start_step + 1):
        sample = next(train_stream)
        step = start_step + step_idx

        decoder_tokens = sample.decoder_tokens.to(device)
        teacher_logits = sample.surrogate_logits.to(device)
        weights = decoder_tokens[..., 0].abs()

        optimizer.zero_grad()
        decoder_logits = decoder(decoder_tokens)
        decoder_logits = decoder_logits[:, : config.C, : config.n]
        loss = criterion(decoder_logits, teacher_logits, weights)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_expected_err += expected_source_index_error(
            decoder_logits.detach(), teacher_logits.detach(), weights.detach(), args.temperature
        )
        batches_since_eval += 1

        if step % args.eval_every == 0 or step == 1:
            avg_train = running_loss / batches_since_eval
            avg_train_expected_err = running_expected_err / batches_since_eval

            decoder.eval()
            test_loss = 0.0
            test_expected_err = 0.0
            teacher_doubt = 0.0
            decoder_doubt = 0.0
            test_batches = 0
            with torch.no_grad():
                for sample_t in make_test_stream():
                    decoder_tokens_t = sample_t.decoder_tokens.to(device)
                    teacher_logits_t = sample_t.surrogate_logits.to(device)
                    weights_t = decoder_tokens_t[..., 0].abs()
                    decoder_logits_t = decoder(decoder_tokens_t)[:, : config.C, : config.n]
                    test_loss += criterion(decoder_logits_t, teacher_logits_t, weights_t).item()
                    test_expected_err += expected_source_index_error(
                        decoder_logits_t, teacher_logits_t, weights_t, args.temperature
                    )
                    teacher_doubt += average_doubt(teacher_logits_t, args.temperature)
                    decoder_doubt += average_doubt(decoder_logits_t, args.temperature)
                    test_batches += 1

            denom = max(test_batches, 1)
            avg_test = test_loss / denom
            avg_test_expected_err = test_expected_err / denom
            avg_teacher_doubt = teacher_doubt / denom
            avg_decoder_doubt = decoder_doubt / denom
            print(
                f"Step {step} | Train KL: {avg_train:.6f} | Train EIdxErr: {avg_train_expected_err:.4f} | "
                f"Test KL: {avg_test:.6f} | Test EIdxErr: {avg_test_expected_err:.4f} | "
                f"Teacher Doubt: {avg_teacher_doubt:.4f} | Decoder Doubt: {avg_decoder_doubt:.4f}"
            )

            args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "version": "v1_soft_decoder_distill",
                    "checkpoint_role": "decoder",
                    "source_script": "experiments/train_decoder.py",
                    "step": step,
                    "model_state_dict": decoder.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": config.to_dict(),
                    "model_config": model_config,
                    "surrogate_checkpoint": str(args.surrogate_checkpoint),
                    "temperature": args.temperature,
                    "seed": args.seed,
                },
                args.checkpoint,
            )
            print(f"Saved checkpoint to {args.checkpoint}")

            running_loss = 0.0
            running_expected_err = 0.0
            batches_since_eval = 0
            decoder.train()

    print("Training finished.")


if __name__ == "__main__":
    main()
