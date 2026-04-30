"""Fine-tune a pretrained RHDecoder through a differentiable soft-scatter L1 head."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim as optim
from path_utils import project_relative_path, resolve_project_path

from rh_memory.decoder import RHDecoder
from rh_memory.decoder_scatter import SoftScatterReconstructionHead
from rh_memory.surrogate import RHSurrogate

from rh_memory.pipeline import (
    PipelineConfig,
    harmonic_stage,
    iter_take,
    surrogate_stage,
)
from rh_memory.training_seed import apply_training_seed


def relative_l1_percent(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    err = (pred - target).abs().sum(dim=1)
    den = target.abs().sum(dim=1).clamp_min(eps)
    return ((err / den).mean().item()) * 100.0


def retained_l1_energy_ratio(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-12) -> float:
    pred_energy = pred.abs().sum(dim=1)
    target_energy = target.abs().sum(dim=1).clamp_min(eps)
    return (pred_energy / target_energy).mean().item()


def cosine_similarity(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    num = (pred * target).sum(dim=1)
    den = torch.linalg.vector_norm(pred, ord=2, dim=1) * torch.linalg.vector_norm(target, ord=2, dim=1)
    return (num / den.clamp_min(eps)).mean().item()


def load_surrogate(checkpoint_path: Path, device: torch.device):
    checkpoint_path = resolve_project_path(checkpoint_path)
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


def load_pretrained_decoder(checkpoint_path: Path, device: torch.device):
    checkpoint_path = resolve_project_path(checkpoint_path)
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
    return model, config, ckpt


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune RHDecoder by soft-scatter L1 reconstruction.")
    p.add_argument("--surrogate-checkpoint", type=Path, default=Path("scripts/checkpoints/surrogate_ce_checkpoint.pt"))
    p.add_argument("--decoder-checkpoint", type=Path, default=Path("scripts/checkpoints/decoder_soft_distill_checkpoint.pt"))
    p.add_argument("--n", type=int, default=None, help="Override sequence length (default: from surrogate meta)")
    p.add_argument("--C", type=int, default=None, help="Override bucket count (default: from surrogate meta)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--surrogate-temperature", type=float, default=1.0)
    p.add_argument("--init-scatter-temperature", type=float, default=1.0)
    p.add_argument("--min-scatter-temperature", type=float, default=0.05)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--total-steps", type=int, default=50_000_000 // 128)
    p.add_argument("--eval-every", type=int, default=50_000 // 128)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--checkpoint", type=Path, default=Path("scripts/checkpoints/decoder_soft_scatter_l1_checkpoint.pt"))
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    args.checkpoint = resolve_project_path(args.checkpoint)
    args.surrogate_checkpoint = resolve_project_path(args.surrogate_checkpoint)
    args.decoder_checkpoint = resolve_project_path(args.decoder_checkpoint)

    ckpt = None
    if args.checkpoint.exists():
        print(f"Loading checkpoint from {args.checkpoint}...")
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        if ckpt.get("version") != "v1_decoder_soft_scatter_l1":
            raise ValueError(
                f"Unsupported decoder soft-scatter checkpoint version {ckpt.get('version')!r}; "
                "choose a new --checkpoint path."
            )

    training_seed = apply_training_seed(args.seed, ckpt)
    args.seed = training_seed.seed
    print(f"Using seed: {training_seed.seed} ({training_seed.source})")

    surrogate, surrogate_config = load_surrogate(args.surrogate_checkpoint, device)
    print(f"Loaded surrogate from {args.surrogate_checkpoint}")
    decoder, decoder_config, decoder_ckpt = load_pretrained_decoder(args.decoder_checkpoint, device)
    print(f"Loaded pretrained decoder from {args.decoder_checkpoint}")

    n = args.n if args.n is not None else surrogate_config.n
    C = args.C if args.C is not None else surrogate_config.C
    if n != surrogate_config.n or C != surrogate_config.C:
        raise ValueError(
            "Fine-tuning n/C overrides must match the frozen surrogate checkpoint; "
            f"got n={n}, C={C}, checkpoint has n={surrogate_config.n}, C={surrogate_config.C}."
        )
    if n != decoder_config.n or C != decoder_config.C:
        raise ValueError(
            "Fine-tuning n/C overrides must match the pretrained decoder checkpoint; "
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

    scatter_head = SoftScatterReconstructionHead(
        init_temperature=args.init_scatter_temperature,
        min_temperature=args.min_scatter_temperature,
    ).to(device)

    optimizer = optim.AdamW(
        [
            {"params": decoder.parameters()},
            {"params": scatter_head.parameters()},
        ],
        lr=args.lr,
    )

    start_step = 0
    if ckpt is not None:
        decoder.load_state_dict(ckpt["model_state_dict"])
        scatter_head.load_state_dict(ckpt["scatter_head_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_step = ckpt.get("step", 0)
        print(f"Resumed from step {start_step}")

    def make_train_stream():
        base = harmonic_stage(config=config, device=device)
        return surrogate_stage(base, config=config, surrogate=surrogate, temperature=args.surrogate_temperature)

    test_num = min(1024, config.batch_size * 8)
    test_chunks = max(1, (test_num + config.batch_size - 1) // config.batch_size)

    def make_test_stream():
        base = harmonic_stage(config=config, device=device)
        sur = surrogate_stage(base, config=config, surrogate=surrogate, temperature=args.surrogate_temperature)
        return iter_take(sur, test_chunks)

    model_config = decoder_ckpt["model_config"]

    decoder.train()
    scatter_head.train()
    running_loss = 0.0
    running_rel_l1 = 0.0
    batches_since_eval = 0

    train_stream = make_train_stream()

    for step_idx in range(1, args.total_steps - start_step + 1):
        sample = next(train_stream)
        step = start_step + step_idx

        decoder_tokens = sample.decoder_tokens.to(device)
        target_series = sample.raw_inputs.to(device)
        perm_1d = sample.perm_1d.to(device)
        bucket_amplitude = decoder_tokens[..., 0]

        optimizer.zero_grad()
        decoder_logits = decoder(decoder_tokens)[:, : config.C, : config.n]
        pred, _probs, _doubt, _support, _temperature = scatter_head(decoder_logits, bucket_amplitude, perm_1d)
        loss = F.l1_loss(pred, target_series)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_rel_l1 += relative_l1_percent(pred.detach(), target_series.detach())
        batches_since_eval += 1

        if step % args.eval_every == 0 or step == 1:
            avg_train = running_loss / batches_since_eval
            avg_train_rel_l1 = running_rel_l1 / batches_since_eval

            decoder.eval()
            scatter_head.eval()
            test_loss = 0.0
            test_rel_l1 = 0.0
            test_energy = 0.0
            test_cos = 0.0
            test_doubt = 0.0
            test_support = 0.0
            test_batches = 0
            with torch.no_grad():
                for sample_t in make_test_stream():
                    decoder_tokens_t = sample_t.decoder_tokens.to(device)
                    target_t = sample_t.raw_inputs.to(device)
                    perm_1d_t = sample_t.perm_1d.to(device)
                    logits_t = decoder(decoder_tokens_t)[:, : config.C, : config.n]
                    pred_t, _probs_t, doubt_t, support_t, _temperature_t = scatter_head(
                        logits_t,
                        decoder_tokens_t[..., 0],
                        perm_1d_t,
                    )
                    tl = F.l1_loss(pred_t, target_t)
                    test_loss += tl.item()
                    test_rel_l1 += relative_l1_percent(pred_t, target_t)
                    test_energy += retained_l1_energy_ratio(pred_t, target_t)
                    test_cos += cosine_similarity(pred_t, target_t)
                    test_doubt += doubt_t.mean().item()
                    test_support += support_t.mean().item()
                    test_batches += 1

            denom = max(test_batches, 1)
            temperature_value = scatter_head.temperature().item()
            print(
                f"Step {step} | Train L1: {avg_train:.6f} | Train RelL1: {avg_train_rel_l1:.2f}% | "
                f"Test L1: {test_loss / denom:.6f} | Test RelL1: {test_rel_l1 / denom:.2f}% | "
                f"L1Energy: {test_energy / denom:.4f} | Cos: {test_cos / denom:.4f} | "
                f"Doubt: {test_doubt / denom:.4f} | EffSupport: {test_support / denom:.1f}/{config.n} | "
                f"ScatterTemp: {temperature_value:.4f}"
            )

            args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "version": "v1_decoder_soft_scatter_l1",
                    "checkpoint_role": "decoder_soft_scatter_l1",
                    "source_script": "scripts/train_decoder_soft_scatter.py",
                    "step": step,
                    "model_state_dict": decoder.state_dict(),
                    "scatter_head_state_dict": scatter_head.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": config.to_dict(),
                    "model_config": model_config,
                    "surrogate_checkpoint": project_relative_path(args.surrogate_checkpoint),
                    "pretrained_decoder_checkpoint": project_relative_path(args.decoder_checkpoint),
                    "pretrained_decoder_version": decoder_ckpt.get("version"),
                    "surrogate_temperature": args.surrogate_temperature,
                    "init_scatter_temperature": args.init_scatter_temperature,
                    "min_scatter_temperature": args.min_scatter_temperature,
                    "scatter_temperature": temperature_value,
                    "loss": "l1",
                    "lr": args.lr,
                    "seed": args.seed,
                    "seed_source": training_seed.source,
                },
                args.checkpoint,
            )
            print(f"Saved checkpoint to {args.checkpoint}")

            running_loss = 0.0
            running_rel_l1 = 0.0
            batches_since_eval = 0
            decoder.train()
            scatter_head.train()

    print("Training finished.")


if __name__ == "__main__":
    main()
