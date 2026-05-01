# RH-Memory

RH-Memory is a research codebase for experimenting with LPAP-based memory compression, surrogate training, decoder distillation, soft-scatter reconstruction, and image/energy flow matching.

The current implemented stack is an autoencoder-shaped pipeline around an LPAP bottleneck. Synthetic harmonic signals provide an energy-space bootstrap, a surrogate learns LPAP teacher assignments, a decoder is distilled from the surrogate's soft slot distributions, and a soft-scatter head reconstructs unpermuted sequences from decoder logits. A separate symmetric flow script uses the frozen autoencoder stack to train image-to-energy and energy-to-image vector fields over Hilbert-flattened grayscale images. The next experimental phase is an end-to-end LPAP autoencoder that uses these components as initialization while allowing the learned energy geometry to move away from the harmonic bootstrap.

## Repository Layout

- `src/rh_memory/`: library code for LPAP operators, pipeline stages, models, Hilbert utilities, image shards, flow models, and shared training helpers.
- `scripts/`: runnable data-preparation, training, and evaluation entrypoints.
- `scripts/checkpoints/`: default checkpoint location used by the scripts.
- `tests/`: pytest coverage for operators, models, pipeline stages, image shards, flow utilities, and training seed behavior.
- `data/`: local dataset artifacts, including grayscale image shards when prepared.
- `doc_llm/spec/`: code-aligned implementation contracts.
- `doc_llm/notes/`: research notes, rationale, and roadmap material.

## Environment

This project uses Pixi. The workspace is configured in `pixi.toml`; Pixi also sets `PYTHONPATH=$PIXI_PROJECT_ROOT/src` for commands so scripts and tests import the local package without per-file `sys.path` changes.

```bash
pixi install
pixi run test
pixi run lint
pixi run format-check
```

The environment targets Linux with CUDA support:

- Python `>=3.14.4,<3.15`
- PyTorch GPU
- Triton
- jaxtyping
- pytest
- Ruff

## Main Commands

Pixi tasks are the preferred interface:

```bash
pixi run test
pixi run train-surrogate
pixi run train-decoder
pixi run train-decoder-scatter
pixi run prepare-grayscale-data
pixi run train-flow-symmetric
pixi run eval-flow-i2e
pixi run eval-flow-e2i
```

Pass script arguments after `--`:

```bash
pixi run train-surrogate -- --total-steps 1000 --eval-every 100
pixi run train-flow-symmetric -- --total-steps 40000 --batch-size 32
```

## Training Stages

The first stages train individual components against synthetic harmonic LPAP supervision and image/energy flow matching. The WIP end-to-end stage then reuses those checkpoints as initialization for joint image reconstruction through the LPAP bottleneck.

### 1. Surrogate

```bash
pixi run train-surrogate
```

Runs `scripts/train_surrogate.py`.

The surrogate consumes bucket tokens shaped `[B, C, stride]` and predicts logits `[B, C, N]`. Its teacher targets come from LPAP: one sparse permuted-slot target per valid bucket. The loss is weighted cross entropy over valid buckets.

Default checkpoint:

```text
scripts/checkpoints/surrogate_ce_checkpoint.pt
```

### 2. Decoder Distillation

```bash
pixi run train-decoder
```

Runs `scripts/train_decoder.py`.

The decoder consumes soft decoder tokens `[B, C, 3]` built from frozen surrogate logits. The token channels are soft amplitude, normalized displacement-in-bucket, and surrogate doubt. The decoder is trained by weighted soft KL distillation against frozen surrogate logits.

Default checkpoint:

```text
scripts/checkpoints/decoder_soft_distill_checkpoint.pt
```

### 3. Decoder Soft-Scatter L1 Fine-Tuning

```bash
pixi run train-decoder-scatter
```

Runs `scripts/train_decoder_soft_scatter.py`.

This stage loads the frozen surrogate and a pretrained distilled decoder, then fine-tunes the decoder with a learnable-temperature soft-scatter reconstruction head. The reconstruction target is the unpermuted raw harmonic input `[B, N]`, and the loss is L1.

Default checkpoint:

```text
scripts/checkpoints/decoder_soft_scatter_l1_checkpoint.pt
```

### 4. Symmetric Image/Energy Flow

```bash
pixi run train-flow-symmetric
```

Runs `scripts/train_flow_symmetric.py`.

This script trains two time-conditioned `Conv1d` flow models over sequences shaped `[B, 1, 1024]`:

- `image_to_energy_flow`: image sequence toward raw harmonic energy.
- `energy_to_image_flow`: projected energy toward image sequence.

Images are flattened through a `32x32` Hilbert curve before flow training. The energy side uses the frozen surrogate, decoder, and soft-scatter stack.

Default checkpoints:

```text
scripts/checkpoints/symmetric_flow_image_to_energy_checkpoint.pt
scripts/checkpoints/symmetric_flow_energy_to_image_checkpoint.pt
```

### 5. WIP: End-to-End LPAP Autoencoder

The planned end-to-end LPAP autoencoder reconstructs real grayscale images through the full path:

```text
image -> image_to_energy_flow -> LPAP surrogate/decoder/scatter branch -> energy_to_image_flow -> reconstructed image
```

This stage is intentionally different from the symmetric flow pretraining objective. The flow checkpoints initialize the two transport fields, but end-to-end training should let the learned energy geometry drift away from the naive harmonic space when that improves reconstruction through the LPAP-like bottleneck.

The first trainer uses:

- reconstruction MSE on the final image sequence
- LPAP surrogate regularization on the active surrogate logits
- no cycle loss
- no flow-matching side losses

Only the current `C=128` surrogate/decoder/scatter branch exists today. The longer-term LoD idea is to train or load several C-specific bottleneck branches, such as `C=64`, `C=128`, and `C=256`, and sample among them during training. Changing `C` changes the approximate top-C LPAP winners that survive the bottleneck, acting as structured level-of-detail dropout over the learned energy space.

The first script entrypoint is experimental and may change:

```bash
python scripts/train_lpap_autoencoder.py --eval-image-manifest path/to/eval/manifest.json
```

It requires a separate eval image manifest rather than silently evaluating on the training manifest.

## Image Data

The flow script expects a grayscale shard manifest by default:

```text
data/grayscale_32x32_torch/manifest.json
```

Use the preparation task to convert a tar archive of PNG images into grayscale tensor shards:

```bash
pixi run prepare-grayscale-data -- --help
```

The resulting dataset is backed by `GrayscaleImageShardDataset` or preloaded through `InMemoryGrayscaleImageShardDataset`.

## Seeding And Checkpoints

Training scripts default to OS-backed random seeds via `rh_memory.training_seed`. You can still request reproducibility by passing `--seed` explicitly.

On resume:

- If `--seed` is omitted, the script reuses the seed stored in the checkpoint.
- If `--seed` is provided, it overrides the checkpoint seed.
- Checkpoints record both `seed` and `seed_source`.

Current checkpoints restore model and optimizer state. They do not yet provide exact dataloader/RNG-stream continuation for every generated batch.

## Documentation

Start with the code-aligned spec pages:

- `doc_llm/spec/index.md`
- `doc_llm/spec/objectives.md`
- `doc_llm/spec/pipeline.md`
- `doc_llm/spec/models.md`
- `doc_llm/spec/pooling.md`

Research notes and future directions live under `doc_llm/notes/`. When notes conflict with the spec or source code, the spec and source code are authoritative.

## Development

Run the full verification loop before committing changes:

```bash
pixi run lint
pixi run format-check
pixi run typecheck
pixi run test
```

Install the Git hook once per clone to enforce the same Ruff checks on commit:

```bash
pixi run pre-commit-install
```

For narrower test runs, pass pytest arguments through the Pixi task:

```bash
pixi run test -- tests/test_pipeline_stages.py -q
```
