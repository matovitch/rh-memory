# RH-Memory Architecture: System Overview for AI Agents

## Core Constraints and Guarantees

- The RH-Memory architecture maps an input vector of sparse activations (size `[n]`) to a compressed pooled table (size `[C]`), where `C << n`.
- Memory uses a single tier: pure GPU-resident **linear-probing-based amplitude pooling**.
- This tier feeds a decoder backbone that maps the `[C]` table state back to the original `[n]` sparse logits.
- **Rule 1:** The GPU owns time progression counting.
- **Rule 2 (optional / future):** Time-decay can be applied outside the scatter–swap core (e.g. using **`gather`** on `carry_id` into per-source retention tables); the default pooling path does not implement decay in-package.
- **Rule 3:** Decisions use **strict** inequality on **absolute amplitude** \(|\cdot|\) in the pipeline and vs the table (incumbent survives ties). **DIB** tracks displacement for metadata / virtual offsets, not ordering among equal amplitudes.

For motivation (autoencoder framing, saliency priors, surrogate vs discrete teacher, forgetting as acting on the pooled table), see **[philosophy.md](philosophy.md)**.

## Key Lexicon & Variables

- `n`: Input sparse tensor dimension (integer scalar).
- `C`: Compressed bucket capacity dimension (integer scalar).
- `stride`: Defined as `n // C`. Number of probe bands along the pipeline grid.
- `dib` ($\text{DIB}$): Distance to Initial Bucket. Represents the collision-induced shift from the original hash bucket index.
- `carry_id`: Integral payload displaced with each slot (e.g. source index for supervision / `gather`); use a sentinel such as $-1$ for empty.
- `delta_steps` ($\Delta t$): The elapsed integer timestep count since the last update.

## Module Map

1. **[philosophy.md](philosophy.md)**: Project intent—saliency-shaped latents, surrogate vs discrete pooler, implicit alignment, curriculum on regularizers.
2. **[pooling.md](pooling.md)**: Tensor semantics for linear-probing-based amplitude pooling.
3. **[linear_probing_amplitude_pooling.md](linear_probing_amplitude_pooling.md)**: Triton / reference design (scatter-swap, virtual DIB).
4. **[decoder.md](decoder.md)**: Decoder input token structure, Transformer backbone, and loss objective.
5. **[gaussian_information_bottleneck.md](gaussian_information_bottleneck.md)**: Sketch of a GIB-style direction: flow-matched peaks through pooling + decoder, latent reconstruction, and MI-related objectives (with hurdles and mapping logits back toward Gaussian latents).
6. **[experiment_plan_cifar_flow_rh.md](experiment_plan_cifar_flow_rh.md)**: End-to-end plan: CIFAR grayscale, flow-matching encoder, surrogate + regularizer, optional soft relaxation, reconstruction losses, optional few-step reflow, staged training.
7. **[synthetic_harmonic_signals.md](synthetic_harmonic_signals.md)**: Pseudo-harmonic peak synthesis for decoder / future joint pretraining (`SyntheticRHDataset` in `experiments/train_decoder.py`); forward pointer to surrogate + LPAP regularizer.
8. **[immediate_todos.md](immediate_todos.md)**: Short-lived scratch queue only.

## Training Configuration

- **Objective Function:** `BCEWithLogitsLoss` weighted by per-bucket **absolute amplitude** (|[signed table value]| for active buckets).
- **Target:** Sparse support mask vector of size `[n]`.
- **Curriculum Learning Baseline:** Recommended Gumbel-Softmax relaxation on hard bucket assignments annealed linearly to discrete assignment over training steps, maintaining an `L1` sparsity penalty.
