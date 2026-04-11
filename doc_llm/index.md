# RH-Memory Architecture: System Overview for AI Agents

## Core Constraints and Guarantees

- The RH-Memory architecture maps an input vector of sparse activations (size `[n]`) to a compressed pooled memory table (size `[C]`), where `C << n`.
- Memory utilizes a single tier: pure GPU-resident, exact parallel Robin Hood (RH) pooling.
- This single Tier feeds a shared Decoder backbone that maps the `[C]` memory state back to the original `[n]` sparse logits.
- **Rule 1:** The GPU owns global time progression counting.
- **Rule 2:** Decay equations are exponential over time: $\text{state}_{new} = \text{state}_{old} \times \gamma^{\Delta t}$, where $\gamma$ is the retention factor (or gamma).
- **Rule 3:** Magnitude is the main proxy for "salience." Items with larger absolute magnitudes win bucket collisions, followed by Distance to Initial Bucket (DIB) as a tie-breaker (the poor-gets-richer property).

## Key Lexicon & Variables

- `n`: Input sparse tensor dimension (integer scalar).
- `C`: Compressed bucket capacity dimension (integer scalar).
- `stride`: Defined as `n // C`. Number of probe bands in approximate pooling.
- `dib` ($\text{DIB}$): Distance to Initial Bucket. Represents the collision-induced shift from the original hash bucket index.
- `gamma` ($\gamma$): Per-slot scalar retention factor. Bounded $\in [0, 1)$. Controls decay over steps.
- `alpha` ($\alpha$) / `epsilon` ($\epsilon$): Hyperparameters used to compute $\gamma$ calibration during pooling. (Implementation-specific formula).
- `delta_steps` ($\Delta t$): The elapsed integer timestep count since the last flush to CPU slow memory.

## Module Map

1. **[pooling.md](pooling.md)**: Exact Parallel RH Pooling tensor semantics.
2. **[exact_triton_pooling.md](exact_triton_pooling.md)**: Details on the parallel Robin Hood implementation via Triton scatter-swap/DIB.
3. **[decoder.md](decoder.md)**: Decoder input token structure, Transformer backbone, and loss objective.

## Training Configuration

- **Objective Function:** Magnitude-weighted `BCEWithLogitsLoss`.
- **Target:** Sparse support mask vector of size `[n]`.
- **Curriculum Learning Baseline:** Recommended Gumbel-Softmax relaxation on hard bucket assignments annealed linearly to discrete assignment over training steps, maintaining an `L1` sparsity penalty.
