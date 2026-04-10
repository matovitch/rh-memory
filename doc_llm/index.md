# RH-Memory Architecture: System Overview for AI Agents

## Core Constraints and Guarantees
- The RH-Memory architecture maps an input vector of sparse activations (size `[n]`) to a compressed pooled memory table (size `[C]`), where `C << n`.
- Memory is split into two tiers: `fast` (synchronous, GPU-resident, approximate) and `slow` (asynchronous, CPU-resident, exact).
- Both tiers feed a single shared Decoder backbone that maps the `[C]` memory state back to the original `[n]` sparse logits.
- **Rule 1:** The GPU owns global time progression counting.
- **Rule 2:** The CPU is stateless with respect to time. The CPU only updates its temporal state when sent a `delta_steps` ($\Delta t$) scalar by the GPU.
- **Rule 3:** Decay equations are exponential over time: $\text{state}_{new} = \text{state}_{old} \times \gamma^{\Delta t}$, where $\gamma$ is the retention factor (or gamma).
- **Rule 4:** Magnitude is the sole proxy for "salience." Items with larger absolute magnitudes win bucket collisions.

## Key Lexicon & Variables
- `n`: Input sparse tensor dimension (integer scalar).
- `C`: Compressed bucket capacity dimension (integer scalar).
- `stride`: Defined as `n // C`. Number of probe bands in approximate pooling.
- `dib` ($\text{DIB}$): Distance to Initial Bucket. Represents the collision-induced shift from the original hash bucket index.
- `gamma` ($\gamma$): Per-slot scalar retention factor. Bounded $\in [0, 1)$. Controls decay over steps.
- `alpha` ($\alpha$) / `epsilon` ($\epsilon$): Hyperparameters used to compute $\gamma$ calibration during pooling. (Implementation-specific formula).
- `delta_steps` ($\Delta t$): The elapsed integer timestep count since the last flush to CPU slow memory.

## Module Map
1. **[pooling.md](pooling.md)**: Exact and Approximate Pooling tensor semantics.
2. **[memory_protocol.md](memory_protocol.md)**: Fast/Slow memory update and synchronization contract.
3. **[decoder.md](decoder.md)**: Decoder input token structure, Transformer backbone, and loss objective.

## Training Configuration
- **Objective Function:** Magnitude-weighted `BCEWithLogitsLoss`.
- **Target:** Sparse support mask vector of size `[n]`.
- **Curriculum Learning Baseline:** Recommended Gumbel-Softmax relaxation on hard bucket assignments annealed linearly to discrete assignment over training steps, maintaining an `L1` sparsity penalty.