# Decoder, surrogate, and token contract

**`RHDecoder`** maps $C$ bucket dimensions back to the original $n$ sequence axis; **`RHSurrogate`** is the transpose geometry over positions (see below). Synthetic training scripts are summarized at the end.

## Input Topology: Token Formats

The decoder backbone maps a tensor of size `[batch_size, C, token_features_dim]` over an $N$-block Transformer Encoder.

### Token Features (`token_features_dim = 2`)

For every bucket index $i \in \{0, \dots, C-1\}$, each input token is **two** scalar features:

1. **`signed_value`:** Signed amplitude sample from the pooler table (continuous).
2. **`dib` (`DIB`):** Distance to Initial Bucket (or equivalent tracking float) from the pooler table.

**Linear-probing-based amplitude pooling** exposes an integral **`carry_id`** per bucket (and per pipeline slot) for tracing and **building supervision targets** (e.g. source index $\mapsto$ bucket); it is **not** fed to the decoder — only **`signed_value`** and **`dib`** are.

### Surrogate backbone (separate from the decoder)

Implementation: **`RHSurrogate`** in **`src/rh_memory/surrogate.py`** shares the same RoPE encoder blocks as **`RHDecoder`** (`rope_bucket_transformer`), but the **sequence axis is length \(n\)**, not \(C\).

| Model | Role of axis | Input | Logits |
|-------|----------------|-------|--------|
| **`RHSurrogate`** | attend over **positions** | **`[batch_size, n, 1]`** (signed amplitude per index) | **`[batch_size, n, C]`** |
| **`RHDecoder`** | attend over **buckets** | **`[batch_size, C, 2]`** (signed value + DIB) | **`[batch_size, C, n]`** |

Same total logits **`B\cdot n\cdot C`**; geometries are **transposes**: surrogate answers “which bucket(s)” **per sequence index**, decoder answers “which index” **per bucket**. The surrogate does **not** take **DIB**—only **one amplitude channel** before **`Linear(1, D_\text{model})`**.

The **decoder**, when consuming **teacher** or **pooler-aligned** table state, may still use **`[signed_value, dib]`** when DIB is part of the story you want the decoder to see (see below on **where DIB comes from**).

### Stage-2 pipelined training (surrogate-aligned)

Script **`experiments/train_decoder_surrogate.py`** feeds **`RHDecoder`** bucket tokens **`[signed_value, dib]`** built **only** from the **frozen surrogate** (`surrogate_bucket_tokens_from_logits`: per-column **`argmax` over permuted slots \(j\)**). Supervision places a one-hot on **`j^\star(c)` from that same forward pass** (via **`decoder_targets_from_j_star`** in **`experiments/synthetic_lpap_pipeline.py`**). **LPAP is not run** in that dataset loop—the task is **invert the surrogate’s routing**, not recover pooler **`carry_id`**. Decoder training **from pooler** `out_values` / **`out_dib`** with LPAP-derived sparse targets requires a **separate** experiment script composing **`lpap_pool`** + **`RHDecoder`** (targets from **`carry_id`** on the pooler side only).

### Surrogate / experimental mode via `dib`

For tokens produced by a **surrogate** path or other experiments, **`dib` can take a negative sentinel value** (or another reserved scalar outside the physical DIB range) so the model can distinguish **real pooler metadata** from **synthetic / surrogate** tokens. When doing so:

- Document the sentinel(s) and keep them **consistent** across training.
- Avoid **normalizing** sentinel values into the same statistics as real DIB unless intentional.

### DIB, discrete routing, and gradients

For a **fixed permutation seed** and LPAP rules, **which source indices land in which buckets** is determined by the **input amplitudes** (the discrete scatter–swap trajectory). **DIB** at a bucket is **metadata about displacement along that trajectory**; if you define it **consistently from the routing outcome**, it is (given the seed) a **deterministic function of the realized assignment**—not an independent continuous knob.

If **decoder DIB inputs** are filled **a posteriori**—for example from **argmax positions of surrogate logits** or from **discrete pooler outputs**—then **gradients do not flow through those DIB values into earlier layers along that path** (argmax / discrete routing break differentiability). That is **often acceptable**: learning pressure remains on **amplitudes and logits** where competition actually happens; DIB then acts as **consistent contextual metadata** for the decoder rather than a quantity the network must differentiate through. If end-to-end sensitivity to small DIB perturbations is required, you must expose DIB through a **smooth** construction (e.g. expectations under soft assignments, or direct pooler DIB tensors in the forward pass).

### Interaction horizon and attention (optional)

LPAP is run for a **finite number of steps** \(k\) (often scaled with capacity, e.g. order \(k \propto \log C\) in experiments). Influence does **not** reduce to “each permuted index only talks to \(\pm k\) neighbors along a line”: after reshaping to **`stride`** probe rows \(\times\) **`C`** bucket columns, scatter–swap and rolling couple a slot with competitors **across strides** as well as along the bucket axis. A token’s effective interaction set is tied to **\(k\) rounds** of dynamics on that grid, **not** a trivial sliding window on a 1D reordering alone.

Because of that structure, an attention mask intended to mirror **exactly** which amplitudes can interact under LPAP (especially if inputs are **seed-permuted** to align with gather order) would be **more involved** than a plain band or local window—it must reflect stride-resolved coupling, not only index distance along the permuted sequence.

For **simplicity**, current models keep **full self-attention** (decoder over \(C\), surrogate over \(n\)). **Restricted or sparse attention** remains an optional future optimization once a mask definition is pinned to this geometry; it is **not** required by the pooler spec.

### Structural Constraints

- **No Head/Absolute Positional Embedding:** Bucket position indexing is entirely implicit.
- **Positional Handling (RoPE):** Rotary Position Embedding (RoPE) is applied solely over queries (`q`) and keys (`k`) inside attention layers across the `C` bucket dimension using standard `0` to `C-1` indices. The spatial geometry maps back cleanly to the sequence space via attention.
- **Causal Masking:** The Transformer Encoder operates bi-directionally over the `C` buckets (sequence axis). No autoregressive or causal mask is applied.
- **Transformer Encoder Baseline (decoder):** The **two** continuous scalar features map via a raw linear projection **`Linear(2, D_model)`**, flowing through $N$ feedforward parameterised sub-layers before un-pooling. **`RHSurrogate`** uses **one** feature per **sequence position** (`[B, n, 1]`) via **`Linear(1, D_model)`** over length \(n\).

## Output Target & Loss

- The dense head projects the backbone outputs linearly to logits representing the full original sequence: `[batch_size, C, D_model] -> [batch_size, C, n]`.
- **Reconstruction:** To recover the uncompressed `[batch_size, n]` sequence during evaluation or inference, the model determines the single most likely position for each bucket using a max reduction over the sequence dimension: `max_logits, predicted_indices = torch.max(logits, dim=2)`. These predictions are then projected back to the original full sequence via a `scatter` (or `scatter_reduce`) operation onto a neutral `[batch_size, n]` tensor.
- **Target Map:** The original non-pooled subset indices (sparse support mask of `1.0` vs `0.0`). The true target is a `[batch_size, C, n]` tensor where each bucket predicts a single original true index (and zero elsewhere). Note: The memory table is an invariant and is always considered filled, so unassigned slots do not occur.
- **Loss Formulation:** Use `BCEWithLogitsLoss`. The loss is applied strictly *per element* on the `[batch_size, C, n]` logits **before** any max reduction occurs. Crucially, the BCE loss for each element *must* be sample-weighted by per-bucket **absolute amplitude** (typically `|[signed_value]|` for active buckets). Gradient flow happens at the element level per bucket. Implementations: **`RHDecoderLoss`** in **`decoder.py`**, and **`RHSurrogateLoss`** in **`surrogate.py`** for logits **`[B, n, C]`** weighted by **`abs_amplitude`** of shape **`[B, n]`**.

## Training scripts (synthetic LPAP + surrogate pipeline)

Harmonic batches come from **`harmonic_raw_batch`** in **`experiments/synthetic_lpap_pipeline.py`**. Signal math: **[synthetic_harmonic_signals.md](synthetic_harmonic_signals.md)**.

| Stage | Script | Summary |
|-------|--------|---------|
| **1 — Surrogate** | **`experiments/train_surrogate.py`** | **`RHSurrogate`** trained with LPAP bucket targets on the **gather-permuted** stream (`RHSurrogateLoss`). Checkpoint **`meta`** stores **`C`**, **`seq_max`**, harmonics hyperparameters, and architecture fields for stage 2. |
| **2 — Decoder** | **`experiments/train_decoder_surrogate.py`** | Load frozen surrogate → build **`[B, C, 2]`** tokens via **`surrogate_bucket_tokens_from_logits`** → train **`RHDecoder`** with **`decoder_targets_from_j_star`** (**no LPAP** in this loop; invert surrogate routing only). |

**Pixi:** `pixi run train-surrogate`, `pixi run train-decoder-surrogate`.

Pooler-aligned decoder training (table **`out_values` / `out_dib`** + sparse targets from **`carry_id`**) is not bundled as a script today; compose **`lpap_pool`** + **`RHDecoder`** in a new experiment if needed.
