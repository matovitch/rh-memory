# Decoder Backbone & Token Contract

The decoder structure maps the $C$ bucket dimensions back to the original $n$ sequence-length structure.

## Input Topology: Token Formats

The decoder backbone maps a tensor of size `[batch_size, C, token_features_dim]` over an $N$-block Transformer Encoder.

### Token Features (`token_features_dim = 2`)

For every bucket index $i \in \{0, \dots, C-1\}$, each input token is **two** scalar features:

1. **`signed_value`:** Signed amplitude sample from the pooler table (continuous).
2. **`dib` (`DIB`):** Distance to Initial Bucket (or equivalent tracking float) from the pooler table.

**Linear-probing-based amplitude pooling** exposes an integral **`carry_id`** per bucket (and per pipeline slot) for tracing and **building supervision targets** (e.g. source index $\mapsto$ bucket); it is **not** fed to the decoder — only **`signed_value`** and **`dib`** are.

### Surrogate / experimental mode via `dib`

For tokens produced by a **surrogate** path or other experiments, **`dib` can take a negative sentinel value** (or another reserved scalar outside the physical DIB range) so the model can distinguish **real pooler metadata** from **synthetic / surrogate** tokens. When doing so:

- Document the sentinel(s) and keep them **consistent** across training.
- Avoid **normalizing** sentinel values into the same statistics as real DIB unless intentional.

### Structural Constraints

- **No Head/Absolute Positional Embedding:** Bucket position indexing is entirely implicit.
- **Positional Handling (RoPE):** Rotary Position Embedding (RoPE) is applied solely over queries (`q`) and keys (`k`) inside attention layers across the `C` bucket dimension using standard `0` to `C-1` indices. The spatial geometry maps back cleanly to the sequence space via attention.
- **Causal Masking:** The Transformer Encoder operates bi-directionally over the `C` buckets (sequence axis). No autoregressive or causal mask is applied.
- **Transformer Encoder Baseline:** The **two** continuous scalar features map via a raw linear projection **`Linear(2, D_model)`**, flowing through $N$ feedforward parameterised sub-layers before un-pooling.

## Output Target & Loss

- The dense head projects the backbone outputs linearly to logits representing the full original sequence: `[batch_size, C, D_model] -> [batch_size, C, n]`.
- **Reconstruction:** To recover the uncompressed `[batch_size, n]` sequence during evaluation or inference, the model determines the single most likely position for each bucket using a max reduction over the sequence dimension: `max_logits, predicted_indices = torch.max(logits, dim=2)`. These predictions are then projected back to the original full sequence via a `scatter` (or `scatter_reduce`) operation onto a neutral `[batch_size, n]` tensor.
- **Target Map:** The original non-pooled subset indices (sparse support mask of `1.0` vs `0.0`). The true target is a `[batch_size, C, n]` tensor where each bucket predicts a single original true index (and zero elsewhere). Note: The memory table is an invariant and is always considered filled, so unassigned slots do not occur.
- **Loss Formulation:** Use `BCEWithLogitsLoss`. The loss is applied strictly *per element* on the `[batch_size, C, n]` logits **before** any max reduction occurs. Crucially, the BCE loss for each element *must* be sample-weighted by per-bucket **absolute amplitude** (typically `|[signed_value]|` for active buckets). Gradient flow happens at the element level per bucket.

## Implementation note

`experiments/train_decoder.py` builds **`[B, C, 2]`** tokens (`out_values`, `out_dib`) from pooler outputs; **`carry_id`** remains on the pooler side for target construction only. For the **synthetic harmonic** input distribution used in that script, see **[synthetic_harmonic_signals.md](synthetic_harmonic_signals.md)**.
