# Decoder Backbone & Token Contract

The decoder structure maps the $C$ bucket dimensions back to the original $n$ sequence-length structure.

## Input Topology: Token Formats

The decoder backbone maps a tensor of size `[batch_size, C, token_features_dim]` over an $N$-block Transformer Encoder.

### Token Features (`token_features_dim = 3`)

For every bucket index $i \in \{0, \dots, C-1\}$, the input token must encapsulate strictly scalar embeddings extracted from pooling outputs:

1. `signed_value`: The explicitly signed continuous tracking feature (`sign` mapping $\times$ `magnitude`).
2. `gamma` ($\gamma$): The calculated retention factor.
3. `dib` ($\text{DIB}$): The Distance to Initial Bucket tracking float.

### Structural Constraints

- **No Head/Absolute Positional Embedding:** Bucket position indexing is entirely implicit.
- **Positional Handling (RoPE):** Rotary Position Embedding (RoPE) is applied solely over queries (`q`) and keys (`k`) inside attention layers across the `C` bucket dimension using standard `0` to `C-1` indices. The spatial geometry maps back cleanly to the sequence space via attention.
- **Causal Masking:** The Transformer Encoder operates bi-directionally over the `C` parallel buckets. No autoregressive or causal mask is applied.
- **Transformer Encoder Baseline:** The 3 continuous scalar features map via a simple raw linear projection `Linear(3, D_model)`, flowing through $N$ feedforward parameterised sub-layers before un-pooling.

## Output Target & Loss

- The dense head projects the backbone outputs linearly to logits representing the full original sequence: `[batch_size, C, D_model] -> [batch_size, C, n]`.
- **Reconstruction:** To recover the uncompressed `[batch_size, n]` sequence during evaluation or inference, the model determines the single most likely position for each bucket using a max reduction over the sequence dimension: `max_logits, predicted_indices = torch.max(logits, dim=2)`. These predictions are then projected back to the original full sequence via a `scatter` (or `scatter_reduce`) operation onto a neutral `[batch_size, n]` tensor.
- **Target Map:** The original non-pooled subset indices (sparse support mask of `1.0` vs `0.0`). The true target is a `[batch_size, C, n]` tensor where each bucket predicts exactly its one original true index (and zero elsewhere). Note: The memory table is an invariant and is always considered filled, so unassigned slots do not occur.
- **Loss Formulation:** Use `BCEWithLogitsLoss`. The loss is applied strictly *per element* on the `[batch_size, C, n]` logits **before** any max reduction occurs. Crucially, the BCE loss for each element *must* be sample-weighted by its original source **magnitude**. Gradient flow happens at the element level per bucket.
