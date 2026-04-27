# Pipeline Contracts Spec

Normative source:

- `src/rh_memory/pipeline/config.py`
- `src/rh_memory/pipeline/types.py`
- `src/rh_memory/pipeline/stage_harmonic.py`
- `src/rh_memory/pipeline/stage_surrogate.py`
- `src/rh_memory/pipeline/adapters.py`
- `src/rh_memory/decoder_scatter.py`

## Pipeline Config

`PipelineConfig` is the source of truth for pipeline dimensions and synthetic-data parameters:

- `N` (stored as `n` in config/API for compatibility), `C`, `batch_size`, `seed`, `fast_k`
- harmonic generator parameters
- derived: `stride = N // C`, `sequence_length = N`, `bucket_count = C`, `k_eff = max(1, int(fast_k * log(C)))`

## Stage Outputs

### `harmonic_stage(...) -> Iterator[HarmonicSample]`

Precomputes one grouped permutation per stream/run and reuses it for all generated batches.

`HarmonicSample` fields:

- `raw_inputs`: `[B, N]` (unpermuted)
- `perm_1d`: `[N]`
- `x_perm`: `[B, N]`

### `surrogate_stage(...) -> Iterator[SurrogateInferenceSample]`

Consumes `HarmonicSample`, runs a surrogate, and emits soft decoder-ready tokens.
The active bridge is differentiable and avoids discrete index selection.

`SurrogateInferenceSample` fields:

- `raw_inputs`: `[B, N]`
- `perm_1d`: `[N]`
- `x_perm`: `[B, N]`
- `surrogate_logits`: `[B, C, N]`
- `decoder_tokens`: `[B, C, 3]` with channels `[soft_amplitude, soft_normalized_dib, surrogate_doubt]`

Soft decoder-token construction:

1. `probs = softmax(surrogate_logits / temperature, dim=-1)`
2. `soft_amplitude = einsum("bcn,bn->bc", probs, x_perm)`
3. scalar DIB table: `((bucket_id - (slot_id % C)) % C) / max(1, C - 1)`
4. `soft_normalized_dib = einsum("bcn,cn->bc", probs, dib_table)`
5. `surrogate_doubt = normalized_entropy(probs)`

### Decoder soft-scatter reconstruction

Decoder L1 fine-tuning does not use a learned sequence reconstruction stage. It consumes `SurrogateInferenceSample`, runs `RHDecoder` on `decoder_tokens`, and reconstructs directly with `decoder_soft_scatter(...)`:

1. `decoder_logits = decoder(decoder_tokens)` with shape `[B, C, N]`
2. `probs = softmax(decoder_logits / temperature, dim=-1)`
3. `weighted_values = probs * decoder_tokens[..., 0].unsqueeze(-1)`
4. scatter-add `weighted_values` through `perm_1d` into unpermuted `[B, N]`

## Adapter Contracts

### `surrogate_training_adapter(...)`

Input stream: `HarmonicSample`  
Output tuple:

1. surrogate model input `bucket_input`: `[B, C, stride]`
2. surrogate target indices `target_idx`: `[B, C]` (LPAP teacher slot per bucket)
3. `valid_bucket`: `[B, C]` bool
4. surrogate weight `weights`: `[B, C]` (bucket-level amplitude weight, zeroed for invalid buckets)

No decoder-to-token reconstruction adapter is part of the current pipeline contract.
