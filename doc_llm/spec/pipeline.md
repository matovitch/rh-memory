# Pipeline Contracts Spec

Normative source:

- `src/rh_memory/pipeline/config.py`
- `src/rh_memory/pipeline/types.py`
- `src/rh_memory/pipeline/stage_harmonic.py`
- `src/rh_memory/pipeline/stage_surrogate.py`
- `src/rh_memory/pipeline/stage_decoder.py`
- `src/rh_memory/pipeline/adapters.py`

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

### `decoder_stage(...) -> Iterator[DecoderInferenceSample]`

Consumes `SurrogateInferenceSample`, runs `RHDecoder`, and emits soft reconstructor-ready tokens.
This stage is used with a trainable decoder in `train_decoder.py` and with a frozen pretrained decoder in `train_reconstructor.py`.

`DecoderInferenceSample` fields:

- `raw_inputs`: `[B, N]`
- `reconstructor_tokens`: `[B, C, 3]` with channels `[soft_amplitude, soft_normalized_source_index, decoder_doubt]`

Soft reconstructor-token construction:

1. `probs = softmax(decoder_logits / temperature, dim=-1)`
2. `soft_amplitude = einsum("bcn,bn->bc", probs, x_perm)`
3. `soft_source_idx = einsum("bcn,n->bc", probs, perm_1d.float())`
4. `soft_normalized_source_index = soft_source_idx / max(1, N - 1)`
5. `decoder_doubt = normalized_entropy(probs)`

## Adapter Contracts

### `surrogate_training_adapter(...)`

Input stream: `HarmonicSample`  
Output tuple:

1. surrogate model input `bucket_input`: `[B, C, stride]`
2. surrogate target indices `target_idx`: `[B, C]` (LPAP teacher slot per bucket)
3. `valid_bucket`: `[B, C]` bool
4. surrogate weight `weights`: `[B, C]` (bucket-level amplitude weight, zeroed for invalid buckets)

### `reconstructor_training_adapter(...)`

Input stream: `DecoderInferenceSample`  
Output tuple:

1. reconstructor input tokens `[B, C, 3]`
2. MSE target `raw_inputs` `[B, N]`
