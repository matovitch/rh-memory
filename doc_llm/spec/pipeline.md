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

## Planned End-to-End LPAP Autoencoder Contract

This section records the next implementation target. It is a planned contract, not yet implemented by the current pipeline stages.

The LPAP autoencoder composes the pretrained components into a trainable image reconstruction path:

1. `image_seq`: Hilbert-flattened grayscale image sequence `[B, 1, N]`.
2. `learned_energy`: image-to-energy Euler integration output `[B, 1, N]`.
3. `energy_values`: squeezed learned energy `[B, N]`.
4. `energy_perm`: `energy_values` gathered by the active branch permutation `[B, N]`.
5. `surrogate_tokens`: `reshape_permuted_to_bucket_tokens(energy_perm, C)` with shape `[B, C, stride]`.
6. `surrogate_logits`: active `RHSurrogate` output `[B, C, N]`.
7. `decoder_tokens`: soft tokens from `decoder_tokens_from_surrogate_logits_soft(energy_perm, surrogate_logits)` with shape `[B, C, 3]`.
8. `decoder_logits`: active `RHDecoder` output `[B, C, N]`.
9. `projected_energy`: active `SoftScatterReconstructionHead` output `[B, N]`, then unsqueezed to `[B, 1, N]`.
10. `reconstructed_image_seq`: energy-to-image Euler integration output `[B, 1, N]`.

The image-to-energy flow produces unpooled learned energy. The surrogate/decoder/scatter branch pools it through the LPAP-like bottleneck. The energy-to-image flow starts from the pooled/projected energy.

### Bottleneck Branches And LoD

The first implementation may only have one available bottleneck branch, currently `C=128`. The contract should still leave room for multiple active branches:

- one surrogate per `C`, for example `C=64`, `C=128`, `C=256`
- one decoder/scatter path per `C`
- one active branch sampled or selected per forward pass

Changing `C` changes the approximate top-`C` LPAP winners that pass through the bottleneck. This is the intended level-of-detail axis and acts as structured dropout pressure on the learned energy geometry.
