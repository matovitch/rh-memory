# Pipeline Contracts Spec

Normative source:
- `experiments/pipeline/types.py`
- `experiments/pipeline/stage_harmonic.py`
- `experiments/pipeline/stage_surrogate.py`
- `experiments/pipeline/stage_decoder.py`
- `experiments/pipeline/adapters.py`

## Stage Outputs

### `harmonic_stage(...) -> Iterator[HarmonicSample]`

`HarmonicSample` fields:
- `raw_inputs`: `[B, n]` (unpermuted)
- `perm_1d`: `[n]`
- `x_perm`: `[B, n]`
- `n`, `C`, `chunk_size`

### `surrogate_stage(...) -> Iterator[SurrogateSample]`

Consumes `HarmonicSample`, computes:
- `k_eff = max(1, int(fast_k * log(C)))`
- `surrogate_tokens`: `[B, C, stride]`
- frozen-surrogate logits internal to stage
- `decoder_tokens_sur`: `[B, C, 2]`
- `j_star_sur`: `[B, C]`

`SurrogateSample` fields:
- passthrough: `raw_inputs`, `perm_1d`, `x_perm`, `n`, `C`, `chunk_size`
- new: `k_eff`, `surrogate_tokens`, `decoder_tokens_sur`, `j_star_sur`

### `decoder_stage(...) -> Iterator[DecoderSample]`

Consumes `SurrogateSample`, computes decoder logits internally and emits:
- `reconstructor_tokens`: `[B, C, 3]`

`DecoderSample` fields:
- `raw_inputs`: `[B, n]`
- `reconstructor_tokens`: `[B, C, 3]`

## Adapter Contracts

### `surrogate_training_adapter(...)`

Input stream: `SurrogateSample`  
Output tuple:
1. surrogate model input `bucket_input`: `[B, C, stride]`
2. surrogate target `targets_bCn`: `[B, C, n]` (LPAP teacher slot one-hot per bucket)
3. surrogate weight `weights`: `[B, C]` (bucket-level amplitude weight)

### `decoder_training_adapter(...)`

Input stream: `SurrogateSample`  
Output tuple:
1. decoder input `decoder_tokens`: `[B, C, 2]`
2. decoder target `targets`: `[B, C, n]` (one-hot at surrogate `j_star`)
3. decoder weight `abs_amplitude`: `[B, C]`
4. `j_target`: `[B, C]` (for metric)
5. `valid_bucket`: `[B, C]` bool

### `reconstructor_training_adapter(...)`

Input stream: `DecoderSample`  
Output tuple:
1. reconstructor input tokens `[B, C, 3]`
2. MSE target `raw_inputs` `[B, n]`
