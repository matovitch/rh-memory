# Training Objectives Spec

Normative source:

- `experiments/train_surrogate.py`
- `experiments/train_decoder.py`
- `experiments/train_decoder_soft_scatter.py`
- `src/rh_memory/pipeline/adapters.py`
- `src/rh_memory/pipeline/primitives_targets.py`
- `src/rh_memory/pipeline/primitives_tokens.py`
- `src/rh_memory/decoder_scatter.py`

Scope note:

- This page describes implemented training objectives only; roadmap regularizers and future memory dynamics belong in `doc_llm/notes/*`.

## Stage 1: Surrogate Objective

Model:

- `RHSurrogate`

Input:

- `surrogate_tokens` `[B, C, stride]`

Target:

- `target_idx` `[B, C]` from LPAP teacher (`out_slot_id` -> permuted slot index `j` per bucket).
- `valid_bucket` `[B, C]` marks buckets with valid LPAP teacher assignments.

Loss:

- `RHSurrogateLoss(logits, target_idx, weights_bn, valid_bucket)`
- weighted cross entropy over logits `[B, C, N]` with sparse targets `[B, C]`.

Metric used in script:

- bucket-slot accuracy from `argmax(logits, dim=2)` vs `target_idx` on valid buckets.

## Stage 2: Decoder Distillation Objective

Model:

- `RHDecoder`

Input:

- `decoder_tokens` `[B, C, 3]` from surrogate logits:
  - soft amplitude from probability-weighted `x_perm`
  - soft scalar DIB in `[0, 1]`
  - surrogate doubt from normalized entropy

Output:

- decoder logits `[B, C, N]`.

Teacher:

- frozen surrogate logits `[B, C, N]` from the same `SurrogateInferenceSample`.

Loss:

- `RHDecoderDistillationLoss(decoder_logits, surrogate_logits, weights)`.
- teacher probabilities: `softmax(surrogate_logits / temperature)`.
- student log-probabilities: `log_softmax(decoder_logits / temperature)`.
- weighted soft KL over `[B, C, N]` distributions.
- bucket weights are `abs(soft_amplitude)` from decoder tokens.

Scope:

- LPAP targets do not enter decoder training.
- No discrete target selection is used.

## Stage 3: Decoder Soft-Scatter L2 Objective

Model:

- pretrained `RHDecoder`
- `SoftScatterReconstructionHead`

Input:

- `decoder_tokens` `[B, C, 3]` from frozen surrogate logits.
- `perm_1d` `[N]` from the harmonic stream.

Output:

- decoder logits `[B, C, N]` from `RHDecoder`.
- soft-scattered reconstruction `[B, N]`:
  - `probs = softmax(decoder_logits / scatter_temperature, dim=-1)`
  - scatter-add `decoder_tokens[..., 0] * probs` through `perm_1d` into unpermuted source coordinates.

Target:

- `raw_inputs` `[B, N]` in unpermuted sequence space.

Loss:

- MSE over `[B, N]`.

Training setup:

- `train_decoder_soft_scatter.py` loads a frozen surrogate checkpoint and a pretrained soft-distilled decoder checkpoint.
- The surrogate is frozen.
- The decoder and single learnable scatter temperature are optimized.
- No distillation/KL regularizer is used in the first implementation.

Metric used in script:

- relative L2 percent, retained energy, cosine, decoder doubt, and effective support.

Soft-bridge note:

- LPAP targets do not enter decoder L2 fine-tuning. The active reconstruction bridge uses full softmax scatter over all decoder slots, not expected-coordinate reduction and not hard argmax.
