# Training Objectives Spec

Normative source:
- `experiments/train_surrogate.py`
- `experiments/train_decoder_surrogate.py`
- `experiments/train_reconstructor.py`
- `experiments/pipeline/adapters.py`
- `experiments/pipeline/primitives_targets.py`
- `experiments/pipeline/primitives_tokens.py`

Scope note:
- This page describes implemented training objectives only; roadmap regularizers and future memory dynamics belong in `doc_llm/notes/*`.

## Stage 1: Surrogate Objective

Model:
- `RHSurrogate`

Input:
- `surrogate_tokens` `[B, C, stride]`

Target:
- `targets_bCn` `[B, C, n]` from LPAP teacher (`out_slot_id` -> one-hot per bucket over permuted slot index `j`).

Loss:
- `RHSurrogateLoss(logits, targets_bCn, weights_bn)`
- weighted BCE-with-logits over `[B, C, n]` with per-bucket weights `[B, C]`.

Metric used in script:
- bucket-slot accuracy from `argmax(logits, dim=2)` vs `argmax(targets, dim=2)` on rows with teacher assignment.

## Stage 2: Decoder Objective (surrogate-aligned)

Model:
- `RHDecoder`

Input:
- `decoder_tokens_sur` `[B, C, 2]` from frozen surrogate outputs:
  - channel 0: selected `x_perm` value at `j_star`
  - channel 1: normalized dib-like quantity `((c - (j_star % C)) % C) / k_eff`

Target:
- `targets` `[B, C, n]`, one-hot at `j_star_sur` from frozen surrogate pass (`decoder_targets_from_j_star`).
- No LPAP call in decoder training loop.

Loss:
- `RHDecoderLoss(logits, targets, abs_amplitude)`
- weighted BCE-with-logits over `[B, C, n]` with `[B, C]` weights.

Metric used in script:
- top-1 index accuracy on active buckets (`valid_bucket & (abs_amplitude > 0.01)`).

## Stage 3: Reconstructor Objective

Model:
- `RHReconstructor`

Input:
- `reconstructor_tokens` `[B, C, 3]` from decoder logits:
  - selected value
  - normalized source index
  - selected decoder logit (confidence)

Target:
- `raw_inputs` `[B, n]` (unpermuted harmonic frame).

Loss:
- `RHReconstructorLoss` (`MSELoss`) over `[B, n]`.

Metric used in script:
- relative L2 percent.
