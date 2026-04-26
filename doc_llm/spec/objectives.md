# Training Objectives Spec

Normative source:

- `experiments/train_surrogate.py`
- `experiments/train_decoder.py`
- `experiments/train_reconstructor.py`
- `src/rh_memory/pipeline/adapters.py`
- `src/rh_memory/pipeline/primitives_targets.py`
- `src/rh_memory/pipeline/primitives_tokens.py`

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

## Stage 3: Reconstructor Objective

Model:

- `RHReconstructor`

Input:

- `reconstructor_tokens` `[B, C, 3]` from decoder logits:
  - soft amplitude from probability-weighted `x_perm`
  - soft normalized unpermuted source index in `[0, 1]`
  - decoder doubt from normalized entropy

Target:

- `raw_inputs` `[B, N]` (current pipeline target in unpermuted sequence space).

Loss:

- `RHReconstructorLoss` (`MSELoss`) over `[B, N]`.

Training setup:

- `train_reconstructor.py` loads a frozen surrogate checkpoint and a frozen soft-distilled decoder checkpoint.
- Only `RHReconstructor` is optimized in this stage.

Metric used in script:

- relative L2 percent.

Soft-bridge note:

- The active bridge uses softmax-weighted expectations, not discrete index selection. Current staged scripts freeze surrogate during decoder distillation and freeze both surrogate and decoder during reconstructor training.
