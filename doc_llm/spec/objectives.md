# Training Objectives Spec

Normative source:

- `scripts/train_surrogate.py`
- `scripts/train_decoder.py`
- `scripts/train_decoder_soft_scatter.py`
- `src/rh_memory/pipeline/adapters.py`
- `src/rh_memory/pipeline/primitives_targets.py`
- `src/rh_memory/pipeline/primitives_tokens.py`
- `src/rh_memory/decoder_scatter.py`

Scope note:

- Stage 1 through Stage 3 describe implemented training objectives.
- The end-to-end LPAP autoencoder section describes the next planned objective and should be updated when implementation lands.

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

## Stage 3: Decoder Soft-Scatter L1 Objective

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

- Mean absolute error over `[B, N]`.
- The relative reconstruction metric uses L1 magnitude: `sum(abs(pred - target)) / sum(abs(target))`.
- The retained-energy diagnostic uses L1 magnitude mass: `sum(abs(pred)) / sum(abs(target))`.

Training setup:

- `train_decoder_soft_scatter.py` loads a frozen surrogate checkpoint and a pretrained soft-distilled decoder checkpoint.
- The surrogate is frozen.
- The decoder and single learnable scatter temperature are optimized.
- No distillation/KL regularizer is used in the first implementation.

Metric used in script:

- relative L1 percent, retained L1 energy, cosine, decoder doubt, and effective support.

Soft-bridge note:

- LPAP targets do not enter decoder L1 fine-tuning. The active reconstruction bridge uses full softmax scatter over all decoder slots, not expected-coordinate reduction and not hard argmax.

## Planned Stage 4: End-to-End LPAP Autoencoder Objective

Model:

- image-to-energy flow
- active `RHSurrogate`
- active `RHDecoder`
- active `SoftScatterReconstructionHead`
- energy-to-image flow

Input:

- real grayscale image shards, Hilbert-flattened to `[B, 1, N]`.

Path:

- image sequence -> image-to-energy Euler integration -> learned unpooled energy -> surrogate -> decoder -> soft scatter -> pooled/projected energy -> energy-to-image Euler integration -> reconstructed image sequence.

Loss:

- reconstruction MSE between reconstructed image sequence and input image sequence.
- weighted LPAP surrogate regularizer, computed as `RHSurrogateLoss`-style cross entropy between active surrogate logits and LPAP teacher slot targets from the current image-to-energy output.

Explicit exclusions for the first implementation:

- no cycle consistency loss.
- no image-to-energy flow-matching side loss against raw harmonic energy.
- no energy-to-image flow-matching side loss against frozen projected-energy/image pairs.
- no latent-energy L1 reconstruction regularizer inside the bottleneck.

Intent:

- Existing harmonic/flow training initializes the system, but end-to-end training should be free to discover a better amplitude-ordered energy geometry for reconstruction through the LPAP bottleneck.
- LPAP regularizes the surrogate so it remains an approximation of the discrete operator; LPAP does not pin the flow output to the harmonic pretraining distribution.
