# Model Contracts Spec

Normative source:

- `src/rh_memory/surrogate.py`
- `src/rh_memory/decoder.py`
- `src/rh_memory/reconstructor.py`

## RHSurrogate

Input:

- `bucket_tokens`: `[B, C, stride]`

Output:

- logits: `[B, C, N]`

Attention behavior:

- Uses RoPE-enabled self-attention blocks.
- Uses ring-causal local mask with horizon:
  - `k_tokens = max(1, int(fast_k * log(C)))`
  - bucket `c` attends to `{c, c-1, ..., c-(k_tokens-1)} mod C`.

Loss module:

- `RHSurrogateLoss`: weighted cross entropy over logits `[B, C, N]`, sparse target indices `[B, C]`, valid bucket mask `[B, C]`, and weights `[B, C]`.

## RHDecoder

Input:

- `decoder_tokens`: `[B, C, 3]`
- channels: `[soft_amplitude, soft_normalized_dib, surrogate_doubt]`

Output:

- logits: `[B, C, N]` over permuted slots.

Architecture behavior:

- Uses RoPE-enabled self-attention blocks over the `C` bucket tokens.
- The training bridge is soft/differentiable; no discrete index-selection path is part of the active contract.

Loss module:

- `RHDecoderDistillationLoss`: weighted soft KL distillation from frozen surrogate logits `[B, C, N]` to decoder logits `[B, C, N]`.
- Bucket weights are `[B, C]`, typically `abs(soft_amplitude)` from decoder input tokens.

## RHReconstructor

Input:

- `bucket_tokens`: `[B, C, 3]`
- soft decoder path channels: `[soft_amplitude, soft_normalized_source_index, decoder_doubt]`

Output:

- reconstructed sequence: `[B, N]`

Architecture behavior:

- Token encoder is permutation-equivariant over bucket tokens (no RoPE).
- Learnable query table of length `N` cross-attends into encoded bucket memory.

Loss module:

- `RHReconstructorLoss`: MSE over `[B, N]`.
