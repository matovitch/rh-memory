# Model Contracts Spec

Normative source:
- `src/rh_memory/surrogate.py`
- `src/rh_memory/decoder.py`
- `src/rh_memory/reconstructor.py`

## RHSurrogate

Input:
- `bucket_tokens`: `[B, C, stride]`

Output:
- logits: `[B, C, n]`

Attention behavior:
- Uses RoPE-enabled self-attention blocks.
- Uses ring-causal local mask with horizon:
  - `k_tokens = max(1, int(fast_k * log(C)))`
  - bucket `c` attends to `{c, c-1, ..., c-(k_tokens-1)} mod C`.

Loss module:
- `RHSurrogateLoss`: weighted BCE-with-logits over `[B, C, n]`, weighted by `[B, C]`.

## RHDecoder

Input:
- `bucket_tokens`: `[B, C, 2]`
- channel 0: signed value
- channel 1: dib-like scalar (in stage-2 currently derived from surrogate `j_star`)

Output:
- logits: `[B, C, n]`

Attention behavior:
- Uses RoPE-enabled self-attention blocks.
- Full self-attention over `C` tokens (no explicit mask in current decoder forward).

Loss module:
- `RHDecoderLoss`: weighted BCE-with-logits over `[B, C, n]`, weighted by `[B, C]`.

## RHReconstructor

Input:
- `bucket_tokens`: `[B, C, 3]`
- channels: `[selected_value, normalized_source_index, confidence]`

Output:
- reconstructed sequence: `[B, n]`

Architecture behavior:
- Token encoder is permutation-equivariant over bucket tokens (no RoPE).
- Learnable query table of length `n` cross-attends into encoded bucket memory.

Loss module:
- `RHReconstructorLoss`: MSE over `[B, n]`.
