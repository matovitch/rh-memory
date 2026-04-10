# RH-Memory Decoder Architecture

## Purpose

This document records the current baseline decoder design for RH-Memory.

The decoder is shared across fast and slow memory and consumes one token per bucket.

## Design Goals

- keep the decoder shared between fast and slow memory
- preserve the bucket geometry created by Robin Hood pooling
- make the decoder standard enough to train with a conventional Transformer stack
- predict original-slot logits directly with a dense output head

## Token Schema

Each bucket token contains:

- signed value, represented as a separate sign feature plus absolute magnitude
- gamma, used as a retention or freshness signal
- raw dib, represented as a scalar float
- memory type, encoded as a signed scalar so the shared decoder can distinguish fast from slow memory

The first version does not include a separate head embedding.

## Positional Handling

We use RoPE inside attention rather than appending an explicit bucket position embedding to the token.

This is the default positional mechanism for the decoder.

Bucket index is therefore handled implicitly through attention geometry instead of as an extra absolute token feature.

## Backbone

The decoder backbone is a standard Transformer encoder stack operating over the C bucket tokens.

Recommended baseline:

- input projection from token features to model width
- 1 to N pre-norm Transformer blocks, where N is a design hyperparameter
- multi-head attention with RoPE applied to q and k
- feedforward sublayer in each block

## Output Head

The backbone output is mapped to a dense n-logit head.

This head predicts original-slot logits directly and is trained as a reconstruction objective.

For the initial implementation, a magnitude-weighted BCEWithLogitsLoss is the preferred starting loss.
The target is the original sparse support mask over the n slots, with the positive class weighted by the source magnitude.

## Current Baseline Decisions

- shared decoder for fast and slow memory
- one token per bucket
- signed value split into sign and magnitude
- raw dib as a scalar
- memory type included in the token
- RoPE inside attention
- no head embedding
- dense n-logit output head
- magnitude-weighted BCEWithLogitsLoss for the first pass
- number of Transformer blocks left as a hyperparameter
- different alpha and epsilon values for fast and slow gamma calibration
- these are the current working defaults unless changed by experiment

## Open Questions

None for the first implementation.

Multi-head RH pooling is deferred, so the baseline decoder has no remaining architecture-level open questions.
