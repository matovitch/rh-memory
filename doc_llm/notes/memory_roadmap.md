# RH-Memory Roadmap Notes

This page is non-normative and captures design intent for the next architectural phase.

Current implemented contracts remain in `doc_llm/spec/*`.

## Why Magnitude-Ordered Compression Matters

- LPAP-style competition is ordered by absolute magnitudes, which makes retained content rank-aware.
- Varying the `C/n` ratio suggests a natural progressive-compression axis: smaller `C` yields coarser retained structure; larger `C` preserves more detail.
- This motivates LoD-like (level-of-detail) behavior as a future memory capability.

## Why L2 / Energy Framing Is Useful

- Magnitude-squared accumulation maps naturally to an energy interpretation.
- An L2-oriented reconstruction anchor is a practical way to keep latent representations meaningful as end-to-end objectives evolve.
- Endpoint L2 matching in flow-related stages can be interpreted as continuity between upstream generation and downstream magnitude-ranked bottleneck usage.

## Forgetting / Retention Dynamics (Design Intent)

- Exponential decay on latent magnitudes can model retention half-life behavior.
- Thresholding low-magnitude values can act as vacuum/cleanup for explicit forgetting.
- Combined with LoD controls (`C/n`, thresholds), this gives interpretable tradeoffs between persistence, memory budget, and reconstruction quality.

## Status

- These dynamics are target design directions.
- They are not yet normative implementation contracts.
