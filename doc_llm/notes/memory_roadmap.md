# RH-Memory Roadmap Notes

This page is non-normative and captures design intent for the next architectural phase.

Current implemented contracts remain in `doc_llm/spec/*`.

Terminology used here:
- **energy space**: representation regime where magnitude/L2-energy interpretation is meaningful (often task/domain dependent).
- **compressed space**: LPAP bottleneck-side representation controlled by `C` and routing/thresholding policies.

## Why Magnitude-Ordered Compression Matters

- LPAP-style competition is ordered by absolute magnitudes, which makes retained content rank-aware.
- Varying the `C/n` ratio in compressed space suggests a natural progressive-compression axis: smaller `C` yields coarser retained structure; larger `C` preserves more detail.
- This motivates LoD-like (level-of-detail) behavior as a future memory capability.

## Why L2 / Energy Framing Is Useful

- Magnitude-squared accumulation maps naturally to an energy interpretation.
- An L2-oriented reconstruction anchor is a practical way to keep energy-space representations meaningful as end-to-end objectives evolve.
- Endpoint L2 matching in flow-related stages can be interpreted as continuity between upstream generation and downstream magnitude-ranked bottleneck usage.
- This endpoint-L2 framing is likely most useful for domains where energy semantics matter (for example grayscale/luminosity), and may be less informative for arbitrary activation spaces.

## Forgetting / Retention Dynamics (Design Intent)

- Exponential decay on latent magnitudes can model retention half-life behavior.
- Thresholding low-magnitude values can act as vacuum/cleanup for explicit forgetting.
- Combined with LoD controls (`C/n`, thresholds) in compressed space, this gives interpretable tradeoffs between persistence, memory budget, and reconstruction quality.

## Status

- These dynamics are target design directions.
- They are not yet normative implementation contracts.
