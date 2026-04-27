# RH-Memory Roadmap Notes

This page is non-normative and captures design intent for the next architectural phase.

Current implemented contracts remain in `doc_llm/spec/*`.

Terminology used here:

- **energy space**: representation regime where absolute magnitude/L1-energy interpretation is meaningful (often task/domain dependent).
- **compressed space**: LPAP bottleneck-side representation controlled by `C` and routing/thresholding policies.

## Why Magnitude-Ordered Compression Matters

- LPAP-style competition is ordered by absolute magnitudes, which makes retained content rank-aware.
- Varying the `C/N` ratio in compressed space suggests a natural progressive-compression axis: smaller `C` yields coarser retained structure; larger `C` preserves more detail.
- This motivates LoD-like (level-of-detail) behavior as a future memory capability.
- A concrete training direction is to maintain multiple surrogate bottlenecks for the same source length (for example `N=1024` with `C=64`, `C=128`, and `C=256`) and swap among them during end-to-end flow-matching training, forcing upstream generation and downstream reconstruction to tolerate multiple compression levels.

## Variable-C LoD Training Sketch

For CIFAR-like experiments, a multi-`C` setup could go beyond a single LPAP approximate top-`C` prior:

1. Train or checkpoint surrogate variants at several `C` values.
2. During end-to-end training, sample a `C` value and route through the corresponding surrogate bottleneck.
3. Decode with the decoder/scatter path assigned to that `C` value.

This would make `C` an explicit compression/detail control rather than a fixed architectural constant.
Distinct per-`C` decoder/scatter paths are acceptable, and likely preferable at this research stage, because the core objective is to discover/train a LoD-compressible energy space rather than to prove that one decoder can handle arbitrary token counts.
Keeping the decoders separate helps isolate whether the upstream representation remains meaningful under different compression budgets; a universal variable-token decoder/scatter path can be revisited later as a distillation or deployment convenience.

## Why L1 / Energy Framing Is Useful

- Absolute-magnitude accumulation maps naturally to LPAP's amplitude-ranked retention mechanism.
- An L1-oriented reconstruction anchor is a practical way to keep energy-space representations meaningful as end-to-end objectives evolve without over-emphasizing large residuals via squaring.
- Endpoint L1 matching in flow-related stages can be interpreted as continuity between upstream generation and downstream magnitude-ranked bottleneck usage.
- This endpoint-L1 framing is likely most useful for domains where magnitude semantics matter (for example grayscale/luminosity), and may be less informative for arbitrary activation spaces.

## Forgetting / Retention Dynamics (Design Intent)

- Exponential decay on latent magnitudes can model retention half-life behavior.
- Thresholding low-magnitude values can act as vacuum/cleanup for explicit forgetting.
- Combined with LoD controls (`C/N`, thresholds) in compressed space, this gives interpretable tradeoffs between persistence, memory budget, and reconstruction quality.

## Status

- These dynamics are target design directions.
- They are not yet normative implementation contracts.
