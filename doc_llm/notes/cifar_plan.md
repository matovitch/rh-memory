# CIFAR + Flow Matching Plan Notes

This page captures roadmap ideas and experiment planning, not current mandatory behavior.

Canonical current behavior is documented in `doc_llm/spec/*`.

## Planned Direction

- Domain: grayscale CIFAR flattened to `N=1024`.
- Objective trend: flow/generator path to produce LPAP-compatible signals, then train with surrogate/reconstructor stages.
- Optional endpoint boundary condition: match L2 norm of generated series endpoint to source image norm.

## LoD-Oriented Multi-C Surrogate Idea

To nudge the CIFAR experiment toward level-of-detail-oriented compression, train several surrogate variants with different bottleneck widths, for example:

- `C=64` for coarse compression.
- `C=128` for intermediate compression.
- `C=256` for higher-detail compression.

During end-to-end flow-matching training, swap between these frozen or slowly-updated surrogate variants so the generator sees multiple LPAP bottleneck capacities rather than only one approximate top-`C` prior.

Implications:

- Use distinct reconstructors per `C` rather than forcing one reconstructor to decode variable token counts initially.
- The training loop needs either a `C` curriculum/sampling schedule or an explicit conditioning signal that tells the reconstructor/generator which compression level is active.
- Evaluation should report reconstruction/flow quality per `C`, plus aggregate robustness when `C` is sampled.

Rationale:

- The primary target is not a universal variable-token decoder yet; it is discovery/training of a LoD-compressible energy space.
- Per-`C` reconstructors keep each compression level honest and avoid making reconstructor capacity/conditioning the bottleneck that hides whether the upstream energy-space representation is actually compressible at multiple levels of detail.
- A variable-token reconstructor can remain a later distillation or deployment objective after the per-`C` energy-space behavior is understood.

## Why This Lives in Notes

- It includes alternatives (surrogate variants, soft-pooler alternatives, few-step reflow choices) that are not all implemented simultaneously.
- It contains sequencing decisions and tuning guidance that can evolve independently of the core spec.

## Practical Rule

When roadmap text conflicts with code-level contracts, the `spec/*` pages and source code win.
