# CIFAR + Flow Matching Plan Notes

This page captures roadmap ideas and experiment planning, not current mandatory behavior.

Canonical current behavior is documented in `doc_llm/spec/*`.

## Planned Direction

- Domain: grayscale CIFAR flattened to `n=1024`.
- Objective trend: flow/generator path to produce LPAP-compatible signals, then train with surrogate/decoder/reconstructor stages.
- Optional endpoint boundary condition: match L2 norm of generated series endpoint to source image norm.

## Why This Lives in Notes

- It includes alternatives (surrogate variants, soft-pooler alternatives, few-step reflow choices) that are not all implemented simultaneously.
- It contains sequencing decisions and tuning guidance that can evolve independently of the core spec.

## Practical Rule

When roadmap text conflicts with code-level contracts, the `spec/*` pages and source code win.
