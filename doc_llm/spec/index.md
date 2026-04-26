# RH-Memory Spec Index

This directory is the authoritative, code-aligned specification for current behavior.
It captures the current autoencoder-stack baseline and losses as implemented today.

Scope rules:
- Normative: tensor contracts, objectives, and dataflow that match current code.
- Non-goals: roadmap ideas, alternatives, and historical experiments.

## Reading Order

1. [objectives.md](objectives.md)
2. [pipeline.md](pipeline.md)
3. [models.md](models.md)
4. [pooling.md](pooling.md)

## Spec Pages

- [objectives.md](objectives.md): surrogate / decoder / reconstructor training inputs, targets, losses, and metrics.
- [pipeline.md](pipeline.md): stage-by-stage contracts and adapter outputs for `src/rh_memory/pipeline/*`.
- [models.md](models.md): `RHSurrogate`, `RHDecoder`, `RHReconstructor` I/O, masking, and loss modules.
- [pooling.md](pooling.md): LPAP operator semantics and API contract.
