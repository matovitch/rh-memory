# RH-Memory Philosophy Notes

This page is intentionally non-normative. It records conceptual framing and research intuition.

For executable contracts, use:
- `doc_llm/spec/models.md`
- `doc_llm/spec/pipeline.md`
- `doc_llm/spec/objectives.md`
- `doc_llm/spec/pooling.md`

Terminology used in notes:
- **energy space**: input/output representation regime where magnitude and L2-energy interpretation are meaningful.
- **compressed space**: LPAP bottleneck-side representation determined by `C` and routing dynamics.

## High-Level Framing

- RH-memory can be viewed as an autoencoder-shaped stack mapping through an energy space and a structured compressed space (`C` buckets), rather than relying on a single unconstrained latent abstraction.
- LPAP competition over absolute amplitudes is the core bottleneck mechanism.
- Surrogate-first training is a practical strategy to keep forward passes differentiable while staying anchored to LPAP behavior.

## Strategic Trajectory (Autoencoder First, Memory Next)

- The current autoencoder stack is a first step, not the final architecture goal.
- The immediate objective is to discover a useful energy-space geometry that can later be integrated into a neural memory design.
- Peak-like harmonic signals act as a bootstrap prior for that energy space; the representation is expected to evolve under end-to-end task training plus regularizers.
- An LPAP-alignment regularizer is intended to keep surrogate behavior close to the operator semantics as training evolves.
- An L2 reconstruction anchor keeps the autoencoder stack grounded while the energy-space representation adapts to downstream tasks.

## Design Intuition (Not Guarantees)

- Saliency-biased generators (peak-like structure) may align better with LPAP competition.
- “Forgetting” is most interpretable when it acts on bottleneck-relevant state, not arbitrary hidden activations.
- Curriculum and regularization schedules are experiment levers, not architectural invariants.

## Caveat

Historical formulations (older surrogate axis/layout alternatives) are research notes only and should not be read as current implementation.
