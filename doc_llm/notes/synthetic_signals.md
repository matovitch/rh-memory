# Synthetic Signal Notes

This page summarizes the synthetic harmonic signal generator and experimentation context.

Normative dataflow and objective contracts are in:

- `doc_llm/spec/pipeline.md`
- `doc_llm/spec/objectives.md`

## Generator Context

- Signals are pseudo-harmonic sums with randomized amplitudes, phases, and envelope sharpness.
- Generation is chunked and used as synthetic input for surrogate and decoder/scatter training scripts.
- Grouped permutation (`N % C == 0`) is used in current synthetic pipeline to balance bucket collision structure.

## Experiment Intent

- Synthetic signals serve as controllable scaffolding for training and validating bottleneck behavior.
- They are a stepping stone toward image-coupled experiments (e.g., CIFAR flow roadmap).

## Non-Normative Reminder

Parameter ranges, curriculum choices, and auxiliary regularization ideas may change between experiments.
