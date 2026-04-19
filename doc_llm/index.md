# RH-Memory Architecture: System Overview for AI Agents

## Overview

RH-Memory routes a length-`n` signal through **linear-probing-based amplitude pooling (LPAP)** into a length-`C` bottleneck table (**`C << n`**), then maps table state back toward sequence (or image) semantics. The bottleneck is **structured competition under fixed pooling rules**, not a free dense latent.

**Concrete specs:** **[pooling.md](pooling.md)** — permutation routing, scatter–swap, virtual DIB, dtypes, **`python_linear_probing_amplitude_pooling`** / **`triton_linear_probing_amplitude_pooling`**.

**Decoder and surrogate:** **[decoder.md](decoder.md)** — **`RHDecoder`** vs **`RHSurrogate`** layouts, **`BCEWithLogits`**-style losses weighted by amplitude, RoPE axes, **two-stage synthetic training** (`train_surrogate.py`, `train_decoder_surrogate.py`). Sparse sequence targets describe **decoder** supervision; surrogate training uses LPAP-derived teachers on the permuted stream.

**Motivation (saliency, surrogate vs discrete pooler, what “forgetting” can mean, curriculum ideas):** **[philosophy.md](philosophy.md)** — conceptual framing only; it does not duplicate tensor contracts.

## Key Lexicon

- **`n`:** Input / sequence dimension (scalar).
- **`C`:** Bucket count after pooling (scalar).
- **`stride`:** `n // C` — pipeline rows in the `[stride, C]` grid when `n` is divisible by `C`.
- **`dib` (DIB):** Distance-to-initial-bucket metadata along the probe chain (see **pooling.md**).
- **`carry_id`:** Integral payload per slot (e.g. source index for supervision); use a sentinel such as `-1` for empty.

## Module Map

1. **[philosophy.md](philosophy.md)** — Intent: autoencoder reading of the stack, saliency bias, surrogate vs discrete teacher, retention / forgetting as bottleneck-level phenomena.
2. **[pooling.md](pooling.md)** — LPAP semantics and implementations (Python + Triton).
3. **[decoder.md](decoder.md)** — Token contracts, **`RHDecoder`** / **`RHSurrogate`**, losses, optional attention notes, synthetic training entrypoints.
4. **[experiment_plan_cifar_flow_rh.md](experiment_plan_cifar_flow_rh.md)** — CIFAR-style roadmap: flow matching, surrogate anchoring, staged training (orthogonal to the minimal synthetic scripts above).
5. **[synthetic_harmonic_signals.md](synthetic_harmonic_signals.md)** — Harmonic peak generator (`harmonic_raw_batch` in `experiments/synthetic_lpap_pipeline.py`).
6. **[immediate_todos.md](immediate_todos.md)** — Short-lived scratch queue only.
