# Pivot to Exact Parallel RH RNN

## Background

The previous architecture implemented a **dual-tier memory**:

- A "Fast" GPU-resident approximate memory (`BatchedFastMemoryState`).
- A "Slow" CPU-resident exact memory (`BatchedSlowMemoryState`).

This dichotomy existed because exact Robin Hood semantics (where larger displaced items continue probing and retain priority) seemed sequentially bottlenecked and ill-suited for the GPU. The CPU tier acted as an exact fallback.

## The Paradigm Shift

The new exact parallel RH pooling design (`exact_triton_pooling.md`) introduces two critical innovations for a purely GPU-based Triton kernel:

1. **Scatter-Swap Mechanism**: Displaced incumbents are seamlessly gathered and re-inserted right into the probing "conveyor belt", effectively continuing linear probing down the bucket line without getting dropped.
2. **Virtual DIB Tracker**: Distance to Initial Bucket is handled analytically (`base_dib_offset = table_dib - step`), circumventing the need to linearly and incrementally update DIBs.

These eliminate the need for the approximate GPU algorithm and the exact CPU fallback tier. Robin Hood properties (poor-gets-richer) can be guaranteed natively on the GPU while maintaining high SIMT parallelism.

## Unified Architecture

The system has pivoted to a **pure GPU streaming RNN model** with a single, exact memory block.

Key simplifications:

- **No CPU Tier**: Transmitting truncated, sorted tensors over PCIe is deprecated.
- **Unified Tokens**: The Decoder architecture now acts on unified `[values, gamma, dib]` tensors (3D tokens). The obsolete `memory_type` feature has been removed.
- **Single Training Pathway**: Removed fast/slow testing branches from the training loop, fully utilizing the single correct exact parallel implementation.
