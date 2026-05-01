# LPAP Spec

Normative source:

- `src/rh_memory/_python_ops.py`
- `src/rh_memory/_triton_ops.py`
- `src/rh_memory/pooling_utils.py`

## Contract

`lpap_pool(...)` routes an already-shuffled incoming stream into table buckets via linear-probing amplitude competition.

Expected dtypes:

- `table_values`, `incoming_values`: `float32`
- `table_dib`, `table_carry_id`, `incoming_carry_id`: `int32`

Expected shapes:

- table tensors: `[B, C]`
- incoming tensors: `[B, N]`

Assumptions:

- `incoming_values` is already shuffled upstream.
- `incoming_carry_id` is position-aligned payload (often contiguous `0..N-1` per row for teacher targets).

## Routing Semantics

- Incoming stream is reshaped to `[B, stride, C]` where `stride = N // C` (in current training scripts `N % C == 0` is enforced upstream).
- Winner selection is by absolute amplitude.
- Table update uses `>=` against incumbent magnitude.
- On winner replacement, displaced incumbent is scattered back into pipeline and probe state rolls along bucket axis.
- DIB metadata is tracked as displacement context; DIB does not decide winners.

## Implementations

- Python and Triton paths are intended to match semantics.
- Triton may stage contiguous buffers internally before writing results back.

## Planned End-to-End Role

In the planned LPAP autoencoder, LPAP remains the discrete bottleneck operator that defines bucket winners by amplitude. The end-to-end model does not backprop through hard LPAP routing directly. Instead, LPAP is used to compute teacher slot targets from the current learned energy sequence, and the active surrogate is regularized to predict those targets.

This keeps the surrogate tied to the intended LPAP semantics while allowing the image-to-energy flow to discover a learned energy geometry that may differ from the harmonic pretraining distribution.

The long-term level-of-detail axis is `C`: using multiple surrogate + decoder/scatter branches with different bucket counts changes how many approximate top-amplitude winners survive the bottleneck. Sampling or swapping among branches such as `C=64`, `C=128`, and `C=256` should pressure the learned energy space to be amplitude ordered across levels of detail.
