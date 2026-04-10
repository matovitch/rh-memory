# RH-Memory Pooling Details

## RH Pooling Operator

Given a 1D sparse tensor `x` of length `n`, we compress it into a word of capacity `C << n` by simulating Robin Hood hash insertion. Elements are effectively processed in descending magnitude order, so high-magnitude items win bucket conflicts.

The output is two vectors:

- `values[C]` - the winning element per bucket
- `dib[C]` - distance to initial bucket, encoding how many shifts the winner traveled from its natural bucket

The original position is recoverable as `(bucket_idx - dib) % C`.

## Hash Function

The default hash is the modulo mapping `h(i) = i % C`, which folds the input into a `(C, stride)` matrix where `stride = n // C`. This can align badly with periodic patterns, so an affine hash is a cleaner default for experiments:

```text
h(i) = (a * i + b) % C
```

The important property is that `dib` stays interpretable in the same modular space.

## Approximate GPU Implementation

The current approximate implementation in [../src/rh_memory/_cpu_ops.py](../src/rh_memory/_cpu_ops.py) uses repeated per-bucket maxima, zeroing, and rolling to mimic displacement.

This is a good fit for GPU execution because it is batch-friendly and easy to vectorize across buckets.

## Multi-Head Pooling

The experimental multi-head variant in [../exp/multihead_rh.py](../exp/multihead_rh.py) uses multiple affine hashes over the same input to produce several compressed views. That is useful for testing whether multiple hash views improve reconstruction fidelity or reduce collision artifacts.
