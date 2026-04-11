# Exact Parallel Robin Hood Pooling

This document outlines the theoretical design for an **Exact Parallel Robin Hood (RH) Pooling** operator implemented as a Triton kernel.

Unlike the approximate "fast" GPU pooling—which lossily discards displaced incumbents during collisions—this approach perfectly mimics the sequential exact RH algorithm on highly parallel GPU architectures.

## Motivation

The initial GPU translation of Robin Hood hashing parallelized insertions by checking `k` probe steps simultaneously over bands (strides). However, when a collision occurred and a larger incoming element displaced an incumbent from the memory table, the incumbent was simply dropped. True Robin Hood hashing requires this displaced item to be pushed to the next bucket (open addressing) and continue probing, prioritized by its Distance to Initial Bucket (DIB).

## Core Mechanisms

To achieve exact RH properties in parallel without sequential bottlenecks, the Triton kernel employs two novel mechanisms: **The Scatter-Swap** and the **Virtual DIB Tracker**.

### 1. The Conveyor Belt and Scatter-Swap

Instead of viewing the incoming rows simply as arrays of fresh items, we treat the `[stride, C]` grid as a "conveyor belt" of memory slots evaluating against the persistent table.

At any step $s$:

1. The kernel finds the maximum element (by magnitude and DIB) across the `stride` dimension targeting a specific table bucket `c`.
2. If this maximum element wins against the table's incumbent, it takes the slot.
3. **The Swap:** Instead of zeroing out the pipeline slot that the winner vacated, the kernel **scatters the displaced incumbent** directly into that vacated pipeline slot.
4. Because this slot on the conveyor belt will naturally shift to target bucket `c + 1` in step $s + 1$, the displaced incumbent seamlessly continues its linear probe!

### 2. Virtual DIB Tracker (Zero-Overhead DIB)

A core part of RH hashing is tracking the DIB to prioritize "poorer" elements. Incrementing the DIB for every item in the pipeline at every step is mathematically slow. Instead, we use an implicit DIB derived from the current step $s$.

We introduce a `base_dib_offset` pipeline tensor, initialized to zeros:

* **Fresh Elements:** For a fresh incoming element, `base_dib_offset = 0`. At step $s$, its effective DIB is simply $s + 0 = s$.
* **Displaced Elements:** When an incumbent with an existing `table_dib` is displaced at step $s$, it enters the pipeline. We compute its new offset as:
  $$ \text{base\_dib\_offset} = \text{table\_dib} - s $$
  When the kernel evaluates this displaced element in the next step ($s + 1$), its effective DIB is calculated as:
  $$ \text{Effective DIB} = (s + 1) + \text{base\_dib\_offset} $$
  $$ \text{Effective DIB} = (s + 1) + (\text{table\_dib} - s) = \text{table\_dib} + 1 $$

This algebraic trick guarantees that the DIB increments exactly by 1 for every subsequent probe step, without requiring any continuous array mutations!

### 3. Exact Collision Resolution

During the max-reduction over the stride dimension, the "winning" element is determined exactly as it would be sequentially:

1. **Absolute Magnitude:** Higher magnitude wins.
2. **Tie-Breaker (Robin Hood Property):** If magnitudes are equal, the element with the higher Effective DIB wins.

### 4. Practical Probe Limits (k)

Theoretically, the absolute maximum boundary is $k = C$; probing $C$ times means an element has checked every single slot in the table, and any further steps would just wrap around.

However, running a full $k=C$ loop (e.g., 128 steps) is still too hostile to GPU performance. A smaller bound must be chosen to minimize warp execution times.

Because exact Robin Hood hashing actively minimizes the variance of probe lengths (the "poor-gets-richer" principle), its DIB distribution is expected to be incredibly tight. However, the exact distribution—and thus the safe empirical threshold for $k$—depends heavily on the input sequence length ($n$) relative to the table capacity ($C$), as well as the sparsity of the incoming signal. If $n \gg C$ without sufficient sparsity, collisions will naturally force longer probes. For a given distribution, an empirical study must first determine a safe limit. For example, if a study on a specific $n$ and $C$ shows over 99% of elements settle within a $DIB \le 20$, setting $k \approx 24$ will capture effectively all non-noise insertions while providing massive GPU speedups.

## Conclusion

By swapping displaced table incumbents back into the probing pipeline and virtually projecting their DIB offsets, this Triton kernel design fully rehabilitates the open-addressing and poor-gets-richer properties of classical Robin Hood hashing. It achieves the exact same stable state as the strictly sequential CPU operation, while fully exploiting GPU SIMT parallelism.
