# Linear-probing-based amplitude pooling

This document outlines the design for **linear-probing-based amplitude pooling**, implemented as a Python reference (`python_linear_probing_amplitude_pooling`, in-place on caller tensors when possible) and a Triton kernel (`triton_linear_probing_amplitude_pooling`).

Unlike coarse approximate pooling—which can drop displaced incumbents during collisions—this routine follows an **open-addressing, linear-probing–style** interaction: collisions resolve by **strict** comparisons on **absolute amplitude** \(|\cdot|\) (stored samples remain **signed**). The table keeps the incumbent unless the candidate is **strictly** larger in \(|\cdot|\). Among pipeline slots with identical \(|v|\), the **first in scan order** among maxima wins (no extra tie key). **Scatter-swap** re-enters displaced slots in the pipeline rather than dropping them.

There is **no guarantee** that the globally largest amplitudes all land in the table within a bounded number of steps \(k\); \(k\) is chosen empirically for accuracy vs cost.

## Motivation

True linear probing requires pushing the incumbent along the probe sequence. This design **swaps** displaced incumbents back into the pipeline slot the winner vacated, then **rolls** the pipeline so probing continues consistently with a sequential open-addressing story.

## Core mechanisms

### 1. Conveyor belt and scatter-swap

The incoming tensor is viewed as a `[stride, C]` grid (after a fixed permutation) probed against a persistent table of size \(C\).

At step \(s\):

1. Find the winning pipeline element per bucket \(c\) by **strictly greater** absolute amplitude when scanning rows; equal \(|\cdot|\) leaves the running best unchanged (**first** maximal row in scan order wins among ties).
2. If it beats the table incumbent, it takes the slot.
3. **Scatter-swap:** the displaced incumbent is written into the **same pipeline slot** the winner came from, so it continues probing as the conveyor **rolls**.

### 2. Virtual DIB tracker

Effective DIB is tracked without mutating every pipeline cell each step, using a `base_dib_offset` pipeline tensor (see Python reference). Displaced incumbents get an updated offset so that at the next step, effective DIB advances by one probe along the chain.

### 3. Collision resolution

During the max-reduction over `stride`:

1. **Pipeline:** only **strictly** larger \(|\text{value}|\) replaces the running best over `stride` (ties keep the current best; first row reaching the max in scan order). Values with **zero** magnitude are included—no separate “active” mask for nonzero samples.
2. **Vs table:** only **strictly** larger \(|\cdot|\) replaces the incumbent; **ties keep the incumbent**.

### 4. Probe limit \(k\)

The theoretical worst case involves many wraps; in practice a finite \(k\) is used. The safe \(k\) depends on \(n\), \(C\), sparsity, and the input distribution; tune empirically (e.g. studies on DIB quantiles).

## Conclusion

Scatter-swap plus virtual DIB tracking yields behaviour aligned with **linear-probing-based amplitude pooling** on SIMT hardware, matching the sequential reference semantics from the Python implementation when \(k\) is sufficient.
