# RH-Memory: Pooling Operators

This document defines the two mathematical variants of Robin Hood (RH) memory pooling mapping `[n]` sparse activations to `[C]` buckets.

## Hashing Function Definition
The mapping for an input index $i \in \{0, \dots, n-1\}$ into a container of capacity $C$ is defined by a quotient-mixed hash:
$$ h(i) = \left( a \cdot \lfloor i / C \rfloor + i \right) \pmod C $$

## 1. Fast Operator: Approximate GPU Pooling
Implemented as a tensorized loop acting on folded feature matrices. Simulates open-addressing across `stride` bands.

### Initialization & Constraints
- Let $stride = n // C$.
- The GPU table is always initialized. Starting states for empty slots: $values=0.0$, $dib=0$, $\gamma=0.0$.
- Time decay is explicitly decoupled. The persistent table must be decayed via a separate `advance_time` step prior to executing this spatial pooling loop.

### Compute Matrix Loop (Pseudocode Semantics)
1. Reshape the input length `n` vector into matrix representations of size `[stride, C]`. This matrix defines the probe bands.
2. For each row $r \in [0, \dots, stride-1]$, roll the row by $r \times a$ (quotient-mixed offset).
3. Maintain aligned matrices for: `values`, `gamma`.
4. Over $k$ max-erase-roll iterations:
   a. Extract `current_winners_absolute = max(abs(values_matrix), axis=0)`. Compare these against the already-decayed GPU table state recursively.
   b. Write updating columns for `[C]` to the persistent table:
      - `values[C]`: The winning signed value.
      - `dib[C]`: The loop iteration index $k$ (analogous to Distance to Initial Bucket).
      - `gamma[C]`: The retention factor matching the winning item.
   c. Scatter $0.0$ to the locations of the selected winners in `values_matrix` to prevent repeat victories.
   d. Roll the matrix rows by 1 to simulate subsequent linear probe steps.

## 2. Slow Operator: Exact CPU Pooling
Implemented as a scalar element-wise Exact Robin Hood hash insertion logic.

### Insertion Logic
- **Input Constraint:** The CPU strictly receives a batch that has **already been sorted** in descending magnitude and **thresholded/truncated** on the GPU. The CPU does no sorting.
- Loop over elements:
  - Hash elements to find the target bucket.
  - If a collision occurs, the element with the larger `dib` (the poorer item) takes the bucket, and the displaced item is pushed to the next bucket ($dib_{displaced} \gets dib_{displaced} + 1$).
  - (Note: The incumbent magnitudes are already pre-aged by a separate `advance_time` pass before this insertion process begins).
- Output states managed: `values[C]`, `dib[C]`, `gamma[C]`.