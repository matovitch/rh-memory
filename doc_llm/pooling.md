# RH-Memory: Pooling Operators

This document defines **linear-probing-based amplitude pooling**: mapping a length-`[n]` vector of signed samples into `[C]` table buckets via an open-addressing, scatter-swap dynamics (Python: `python_linear_probing_amplitude_pooling`; GPU: `triton_linear_probing_amplitude_pooling`). Both implementations require **``torch.float32``** ``table_values`` / ``incoming_values`` and **``torch.int32``** ``table_dib``, ``table_carry_id``, and ``incoming_carry_id``. The **Python** reference updates **table** state in place; **incoming** tensors are only **read** (permuted pipeline in gather buffers). **Triton** shares that gather and may stage **table** tensors to contiguous buffers when convenient, run the kernel (stride-axis **vectorized max** on absolute amplitude, strict compare vs ``|table|``), then copy back—**ownership is an optimization hint**, not a mandate to mutate every buffer in place. **Amplitude** here means the signal can be signed; comparisons use **strict** inequality on \(|\cdot|\) only. Versus the table, only a **strictly larger** amplitude replaces the incumbent; exact ties between winner and incumbent leave the incumbent unchanged. DIB does not enter ordering.

## Position routing (permutation + modulo)

Routing does **not** use a closed-form mixed hash on the raw index. Instead:

1. **Seeded permutation:** build a uniform random permutation `perm` of $\{0,\ldots,n-1\}$ with `torch.randperm(n)` and a fixed generator seed (the pooling API exposes `seed`; default matches the Python/Triton reference). The incoming vector is **gathered** along the sequence axis: destination position $j$ reads source index `perm[j]` (`output[j] = input[perm[j]]`). Thus original index $i$ lands at permuted linear index $k$ satisfying `perm[k]=i`.
2. **Grid reshape:** treat the permuted sequence as a dense matrix of shape $(\textit{stride}, C)$ with $\textit{stride}=n/C$.
3. **Initial bucket:** column $c \in \{0,\ldots,C-1\}$ is bucket $c$. For permuted linear index $k$, the pipeline column is
   $$ c = k \bmod C,\qquad r = \lfloor k / C \rfloor \in \{0,\ldots,\textit{stride}-1\} $$
   so each bucket receives exactly $\textit{stride}$ competing slots (one per row).

It is heavily advised that $n$ be perfectly divisible by $C$ ($n \pmod C = 0$). Then every bucket has the same pipeline depth $\textit{stride}=n/C$, which balances load across buckets for the decoder’s Transformer over the bucket axis (RoPE).

The permutation intentionally **breaks trivial locality** in raw index order while keeping the regular $(\textit{stride}, C)$ geometry the scatter–swap dynamics rely on.

## Operator overview

Implemented as a Triton kernel (and Python reference) on a persistent table.

### Initialization & constraints

- Let $stride = n // C$.
- The GPU table is initialized with: $values=0.0$, $dib=0$, and `carry_id` (integral), often a sentinel such as $-1$ until real payloads are written. Comparison vs incumbents uses **absolute amplitude** of stored values only—there is **no** separate “empty slot” amplitude sent like $-1$ for the table.
- (Time-decay of the table is not part of the current Python/Triton path; a future design can `gather` per-source factors from `carry_id` if needed.)

### Scatter-swap & virtual DIB tracking

Unlike pooling that discards displaced elements, this operator keeps displaced incumbents in play:

1. **Scatter-swap**: When an incoming element wins a bucket, the displaced incumbent is written back into the probing pipeline slot the winner vacated.
2. **Virtual DIB tracker**: DIB is tracked via an algebraic offset (`base_dib_offset = table_dib - step`) so effective DIB composes correctly across steps without mutating every cell each iteration.

### Compute loop (semantics)

1. Reshape the length-`n` vector into `[stride, C]` (after the fixed permutation).
2. Initialize the `base_dib_offset` pipeline tensor to zeros.
3. Over $k$ iterations (chosen empirically):
   a. Compute effective DIB for pipeline cells.
   b. Per bucket $c$, reduce over `stride` to select a winner by \(|\cdot|\) (strict `>` vs the table). **Zero** values participate like any other (\(|0|=0\)); they are not masked out of the reduction.
   c. Replace a table slot only if the winner has **strictly larger** \(|\cdot|\) than the incumbent.
   d. On update: write `values`, `carry_id`, effective DIB; scatter displaced incumbent back; roll the pipeline along $C$.

There is **no** strict guarantee that the globally top-$C$ amplitudes occupy the table after finite $k$; increase $k$ or adjust data sparsity as needed.
