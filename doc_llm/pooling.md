# Linear-probing-based amplitude pooling (LPAP)

This document specifies **tensor semantics**, **routing geometry**, and **algorithm design** for mapping a length-`[n]` vector of signed samples into `[C]` table buckets via open-addressing **scatter–swap** dynamics.

**Implementations:** Python reference **`python_linear_probing_amplitude_pooling`** (`src/rh_memory/_python_ops.py`, table updated in place when possible) and GPU **`triton_linear_probing_amplitude_pooling`** (`src/rh_memory/_triton_ops.py`). Both require **`torch.float32`** for `table_values` / `incoming_values` and **`torch.int32`** for `table_dib`, `table_carry_id`, and `incoming_carry_id`. The Python path applies the permuted pipeline in gather buffers and updates the **table** in place; **incoming** tensors are read-only. Triton may stage table tensors to contiguous buffers, run the kernel (stride-axis **vectorized max** on absolute amplitude, strict compare vs `|table|`), then copy back—**ownership is an optimization hint**, not a mandate to mutate every buffer in place.

**Amplitude** means the signal can be **signed**; all **ordering** uses **strict** inequality on \(|\cdot|\) only. Versus the table, only a **strictly larger** amplitude replaces the incumbent; exact ties leave the incumbent. **DIB** does not decide ordering.

---

## Position routing (permutation + modulo)

Routing does **not** use a closed-form mixed hash on the raw index.

1. **Seeded permutation:** build a uniform random permutation `perm` of \(\{0,\ldots,n-1\}\) with `torch.randperm(n)` and a fixed generator seed (the pooling API exposes `seed`; default matches the Python/Triton reference). The incoming vector is **gathered** along the sequence axis: destination position \(j\) reads source index `perm[j]` (`output[j] = input[perm[j]]`). Thus source index \(i\) sits at permuted linear index \(k\) with `perm[k]=i`.
2. **Grid reshape:** treat the permuted sequence as a dense matrix of shape \((\textit{stride}, C)\) with \(\textit{stride}=n/C\) (when \(n\) is divisible by \(C\)).
3. **Initial bucket:** column \(c \in \{0,\ldots,C-1\}\). For permuted linear index \(k\), pipeline column \(c = k \bmod C\) and row \(r = \lfloor k / C \rfloor\).

Prefer \(n\) divisible by \(C\) so each bucket has equal pipeline depth \(\textit{stride}=n/C\)—this balances load for decoders that use RoPE over the bucket axis.

The permutation **breaks trivial locality** in raw index order while preserving the regular \((\textit{stride}, C)\) geometry scatter–swap relies on.

---

## Scatter–swap dynamics and virtual DIB

Unlike pooling that drops displaced entries on collision, LPAP **keeps** displaced incumbents in play:

1. **Scatter-swap:** When an incoming element wins a bucket, the displaced incumbent is written back into the **pipeline slot the winner vacated**, then the pipeline **rolls** along \(C\) so probing continues like sequential open addressing.
2. **Virtual DIB tracker:** Effective DIB is tracked via an algebraic offset (`base_dib_offset`) so displacement composes across steps without mutating every pipeline cell each iteration.

### Conveyor belt intuition

The incoming tensor is viewed as `[stride, C]` after permutation. Each step: per bucket \(c\), pick the pipeline row winner by \(|\cdot|\) over `stride`; **ties** among pipeline maxima resolve by **scan order** (first maximal row wins). **Vs table:** replace incumbent only on **strictly larger** \(|\cdot|\); ties keep the table. **Zeros** participate (\(|0|=0\)); there is no separate “inactive” mask for amplitude.

There is **no guarantee** that the globally largest amplitudes all occupy the table after finite \(k\); tune \(k\) empirically vs cost (e.g. relative to \(\log C\) in experiments).

---

## Operator compute loop (semantics)

Implemented as an outer loop over \(k\) iterations:

1. Reshape the permuted length-\(n\) vector to `[stride, C]`.
2. Initialize pipeline `base_dib_offset` to zeros.
3. Repeat \(k\) times:
   - Effective DIB for pipeline cells combines offsets with the step index.
   - Per bucket \(c\), reduce over `stride` to select a winner by \(|\cdot|\) (strict `>` vs the table for updates).
   - On update: write `values`, `carry_id`, effective DIB; scatter displaced incumbent back; **roll** the pipeline along \(C\).

Finite \(k\) does not strictly guarantee the global top-\(C\) amplitudes fill the table—increase \(k\) or adjust sparsity as needed.

---

## Python vs Triton

Semantics are aligned: same permutation gather, same strict-\(|\cdot|\) rules, same scatter–swap and roll. Triton parallelizes over batch (and may vectorize along `stride`); choose either path for correctness checks (`tests/test_batched_rh.py` parity).
