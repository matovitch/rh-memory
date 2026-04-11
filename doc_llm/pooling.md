# RH-Memory: Pooling Operators

This document defines the mathematical variant of Robin Hood (RH) memory pooling mapping `[n]` sparse activations to `[C]` buckets. We use a **pure GPU-resident Exact Parallel Robin Hood** operator.

## Hashing Function Definition

The mapping for an input index $i \in \{0, \dots, n-1\}$ into a container of capacity $C$ is defined by a quotient-mixed hash:
$$ h(i) = \left( a \cdot \lfloor i / C \rfloor + i \right) \pmod C $$

**Recommended Configuration for $a$:**

- It is heavily advised that $n$ be perfectly divisible by $C$ ($n \pmod C = 0$). This guarantees that all $C$ buckets receive the exact same number of source tokens per sequence ($stride = n/C$), making the expected magnitude identically distributed instead of starving trailing buckets.
- Given that $n$ is divisible by $C$, setting the coefficient **$a = n / C$** is mathematically optimal. This defines a uniform Torus-like topology framing across blocks where every $C$-length subgroup is shifted symmetrically by exactly $stride$. This structural regularity is extremely beneficial when paired with the Decoder's Transformer backbone and Relative Positional Encodings (RoPE), as the attention layers can natively leverage these translational equivariant offsets.

## Exact Parallel RH Operator

Implemented as a Triton kernel acting on the persistent memory table.

### Initialization & Constraints

- Let $stride = n // C$.
- The GPU table is initialized with: $values=0.0$, $dib=0$, $\gamma=0.0$.
- Time decay is explicitly decoupled. The persistent table must be decayed via a separate `advance_time` step prior to executing this spatial pooling loop.

### Parallel Scatter-Swap & Virtual DIB Tracking

Unlike approximate pooling that discards displaced elements, this exact parallel operator ensures no elements are unfairly dropped during collisions by utilizing two core mechanisms:

1. **Scatter-Swap**: When a collision occurs and an incoming element displaces an incumbent, the incumbent is *scattered* back into the probing pipeline slot vacated by the winner. It seamlessly continues probing the next buckets.
2. **Virtual DIB Tracker**: To avoid repeatedly updating DIB values in the tracking pipeline, DIB is tracked via an algebraic offset: `base_dib_offset = table_dib - step`. For any displaced element, its effective DIB during subsequent probe steps is evaluated as `Effective DIB = (step + 1) + base_dib_offset = table_dib + 1`.

### Compute Loop (Semantics)

1. Reshape the input length `n` vector into matrix representations of size `[stride, C]`.
2. Initialize the `base_dib_offset` pipeline tensor to zeros.
3. Over $k$ max iterations (where $k$ is empirically chosen to capture $>99\%$ of settlements):
   a. Compute the `Effective DIB` for all elements currently in the pipeline.
   b. Extract the max elements across the `stride` dimension targeting each bucket `c`, prioritized first by absolute magnitude, then by `Effective DIB` as a tie-breaker.
   c. Compare the winning pipeline elements against the current table incumbents at bucket `c` (again using magnitude, then DIB).
   d. For any pipeline element that wins:
      - Write its `values`, `gamma`, and `Effective DIB` to the table at bucket `c`.
      - Gather the displaced incumbent's `values`, `gamma`, and `dib`.
      - Compute the incumbent's `base_dib_offset = dib - step`.
      - Scatter these displaced attributes back into the pipeline slot that the winner vacated.
   e. Zero out any pipeline elements that successfully wrote and did *not* displace an incumbent (or drop losers that couldn't beat the table natively, although this implies they either lacked magnitude or DIB).
   f. Roll the pipeline row by 1 step to simulate the next linear probe for the remaining/displaced elements.
