# Memory Architecture: Fast and Slow Protocols

## Objective
This document outlines the state sync contract and memory parameters defined within RH-Memory, enforcing the synchronous (GPU) and asynchronous (CPU) split.

## Tier 1: Fast Memory (GPU)
- **Constraint:** Completely synchronous. Evaluates over every sequence simulation step ($\Delta t_{GPU} \equiv 1$).
- **Protocol:** API symmetry with Slow Memory is strictly enforced. Temporal decay is applied as an explicit, separate `advance_time(delta_steps=1)` procedure before invoking the spatial pooling operator. Pre-existing table values (`values[i]`) are updated to `values[i] * gamma[i]`.

## Tier 2: Slow Memory (CPU)
- **Constraint:** Asynchronous. Receives incomplete batches and explicit timeline jumps.
- **Protocol:** The CPU is stateless regarding wall-clock time. The elapsed step count since the previous memory transfer ($\Delta t$) is enforced by the GPU pushing a `delta_steps` integer variable to the C++ bindings (`/src/rh_memory/_cpu_ops.py`).

### Slow Memory Updates
The batch tensor sent over PCIe is strictly truncated based on a running bound.
- Batched elements are strictly sorted in descending magnitude.
- Elements explicitly carry their unique write-time $\gamma$.

#### CPU Reception Semantics:
Before performing an exact Robin Hood insert of the new batch, the CPU ages its current `values[C]` vector using the received `delta_steps` using exponential decay:
$$ values\_aged_{i} = values_{i} \times \left( \gamma_{i} \right)^{\Delta t_{recv}} $$

### Cutoff Logic Formulation
To minimize PCIe transfer width asynchronously, the GPU uses `cutoff_bound_slow_mag` to cleanly truncate newly generated activations.

- During the `advance_time` operation, the CPU iterates exactly over the aged table (`O(C)`) and computes the exact global minimum absolute magnitude of the updated array.
- The GPU holds this exact scalar (`cutoff_bound_slow_mag`) and strictly discards any new activations that fall below it.
- This effectively bounds the transfer size per update.