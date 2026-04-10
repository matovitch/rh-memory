# RH-Memory Slow-Memory Protocol

## Slow State

The slow memory stores words of the form `(values, dib, gamma)` on the CPU side. The CPU owns the table and updates it asynchronously.

The current Python container is [../src/rh_memory/memory.py](../src/rh_memory/memory.py), and the CPU operator binding is in [../src/rh_memory/_cpu_ops.py](../src/rh_memory/_cpu_ops.py) with the C++ implementation in [../csrc/rh_cpu.cpp](../csrc/rh_cpu.cpp).

## Contract Between GPU and CPU

The CPU does not infer time by itself.

- The GPU maintains the global timestep counter.
- The GPU can explicitly advance the slow-memory clock through a helper before the next write batch is prepared.
- The GPU sends only the truncated write batch, not the whole slow-memory table.
- The GPU sends the elapsed timestep delta to that helper.
- The CPU applies the stored per-slot gammas over that delta to decay its slow state before comparison and eviction.

This keeps the CPU asynchronous and stateless with respect to global wall-clock time.

## What Is Fixed and What Changes

- `gamma` is fixed at write time and acts as the per-slot retention factor.
- the stored value is aged in place by the GPU-side clock step.
- the stored value changes over time through the stored gamma applied over the elapsed delta.
- `cutoff_bound_slow_mag` is carried forward on the GPU, tightened by successful writes, and returned so the next write batch can be truncated.
- `cutoff_bound_slow_gamma` is the gamma of the slot that defined that cutoff bound.

## Write Batch Shape

The slow-memory write batch should include:

- sorted values in descending magnitude order
- the truncated sort permutation indices
- per-item write gammas
- the timestep delta since the last slow-memory update

The first slow-memory write batch is expected to contain at least `capacity` items so the table becomes fully occupied on bootstrap. After that, subsequent batches may be smaller and will update the already full table.

The CPU receives `delta_steps` through the slow-memory container helper. It uses the stored per-slot gammas to decay the current slow table before inserting the next batch, which keeps the CPU stateless with respect to global time and avoids sending the full memory back from the GPU.

The GPU-side slow-memory container also ages `cutoff_bound_slow_mag` with `cutoff_bound_slow_gamma` so the truncation threshold can advance without rescanning the full table.

The CPU then performs the actual insertion and eviction logic using the existing table and the new batch.

## Cutoff Logic

The GPU may use `cutoff_bound_slow_mag` to truncate future write batches before transfer. That makes the PCIe transfer smaller over time and preserves the design where only the new writes are sent.

The cutoff bound is stored as a raw magnitude from the current slow table and is aged over time with the gamma of the slot that originally set it.

The write path only tightens the bound when the last successfully inserted item is smaller than the current bound, which keeps the update $O(1)$ after the insertion loop.

## Decay Interpretation

The write-time gamma is both a write-time retention factor and the aging rate applied by the explicit clock step. Time evolution comes from the delta step acting on the stored gammas, not from a global decay knob or timestamp field.

## Helper Contract

The explicit time-advance helper is a convenience wrapper for the GPU-side scheduler. It advances the stored slow-memory values by the elapsed delta using the stored gammas, without inserting new items, which keeps the write path focused on the incoming batch.
