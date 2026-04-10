# RH-Memory: A Two-Speed Memory Architecture for Sequence Models

## Core Idea

RH-Memory compresses sparse activations into compact words using Robin Hood hashing-inspired pooling, then stores those words in a two-speed memory system with a fast GPU-side working memory and an asynchronous CPU-side slow memory.

The main design contract is simple:

- the GPU computes new writes and their write-time salience
- the GPU sends the elapsed timestep delta for the current slow-memory update window
- the CPU receives only the write batch, not the full memory table
- the CPU owns the persistent slow table and returns the updated slow-memory summary, including `cutoff_bound_slow_mag` and `cutoff_bound_slow_gamma`

## Reading Map

- [Pooling details](rh_memory_v5_pooling.md)
- [Slow-memory protocol and timestep contract](rh_memory_v5_slow_memory.md)
- [Decoder architecture](rh_memory_v5_decoder.md)
- [Training and decoder details](rh_memory_v5_training.md)
- [Validation plan](rh_memory_v5_validation.md)

## Assumptions

- magnitude is the main salience signal
- the GPU owns time progression
- the CPU is asynchronous and stateless with respect to global time
- the slow-memory update receives only the truncated write batch plus the timestep delta since the last update
