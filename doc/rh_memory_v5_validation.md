# RH-Memory Validation Plan

## What to Measure

- DIB scaling against `log(n)`
- reconstruction fidelity versus compression ratio
- bucket utilization and dead-bucket collapse
- synthetic sequence recall performance
- the effect of time decay and salience gamma ablations

## Suggested Ablations

- with and without gamma-based decay
- scalar versus magnitude-dependent retention
- with and without DIB
- fast state only versus two-speed memory
- shared versus separate decoders
- affine versus modulo hash
