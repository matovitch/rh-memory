# RH-Memory Training and Decoder Notes

## Decoder

The full decoder specification lives in [rh_memory_v5_decoder.md](rh_memory_v5_decoder.md).

Both fast and slow memory words can feed a shared decoder because they use the same word structure and hash geometry. The decoder can condition on the memory type and bucket metadata.

The current baseline loss is magnitude-weighted BCEWithLogitsLoss over the original-slot support mask.
Memory type is encoded as a signed scalar feature in the decoder token.

## Training Stability

The architecture is sensitive to hard discrete changes in bucket assignment, so a soft-start curriculum is still a sensible choice:

- start with a smooth relaxation such as Gumbel-Softmax
- anneal toward hard assignments later
- keep the L1 schedule to encourage sparse salience concentration

## Magnitude as Salience

The architecture assumes that important information is expressed through magnitude. This should be treated as an explicit inductive bias rather than a theorem.
