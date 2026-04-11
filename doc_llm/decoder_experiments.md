# Decoder Training Experiments: Hashing, Plateaus, and Grokking

## Architecture Overview
The system employs a Transformer spatial decoder tasked with reversing a Robin Hood hashing operation, continuously taking in features mapped from 1024 sequence positions compressed down to 128 buckets. The model is forced to perform continuous regression (finding the input wave signal) while simultaneously performing discrete modular arithmetic (using the bucket index and Distance-to-Initial-Bucket `DIB`).

## Finding 1: The 8% Accuracy Plateau
During initial training runs, the model plateaued at approximately 8% accuracy.
- **Root Cause**: The network successfully learned the base modulo-based hash routing function ($Position \pmod{128}$). Since sequence length $n=1024$ and capacity $C=128$, there are exactly $1024 / 128 = 8$ sequence positions that map to any given bucket modulo.
- **Conclusion**: The model correctly filtered out the 1016 incorrect locations, leaving the 8 candidates. This yields a 1-in-8 chance, resulting in the ~8-12.5% plateau.

## Finding 2: The 50% Accuracy Plateau
To break the 1-in-8 tie, we optimized the dataset signal generation to use a squared-abs inverted sine signal: `(1 - |sin|)^2` with less noise to provide sharper, cleaner positional gradients. We also shifted our evaluation logic to output a $1024 \times 1024$ conditional prediction confusion matrix.
- **Root Cause**: This change pushed training accuracy to ~50%. Visualizations of the confusion matrix revealed a sharp main diagonal but identical symmetric "ghost" diagonals. The 1-in-2 plateau occurs because of the left/right mathematical symmetry of the wave function—it maps identical magnitudes to two valid positions within a cycle.
- **Conclusion**: The model successfully isolates the single correct wave cycle but struggles to resolve the final 50/50 symmetric ambiguity.

## Hypothesis & Next Steps
The model mathematically processes everything it needs to break this tie: it has the exact continuous magnitude and the discrete bucket index, which together uniquely define the position. However, fusing continuous regression with integer modular geometry is notoriously difficult for transformers and requires deep algorithmic grokking to emerge.

**Action Plan:** Let the model train significantly longer (e.g., 5,000,000 steps) to see if grokking occurs. To support this safely, we will implement checkpoint saving and loading in the training script (`train_decoder.py`) to periodically dump the `decoder` and `optimizer` state parameters.