# Image <-> Latent-Energy Flow Matching Notes

This page is LLM memory for the next experimental direction. It is exploratory, not a current implementation contract.

Canonical implemented behavior remains in `doc_llm/spec/*`.

## Current Picture

The current LPAP autoencoder stack is not the first thing to train in the flow experiment. It is used as a frozen projection operator for the energy-to-image direction:

1. Sample a grayscale `32x32` image and flatten it to `N=1024`.
2. Sample a harmonic energy series with the current synthetic harmonic pipeline.
3. Pass the harmonic series through the frozen surrogate + decoder + soft-scatter stack.
4. Use both the raw harmonic series and the resulting reconstruction in the symmetric flow batch.

The first flow-matching stage trains two directional models with independent image/harmonic pairings:

- `image_to_energy_flow`: maps image-space samples toward raw harmonic energy samples.
- `energy_to_image_flow`: maps frozen soft-scatter projected energy samples toward image-space samples.

Avoid the name `decoder_flow` for the second model because `decoder` already names the LPAP autoencoder decoder.

## Flow Architecture And Ordering

Use the same model architecture for both directional flows. Start with two separate parameter sets rather than shared weights:

- `image_to_energy_flow`: one instance of the common flow architecture.
- `energy_to_image_flow`: another instance of the same architecture.

The first implementation preference is a small time-conditioned dilated residual `Conv1d` vector field over `[B, 1, 1024]`. This keeps implementation surface low while still giving a wide receptive field through dilations. A `1d` U-Net remains a reasonable second experiment if the residual ConvNet lacks multiscale capacity.

Because images are naturally `2d` and harmonic/latent-energy samples are naturally `1d`, define a canonical image sequence order instead of leaving flattening implicit. Use a `32x32` Hilbert curve permutation for image flattening:

1. image tensor `[B, 1, 32, 32]` -> Hilbert-ordered sequence `[B, 1, 1024]` before flow training/inference.
2. sequence `[B, 1, 1024]` -> inverse Hilbert order -> image tensor `[B, 1, 32, 32]` after `energy_to_image_flow` sampling.

Store the Hilbert permutation and inverse permutation as explicit utilities or metadata, not as hidden training-script behavior. The raw-energy endpoint is `HarmonicSample.raw_inputs`; the projected-energy endpoint should use the reconstructed unpermuted `[N]` output produced by the current surrogate + decoder + soft-scatter stack.

## Symmetric Time Pairing

For a sampled triple `(image, raw_energy, projected_energy)`:

- Train `image_to_energy_flow` at time `t` along the image-to-raw-energy interpolation.
- Train `energy_to_image_flow` at time `1 - t` along the projected-energy-to-image interpolation.

This keeps both models at matching progress along opposite directions. Start with a simple time distribution before adding curriculum complexity.

Suggested first baseline:

- sample `t ~ Uniform(eps, 1 - eps)` with `eps` around `1e-4` or `1e-3`.

If endpoint behavior is weak, use a mixture that spends extra probability near both endpoints:

- with probability `0.8`, sample `Uniform(eps, 1 - eps)`.
- with probability `0.2`, sample `Beta(0.5, 0.5)` clamped into `[eps, 1 - eps]`.

## Reflow Stage

After initial flow matching, apply reflow independently to both directional models:

1. Use the learned model to generate trajectories between endpoints.
2. Treat model-generated start/end pairs or intermediate trajectories as straighter supervision.
3. Retrain the flow field on these improved paths.

The goal is to make trajectories straight enough that inference eventually needs very few integration steps. Step-count diagnostics belong in a later dedicated evaluator rather than in the first symmetric training loop.

## End-to-End LPAP Autoencoder Stage

After the directional flows are useful as initializers, train the larger LPAP autoencoder stack end-to-end on real grayscale images:

1. Flatten image tensors through the canonical Hilbert order to `[B, 1, 1024]`.
2. Integrate the image-to-energy flow to produce unpooled learned energy `[B, 1, 1024]`.
3. Feed the learned energy through the active LPAP bottleneck branch: surrogate, decoder, and soft scatter.
4. Treat the soft-scatter output as pooled/projected energy `[B, 1, 1024]`.
5. Integrate the energy-to-image flow to reconstruct the image sequence.

In this stage, harmonic/raw-energy flow matching is only an initialization story. The learned energy geometry should be allowed to drift away from the naive harmonic space if doing so improves reconstruction through the LPAP-like bottleneck. The objective is not to keep the image-to-energy flow close to raw harmonic samples, and it is not to keep the energy-to-image flow close to the projected-energy distribution from the frozen pretraining stack.

The first end-to-end objective should contain:

- Image reconstruction MSE on the final energy-to-image output.
- A surrogate LPAP regularizer computed from the current image-to-energy output, keeping surrogate logits aligned with the discrete LPAP operator.

The first end-to-end objective should exclude:

- Cycle consistency loss.
- Flow-matching side losses for either direction.
- A latent-energy L1 reconstruction regularizer on the bottleneck itself.

The LPAP regularizer is a constraint on the surrogate branch, not a target that pins the learned energy geometry to the harmonic initializer.

## Pairing Caveat

If images and harmonic series are sampled independently, the first flow stage is distribution-level transport, not identity-preserving supervised encoding. That may be acceptable for learning how to enter and leave the latent-energy distribution.

Later end-to-end training, cycle losses, or image-conditioned harmonic sampling would be needed if the goal becomes preserving image identity through the full image -> energy -> image path.

## Diagnostics To Track

- image -> energy -> image relative L1.
- energy -> image -> energy relative L1.
- endpoint L1 magnitude / retained-energy ratios.
- cosine similarity between endpoints and reconstructions.
- distribution statistics for generated flow endpoints once a dedicated evaluator exists.
- LPAP surrogate accuracy / regularizer value during end-to-end training.
- latent-energy autoencoder L1 reconstruction during end-to-end training.

## LoD-Oriented Multi-C Surrogate Idea

To nudge the experiment toward level-of-detail-oriented compression, train several surrogate variants with different bottleneck widths, for example:

- `C=64` for coarse compression.
- `C=128` for intermediate compression.
- `C=256` for higher-detail compression.

During later end-to-end flow training, swap between these surrogate + decoder + scatter branches so the flows see multiple LPAP bottleneck capacities rather than only one approximate top-`C` prior. This acts like structured dropout over level of detail: the energy space is useful only if reconstruction survives after different numbers of high-amplitude winners pass through LPAP.

Implications:

- Use distinct surrogate + decoder/scatter paths per `C` rather than forcing one variable-token branch to handle every compression level initially.
- The training loop needs either a `C` curriculum/sampling schedule or an explicit conditioning signal that tells the flow/autoencoder path which compression level is active.
- Evaluation should report reconstruction/flow quality per `C`, plus aggregate robustness when `C` is sampled.

Rationale:

- The primary target is not a universal variable-token decoder yet; it is discovery/training of a LoD-compressible energy space.
- Per-`C` decoder/scatter paths keep each compression level honest and avoid making variable-token conditioning the bottleneck that hides whether the upstream energy-space representation is actually compressible at multiple levels of detail.
- A universal variable-token decoder/scatter path can remain a later distillation or deployment objective after the per-`C` energy-space behavior is understood.

## Practical Rule

When roadmap text conflicts with code-level contracts, the `spec/*` pages and source code win.
