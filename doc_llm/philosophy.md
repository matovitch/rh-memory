# RH-Memory: Project philosophy

This note states **intent**—how the pieces fit together conceptually—without duplicating tensor-level specs ([pooling.md](pooling.md), [decoder.md](decoder.md)). It complements the concrete experiment roadmap in [experiment_plan_cifar_flow_rh.md](experiment_plan_cifar_flow_rh.md).

---

## Autoencoder shape

RH-memory is organized as an **autoencoder** over a high-dimensional signal:

- **Encoder / generator path:** maps a simple base distribution (e.g. Gaussian noise) toward **inputs the pooler can act on**—typically **saliency-structured**: most mass in a small set of localized, high-contrast regions rather than diffuse fields.
- **Bottleneck:** **linear-probing-based amplitude pooling** compresses length-`n` inputs to a **table of size `C`** with explicit scatter–swap dynamics.
- **Decoder:** maps table tokens (signed value, DIB, and whatever payload the design exposes) back toward **index or image** space.

The bottleneck is **not** a free dense vector; it is a **small, structured memory** whose slots compete under fixed rules.

---

## Saliency as inductive bias (not a theorem)

We **bias** the generator toward peak-like signals using tools such as **flow matching toward peaks**, and optional **L1**, **total variation (TV)**, or similar penalties. **Total variation** is often applied to **absolute amplitude** \(|\mathbf{x}|\) (per-coordinate \(|x_i|\) on the flattened vector, or on \(|\mathbf{x}|\) reshaped to an image grid)—so smoothness and “peakiness” are expressed in the same **saliency** units the pooler uses when ranking by **absolute amplitude**. These terms **steer** optimization; they do **not** provide a statistical guarantee of sparsity in the worst case. Treat them as **strong priors**, tuned on validation.

Informal analogies (e.g. **wavelets**: localized bumps in frequency–space) can help intuition about **localized saliency**. They are **not** formal commitments: we do not claim an explicit wavelet transform unless one is built into the architecture or loss.

---

## What “forgetting” can mean here

**Decay** or shrinkage on activations is generic unless it targets **the same representation the memory cares about**.

In this stack, **what persists** is implicitly **what wins slots in the pooled table** under the pooler dynamics (or under a **surrogate** trained to approximate those dynamics). **Forgetting** is interpretable when **time evolution** (decay, eviction, or fewer integration steps) acts on **that** bottleneck state—the same objects the decoder and task use—not only on an unconstrained dense hidden state elsewhere.

This does not solve **identifiability** of latents by itself; it clarifies **where** retention is decided.

---

## Surrogate, discrete pooler, and implicit alignment

End-to-end training through the **discrete** pooler is awkward (piecewise choices). A **differentiable surrogate** can sit in the forward path for the generator and decoder while the **discrete pooler** supplies **targets** and a **regularizer** so the surrogate does not drift arbitrarily.

**Separate input contracts.** **`RHSurrogate`** consumes **one amplitude per sequence index** (`[B, n, 1]`) and emits **`[B, n, C]`** logits—a **transpose** of the decoder layout (`[B, C, n]`). It does not take **DIB**. The **decoder** still uses **`[signed_value, dib]`** per bucket (`[B, C, 2]`). If DIB is reconstructed **after** discrete choices (e.g. from argmax routes), gradients do **not** flow through DIB along that path; see **[decoder.md](decoder.md)** for why that is usually fine.

**Routing determinism given a seed.** For fixed LPAP hyperparameters—including the **permutation seed**—the mapping from **input amplitudes** to **winner indices** is **determined by the operator**. **Which indices appear** depends on **amplitudes**; **DIB** values, when defined consistently, depend on **those outcomes** (and the same seed), not on a separate latent that must receive gradient through every gate.

**Implicit alignment:** the **memory encoder** (flow / few-step integrator) is not required to match a hand-written summary statistic. It can **adapt** to whatever selection policy the surrogate implements—as long as **task losses** and **regularization to LPAP** jointly favor useful, operator-consistent behavior. Distillation terms are explicit; **task-driven** agreement is implicit and must be monitored (e.g. discrete pooler → decoder evaluation).

A practical **surrogate-first** stance (see the experiment plan): train the surrogate to recover **bucket-level structure** (e.g. who won which bucket), build **decoder inputs** from the surrogate, and pair **predicted `carry_id`** with a **`dib` computed consistently under the routing story for that prediction**—so tokens stay **internally coherent** even when they disagree with the discrete teacher. The discrete pooler then remains a **teacher and regularizer**, not the only trainable forward path.

---

## Curriculum on constraints

Early training can lean on **stronger** sparsity / smoothness (including TV on **\(|\mathbf{x}|\)** if used) / decay weights to keep the generator in a sensible regime. **Anneal** those weights as the **surrogate** and **task** losses take over, so late training is shaped mainly by **what the surrogate selects** and by **few-step reflow** of flow matching rather than by hand-tied penalties.

Avoid dropping regularizers **too fast** before the surrogate is stable: validation on surrogate quality and on the discrete regularizer term is a useful guard.

---

## Related reading

- [index.md](index.md) — module map and lexicon  
- [experiment_plan_cifar_flow_rh.md](experiment_plan_cifar_flow_rh.md) — staged training and CIFAR / flow-matching roadmap  
