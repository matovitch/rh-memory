# Experiment Plan: CIFAR Grayscale, Flow Matching, and Surrogate Pooling

This document records a **coherent research plan** from the existing **linear-probing-based amplitude pooling** operator (`python_linear_probing_amplitude_pooling` / `triton_linear_probing_amplitude_pooling`) through **training an encoder** with **flow matching**, with **CIFAR-style grayscale** images as the data domain.

**Primary plan for gradients:** replace the discrete pooler with a **differentiable surrogate** (e.g. a **transformer** over buckets, analogous in role to the decoder), and **anchor** it to the real operator with a **regularization / distillation loss** against **discrete pooler** outputs. A **fully relaxed “soft pooler”** (softmax winners, soft scatter) remains an **alternative** if you want algorithmic fidelity without a learned middle layer.

Related notes: [philosophy.md](philosophy.md) (motivation and surrogate-first stance), [decoder.md](decoder.md), [linear_probing_amplitude_pooling.md](linear_probing_amplitude_pooling.md), [gaussian_information_bottleneck.md](gaussian_information_bottleneck.md).

---

## 1. Starting point: what the stack is today

### 1.1 Linear-probing-based amplitude pooling

- **Input:** a 1D vector of length `n` (e.g. flattened image or synthetic signal), with optional per-position **`carry_id`** (integral) for supervision / gather.
- **Geometry:** after a **fixed permutation** (hash-like reordering), the vector is viewed as `[stride, C]` with `stride = n // C`. Each **column** is a **bucket** with `stride` competing “pipeline” slots.
- **Dynamics:** for `k` rounds, each round **selects a winner per bucket** (**strict** \(|\cdot|\) only—pipeline scan order resolves exact ties among equals; vs table **strict** \(|\cdot|\), incumbent on tie), **updates** the table, **writes displaced mass back** into the pipeline (scatter-swap), then **rolls** along the bucket axis so mass moves between buckets.
- **Output:** table values and metadata (e.g. DIB, carried source indices) of shape `[C]` per batch row.

The **reference implementation** is **discrete**: hard maxima, hard masks, hard scatter indices. See [linear_probing_amplitude_pooling.md](linear_probing_amplitude_pooling.md) and the Python baseline in the codebase.

### 1.2 Decoder

- **Input:** table state encoded as **two** scalars per bucket — **`signed_value`** and **`dib`** — shape **`[B, C, 2]`**. **`carry_id`** is handled **inside** the pooler only (integral); not passed to the decoder.
- **Surrogate / mode flag:** use a **negative `dib`** (or another reserved scalar) on tokens that come from the **surrogate** path so the network can distinguish **physical DIB** (≥ 0, or the operator’s real range) from **experimental / surrogate** tokens. Document the sentinel and avoid normalizing it away unintentionally.
- **Backbone:** RoPE transformer over the **bucket** axis (length `C`), then a projection to logits over the **original** index dimension `n` (or, in a reconstruction-oriented variant, to something that can be reshaped to an image).
- **Current training objective (baseline):** absolute-amplitude–weighted BCE for **index recovery** (sparse one-hot targets over `n`). See [decoder.md](decoder.md).

This plan **does not** require throwing away that baseline; it **extends** the pipeline with an **image domain** and an **encoder**.

---

## 2. Target domain: CIFAR grayscale

**Choice:** **grayscale** first (single channel) to avoid batch/channel reshaping decisions and to keep **`n` small**.

- **Spatial size:** `32 × 32 = 1024` pixels.
- **Flattened length:** `n = 1024`, which **aligns** with existing decoder/pooler experiments that use `n = 1024` and e.g. `C = 128` → `stride = 8`.
- **Consistency:** pick **one** flatten order (e.g. **row-major**) and use it everywhere: pooler input, **carry_id** / index targets, visualization, reconstruction reshape. A global **permutation** inside the pooler **scrambles** locality in pipeline space but **does not** remove the need for a **fixed** convention in **index space** for labels and losses.

**Data:** standard CIFAR-10/100 (or similar) converted to grayscale by a fixed linear map (e.g. luminance weights on RGB), with the usual train/val split.

---

## 3. High-level goal of the new experiment

**Rough objective:** learn a **generative map** from a **simple base** (e.g. Gaussian noise) to **peak-like / salient** signals that are plausible inputs to the pooler, then route them through **pooler → decoder**, and train end-to-end or in phases toward a **reconstruction** target in **image space** (e.g. L2 / SSIM / perceptual loss on `32×32`), instead of starting from the heavier **mutual information** formulation.

**Why reconstruction first:** stable gradients, clear metrics, standard tooling. Information-bottleneck or contrastive terms can be **added later** once the pipeline works.

For how this experiment fits the broader **autoencoder / saliency / surrogate-vs-teacher** outlook, see [philosophy.md](philosophy.md).

---

## 4. Encoder: flow matching (and optional few-step reflow)

### 4.1 Role of the encoder

The **encoder** here is the **generative path** from noise to data:

- Sample `z` (e.g. `z ∼ N(0, I)` with dimension matching the flattened image, or a lower-dimensional latent mapped up).
- Produce `x = f_θ(z)` or integrate a **flow-matching** vector field from noise to a **target distribution** matching **training images** or **intermediate “peak”** statistics.

**Pretraining intuition:** first align `f_θ` with **simple synthetic peaks** (as in current 1D synthetic datasets) or with **direct image reconstruction** so the pooler sees well-scaled, meaningful inputs before the full stack is coupled.

### 4.2 Few-step / reflow (optional)

Full continuous-time integration is expensive. **Reflow**, **shortcut models**, or **distillation** can compress the map to **few steps** while staying close to the learned field. Benefits for this project:

- **Fewer** forward operations through the **surrogate** (and encoder) during joint training.
- Potentially **simpler** graph for debugging and for matching **wall-clock** to the discrete pooler experiments.

This is **not** required for a first proof of concept; it is a **later** speed and deployment knob. **Backprop through few-step unrolled flow matching** is standard when each step is differentiable; combine with **staged** training if the chain is long or unstable.

---

## 5. Why the hard discrete pooler blocks the encoder (and what we do instead)

To train `θ` through the bottleneck, gradients must flow from the loss to `x` (and then to `z` / the flow network).

The **hard** pooler implementation is **piecewise constant** where it matters:

- **Winner selection** over `stride` rows is **argmax**-like.
- **Update vs keep** uses **hard** masks.
- **Displacement** uses **discrete scatter** indices.

**`torch.roll` along buckets is differentiable** (backward is an inverse roll). The **non-differentiable** parts are the **discrete choices** above, not the roll. **`carry_id` / index tracking** helps **labels** but does **not** differentiate the winner.

**Chosen approach:** a **learned surrogate** \(g_\phi\) that sees the **flattened input** \(x\) (same convention as the discrete pooler) and predicts a **dual** supervision tensor aligned with \(\mathrm{LPAP}\) (**linear-probing-based amplitude pooling**). **Forward** uses \(g_\phi\) for training the encoder; the **discrete pooler** supplies **targets** and **regularization**, and **evaluation** can still use **discrete pooler → decoder** when measuring fidelity.

---

## 6. Surrogate transformer + regularizer for linear-probing-based amplitude pooling (primary path)

### 6.1 Dual outputs: `[B, n, C]` (transpose of the decoder logits)

Avoid **pooling** from \(n\) tokens down to \(C\) bucket summaries by supervising the **transpose** of the decoder prediction layout:

| Role | Sequence axis | Logits over | Shape |
|------|----------------|-------------|--------|
| **Decoder** | \(C\) buckets | which **source index** \(0..n-1\) won each bucket | `[B, C, n]` |
| **Surrogate** | \(n\) positions | which **bucket** \(0..C-1\) that position **landed in** if it **won**; **losers** → target **zero** everywhere | **`[B, n, C]`** |

- **Same numel** as `[B, C, n]` (swap axes).
- **Per-position** logits need **\(n\)** transformer tokens (one per flattened index); **no** aggregation step from sequence length \(n\) to \(C\) in the head.
- **Weighted BCE** (same spirit as `RHLoss`): weight **per position** (e.g. by **absolute amplitude** \(|x_i|\) or masks from the **discrete pooler**). **Loser** rows stay all-zero targets.

**Consistency:** targets come from \(\mathrm{LPAP}\) (**linear-probing-based amplitude pooling**): each **winning** source index maps to **one** bucket (injectivity of “winner position → bucket” in the usual case). If two buckets ever shared the same **carry_id** in edge cases, clarify target construction or use multi-label BCE for that row.

### 6.2 Sharing weights with the decoder

The surrogate does **not** require a wholly separate giant model:

- **Shared:** same **block** implementation, `d_model`, depth, often **RoPE** with **max length \(\geq \max(n, C)\)** if one codebase hosts both forwards.
- **Different:** **sequence length** in the forward (`n` vs `C`), **`input_proj`** from **per-token** features built from \(x\) (and optional sentinel **dib** on surrogate tokens), and **`output_proj`**: **`Linear(d_model → C)`** for surrogate vs **`Linear(d_model → n)`** for the decoder.

Optional **task embedding** only if 128 vs 1024 is not enough to disambiguate modes empirically.

### 6.3 Regularizer / distillation against the discrete pooler

On sampled \(x\), compute \(\mathrm{LPAP}(x)\) **without** gradients through the pooler for the teacher (stop-gradient into the encoder when chaining).

**Primary surrogate loss:** **weighted BCE** on **`[B, n, C]`** logits vs targets derived from the **discrete pooler** (bucket assignment per winning index; losers zero). This mirrors the **decoder’s** weighted BCE philosophy but on the **dual** tensor.

**Optional extras:** MSE on **predicted table values** if the surrogate also emits a table-shaped head; usually **secondary** once the dual logit loss is stable.

**Weight** \(\lambda_{\mathrm{lpap}}\): high during surrogate pretraining, then **annealed** or **interleaved** with downstream reconstruction so \(g_\phi\) does not **forget** the discrete pooler when the encoder moves \(x\) off the training manifold.

### 6.4 Staged training (recommended)

The **discrete pooler** is the **teacher and regularizer**; the **surrogate** carries the trainable forward path for the encoder. Rationale and curriculum on penalties (sparsity, TV, decay) are in [philosophy.md](philosophy.md).

**Baseline ordering (still valid):**

1. **Train or load** the **decoder** (discrete pooler → decoder) on your baseline task.
2. **Pretrain** \(g_\phi\): minimize **dual** distillation loss to \(\mathrm{LPAP}\) on **fixed** \(x\) (synthetic peaks, CIFAR flatten, or encoder outputs with encoder **frozen**). Goal: surrogate **`[B, n, C]`** matches LPAP-derived targets on held-out inputs.
3. **Pretrain** the **flow / encoder** on images or peaks **without** the bottleneck (no pooler in graph, or pooler **detached**).
4. **Joint:** reconstruction (and task losses) through **\(g_\phi\) → (table or decoder inputs) → decoder** as your architecture defines; add **regularizer** on **dual** logits vs \(\mathrm{LPAP}\). Optionally **alternate** batches: downstream loss only vs **distillation-only** micro-steps.
5. **Evaluate** with **discrete pooler → decoder** on the same \(x\) to measure **sim-to-real** gap.
6. **Optional:** **reflow / few-step** encoder; revisit **MI** or **contrastive** terms later.

**Surrogate-first emphasis (optional tightening):** prioritize training the **surrogate** on bucket-level assignment / **`carry_id`** (e.g. logits over indices or dual **`[B, n, C]`** supervision). Feed the **decoder** from **surrogate-produced** table semantics so gradients flow smoothly; keep \(\mathrm{LPAP}\) as **regularizer** only if you accept higher drift risk without it. When forming decoder tokens, pair **predicted `carry_id`** with a **`dib` that is coherent for that prediction** under the seeded permutation routing—not necessarily the discrete teacher’s winner—so each forward pass presents **internally consistent** `(value, dib)` geometry for weighted BCE.

**Curriculum:** start with **stronger** L1 / total-variation / decay-style weights if used; **anneal** them as the surrogate and task losses stabilize, and **fine-tune** the **few-step** flow integrator in late stages. **TV** can be computed on **absolute amplitude** \(|\mathbf{x}|\) (see [philosophy.md](philosophy.md)) so regularization matches how **peaks** and the pooler treat salience. Tie annealing to **validation** (surrogate fidelity, regularizer term) so penalties do not vanish before the surrogate is usable.

Freezing the **decoder** alone does not fix the bottleneck; the **surrogate** is what makes the encoder path smooth. The **regularizer** is what keeps \(g_\phi\) tied to the **reference** operator.

### 6.5 Risks specific to the surrogate

| Topic | Mitigation |
|--------|------------|
| **Train–test gap** | Surrogate may match the discrete pooler **in-distribution** but fail **off-distribution** when the encoder moves \(x\). Keep \(\mathcal{L}_{\mathrm{reg}}\) non-negligible; **periodic** discrete-pooler eval. |
| **Shortcut solutions** | Large \(g_\phi\) may memorize; use **validation** on the discrete pooler, **weight decay**, and **diverse** \(x\) in distillation. |
| **What to match** | Primary: **dual** `[B, n, C]` BCE vs discrete pooler; optional table-value MSE. |
| **`carry_id` in the pooler** | Integral lane for **target construction**; **not** a **decoder input** (tokens stay 2-D). |

---

## 7. Alternative: fully relaxed “soft pooler”

If you prefer **algorithmic** relaxation instead of a learned \(g_\phi\): replace argmax / hard scatter with **softmax** weights, **soft** table updates, and **soft** write-back; **`torch.roll` stays** as in the reference (already differentiable). **Anneal** temperature toward hard behavior or **evaluate** with the **discrete pooler**. This path is **more faithful** to one smooth extension of the same math but **more implementation** work than a transformer surrogate.

---

## 8. Losses (first iteration)

- **Downstream:** **Image reconstruction** — L2 (MSE) on `32×32` after reshaping logits or a dedicated image head; optionally **SSIM** or **perceptual** (LPIPS-style) loss.
- **Surrogate anchor:** **Weighted BCE** on **`[B, n, C]`** dual logits vs LPAP-derived targets (same **spirit** as decoder `RHLoss` on **`[B, C, n]`**).
- **Optional:** small terms for **peakiness** / **sparsity** if the flow collapses to flat fields (e.g. L1 or **TV on \(|\mathbf{x}|\)** for alignment with absolute-amplitude salience); auxiliary MSE on predicted **table values** if you add such a head.

**MI / IB:** out of scope for v1; see [gaussian_information_bottleneck.md](gaussian_information_bottleneck.md).

---

## 9. Risks and open questions (global)

| Topic | Risk / question |
|--------|------------------|
| **Surrogate vs discrete** | Mitigate with **regularizer**, **staged** training, and **discrete pooler** evaluation. |
| **Soft pooler alternative** | Fidelity to mechanics vs **engineering** cost; surrogate is often **faster to prototype**. |
| **Capacity** | Reconstruction alone may not stress the bottleneck; later **IB-style** or noise terms if needed. |
| **Compute** | Surrogate + decoder + optional few-step flow; reflow **later**. |
| **Decoder interface** | **`[B, C, 2]`** tokens + logits **`[B, C, n]`**; surrogate **`[B, n, C]`** + reshape / branch as needed for reconstruction. |

---

## 10. Summary roadmap (checklist)

1. **Grayscale CIFAR** → `[B, 1024]`, **fixed** flatten order; `n % C == 0` (e.g. `C=128`).
2. **Baseline:** **discrete pooler + decoder** with **`[B, C, 2]`** tokens (value + dib; sentinel dib for surrogate path when applicable).
3. **Surrogate \(g_\phi\)** (transformer): **\(n\)** tokens in → **`[B, n, C]`** logits; **pretrain** with **weighted BCE** vs \(\mathrm{LPAP}\)-derived dual targets; validate on held-out \(x\).
4. **Flow-matching encoder:** **pretrain** without the pooler in the graph (or pooler **detached**).
5. **Joint training:** reconstruction through **\(g_\phi\) → decoder** + **regularizer** to the discrete pooler; **monitor** discrete pooler → decoder **eval**.
6. **Optional:** **reflow / few-step** encoder; **soft pooler** only if abandoning surrogate; revisit **MI** / **contrastive** terms once stable.

This file is the **planning umbrella** for that line of work; implementation details belong in code and smaller, module-specific notes as they land.
