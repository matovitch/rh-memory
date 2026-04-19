# Gaussian Information Bottleneck (GIB) with Flow Matching, Linear-Probing-Based Amplitude Pooling, and the Decoder

This note sketches a research direction: compress a **latent Gaussian** through a **generative path** (flow matching) into **peak-like signals**, route them through **linear-probing-based amplitude pooling** (`python_linear_probing_amplitude_pooling` / `triton_linear_probing_amplitude_pooling`) and the **decoder**, then **reconstruct the latent** and optimize a **mutual information**–style objective related to the **information bottleneck**. It also lists practical hurdles and concrete options for mapping decoder outputs back toward Gaussian samples.

Inductive bias here comes from **flow matching toward peaks**, optional **sparsity / smoothness penalties** (total variation can be applied to **absolute amplitude** \(|\mathbf{x}|\) to match salience-side definitions used with the pooler), and the **pooling bottleneck**—not from an explicit wavelet or other hand-picked transform. Comparisons to wavelets are **informal** (localized bumps), as in [philosophy.md](philosophy.md).

## Motivation

The current decoder training pipeline (see [decoder.md](decoder.md)) supervises bucket-wise recovery of **source indices** with **absolute-amplitude–weighted** BCE (weights use \(|x_i|\); table tokens still carry **signed** amplitude). That isolates “can the transformer invert the pooled table?” A natural next step is to ask: **how much information about a low-dimensional or structured latent survives** the bottleneck \(n \mapsto C \mapsto n\) when the **input distribution** is not hand-designed peaks but **samples from a prior** (e.g. Gaussian) transformed by a **learned** flow into peak-like waveforms.

The **Gaussian information bottleneck** (and related variational IB) formalizes a tradeoff: compress a representation \(T\) of input \(X\) while preserving a target \(Y\) (here, the Gaussian latent or another supervised signal). A typical surrogate objective combines prediction error with a penalty on **information** \(I(X;T)\) or uses variational bounds on \(I(T;Y)\).

Rough end-to-end picture:

1. **Latent:** \(z \sim \mathcal{N}(0, I)\) (or a structured Gaussian).
2. **Flow matching / flow:** \(x = f_\theta(z)\) (or continuous-time flow matching giving \(f_\theta\)), producing **peaks** in \(\mathbb{R}^n\) analogous to the synthetic dataset in `experiments/train_decoder.py`.
3. **Pooler:** \(m = \mathrm{LPAP}(x)\) (table values, DIB, etc.; **LPAP** = linear-probing-based amplitude pooling); **decoder:** \(\hat{\ell} = \mathrm{Dec}(m)\) (e.g. logits over positions or reconstructed amplitudes).
4. **Reconstruction toward Gaussian:** build \(\hat{z}\) or a distribution \(q(\hat{z} \mid \hat{\ell}, m, \ldots)\).
5. **Loss:** combine **reconstruction** (e.g. \(\lVert z - \hat{z}\rVert^2\)) with an **MI-related** term between **bottleneck representation** and **\(z\)**, or a variational bound that penalizes \(I(X;T)\).

**Pretraining / fine-tuning:** Pretrain the **flow** (and optionally a simple head) on **synthetic peaks** matching the current pooler pipeline so \(f_\theta\) starts in a good region of peak shapes. Then **jointly fine-tune** flow + decoder (and, if needed, a small “latent head”) on the **GIB-style** objective.

**Reflow / few-step unrolling:** Flow matching often yields a **single-step** or **few-step** sampler after distillation (“reflow,” consistency models, shortcut models). Unrolling a **short** integration is attractive here for speed and differentiability through fewer operations; the same idea applies if you distill the **Gaussian \(\to\) peaks** map to a small network.

---

## Is this worth exploring?

**Yes, as an exploratory line:** it connects representation learning, generative modeling, and your existing pooler+decoder stack. You already have a **differentiable decoder** and a **deterministic** (or fixed-seed) pooler implementation for analysis.

**Caveats:** the experimental design must separate **(a)** “the decoder recovers indices / waveforms” from **(b)** “the bottleneck actually limits information in a measurable way.” Without careful MI estimation or controlled capacity, runs can collapse to “good reconstruction + unclear IB effect.”

---

## Hurdles to be mindful of

### 1. Identifiability and the role of \(z\)

A powerful flow can make **many** \(z\) map to similar peaks after thresholding or pooling. **\(z\)** may not be uniquely recoverable from **pooler output alone**. The GIB objective should assume **partial** information, or use **stochastic encoders** \(q(z \mid x)\) so “information retained” is well-defined.

### 2. Mutual information is hard to estimate

Direct computation of **mutual information** between high-dimensional variables is intractable. Common stand-ins:

- **Variational IB (VIB):** KL terms that bound \(I(X;T)\) when \(T\) is parameterized as \(q(t \mid x)\).
- **Contrastive / InfoNCE** between **bottleneck embeddings** and **\(z\)** (or slices of \(z\)).
- **MINE / NWJ** and related critics (unstable; need careful baselines and critics).
- **Linear Gaussian IB** in a **probe subspace:** project representations to low dimension and use **closed-form** MI under Gaussian assumptions (only a sanity check, not a full story).

For a first pass, a **reconstruction loss** \(\mathbb{E}[\lVert z - \hat{z}\rVert^2]\) plus an explicit **capacity** constraint on an intermediate representation is often more reliable than claiming “we optimized true MI.”

### 3. Differentiability of the discrete pooler

The **Python/Triton** linear-probing-based amplitude pooling path may be **non-differentiable** (argmax-style winners, discrete scatter). Options:

- **Straight-through** or **soft relaxations** of winners (high engineering cost).
- **Train the decoder** on **hard** pooler outputs and **detach** the pooler for the flow (two-stage), or differentiate only through **decoder + reconstruction head**.
- **Surrogate gradients** only where you need them for the flow.

Clarify early whether the experiment needs **end-to-end** gradients through the pooler or a **stochastic / relaxed** pooling.

### 4. Alignment: what is \(Y\) in the IB?

Pick one primary story:

- **\(Y = z\)** (latent): IB asks how much **table/decoder state** retains about **\(z\)** under a compression penalty.
- **\(Y = x\)** (full peak vector): closer to classical reconstruction, with **\(z\)** entering only through the flow.

Mixing both is fine if the loss weights are explicit.

### 5. Scale and variance of losses

BCE on logits, flow matching noise prediction, and **MI surrogates** live on different scales. **Gradient balancing**, **stop-grad** on some branches, and **warmup** schedules help avoid one term dominating.

---

## From decoder output back to a “Gaussian-like” sample

The decoder currently maps **table tokens** \(\to\) **logits** \([B, C, n]\). A Gaussian lives in \(\mathbb{R}^{d_z}\), not in “position logits.” You need an explicit **bridge**. Options (combinable):

### A. Inverse flow on a reconstructed \(\hat{x} \in \mathbb{R}^n\)

1. Build a **full-length estimate** \(\hat{x}\) from decoder outputs, e.g.:

   - **Hard:** `reconstruct()`-style scatter of max logits (see `RHDecoder.reconstruct` in `decoder.py`), or argmax per bucket mapped to amplitudes.
   - **Soft:** weighted combination of basis vectors using **softmax(logits)** over \(n\) per bucket, then **aggregate** buckets (e.g. sum/max where the pooler is injective in your regime — this is **ill-defined** where buckets collide; you may need **the same** deterministic aggregation you use at train time).

2. Apply the **inverse flow** \( \hat{z} = f_\theta^{-1}(\hat{x}) \) (closed-form for invertible flows, or approximate for continuous normalizing flows).

**Pros:** Uses the **same** \(f_\theta\) for forward and backward. **Cons:** \(\hat{x}\) from bucket logits may be **ambiguous**; inverse can amplify noise.

### B. Direct latent head (recommended for prototyping)

Add a small network \(g_\psi\) that maps **pooled decoder state** (e.g. mean over buckets of last-layer activations, or flatten of last hidden \([B,C,d]\)) to \(\hat{z}\):

\[
\hat{z} = g_\psi(\mathrm{Pool}(\mathrm{Dec}_{\mathrm{hidden}}(m))).
\]

Train with \(\lVert z - \hat{z}\rVert^2\) (and optionally adversarial / perceptual terms on \(\hat{x} = f_\theta(\hat{z})\)).

**Pros:** Simple, stable gradients. **Cons:** Does not by itself prove “inversion through the bottleneck” unless you **constrain** \(g_\psi\) or regularize **information** in the pooled vector.

### C. Variational encoder alongside the flow

Define \(q_\phi(z \mid x)\) (encoder) and keep \(p_\theta(x \mid z) = \delta(x - f_\theta(z))\) or a thin noise model. The **bottleneck** is then **table \(m\)** or **decoder features**; you add a term like \(\mathbb{E}_{q_\phi}[\log p_\psi(z \mid m)] - \mathrm{KL}(q_\phi(z\mid x) \| p(z))\). This is classic **VIB** flavor with the pooler in the middle.

### D. Contrastive MI without full \(\hat{z}\)

Pair **\(z\)** with **embeddings** \(e(m)\) from the decoder trunk; optimize **InfoNCE** so **\(e(m)\)** predicts **\(z\)** (or a projection of \(z\)). This targets **\(I(z; e)\)** without claiming invertible reconstruction.

---

## Suggested phased experiment

1. **Freeze pooler + decoder** (or train decoder as today); **pretrain flow** \(f_\theta\) so \(f_\theta(z)\) matches **synthetic peak statistics** (flow matching or moment matching).
2. Add **\(\hat{z}\)** via **(B)** or **(A)**; minimize \(\lVert z - \hat{z}\rVert^2\) with **small** auxiliary loss on \(\hat{x} = f_\theta(\hat{z})\) vs \(x\) to keep geometry sane.
3. Introduce **capacity** on an intermediate representation (noise in VIB, \(\beta\) on KL, or bottleneck width).
4. Optionally replace / augment reconstruction with **InfoNCE** between **\(z\)** and **pooled features**.
5. If needed, **distill** the flow to **few-step reflow** and repeat with the same outer losses.

---

## Related reading (topics, not citations)

- Information bottleneck (original and variational forms).
- Flow matching and **reflow** / shortcut distillation.
- Contrastive predictive coding / InfoNCE as MI lower bounds.
- Straight-through estimators for non-differentiable pooling (if you pursue end-to-end through the discrete pooler).

---

## Summary

The proposed pipeline—**Gaussian \(\to\) flow-matched peaks \(\to\) linear-probing-based amplitude pooling \(\to\) decoder \(\to\) latent reconstruction + MI-style loss**—is **explorable** as a research thread, especially if you start with a **direct \(\hat{z}\) head** or **inverse flow on \(\hat{x}\)** and treat **true MI** as a long-term goal via variational or contrastive surrogates. The main **hurdles** are **MI estimation**, **differentiability / discreteness of the pooler**, **identifiability of \(z\) through the bottleneck**, and **loss balancing**. Documenting explicit choices for \(Y\) (latent vs waveform) and for the **bridge** from logits to \(\mathbb{R}^n\) or \(\mathbb{R}^{d_z}\) will keep experiments interpretable.
