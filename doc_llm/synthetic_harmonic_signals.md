# Synthetic harmonic training signals

This document records the **pseudo-harmonic peak generator** used for decoder training data (implemented in **`experiments/train_decoder.py`** as `SyntheticRHDataset`). Treating the normalized index \(x\) as living on a **circle** (periodic boundary) matches torus-like routing intuition and avoids privileging endpoints.

## Signal definition

The waveform is a **weighted sum of pseudo-harmonics**, each with its own envelope sharpness \(\exp(a_k)\), plus **random phases** and **Gaussian amplitudes** whose scale decays with \(k\):

$$
s(x)=\sum_{k=1}^{K}\alpha_{k}\left(1-\operatorname{abs}\left(\sin\left(k\pi x+\phi_{k}\right)\right)\right)^{\exp\left(a_{k}\right)}
$$

Illustrative shape (coefficients \(\alpha_k\), phases \(\phi_k\), and sharpnesses \(a_k\) are random in production—here \(a_1,a_2,a_3\) are only placeholders):

$$
\left(1-\operatorname{abs}\left(\sin\left(\pi x-0.3\right)\right)\right)^{\exp\left(a_{1}\right)}-0.7\left(1-\operatorname{abs}\left(\sin\left(2\pi x+0.2\right)\right)\right)^{\exp\left(a_{2}\right)}+0.4\left(1-\operatorname{abs}\left(\sin\left(3\pi x-0.7\right)\right)\right)^{\exp\left(a_{3}\right)}+\cdots
$$

## Parameters and randomness (per row of a batch chunk)

- **Sharpness** \(a_k\sim\mathcal{U}(1,5)\) **independently per harmonic** \(k\). The exponent \(\exp(a_k)\) sharpens peaks inside that harmonic’s envelope.
- **Phases** \(\phi_{k}\): independent draws per harmonic (e.g. \(\mathcal{U}(-\pi,\pi]\)), so peak locations vary while keeping the same ring geometry.
- **Amplitudes** \(\alpha_k\): **Gaussian** with stddev **\(\sigma_k=\gamma^{\,k}\)** (fixed \(\gamma\in(0,1)\)). Draw \(Z_k\sim\mathcal{N}(0,1)\) and set \(\alpha_k=\sigma_k\,Z_k\). Constructor hyperparameters: `harmonic_decay` (\(\gamma\)), `harmonic_amp_threshold` (\(\tau\)), `max_harmonics`.
- **Truncation**: stop before harmonic \(k\) when **\(\sigma_k=\gamma^{\,k}<\tau\)** (e.g. \(\tau=0.1\)), so no term is added for that \(k\) or higher.

## Implementation

- **Chunk-wise** generation: `t = \texttt{linspace}(0,1,n)` expanded to `(chunk_size, n)`; for each harmonic, accumulate `alpha_k * envelope` in **float32**; then **per-position sign flips** (as in the current script) for additional variety.
- **Reference code path:** `SyntheticRHDataset` in **`experiments/train_decoder.py`**.

## Toward joint surrogate + decoder pretraining

An intended next stage is to **pretrain surrogate and decoder together** on these (or related) signals: the **discrete LPAP** operator acts as a **latent regularizer** on the surrogate’s **weighted BCE** (teacher / bottleneck alignment), while the **decoder** trains with a simple **weighted BCE** driven by the **surrogate output**—keeping the discrete pooler as a structured prior rather than the only trainable forward path. Details belong in training scripts and experiment configs as that line of work lands; see **[experiment_plan_cifar_flow_rh.md](experiment_plan_cifar_flow_rh.md)** and **[philosophy.md](philosophy.md)** for surrogate-first framing.
