# RAPID adaptive prior on the RAE baseline

Port of the RAPID class-conditional GMM prior (validated on LightningDiT /
VA-VAE) onto this repository's RAE (DINOv2-B) stage-2 model.

`prior.enable: false`, or no `prior:` section at all, runs the **unmodified
baseline**. Every change is an added branch; no existing function was altered.

---

## 1. The one thing to get right: the time axis

The two codebases run flow matching in opposite directions.

| | LightningDiT / RAPID | RAE (this repo) |
|---|---|---|
| `compute_alpha_t(t)` (data coeff) | `t` | `1 - t` |
| `compute_sigma_t(t)` (noise coeff) | `1 - t` | `t` |
| **noise end** | **t = 0** | **t = 1** |
| `ut` | `x1 - x0` | `x0 - x1` |
| latent | 32x16x16 = 8,192-D | 768x16x16 = **196,608-D** |
| timestep shift | ~1 | `sqrt(196608/4096)` = **6.93** |
| conditioning | LSUN, single class | ImageNet, 1000 classes |

RAPID's schedule is `w(s) = q0 * exp(-decay_alpha * s)`. Copying it verbatim
would inject the GMM at the **data** end. Under `s = 1 - t` it becomes:

```
w(t) = q0 * exp(-decay_alpha * (1 - t))        # w(t=1) = q0 at the noise end
x0_blended = w * x0_gmm + (1 - w) * eps
xt = (1 - t) * x1 + t * x0_blended
ut = x0_blended - x1
```

`tools/verify_port.py` check 3 pins this numerically: `xt` is identical under
`s = 1 - t` and `ut` is exactly sign-flipped.

**Do not** change `w(t)` back to `q0 * exp(-alpha * t)`.

---

## 2. What changed

| File | Change |
|---|---|
| `src/stage2/transport/path.py` | **+** `ICPlan.gmm_weight_t`, `plan_gmm_adaptive`, `plan_gmm_const` |
| `src/stage2/transport/transport.py` | **+** `Transport.training_losses_gmm`, `_ot_pair_within_class` |
| `src/stage2/gmm_prior.py` | **new** — `GMMPrior`, `lowpass_avg`, `wrap_sampler_with_prior` |
| `src/fit_gmm_rae.py` | **new** — three-stage GMM fitting |
| `src/train.py` | **+** import, `prior_cfg`, prior init + sampler wrap, training-loop branch |
| `src/sample.py`, `src/sample_ddp.py` | **+** same sampler wrap, so standalone inference matches training |
| `tools/verify_port.py` | **new** — Gate A |
| `configs/.../*_rapid.yaml` | **new** — prior-enabled variants |

`src/eval/__init__.py` is **untouched**: it already forwards `y` in
`model_kwargs`, and the wrapper only swaps the initial latent.

### `plan_gmm_const` is not decoration

`ut = x0_blended - x1` is the *exact* conditional velocity only when `w` is
constant. `plan_gmm_adaptive` omits the `t * dw/dt * (x0_gmm - eps)` term —
the same approximation the original makes. `verify_port.py` check 5 measures
the gap (≈0.63 at t=0.4 in the synthetic setup) rather than hiding it. Run
`schedule: const` as an ablation; a reviewer will ask.

### An ImageNet-only bug that did not come across

RAPID's `training_losses_learnable_eps2` does `x1 = x1[col_idx]` after
Hungarian matching but leaves `model_kwargs['y']` alone. On single-class LSUN
that is harmless; on ImageNet it trains latent *i* under a **different class
label**. `_ot_pair_within_class` here returns the permutation and the caller
applies it to `x1` **and** `y` together, and matching never crosses a class
boundary.

> If past LightningDiT **ImageNet** runs used that path, those numbers are
> suspect and worth re-checking.

---

## 3. Storage: why fitting was rewritten

At 196,608-D the original artefacts do not fit on disk:

- per-class PCA bases: 1000 x 256 x 196608 x 4B = **201 GB**
- data-space diag-var table: 1000 x K x 196608 x 4B = **15.7 GB** (K=20)

The port keeps the arithmetic and changes the layout:

- **one shared PCA basis** `U (d, D)`, d=256 → 201 MB
- per-class GMM (means/diag-cov/weights) in PCA space → 40 MB
- data-space diagonal variances restored on the fly as `v_k @ U²` — exactly
  the value the original read from its table (`verify_port.py` check 6)
- **cluster means stay full-space** as an fp16 memmap (K=10 → 3.9 GB).
  Projecting 1000 class means onto a 256-D shared basis would destroy the
  class information the prior exists to carry.
- means are **responsibility-weighted full-space** means, not PCA
  reconstructions, so the discarded complement still contributes.

---

## 4. Running it

### Gate A — code verification (no GPU, ~1 s)

```bash
python tools/verify_port.py
# optional cross-check against the original:
python tools/verify_port.py --old-path <RAPID_repo>/transport/path.py
```

Must print `ALL CHECKS PASSED`. If check 3 fails, **stop**.

### Fit the GMM

```bash
OUT=gmm_out_dinov2b_k10

python src/fit_gmm_rae.py --stage pca --config <cfg> --data-path <imagenet/train> \
    --out $OUT --k 10 --pca-dim 256 --pca-per-class 8

for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$i python src/fit_gmm_rae.py --stage fit --config <cfg> \
      --data-path <imagenet/train> --out $OUT --shard $i --num-shards 4 &
done; wait

python src/fit_gmm_rae.py --stage merge --out $OUT
```

Smoke-test first with `--max-classes 10 --max-per-class 64` on a single shard.

**K = 10 is the default and the ceiling is 20.** ImageNet has ~1300 images per
class; at K=10 each 256-D diagonal Gaussian gets ~130 samples (~260 with
flips), at K=20 only ~65 and the covariance estimate collapses. The K=30 used
for LSUN does not transfer. `--k > 20` is rejected outright.

**Log these** (they belong in the paper): the `pca` stage's latent
`mean`/`std` and explained-variance ratio, and the `merge` stage's
`global_sigma_scale`.

### Train

Use `configs/stage2/training/ImageNet256/DiTDH-XL_DINOv2-B_rapid.yaml` (or the
`-S` variant), pointing `prior.ckpt_path` at `$OUT/gmm_rae.pkl`.

Rank 0 must log three lines early:

```
[RAPID prior] classes=1000 K=10 pca_dim=256 ... global_sigma_scale=...
[RAPID] enabled | schedule=exp q0=0.5 decay_alpha=1.0 ot_within_class=False
[RAPID prior] var(mu_lpf)=... var(x0_gmm)=... var(z_init)=...
```

**Record `var(z_init)`.** Synthetic validation gives ≈0.51 at q0=0.5 (baseline
`N(0,I)` is 1.0), because the blend is linear, not variance-preserving:
`Var = q0²·Var(x0_gmm) + (1-q0)²`. Training and inference share the
configuration so training is consistent, but the effective SNR differs from
baseline and combines with the shifted schedule. If it drops below ~0.3,
revisit `q0`.

### Gate B — confirm the data path (do this once)

This repo trains **on-the-fly**: `src/train.py` calls `z = rae.encode(images)`
on an `ImageFolder`, and `RAE.encode` applies the stage-1 normalisation
internally. `fit_gmm_rae.py` uses the same `rae.encode` and the same
transform, so the two distributions match by construction.

Verify anyway: print `z.mean()/z.std()/z.shape` for one training batch and
compare against the `pca` stage's output. **They must agree to within ~1%.**
If this repo is ever switched to precomputed latents, replace
`fit_gmm_rae.encode_indices` — the GMM must be fitted on exactly the tensor
the diffusion model sees, post-normalisation.

---

## 5. Experiment matrix

| # | Setting | Purpose | Required |
|---|---|---|---|
| 0 | `prior.enable: false` | same-codebase baseline | **yes** |
| 1 | `q0=0.5, decay_alpha=1.0, schedule=exp` | default | **yes** |
| 2 | `q0=0.3` | mitigate mode collapse (§6) | **yes** |
| 3 | `schedule=const, q0=0.5` | exact-velocity control | **yes** |
| 4 | `stochastic_assign: true` | intra-class diversity → recall | recommended |
| 5 | ODE vs SDE at inference (no retrain) | −22.5% FID on LSUN | **yes** |
| 6 | `lpf_alpha ∈ {0.7, 1.0}` | LPF strength sensitivity | recommended |

Runs 0 and 1 must use the **same epochs, seed, and step count**. Run 5 is two
sampling passes over one checkpoint — the best value for the compute. On a
tight budget, do 0/1/2/5.

Metrics: FID, Precision, Recall, Density, Coverage, plus **FID vs steps
(5/10/20/40)** — the few-step curve is the headline figure.

---

## 6. Traps

1. **The timestep shift amplifies the injection.** With s≈6.93, ~87% of
   training samples land at t>0.5, where `w(t)≈q0` (measured `E[w(t)]=0.41`
   against `q0=0.5`; `verify_port.py` check 8). The GMM is injected far more
   strongly than in LightningDiT, so the precision↑/recall↓ bias is
   structurally worse. This is why `q0=0.3` is required, not optional.
2. **RAE latents already separate classes.** DINOv2 features are
   class-discriminative before any prior. A class-conditional GMM can only add
   *intra-class* structure, so the gain over VA-VAE may be small. That is a
   result, not a defect — "the better the latent space, the less a prior
   buys" is a defensible, testable claim.
3. **OT matching** is off by default. It only means something if batches
   deliberately repeat classes.
4. **`means_fp16.npy` I/O.** `mmap` reads ~25 MB/batch at B=64; put it on
   **local SSD**. `means_device: cpu` is faster with few ranks but replicates
   ~4 GB per rank — never use it on an 8-rank node.
5. **fp32 is enforced inside `GMMPrior`** (`_fp32()`), because bf16 autocast
   breaks the PCA projection and the diagonal-variance restore. `train.py`
   also wraps the call in `autocast(enabled=False)`. Do not remove either.
6. **Flip augmentation must match.** Training uses `RandomHorizontalFlip`, so
   the GMM is fitted with `--include-flip 1` (default). Disabling it biases
   posterior assignment.
7. **`w(t)` uses the shifted `t`.** `Transport.sample()` returns an already
   shifted `t`, and the interpolation coefficients use that value; `w` uses
   the same one, which preserves the SNR relationship. Do not substitute the
   pre-shift `t`.

### Never do this at inference

```python
z = z / z.std(...)   # NO
```

Std-normalising the initial latent silently cancels the GMM/eps ratio the
model was trained on. It was the direct cause of the 13th–15th experiment
failures. All three sampling entry points (`train.py` eval, `sample.py`,
`sample_ddp.py`) go through `wrap_sampler_with_prior`, which does not
renormalise — keep it that way.

---

## 7. Open items

- **`lpf_alpha`.** Training uses the configured value (1.0 for ImageNet).
  `estimate_lpf_alpha_minus3db()` is a **diagnostic only**: it was never
  confirmed whether the thesis derives −3 dB on a power ratio (`R(f)=0.5`) or
  an amplitude ratio (`|H|=1/√2`), which shifts α* by ~0.7x. It implements the
  power convention. Resolve against the thesis before quoting α*.
- **Past LightningDiT ImageNet runs** may carry the OT label-shuffling bug
  (§2). Worth re-checking before reusing those numbers.

## 8. Rollback

Set `prior.enable: false` — the code path becomes identical to baseline.
