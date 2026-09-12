# RAPID adaptive prior — experiment code

All experiment code lives under `learnable_eps/`. **`src/` is byte-identical to
the published RAE baseline** — `learnable_eps/verify_port.py` check 0 enforces
this by grepping `src/` for any prior code. That is what makes the
`prior.enable: false` arm a genuine control.

```
learnable_eps/
  rapid_prior/          the prior itself
    plans.py            gmm_weight_t, plan_gmm_adaptive, plan_gmm_const  (free functions)
    losses.py           training_losses_gmm                              (free function)
    gmm_prior.py        GMMPrior, wrap_sampler_with_prior, lowpass_avg
  convert_gmm.py        gmm_fit.py output  ->  prior checkpoint   ← start here
  fit_gmm_rae.py        fallback: fit from scratch, encoding on the fly
  train.py              stage-2 training on pre-extracted latents
  sample_ddp.py         prior-aware distributed sampling
  verify_port.py        Gate A, no GPU needed
  configs/              *_baseline.yaml (prior off) and *_rapid.yaml (prior on)
```

The path/transport extensions are **free functions**, not methods bolted onto
`ICPlan` / `Transport`, precisely so the baseline files need no edit.

---

## 1. The time axis — the one thing to get right

| | LightningDiT / RAPID | RAE (this repo) |
|---|---|---|
| `compute_alpha_t(t)` (data) | `t` | `1 - t` |
| `compute_sigma_t(t)` (noise) | `1 - t` | `t` |
| **noise end** | **t = 0** | **t = 1** |
| `ut` | `x1 - x0` | `x0 - x1` |
| latent | 32×16×16 = 8,192-D | 768×16×16 = **196,608-D** |
| timestep shift | ~1 | `sqrt(196608/4096)` = **6.93** |

RAPID's schedule `w(s) = q0·exp(-decay_alpha·s)` copied verbatim would inject
the GMM at the **data** end. Under `s = 1 - t`:

```
w(t) = q0 * exp(-decay_alpha * (1 - t))      # w(t=1) = q0, at the noise end
x0_blended = w * x0_gmm + (1 - w) * eps
xt = (1 - t) * x1 + t * x0_blended
ut = x0_blended - x1
```

`verify_port.py` check 3 pins it: `xt` identical under `s = 1 - t`, `ut`
exactly sign-flipped. **Do not** revert `w(t)` to `q0*exp(-alpha*t)`.

---

## 2. Reusing the GMM you already fitted

`gmm_fit.py` has already done the expensive part. **Do not refit** — convert:

```bash
python learnable_eps/convert_gmm.py \
    --gmm gmm_imagenet_256/20_diag/gmm_clusters.pkl \
    --out gmm_prior_k20 \
    --latent-shape 768 16 16
```

Handles both `--use-pca false` (covs already data-space) and `--use-pca true`
(restores data-space diagonals as `v_k @ U_c²`, the same table RAPID's
`precompute_sigma_scale` built). Output:

```
gmm_prior_k20/gmm_rae.pkl        shared basis + projected GMM + scalars
gmm_prior_k20/means_fp16.npy     (C, K, D) exact full-space means
gmm_prior_k20/vars_fp16.npy      (C, K, D) exact full-space diagonal variances
```

**RAM:** `pickle.load` is monolithic, so this needs roughly the pkl's own size
in RAM (~63 GB for K=20, `--use-pca false`, sklearn's float64). Run it on a
big-RAM node. `labels` and `responsibilities` are dropped immediately after
load and each class streams straight to the memmaps.

**Disk:** two ~7.9 GB memmaps at K=20. **Local SSD**, not NFS.

### Why the split (PCA for assignment, full space for the draw)

- The **posterior** must score all K clusters of a class at once. In full space
  that is `B·K·D·2B` of reads per step — 500 MB at B=64, K=20. Unusable. So it
  runs in a shared 256-D PCA space whose tensors stay GPU-resident. This is
  exactly why RAPID had a PCA: its `get_cluster_gmm` scores in PCA space too.
- The **draw** touches only the selected cluster — two `(B, D)` gathers, ~50 MB
  per step. Cheap, so it stays in **full space and is exact**.

Nothing about the sampled `x0_gmm` is approximated. Only the choice of `k` goes
through the projection.

### A caveat on K=20

`gmm_fit.py` defaults to `--num-clusters 20`. With ~1300 images/class that is
~65 samples per component to estimate a diagonal covariance over 196,608 dims.
`reg_covar=1e-6` keeps it non-singular, but the estimate is poorly conditioned.
The converter warns. If the prior underperforms, refitting at **K=10** is the
first thing to try — and `fit_gmm_rae.py` can do that without re-extracting.

---

## 3. Running it

### Check the latent extraction first (no GPU, seconds)

```bash
python learnable_eps/check_latents.py --latent-path /mnt/aisha/ahlee-rae
```

`extract_z.py` preallocates `latents_rank{R}.dat` at full size **before** its
encode loop and writes `labels_rank{R}.npy` only **after** it finishes. A rank
that dies early therefore leaves a full-size, zero-filled `.dat` with no labels
beside it — neither the file's presence nor its size tells you anything. This
script checks every group/rank and names the ones that need re-extracting.
Re-run the **whole group** with the same `--nproc_per_node` as the original:
the per-rank split comes from a DistributedSampler, so it only reproduces when
the same world_size runs together.

### Gate A (no GPU, ~1 s)

```bash
python learnable_eps/verify_port.py
# cross-check against the original implementation, if you have it:
python learnable_eps/verify_port.py --old-path <RAPID_repo>/transport/path.py
```

Must print `ALL CHECKS PASSED`. If check 3 fails, **stop**.

### Training

`learnable_eps/train.py` reads **pre-extracted latents** (`extract_z.py`
output), not images — that is the one structural difference from
`src/train.py`, and the reason `--latent-path` replaces `--data-path`.

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=8 \
  learnable_eps/train.py \
  --config learnable_eps/configs/DiTDH-S_DINOv2-B_rapid.yaml \
  --latent-path /mnt/disk1/ahlee-rae \
  --results-dir ckpts/stage2 \
  --precision fp32 \
  --compile \
  --global-seed 42
```

**Run name** resolves as `--experiment-name` > `training.experiment_name` in
the config > `$EXPERIMENT_NAME`, and decides `<results-dir>/<name>/`. Each
config ships with a name, so no environment variable is needed. Reusing a name
**auto-resumes** that directory, so give every variant its own:

```bash
# same config, different variant -> different name, or it continues the old run
--experiment-name exp2_rapid_q0.3_S
```

Three gotchas that are not in the README at repo root:

1. **`--compile` is mandatory** — `train.py` raises `NotImplementedError` without it.
2. **A run name is mandatory** (config, flag, or env — see above), and reusing
   one auto-resumes. Use a fresh name per experiment.
3. **The `eval:` block's paths are opened at startup.** If
   `data/imagenet/val/` does not exist the run dies immediately; fix the paths
   or delete the block.

Expected rank-0 log:

```
[RAPID prior] classes=1000 K=20 pca_dim=256 ... exact_vars=True global_sigma_scale=...
[RAPID] enabled | schedule=exp q0=0.5 decay_alpha=1.0
[RAPID prior] var(mu_lpf)=... var(x0_gmm)=... var(z_init)=...
```

**Record `var(z_init)`.** Synthetic validation gives ≈0.51 at q0=0.5 (baseline
`N(0,I)` = 1.0), because the blend is linear, not variance-preserving:
`Var = q0²·Var(x0_gmm) + (1-q0)²`. Note this means **smaller q0 gives larger
variance** (0.58 at q0=0.3). Below ~0.3, revisit q0.

### Sampling

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=8 \
  learnable_eps/sample_ddp.py --config <same config with prior block> ...
```

A model trained with the prior **must** be sampled through this wrapper.
Sampling it from `N(0,I)` silently mismatches training.

---

## 4. Experiment matrix

Every variant below needs its own `experiment_name`.

| # | Setting | Purpose | Required |
|---|---|---|---|
| 0 | `configs/*_baseline.yaml` | same-codebase control | **yes** |
| 1 | `q0=0.5, decay_alpha=1.0, schedule=exp` | default | **yes** |
| 2 | `q0=0.3` | mitigate mode collapse (§5) | **yes** |
| 3 | `schedule=const, q0=0.5` | exact-velocity control | **yes** |
| 4 | `stochastic_assign: true` | intra-class diversity → recall | recommended |
| 5 | ODE vs SDE at inference (no retrain) | −22.5% FID on LSUN | **yes** |
| 6 | `lpf_alpha ∈ {0.7, 1.0}` | LPF strength | recommended |

Runs 0 and 1 need the **same epochs, seed, and step count**. Run 5 is two
sampling passes over one checkpoint — best value per GPU-hour. Tight budget:
0/1/2/5.

Metrics: FID, Precision, Recall, Density, Coverage, plus **FID vs steps
(5/10/20/40)** — the few-step curve is the headline figure.

### `plan_gmm_const` is not decoration

`ut = x0_blended - x1` is the *exact* conditional velocity only when `w` is
constant. `plan_gmm_adaptive` omits the `t·dw/dt·(x0_gmm - eps)` term — the
same approximation the original makes. `verify_port.py` check 5 measures the
gap rather than hiding it. Run experiment 3; a reviewer will ask.

---

## 5. Traps

1. **The timestep shift amplifies the injection.** With s≈6.93, ~87% of
   training samples land at t>0.5 where `w(t)≈q0` (measured `E[w(t)]=0.41`
   against `q0=0.5`; check 8). The GMM is injected far more strongly than in
   LightningDiT, so the precision↑/recall↓ bias is structurally worse. This is
   why `q0=0.3` is required, not optional.
2. **RAE latents already separate classes.** DINOv2 features are
   class-discriminative before any prior, so a class-conditional GMM can only
   add *intra-class* structure. The gain over VA-VAE may be small — that is a
   result, not a defect. "The better the latent space, the less a prior buys"
   is a defensible, testable claim.
3. **No flip augmentation on the latent path.** `extract_z.py` deliberately
   omits `RandomHorizontalFlip` (its cache is fixed), so training from latents
   has no flip augmentation, unlike the baseline image path. The GMM is fitted
   on the same cache so prior and data agree — but when comparing against an
   image-path baseline this is a real confound. Keep both arms on the latent
   path.
4. **`means_fp16.npy` / `vars_fp16.npy` I/O.** ~50 MB/step at B=64 in `mmap`
   mode; **local SSD**. `means_device: cpu` is faster with few ranks but
   replicates ~16 GB per rank — never on an 8-rank node.
5. **fp32 is enforced inside `GMMPrior`** (`_fp32()`) because bf16 autocast
   breaks the PCA projection. `train.py` also wraps the call in
   `autocast(enabled=False)`. Do not remove either.
6. **`w(t)` uses the shifted `t`.** `Transport.sample()` returns an already
   shifted `t` and the interpolation coefficients use that value; `w` uses the
   same one, preserving the SNR relationship. Do not substitute pre-shift `t`.

### Never do this at inference

```python
z = z / z.std(...)   # NO
```

It silently cancels the GMM/eps ratio the model was trained on — the direct
cause of the 13th–15th experiment failures. Neither `train.py`'s eval path nor
`sample_ddp.py` renormalises. Keep it that way.

---

## 6. Open items

- **`lpf_alpha`** — training uses the configured value (1.0 for ImageNet).
  `estimate_lpf_alpha_minus3db()` is a **diagnostic only**: it was never
  confirmed whether the thesis derives −3 dB on a power ratio (`R(f)=0.5`) or
  an amplitude ratio (`|H|=1/√2`), which shifts α* by ~0.7×. It implements the
  power convention. Resolve against the thesis before quoting α*.
- **OT matching is not implemented.** Dropped by decision. Note for the record
  that RAPID's `training_losses_learnable_eps2` permutes `x1` without
  permuting `model_kwargs['y']` — harmless on single-class LSUN, but it trains
  ImageNet latents under the wrong label. Any past LightningDiT **ImageNet**
  run that used that path is suspect.
- **CFG path is broken in the baseline** (pre-existing, unrelated):
  `train.py` reads `ys` before assignment, and `src/eval/__init__.py` uses an
  undefined `null_label`. Both only fire when `guidance.scale > 1.0`. Fix
  before running any CFG experiment.

## 7. Dependencies

Beyond `requirements.txt`: `scikit-learn` (GMM fitting). `scipy` is not needed
— OT is not implemented.

## 8. Rollback

Use a `*_baseline.yaml` config, or set `prior.enable: false`.
