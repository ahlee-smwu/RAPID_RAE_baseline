# RUNBOOK — latent repair → GMM repair → training

Copy-paste procedure for the GPU server. Work through it in order; each step
has a **gate** that must pass before the next one.

Set these once per shell:

```bash
cd ~/RAE                      # repo root
git checkout train && git pull origin train

export LAT=/mnt/aisha/ahlee-rae                              # extract_z.py --out-dir
export GMM=gmm_imagenet_256/20_diag/gmm_clusters.pkl         # existing gmm_fit.py output
export PRIOR=gmm_prior_k20                                   # converter output (local SSD!)
```

**Why all this:** an extraction rank died partway (`group001/rank002` had a
`.dat` but no `labels_*.npy`). Two things follow, and both are silent:

- `extract_z.py` preallocates `latents_rank{R}.dat` at **full size before**
  encoding and writes labels **after**, so a dead rank leaves a full-size,
  zero-filled `.dat`. File size proves nothing.
- `gmm_fit.py`'s `build_class_index` does `if not label_path.exists(): continue`,
  so it **skipped that rank without warning**. Every class in that group was
  fitted on a fraction of its images.

---

## Step 1 — check the extraction

```bash
python learnable_eps/check_latents.py --latent-path $LAT
```

Per group and rank: `ok` / `INCOMPLETE` / `MISSING` / `MISMATCH` / `SUSPECT`.

**Gate:** `ALL GROUPS COMPLETE` (exit 0) → skip to Step 3.
Otherwise note the listed groups and continue.

---

## Step 2 — re-extract the broken groups

The script prints the exact command per broken group. It looks like:

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=<WORLD_SIZE> \
  learnable_eps/extract_z.py --out-dir $LAT \
  --classes-per-group <N> --group-idx <G> \
  <the original flags: --data-path, --dtype, --batch-size, ...>
```

Three rules:

1. **`--nproc_per_node` must equal that group's `meta.json` `world_size`**
   (check_latents prints it). Each rank's slice comes from a
   `DistributedSampler` over the group, so a different world size reshuffles
   *every* rank, corrupting the ranks that were already fine.
2. **Re-run the whole group, not just the dead rank.** A single rank cannot
   rejoin an existing group — the other ranks have to be in the process group.
3. **Use the same `--dtype` and `--data-path` as the original.** check_latents
   prints the group's dtype; a mismatch is rejected in Step 1 on re-check.

Confirm the GPUs exist before launching — the original failure was very likely
a launch with more ranks than visible GPUs:

```bash
nvidia-smi -L
python -c "import torch; print(torch.cuda.device_count())"
```

**Gate:** re-run Step 1 until it prints `ALL GROUPS COMPLETE`.

---

## Step 3+4+5 — audit the GMM, and repair only what is stale

Do these as **one command**. It audits first and exits without writing when
nothing is stale, so you pay the (large) pickle load once instead of twice:

```bash
python learnable_eps/gmm_doctor.py --mode repair \
    --gmm $GMM --latent-path $LAT \
    --out gmm_imagenet_256/20_diag/gmm_clusters_repaired.pkl
```

How it decides: `gmm_fit.py` stores `labels[cls]`, one entry per sample it
actually saw. Comparing `len(labels[cls])` against the images that class has in
the repaired shards names exactly the stale classes.

Outcomes:

- **`GMM IS COMPLETE`** → the fit already used all the good latents.
  **Step 5 is skipped; keep using `$GMM`.** This happens if `gmm_fit.py` ran
  *before* the extraction broke, or the broken rank held no images for any
  fitted class.
- **`REPAIRED — N classes`** → only the stale classes were refit, with the
  hyperparameters read back out of the pkl (`num_components`, `cov_type`, PCA)
  via `gmm_fit.py`'s own `process_class`. Healthy classes are copied through
  byte-for-byte. Then:

  ```bash
  export GMM=gmm_imagenet_256/20_diag/gmm_clusters_repaired.pkl
  ```

To look without writing, use `--mode audit` (exit 1 = needs repair).

**RAM:** `pickle.load` is monolithic — roughly the pkl's own size (~63 GB at
K=20, `--use-pca false`, sklearn float64). Run on a big-RAM node.

**Gate:** `gmm_doctor.py --mode audit --gmm $GMM --latent-path $LAT` exits 0.

---

## Step 5b — convert the GMM into a prior checkpoint

```bash
python learnable_eps/convert_gmm.py \
    --gmm $GMM --out $PRIOR --latent-shape 768 16 16
```

Writes `gmm_rae.pkl`, `means_fp16.npy`, `vars_fp16.npy`. **Keep all three
together on local SSD** — training reads two `(B, 196608)` fp16 gathers per
step, and NFS will bottleneck it.

Record the printed `global_sigma_scale` and `explained_variance_ratio`; both
belong in the paper table.

Then point the configs at it:

```bash
sed -i "s|^  ckpt_path: .*|  ckpt_path: $PRIOR/gmm_rae.pkl|" \
    learnable_eps/configs/*_rapid.yaml
grep -n "ckpt_path" learnable_eps/configs/*_rapid.yaml
```

**Gate:**

```bash
python learnable_eps/verify_port.py     # must print ALL CHECKS PASSED (21/21)
```

---

## Step 6 — train on 3 of the 5 GPUs

### 6a. Pick the idle ones

```bash
python learnable_eps/pick_gpus.py --want 3
```

It prints a table to stderr and the selection to stdout. Capture it:

```bash
export CUDA_VISIBLE_DEVICES=$(python learnable_eps/pick_gpus.py --want 3)
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
```

Exits non-zero if fewer than 3 are idle, so a launch script stops rather than
colliding with someone else's job.

Once `CUDA_VISIBLE_DEVICES` is set, torch renumbers the visible GPUs to 0,1,2 —
so **`--nproc_per_node=3` regardless of the physical ids**.

### 6b. Fix the batch size for world_size=3 — REQUIRED

`train.py` asserts `global_batch_size % world_size == 0`, and the shipped
`global_batch_size: 1024` is **not divisible by 3**. Training dies immediately
otherwise. Use 768 (= 3 × 256) and recover the effective batch with gradient
accumulation:

```bash
sed -i -e 's|^  global_batch_size: .*|  global_batch_size: 768|' \
       -e 's|^  grad_accum_steps: .*|  grad_accum_steps: 4|' \
       learnable_eps/configs/DiTDH-S_DINOv2-B_baseline.yaml \
       learnable_eps/configs/DiTDH-S_DINOv2-B_rapid.yaml
grep -n "global_batch_size\|grad_accum_steps" learnable_eps/configs/DiTDH-S*.yaml
```

`micro_batch = global_batch_size / (world_size × grad_accum_steps)`:

| grad_accum_steps | micro batch / GPU | note |
|---|---|---|
| 2 | 128 | fastest, most memory |
| **4** | **64** | **start here (S)** |
| 8 | 32 | for XL, or if 64 OOMs |

**Apply the identical change to baseline and rapid** (the `sed` above does
both). Different effective batch sizes make experiments 0 and 1 incomparable.

### 6c. Run

Experiment 0 — baseline control:

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=3 \
  learnable_eps/train.py \
  --config learnable_eps/configs/DiTDH-S_DINOv2-B_baseline.yaml \
  --latent-path $LAT \
  --results-dir ckpts/stage2 \
  --precision fp32 --compile --global-seed 42
```

Experiment 1 — RAPID prior:

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=3 \
  learnable_eps/train.py \
  --config learnable_eps/configs/DiTDH-S_DINOv2-B_rapid.yaml \
  --latent-path $LAT \
  --results-dir ckpts/stage2 \
  --precision fp32 --compile --global-seed 42
```

Only the config differs. Same seed, same epochs, same batch.

### 6d. Confirm the first ~100 steps

```
[run]  experiment_name=exp1_rapid_q0.5_S -> ckpts/stage2/exp1_rapid_q0.5_S
[data] /mnt/aisha/ahlee-rae: 1281167 samples, N shards across 4 group(s) [...]
[RAPID prior] classes=1000 K=20 pca_dim=256 ... exact_vars=True global_sigma_scale=...
[RAPID] enabled | schedule=exp q0=0.5 decay_alpha=1.0
[RAPID prior] var(mu_lpf)=... var(x0_gmm)=... var(z_init)=...
```

Check, in order:

1. **`[data]` shows every group** and the expected total. A missing group here
   means Step 1 was not clean.
2. **`experiment_name` is new.** Reusing a name **auto-resumes** that
   directory — a variant would silently continue the previous run. Override
   per variant with `--experiment-name exp2_rapid_q0.3_S`.
3. **`var(z_init)` ≈ 0.51** at q0=0.5 (baseline `N(0,I)` is 1.0; the blend is
   linear, so `Var = q0²·Var(x0_gmm) + (1-q0)²`). Record it. Below ~0.3,
   revisit q0.
4. The baseline run must instead log
   `[RAPID] disabled | running the unmodified baseline path.`

---

## Gotchas

| Symptom | Cause |
|---|---|
| `invalid device ordinal` | `--nproc_per_node` > visible GPUs. Use `pick_gpus.py`. |
| `global_batch_size must be divisible by world_size` | Step 6b not applied. |
| `NotImplementedError: ARGS>COMPILE` | `--compile` is mandatory. |
| `FileNotFoundError: .../meta.json` | `--latent-path` must be the **parent** holding `group000/`, not a group. |
| dies at startup on `data/imagenet/val/` | the config's `eval:` block opens its paths at startup. Fix them or delete the block. |
| OOM | raise `grad_accum_steps` (6b), keeping `global_batch_size` fixed. |

Never add `z = z / z.std(...)` anywhere in the inference path — it cancels the
GMM/eps ratio the model was trained on.

---

## One-shot gate check

```bash
python learnable_eps/check_latents.py --latent-path $LAT \
 && python learnable_eps/gmm_doctor.py --mode audit --gmm $GMM --latent-path $LAT \
 && python learnable_eps/verify_port.py \
 && echo "ALL GATES PASSED — safe to train"
```
