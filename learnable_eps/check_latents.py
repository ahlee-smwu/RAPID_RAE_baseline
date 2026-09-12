"""Pre-flight check for an extract_z.py output directory.

    python learnable_eps/check_latents.py --latent-path /mnt/aisha/ahlee-rae

Reports, per group and per rank, whether a shard is complete -- and exits
non-zero if anything is missing, so it can gate a training launch.

Why a .dat file is not evidence of anything
-------------------------------------------
extract_z.py allocates ``latents_rank{R}.dat`` at its FULL final size before
the encode loop starts (``np.memmap(..., mode="w+")``), and only writes
``labels_rank{R}.npy`` / ``global_index_rank{R}.npy`` after the loop finishes.
So a rank that dies early leaves a full-size .dat holding mostly zeros, with
no labels beside it. Neither the presence of the .dat nor its size tells you
whether that rank ever ran; only the labels file does.

``progress_rank{R}.json`` is written every ~20 batches and rewritten at the end
with ``"completed": true``, so it gives the finer-grained picture: absent means
the rank died in its first few batches (or never started).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

NUMPY_DTYPE = {"fp16": np.float16, "bf16": np.float32, "fp32": np.float32, "int8": np.int8}


def discover_groups(latent_dir: Path):
    group_metas = sorted(latent_dir.glob("group*/meta.json"))
    if group_metas:
        return [m.parent for m in group_metas]
    if (latent_dir / "meta.json").exists():
        return [latent_dir]
    raise FileNotFoundError(
        f"No meta.json under {latent_dir} or {latent_dir}/group*/ -- "
        f"is this really an extract_z.py --out-dir?"
    )


def check_group(gdir: Path):
    """Returns (rows, ok, n_complete_samples)."""
    with open(gdir / "meta.json") as f:
        meta = json.load(f)

    latent_shape = tuple(meta["latent_shape"])
    dtype = meta["dtype"]
    world_size = int(meta["world_size"])
    itemsize = np.dtype(NUMPY_DTYPE[dtype]).itemsize
    bytes_per_sample = int(np.prod(latent_shape)) * itemsize

    rows = []
    ok = True
    total = 0
    for r in range(world_size):
        dat = gdir / f"latents_rank{r:03d}.dat"
        lab = gdir / f"labels_rank{r:03d}.npy"
        idx = gdir / f"global_index_rank{r:03d}.npy"
        prog = gdir / f"progress_rank{r:03d}.json"

        alloc = dat.stat().st_size // bytes_per_sample if dat.exists() else 0
        n_lab = int(np.load(lab, mmap_mode="r").shape[0]) if lab.exists() else None

        done = None
        seen = None
        if prog.exists():
            try:
                p = json.load(open(prog))
                done = bool(p.get("completed", False))
                seen = int(p.get("write_ptr", 0))
            except Exception:
                pass

        if not dat.exists():
            status, detail = "MISSING", "no latents .dat at all"
            ok = False
        elif not lab.exists() or not idx.exists():
            missing = [n for n, e in (("labels", lab.exists()), ("global_index", idx.exists())) if not e]
            if prog.exists():
                detail = f"died after ~{seen}/{alloc} images; missing {', '.join(missing)}"
            else:
                detail = f"never got past its first batches; missing {', '.join(missing)}"
            status = "INCOMPLETE"
            ok = False
        elif n_lab != alloc:
            status = "MISMATCH"
            detail = f"labels have {n_lab} rows but .dat was allocated for {alloc}"
            ok = False
        elif done is False:
            status = "SUSPECT"
            detail = f"labels written but progress says not completed ({seen}/{alloc})"
            ok = False
        else:
            status = "ok"
            detail = f"{n_lab} images"
            total += n_lab

        rows.append((r, status, detail))

    if ok and total != int(meta["total_samples"]):
        rows.append(("-", "MISMATCH",
                     f"ranks sum to {total} but meta.json says {meta['total_samples']}"))
        ok = False

    return meta, rows, ok, total


def main():
    ap = argparse.ArgumentParser(description="Verify an extract_z.py output directory.")
    ap.add_argument("--latent-path", required=True)
    args = ap.parse_args()

    root = Path(args.latent_path)
    groups = discover_groups(root)
    print(f"{root}: {len(groups)} group(s)\n")

    all_ok = True
    grand = 0
    broken = []
    for gdir in groups:
        meta, rows, ok, total = check_group(gdir)
        cg = meta.get("class_group") or {}
        rng = cg.get("class_range")
        label = f"{gdir.name}" + (f"  classes [{rng[0]}, {rng[1]})" if rng else "")
        print(f"── {label}   world_size={meta['world_size']} dtype={meta['dtype']} "
              f"latent_shape={tuple(meta['latent_shape'])}")
        for r, status, detail in rows:
            mark = "  " if status == "ok" else "!!"
            print(f"   {mark} rank{r if r == '-' else f'{r:03d}'}  {status:11s} {detail}")
        print(f"   -> {'OK' if ok else 'NEEDS RE-EXTRACTION'}   ({total} usable images)\n")
        all_ok &= ok
        grand += total
        if not ok:
            broken.append((gdir, cg))

    print("=" * 66)
    if all_ok:
        print(f"ALL GROUPS COMPLETE — {grand} images total")
        return 0

    print(f"INCOMPLETE — {grand} images usable, but these groups must be re-extracted:")
    for gdir, cg in broken:
        gi = cg.get("group_idx")
        cpg = cg.get("classes_per_group")
        print(f"  {gdir}")
        if gi is not None:
            print(f"     torchrun --standalone --nnodes=1 --nproc_per_node=<NGPU> \\")
            print(f"       learnable_eps/extract_z.py --out-dir {gdir.parent} \\")
            print(f"       --classes-per-group {cpg} --group-idx {gi}  <plus the original flags>")
    print()
    print("Re-run the whole group, not just the broken rank: the per-rank split comes")
    print("from a DistributedSampler over the group, so it only reproduces if the same")
    print("world_size runs together. Use the SAME --nproc_per_node as the original run")
    print("(meta.json world_size above), or every rank's slice changes.")
    print("Training on the directory as-is would silently feed zero-filled latents.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
