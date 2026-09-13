"""Make an fp16 copy of an extract_z.py output directory (sequential read/write).

    python learnable_eps/convert_latents_fp16.py \
        --src /mnt/aisha/ahlee-rae --dst /home/ahlee/latents_fp16

Why: the fp32 shards (968 GB) live on a spinning disk that reads at ~108 MB/s
sequentially, so even a perfectly sequential loader needs 2.5 h per epoch,
slower than three GPUs. Halving the bytes (and moving them to a faster array)
makes the loader disappear from the critical path. Values are rounded to
fp16 (relative error ~5e-4 for these O(1) latents); labels/global_index/
progress/meta are copied, with meta['dtype'] set to 'fp16'.

Shards are streamed in --chunk rows at a time, each shard verified by row
count, and the script is idempotent: a shard whose fp16 file already has the
full size and a '.done' marker is skipped.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

NUMPY_DTYPE = {"fp16": np.float16, "bf16": np.float32, "fp32": np.float32}


def convert_shard(src: Path, dst: Path, n: int, latent_shape, src_dtype, chunk: int) -> None:
    done = dst.with_suffix(".done")
    if done.exists():
        print(f"  skip {src.name} (done)", flush=True)
        return
    mm_in = np.memmap(src, dtype=src_dtype, mode="r", shape=(n, *latent_shape))
    mm_out = np.memmap(dst, dtype=np.float16, mode="w+", shape=(n, *latent_shape))
    t0 = time.time()
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        mm_out[s:e] = np.asarray(mm_in[s:e]).astype(np.float16)
        if (s // chunk) % 20 == 0:
            rate = (e * mm_in.dtype.itemsize * int(np.prod(latent_shape))) / max(time.time() - t0, 1e-6) / 1e6
            print(f"  {src.name}: {e}/{n} rows  ({rate:.0f} MB/s read)", flush=True)
    mm_out.flush()
    del mm_out, mm_in
    done.touch()
    print(f"  {src.name}: done in {time.time() - t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--chunk", type=int, default=512, help="rows per read (512 rows = 400 MB fp32)")
    args = ap.parse_args()
    src_root, dst_root = Path(args.src), Path(args.dst)

    groups = sorted(src_root.glob("group*/meta.json"))
    if not groups:
        groups = [src_root / "meta.json"]
    for meta_path in groups:
        gsrc = meta_path.parent
        gdst = dst_root / gsrc.name if gsrc != src_root else dst_root
        gdst.mkdir(parents=True, exist_ok=True)
        meta = json.load(open(meta_path))
        if meta["dtype"] not in ("fp32", "bf16"):
            raise ValueError(f"{gsrc}: dtype {meta['dtype']} -- only fp32/bf16 sources are converted")
        latent_shape = tuple(meta["latent_shape"])
        print(f"== {gsrc} -> {gdst}", flush=True)
        for r in range(int(meta["world_size"])):
            lab = gsrc / f"labels_rank{r:03d}.npy"
            if not lab.exists():
                raise FileNotFoundError(f"{lab} missing: run check_latents.py first")
            n = int(np.load(lab, mmap_mode="r").shape[0])
            convert_shard(gsrc / f"latents_rank{r:03d}.dat", gdst / f"latents_rank{r:03d}.dat",
                          n, latent_shape, NUMPY_DTYPE[meta["dtype"]], args.chunk)
            for name in (f"labels_rank{r:03d}.npy", f"global_index_rank{r:03d}.npy", f"progress_rank{r:03d}.json"):
                if (gsrc / name).exists():
                    shutil.copy2(gsrc / name, gdst / name)
        meta_out = dict(meta)
        meta_out["dtype"] = "fp16"
        meta_out["converted_from"] = {"path": str(gsrc), "dtype": meta["dtype"]}
        with open(gdst / "meta.json", "w") as f:
            json.dump(meta_out, f, indent=2)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    sys.exit(main())
