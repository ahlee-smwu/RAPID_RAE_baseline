"""Pick idle GPUs and print the CUDA_VISIBLE_DEVICES to use.

    python learnable_eps/pick_gpus.py --want 3
    # -> prints e.g.  0,3,4

    # straight into a launch:
    export CUDA_VISIBLE_DEVICES=$(python learnable_eps/pick_gpus.py --want 3)

A GPU counts as idle when its used memory is below --max-used-mb AND its
utilization is below --max-util. Exits non-zero (printing nothing on stdout)
when it cannot find enough, so a launch script stops instead of colliding with
somebody else's job.

Once CUDA_VISIBLE_DEVICES is set, torch renumbers the visible GPUs to
0..N-1, so torchrun always takes --nproc_per_node=<count>, never the physical
ids.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys


def query():
    if shutil.which("nvidia-smi") is None:
        print("nvidia-smi not found", file=sys.stderr)
        raise SystemExit(2)
    out = subprocess.run(
        ["nvidia-smi",
         "--query-gpu=index,memory.used,memory.total,utilization.gpu,name",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    rows = []
    for line in out.splitlines():
        idx, used, total, util, name = [p.strip() for p in line.split(",", 4)]
        rows.append({"index": int(idx), "used": int(used), "total": int(total),
                     "util": int(util), "name": name})
    return rows


def main():
    ap = argparse.ArgumentParser(description="Pick idle GPUs.")
    ap.add_argument("--want", type=int, default=3, help="How many GPUs are needed.")
    ap.add_argument("--max-used-mb", type=int, default=1024,
                    help="A GPU with more memory in use than this is considered busy.")
    ap.add_argument("--max-util", type=int, default=10,
                    help="A GPU busier than this percent is considered busy.")
    ap.add_argument("--show", action="store_true", help="Print the full table to stderr.")
    args = ap.parse_args()

    rows = query()
    if args.show or True:
        print(f"{'gpu':>3s} {'used/total MB':>16s} {'util%':>6s}  {'state':8s} name", file=sys.stderr)
        for r in rows:
            idle = r["used"] <= args.max_used_mb and r["util"] <= args.max_util
            print(f"{r['index']:>3d} {r['used']:>7d}/{r['total']:<8d} {r['util']:>6d}  "
                  f"{'IDLE' if idle else 'busy':8s} {r['name']}", file=sys.stderr)

    idle = [r["index"] for r in rows
            if r["used"] <= args.max_used_mb and r["util"] <= args.max_util]

    if len(idle) < args.want:
        print(f"\nOnly {len(idle)} idle GPU(s) ({idle}), need {args.want}. "
              f"Wait, or lower --want.", file=sys.stderr)
        return 1

    chosen = idle[: args.want]
    print(f"\nUsing GPUs {chosen} -> --nproc_per_node={len(chosen)}", file=sys.stderr)
    print(",".join(str(i) for i in chosen))   # stdout: the only machine-readable line
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
