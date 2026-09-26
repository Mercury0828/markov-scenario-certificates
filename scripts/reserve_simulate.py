"""Monte Carlo runs of the reserve problem (Section 5.2, Table 2).

Four settings (rows of Table 2), each with 20 chunks of 5000 independent stationary trajectories. Chunk c of the setting in
row s + 1 uses NumPy's default generator (PCG64) with seed 20260927 + 1000 s + c. Every chunk is saved separately, so an
interrupted run resumes where it stopped. Output: results/reserve/<setting>_chunk<c>.npz.

Usage: python scripts/reserve_simulate.py [--jobs 4] [--chunks 20] [--settings 0,1,2,3]
Single-threaded, the full run takes about 2.5 hours on a current desktop computer; --jobs runs chunks in parallel.
"""
import argparse
import os
import time
from multiprocessing import Pool

import _common as U
import numpy as np

from mixscen import reserve as RS

OUT = os.path.join(U.RESULTS, "reserve")


def run_chunk(job):
    si, c = job
    name, theta, N = RS.SETTINGS[si]
    path = os.path.join(OUT, f"{name}_chunk{c}.npz")
    if os.path.exists(path):
        return f"{name} chunk {c}: exists"
    t = time.time()
    gaps = RS.gaps_for(N, RS.tmix_formula(theta))
    res = RS.simulate_chunk(np.random.default_rng(RS.seed(si, c)), theta, N, RS.CHUNK_SIZE, gaps)
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, **res)
    os.replace(tmp, path)
    return f"{name} chunk {c}: {time.time() - t:.0f} s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--chunks", type=int, default=RS.CHUNKS)
    ap.add_argument("--settings", default="0,1,2,3")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    jobs = [(int(s), c) for s in a.settings.split(",") for c in range(a.chunks)]
    if a.jobs > 1:
        with Pool(a.jobs) as pool:
            for msg in pool.imap_unordered(run_chunk, jobs):
                print(msg, flush=True)
    else:
        for job in jobs:
            print(run_chunk(job), flush=True)


if __name__ == "__main__":
    main()
