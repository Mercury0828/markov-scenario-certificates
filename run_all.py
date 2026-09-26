"""Regenerate every computed figure, table and number of the paper, then compare them with the printed values.

Usage: python run_all.py [--jobs N] [--use-data]
--jobs N     run the Monte Carlo chunks of the reserve problem on N processes (default 1)
--use-data   do not simulate; build Table 2 from the archived runs in data/reserve (the runs used in the paper)
The simulation resumes where it stopped: chunks already in results/reserve are kept.
"""
import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def run(script, *args):
    t = time.time()
    print(f"\n=== {script} {' '.join(args)}", flush=True)
    r = subprocess.run([sys.executable, os.path.join(HERE, "scripts", script), *args], cwd=HERE)
    print(f"=== {script}: {time.time() - t:.0f} s", flush=True)
    return r.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--use-data", action="store_true")
    a = ap.parse_args()
    for s in ("text_numbers.py", "figure1_sticky_path.py", "figure3_certificate.py", "table3_validity.py"):
        if run(s):
            raise SystemExit(f"{s} failed")
    if a.use_data:
        code = run("table2_reserve.py", "--dir", os.path.join(HERE, "data", "reserve"))
    else:
        if run("reserve_simulate.py", "--jobs", str(a.jobs)):
            raise SystemExit("reserve_simulate.py failed")
        code = run("table2_reserve.py")
    if code:
        raise SystemExit("table2_reserve.py failed")
    raise SystemExit(run("check_paper.py"))


if __name__ == "__main__":
    main()
