#!/usr/bin/env python3
"""
Milestone 12: validate model/perf_model.py against RTL performance counters.

Runs tb_npu_top on Verilator with deterministic memory (fixed latency, always
ready) over directed + random layer shapes, parses PERF_CYCLES / PERF_MACS per
layer, and compares with the model. Deterministic memory makes the RTL
reproducible, so the comparison is cycle-for-cycle.
"""

import argparse
import os
import re
import subprocess
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)
from model.perf_model import ArchCfg, simulate_layer  # noqa: E402

LINE = re.compile(r"\[PASS\] layer (\d+)\s+M=(\d+) N=(\d+) K=(\d+).*lat=(\d+)\+(\d+) rdy=(\d+)%\s+cycles=(\d+)")


def rtl_run(seed, n_random, lat):
    cmd = [sys.executable, "scripts/run_top_test.py", "--simulator", "verilator",
           "--seed", str(seed), "--random", str(n_random), "--fixed-lat", str(lat)]
    out = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True).stdout
    if "ALL TESTS PASSED" not in out:
        sys.exit(f"RTL run failed (seed {seed}, lat {lat})")
    return [tuple(int(x) for x in m.groups()) for m in LINE.finditer(out)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=[1, 2, 3])
    ap.add_argument("--lats", type=int, nargs="*", default=[1, 3, 6])
    ap.add_argument("--random", type=int, default=40)
    args = ap.parse_args()

    rows = []
    for lat in args.lats:
        for seed in args.seeds:
            for (_, M, N, K, l, j, r, cyc) in rtl_run(seed, args.random, lat):
                # Layer 0 always runs with latency 1 (see run_top_test.py)
                model = simulate_layer(M, N, K, ArchCfg(mem_lat=l))["cycles"]
                rows.append((M, N, K, l, cyc, model))

    exact = sum(1 for r in rows if r[4] == r[5])
    errs = [abs(r[5] - r[4]) / r[4] for r in rows]
    print(f"layers compared: {len(rows)}")
    print(f"cycle-exact:     {exact} ({100 * exact / len(rows):.1f}%)")
    print(f"mean |error|:    {100 * sum(errs) / len(errs):.3f}%   max |error|: {100 * max(errs):.3f}%")
    worst = sorted(rows, key=lambda r: -abs(r[5] - r[4]))[:5]
    print("largest differences (M N K lat rtl model):")
    for r in worst:
        print("  ", r)
    return 0 if exact == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
