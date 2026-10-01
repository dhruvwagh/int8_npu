#!/usr/bin/env python3
"""
Full regression: unit suites, integration (16x16 and 8x8, normal and harsh
memory, both simulators), MNIST, performance-model validation, plus
functional-coverage closure.

Every Verilator run has the SVA module bound in (assertion failure = test
failure). COVER counters from all Verilator runs are summed; any coverage
point that is never hit fails the regression.

  python3 scripts/run_regression.py            # full (~5 min)
  python3 scripts/run_regression.py --quick    # smaller random counts
"""

import argparse
import collections
import os
import re
import subprocess
import sys
import time

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PY = sys.executable


def suites(quick):
    n = 20 if quick else 100
    s = [
        ("unit: PE",                [PY, "scripts/run_pe_test.py"], "Exit Criteria Met"),
        ("unit: 4x4 array",         [PY, "scripts/run_array_4x4_test.py"], "Exit Criteria Met"),
        ("unit: 16x16 array",       [PY, "scripts/run_array_16x16_test.py"], "Exit Criteria Met"),
        ("unit: buffers + fetch",   [PY, "scripts/run_buffer_test.py"], "SUCCESS"),
        ("unit: epilogue + obuf",   [PY, "scripts/run_epilogue_test.py"], "SUCCESS"),
    ]
    for seed in (1, 2, 3):
        s.append((f"top 16x16 seed {seed}", [PY, "scripts/run_top_test.py", "--simulator", "verilator",
                                              "--seed", str(seed), "--random", str(n)], "ALL TESTS PASSED"))
    s += [
        ("top 16x16 harsh memory", [PY, "scripts/run_top_test.py", "--simulator", "verilator",
                                    "--harsh-mem", "--seed", "4", "--random", str(n // 2)], "ALL TESTS PASSED"),
        ("top 16x16 iverilog",     [PY, "scripts/run_top_test.py", "--simulator", "iverilog",
                                    "--seed", "5", "--random", "10"], "ALL TESTS PASSED"),
        ("top 8x8 seed 6",         [PY, "scripts/run_top_test.py", "--simulator", "verilator", "--arr", "8",
                                    "--seed", "6", "--random", str(n)], "ALL TESTS PASSED"),
        ("top 8x8 harsh memory",   [PY, "scripts/run_top_test.py", "--simulator", "verilator", "--arr", "8",
                                    "--harsh-mem", "--seed", "7", "--random", str(n // 2)], "ALL TESTS PASSED"),
        ("MNIST",                  [PY, "scripts/run_mnist_test.py", "--skip-accuracy",
                                    "-n", "50" if quick else "300", "--mem-lat", "3", "--mem-jit", "2",
                                    "--mem-rdy", "80"], "MNIST PASSED"),
        ("perf model vs RTL",      [PY, "scripts/validate_perf_model.py", "--seeds", "1", "--lats", "1", "5",
                                    "--random", "20" if quick else "40"], "cycle-exact:"),
    ]
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    cover = collections.Counter()
    seen = set()
    results = []
    t_all = time.time()
    for name, cmd, marker in suites(args.quick):
        t0 = time.time()
        res = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
        out = res.stdout + res.stderr
        ok = res.returncode == 0 and marker in out and "Assertion failed" not in out
        if name == "perf model vs RTL":
            ok = ok and "(100.0%)" in out
        for m in re.finditer(r"^COVER (\S+)\s+(\d+)", out, re.M):
            cover[m.group(1)] += int(m.group(2))
            seen.add(m.group(1))
        results.append((name, ok, time.time() - t0))
        print(f"{'PASS' if ok else 'FAIL'}  {name:28s} {time.time() - t0:6.1f} s", flush=True)
        if not ok:
            tail = "\n".join(out.strip().splitlines()[-15:])
            print("      " + tail.replace("\n", "\n      "))

    print("\nFunctional coverage (summed over Verilator runs):")
    holes = []
    for k in sorted(seen):
        print(f"  {k:30s} {cover[k]:10d}{'   <-- HOLE' if cover[k] == 0 else ''}")
        if cover[k] == 0:
            holes.append(k)

    n_fail = sum(1 for _, ok, _ in results if not ok)
    print(f"\n{len(results) - n_fail}/{len(results)} suites passed, {len(seen) - len(holes)}/{len(seen)} "
          f"coverage points hit, {time.time() - t_all:.0f} s total")
    sys.exit(1 if n_fail or holes else 0)


if __name__ == "__main__":
    main()
