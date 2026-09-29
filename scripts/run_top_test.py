#!/usr/bin/env python3
"""
Test runner for Milestone 7: npu_top end-to-end integration.

1. Generate layer jobs with model/golden_npu.py: directed MNIST-shaped layers,
   K-split, edge tiles and constrained-random shapes, each with its own memory
   latency / jitter / backpressure setting.
2. Write per-layer memory images (input + expected) and tests.txt.
3. Simulate tb/integration/tb_npu_top.sv on Icarus Verilog and/or Verilator.
"""

import argparse
import os
import random
import subprocess
import sys

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from model.golden_npu import LayerCfg, build_layer_job, choose_requant, pad_to  # noqa: E402

VEC_DIR = os.path.join(REPO_ROOT, "tb", "vectors", "top")
FILELIST = os.path.join(REPO_ROOT, "src", "npu.f")
TB = "tb/integration/tb_npu_top.sv"


def make_layer(rng, M, N, K, relu=True, bias=True, signed_act=True, sat_target=100):
    A = rng.integers(-128, 128, (M, K), dtype=np.int8)
    B = rng.integers(-128, 128, (K, N), dtype=np.int8)
    acc = A.astype(np.int64) @ B.astype(np.int64)
    absmax = int(np.abs(acc).max()) if acc.size else 1
    bias_v = rng.integers(-(absmax // 4) - 1, absmax // 4 + 1, M).astype(np.int32)
    mult, shift = choose_requant(absmax + (absmax // 4 if bias else 0), sat_target,
                                 shift=int(rng.integers(12, 31)))
    cfg = LayerCfg(M, N, K, mult, shift, relu, bias, signed_act)
    return build_layer_job(A, B, bias_v, cfg)


def generate(seed, n_random, ideal_mem=False):
    rng = np.random.default_rng(seed)
    pyrng = random.Random(seed)

    directed = [
        # (M, N, K, relu, bias, signed, sat_target, comment)
        (16, 16, 16, False, False, True, 100, "full tile"),
        (8, 784, 9, True, True, True, 100, "MNIST conv1 shape"),
        (16, 196, 72, True, True, True, 100, "MNIST conv2 shape"),
        (10, 1, 784, False, True, True, 100, "MNIST fc shape (K split, N=1)"),
        (20, 40, 300, True, True, True, 100, "K split with N > 16"),
        (1, 1, 1, False, False, True, 100, "minimal"),
        (16, 16, 128, False, True, True, 100, "K == K_TILE"),
        (17, 33, 129, True, False, True, 100, "edges in M, N, K"),
        (33, 17, 256, False, True, True, 400, "saturating requant"),
        (12, 20, 40, True, True, False, 200, "unsigned output mode"),
    ]
    shapes = list(directed)
    for _ in range(n_random):
        M = pyrng.choice([pyrng.randint(1, 16), pyrng.randint(1, 48)])
        N = pyrng.choice([pyrng.randint(1, 16), pyrng.randint(1, 48)])
        K = pyrng.choice([pyrng.randint(1, 32), pyrng.randint(1, 300)])
        shapes.append((M, N, K, pyrng.random() < 0.5, pyrng.random() < 0.7, True,
                       pyrng.choice([100, 100, 300]), "random"))

    os.makedirs(VEC_DIR, exist_ok=True)
    for f in os.listdir(VEC_DIR):
        os.remove(os.path.join(VEC_DIR, f))

    lines = []
    for i, (M, N, K, relu, bias, sgn, sat, note) in enumerate(shapes):
        job = make_layer(rng, M, N, K, relu, bias, sgn, sat)
        nwords = job.mem_exp.used_words()
        with open(os.path.join(VEC_DIR, f"mem_{i}.hex"), "w") as f:
            f.write("\n".join(job.mem_in.to_hex_lines(nwords)) + "\n")
        with open(os.path.join(VEC_DIR, f"exp_{i}.hex"), "w") as f:
            f.write("\n".join(job.mem_exp.to_hex_lines(nwords)) + "\n")

        # Memory behaviour: first layer ideal, then randomised
        if i == 0 or ideal_mem:
            lat, jit, rdy = 1, 0, 100
        else:
            lat, jit, rdy = pyrng.randint(1, 6), pyrng.choice([0, 0, 2, 5]), pyrng.choice([100, 100, 80, 50])
        c = job.cfg
        lines.append(f"{i} {M} {N} {K} {job.a_base:08x} {job.b_base:08x} {job.c_base:08x} "
                     f"{job.bias_base:08x} {c.rq_mult & 0xFFFFFFFF:08x} {c.rq_shift} {c.flags} "
                     f"{nwords} {lat} {jit} {rdy} {M * N * K}")
        print(f"  layer {i:2d}: M={M:3d} N={N:4d} K={K:4d} relu={int(relu)} bias={int(bias)} "
              f"signed={int(sgn)} mem=({lat},{jit},{rdy}%)  {note}")

    with open(os.path.join(VEC_DIR, "tests.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    return len(shapes)


def run(cmd):
    print(f"\n[RUN] {' '.join(cmd)}")
    res = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    out = (res.stdout or "") + (res.stderr or "")
    print(out.strip())
    return res.returncode


def main():
    ap = argparse.ArgumentParser(description="Milestone 7 npu_top integration test")
    ap.add_argument("--simulator", choices=["iverilog", "verilator", "both"], default="both")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--random", type=int, default=20, help="number of random layers")
    ap.add_argument("--vcd", action="store_true")
    ap.add_argument("--ideal-mem", action="store_true",
                    help="1-cycle memory, no jitter or backpressure (for baseline perf numbers)")
    args = ap.parse_args()

    os.chdir(REPO_ROOT)
    os.makedirs("sim_output", exist_ok=True)
    print("== Generating layer jobs (seed %d)" % args.seed)
    generate(args.seed, args.random, args.ideal_mem)

    plus = ["+vcd"] if args.vcd else []
    ok = True
    if args.simulator in ("iverilog", "both"):
        print("\n== Icarus Verilog")
        if run(["iverilog", "-g2012", "-o", "sim_output/tb_npu_top.vvp", "-s", "tb_npu_top",
                "-c", FILELIST, TB]) != 0:
            sys.exit("[FAIL] iverilog compile")
        ok &= run(["vvp", "-n", "sim_output/tb_npu_top.vvp"] + plus) == 0
    if args.simulator in ("verilator", "both"):
        print("\n== Verilator")
        if run(["verilator", "--binary", "--timing", "-j", "0", "-O2",
                "-Wno-TIMESCALEMOD", "-Wno-WIDTHEXPAND", "-Wno-WIDTHTRUNC", "-Wno-INITIALDLY",
                "--top-module", "tb_npu_top", "-f", FILELIST, TB,
                "-Mdir", "sim_output/verilator_top"]) != 0:
            sys.exit("[FAIL] verilator compile")
        ok &= run(["./sim_output/verilator_top/Vtb_npu_top"] + plus) == 0

    if not ok:
        sys.exit("[FAIL] npu_top integration test failed")
    print("\n[SUCCESS] Milestone 7 integration test passed")


if __name__ == "__main__":
    main()
