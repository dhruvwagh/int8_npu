#!/usr/bin/env python3
"""
Test runner for Milestone 8: full MNIST CNN on npu_top, bit-exact.

1. Train (if no checkpoint) and quantize the CNN (compiler/qnet.py).
2. Report float vs int8 accuracy on the full 10k test set.
3. Export: weights/biases in panel layout, raw int8 test images, scratch
   regions for every layer, and golden per-layer outputs for each image.
4. Simulate tb/integration/tb_mnist.sv (the testbench is the host: im2col,
   maxpool, flatten; the NPU runs every GEMM + requant).
"""

import argparse
import os
import subprocess
import sys

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from compiler.qnet import (get_quantized_net, load_mnist, float_accuracy, int_accuracy,  # noqa: E402
                           int_forward, quantize_input)
from model.golden_npu import MemImage, pack_a, pack_bias, pad_to  # noqa: E402

VEC_DIR = os.path.join(REPO_ROOT, "tb", "vectors", "mnist")
FILELIST = os.path.join(REPO_ROOT, "src", "npu.f")
# Assertions + functional coverage, bound into npu_top (Verilator only)
SVA = ["tb/sva/npu_sva.sv", "tb/sva/npu_sva_bind.sv"]
TB = "tb/integration/tb_mnist.sv"

# (M, N, K) of the three GEMMs
SHAPES = [(8, 784, 9), (16, 196, 72), (10, 1, 784)]


def export(layers, x, y, n_img):
    mem = MemImage(1 << 20)
    desc = []
    for L, (M, N, K) in zip(layers, SHAPES):
        a = mem.place(pack_a(L.Wq))
        bias = mem.place(pack_bias(L.bq, M))
        desc.append(dict(a=a, bias=bias, M=M, N=N, K=K, cfg=L.cfg))
    img_base = mem.alloc(n_img * 784)
    for i in range(n_img):
        mem.write(img_base + i * 784, quantize_input(x[i]).tobytes())
    for d in desc:
        d["b"] = mem.alloc(pad_to(d["N"]) * d["K"])
        d["c"] = mem.alloc(pad_to(d["M"]) * d["N"])

    os.makedirs(VEC_DIR, exist_ok=True)
    with open(os.path.join(VEC_DIR, "mem.hex"), "w") as f:
        f.write("\n".join(mem.to_hex_lines()) + "\n")
    with open(os.path.join(VEC_DIR, "params.txt"), "w") as f:
        f.write(f"{n_img:x} {img_base:x}\n")
        for d in desc:
            c = d["cfg"]
            f.write(f"{d['a']:x} {d['bias']:x} {d['b']:x} {d['c']:x} {d['M']} {d['N']} {d['K']} "
                    f"{c.rq_mult & 0xFFFFFFFF:x} {c.rq_shift} {c.flags}\n")

    files = {k: open(os.path.join(VEC_DIR, k), "w")
             for k in ("exp_c1.hex", "exp_c2.hex", "exp_logits.hex", "labels.txt")}
    for i in range(n_img):
        c1, c2, lg = int_forward(layers, quantize_input(x[i]))
        files["exp_c1.hex"].write("\n".join(f"{v:02x}" for v in c1.view(np.uint8).ravel()) + "\n")
        files["exp_c2.hex"].write("\n".join(f"{v:02x}" for v in c2.view(np.uint8).ravel()) + "\n")
        files["exp_logits.hex"].write("\n".join(f"{v:02x}" for v in lg.view(np.uint8).ravel()) + "\n")
        files["labels.txt"].write(f"{y[i]} {int(np.argmax(lg))}\n")
    for f in files.values():
        f.close()
    print(f"  exported {n_img} images, memory image {mem.used_words() * 16 / 1024:.1f} KB")


def run(cmd, quiet=False):
    """Run a command; compile steps (quiet=True) only print output on failure."""
    print(f"\n[RUN] {' '.join(cmd)}")
    res = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    if quiet and res.returncode == 0:
        return 0
    print(((res.stdout or "") + (res.stderr or "")).strip())
    return res.returncode


def main():
    ap = argparse.ArgumentParser(description="Milestone 8: MNIST on the NPU")
    ap.add_argument("--simulator", choices=["iverilog", "verilator", "both"], default="verilator")
    ap.add_argument("-n", "--images", type=int, default=100)
    ap.add_argument("--mem-lat", type=int, default=1)
    ap.add_argument("--mem-jit", type=int, default=0)
    ap.add_argument("--mem-rdy", type=int, default=100)
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument("--skip-accuracy", action="store_true")
    args = ap.parse_args()

    os.chdir(REPO_ROOT)
    os.makedirs("sim_output", exist_ok=True)
    print("== Quantized network")
    model, layers = get_quantized_net(retrain=args.retrain)
    x_te, y_te = load_mnist(False)
    if not args.skip_accuracy:
        print(f"  float accuracy, 10k test images: {100 * float_accuracy(model, x_te, y_te):.2f}%")
        print(f"  int8  accuracy, 10k test images: {100 * int_accuracy(layers, x_te, y_te):.2f}%")

    print("== Exporting")
    export(layers, x_te, y_te, args.images)

    plus = [f"+n={args.images}", f"+mem_lat={args.mem_lat}", f"+mem_jit={args.mem_jit}",
            f"+mem_rdy={args.mem_rdy}"]
    ok = True
    if args.simulator in ("iverilog", "both"):
        if run(["iverilog", "-g2012", "-Itb/integration", "-DTB_MEM_WORDS=65536",
                "-o", "sim_output/tb_mnist.vvp", "-s", "tb_mnist", "-c", FILELIST, TB], quiet=True) != 0:
            sys.exit("[FAIL] iverilog compile")
        ok &= run(["vvp", "-n", "sim_output/tb_mnist.vvp"] + plus) == 0
    if args.simulator in ("verilator", "both"):
        if run(["verilator", "--binary", "--timing", "-j", "0", "-O2", "--assert",
                "-Wno-TIMESCALEMOD", "-Wno-WIDTHEXPAND", "-Wno-WIDTHTRUNC", "-Wno-INITIALDLY",
                "-Itb/integration", "--top-module", "tb_mnist", "-f", FILELIST, TB, *SVA,
                "-Mdir", "sim_output/verilator_mnist"], quiet=True) != 0:
            sys.exit("[FAIL] verilator compile")
        ok &= run(["./sim_output/verilator_mnist/Vtb_mnist"] + plus) == 0
    if not ok:
        sys.exit("[FAIL] MNIST test failed")
    print("\n[SUCCESS] Milestone 8: MNIST runs bit-exact on the NPU")


if __name__ == "__main__":
    main()
