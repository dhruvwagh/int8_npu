#!/usr/bin/env python3
"""
Milestone 12: design-space exploration with the validated performance model.

Sweeps (all on the MNIST network, 50 MHz clock for time figures):
  1. Tensor-memory port width  4 .. 64 B/cycle      -> per-layer utilization, roofline
  2. K tile depth              32 .. 256            -> FC K-splitting cost
  3. Batch size                1 .. 32 images        -> why FC needs batching
Writes docs/results/dse.md (tables) and docs/figures/*.png.

The RTL implements port = 16 B, array = 16x16, K_TILE = 128; the model is
cycle-exact there (scripts/validate_perf_model.py). Other points are model
predictions of the same microarchitecture.
"""

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from model.perf_model import ArchCfg, simulate_layer, MNIST_LAYERS  # noqa: E402

FIG_DIR = os.path.join(REPO_ROOT, "docs", "figures")
RES = os.path.join(REPO_ROOT, "docs", "results", "dse.md")
CLK_MHZ = 50.0

# Validated categorical slots 1-3 (dataviz reference palette, light surface)
C_BLUE, C_ORANGE, C_AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
LAYER_STYLE = {"conv1": (C_BLUE, "o"), "conv2": (C_ORANGE, "s"), "fc": (C_AQUA, "^")}


def mnist(cfg, batch=1):
    """Per-layer model results for one batch of `batch` images."""
    out = {}
    for name, M, N, K in MNIST_LAYERS:
        out[name] = simulate_layer(M, N * batch, K, cfg)
    return out


def traffic_bytes(r, cfg):
    return (r["port_read_beats"] + r["port_write_beats"]) * cfg.port_bytes


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK2)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(RES), exist_ok=True)
    md = ["# Design-space exploration (performance model)", "",
          "Model: `model/perf_model.py`, cycle-exact against RTL at the implemented point "
          "(16x16 array, 16 B/cycle port, K_TILE = 128; 450 layers compared). "
          "Ideal memory (1-cycle latency, always ready). Clock 50 MHz for time figures.", ""]

    # ------------------------------------------------------------------
    # 1. Port width sweep
    # ------------------------------------------------------------------
    widths = [4, 8, 16, 32, 64]
    sweep = {w: mnist(ArchCfg(port_bytes=w)) for w in widths}
    md += ["## 1. Tensor-memory port width", "",
           "| Port (B/cycle) | conv1 cycles | conv2 cycles | fc cycles | Total cycles/image | Array utilization | Images/s @50 MHz |",
           "|---|---|---|---|---|---|---|"]
    for w in widths:
        r = sweep[w]
        tot = sum(x["cycles"] for x in r.values())
        macs = sum(x["macs"] for x in r.values())
        md.append(f"| {w}{' (RTL)' if w == 16 else ''} | {r['conv1']['cycles']:,} | {r['conv2']['cycles']:,} | "
                  f"{r['fc']['cycles']:,} | {tot:,} | {100 * macs / (tot * 256):.1f}% | "
                  f"{CLK_MHZ * 1e6 / tot:,.0f} |")
    md.append("")

    fig, ax = plt.subplots(figsize=(6.4, 3.8), facecolor=SURFACE)
    style_axes(ax)
    for name in ("conv1", "conv2", "fc"):
        col, mk = LAYER_STYLE[name]
        ys = [100 * sweep[w][name]["util"] for w in widths]
        ax.plot(widths, ys, color=col, linewidth=2, marker=mk, markersize=7,
                markeredgecolor=SURFACE, markeredgewidth=1.5)
        ax.annotate(name, (widths[-1], ys[-1]), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=9, color=INK)
    ax.set_xscale("log", base=2)
    ax.set_xticks(widths)
    ax.set_xticklabels([str(w) for w in widths])
    ax.axvline(16, color=INK2, linewidth=0.8, linestyle=(0, (3, 3)))
    ax.text(16, ax.get_ylim()[1] * 0.97, " implemented", fontsize=8, color=INK2, va="top")
    ax.set_xlabel("Tensor-memory port width (bytes/cycle)", color=INK, fontsize=10)
    ax.set_ylabel("Array utilization (%)", color=INK, fontsize=10)
    ax.set_title("MNIST layer utilization vs memory port width", color=INK, fontsize=11, loc="left")
    ax.set_xlim(3.5, 100)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "port_width_sweep.png"), dpi=160, facecolor=SURFACE)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Roofline at the implemented point (+ batched points)
    # ------------------------------------------------------------------
    cfg = ArchCfg()
    peak = cfg.arr * cfg.arr
    pts = []
    for batch in (1, 16):
        r = mnist(cfg, batch)
        for name, M, N, K in MNIST_LAYERS:
            x = r[name]
            ai = x["macs"] / traffic_bytes(x, cfg)
            pts.append((name, batch, ai, x["macs"] / x["cycles"]))
    md += ["## 2. Roofline (16 B/cycle port, 256 MAC/cycle peak)", "",
           "Arithmetic intensity uses the traffic the NPU actually moves (operands, bias, outputs).", "",
           "| Layer | Batch | MACs / byte | Achieved MAC/cycle | Bandwidth ceiling MAC/cycle | Fraction of ceiling |",
           "|---|---|---|---|---|---|"]
    for name, b, ai, perf in pts:
        ceil_ = min(peak, ai * cfg.port_bytes)
        md.append(f"| {name} | {b} | {ai:.2f} | {perf:.1f} | {ceil_:.1f} | {100 * perf / ceil_:.0f}% |")
    md.append("")

    fig, ax = plt.subplots(figsize=(6.4, 4.2), facecolor=SURFACE)
    style_axes(ax)
    import numpy as np
    xs = np.logspace(-1.3, 2.3, 200)
    ax.plot(xs, np.minimum(peak, xs * cfg.port_bytes), color=INK2, linewidth=1.5)
    ax.text(0.07, 0.07 * cfg.port_bytes * 1.25, "16 B/cycle port", color=INK2, fontsize=8, rotation=0)
    ax.text(40, peak * 1.12, f"peak {peak} MAC/cycle", color=INK2, fontsize=8)
    # Label placement: conv1's two points coincide, so they share one label
    labels = {("conv1", 1): ("conv1 (batch 1 and 16)", (9, -3)),
              ("conv2", 1): ("conv2, batch 1", (9, -10)),
              ("conv2", 16): ("conv2, batch 16", (-10, 9)),
              ("fc", 1): ("fc, batch 1", (9, -3)),
              ("fc", 16): ("fc, batch 16", (9, -3))}
    for name, b, ai, perf in pts:
        col, mk = LAYER_STYLE[name]
        ax.plot([ai], [perf], marker=mk, markersize=8, linestyle="none",
                color=col if b == 1 else SURFACE, markeredgecolor=col, markeredgewidth=2)
        if (name, b) in labels:
            text, off = labels[(name, b)]
            ax.annotate(text, (ai, perf), xytext=off, textcoords="offset points", fontsize=8,
                        color=INK, ha="right" if off[0] < 0 else "left")
    ax.plot([], [], marker="o", color=INK2, linestyle="none", markersize=6, label="batch 1 (filled)")
    ax.plot([], [], marker="o", color=SURFACE, markeredgecolor=INK2, markeredgewidth=1.5,
            linestyle="none", markersize=6, label="batch 16 (hollow)")
    leg = ax.legend(loc="upper left", fontsize=8, frameon=False)
    for t in leg.get_texts():
        t.set_color(INK2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Arithmetic intensity (MACs per byte of memory traffic)", color=INK, fontsize=10)
    ax.set_ylabel("Achieved MACs/cycle", color=INK, fontsize=10)
    ax.set_title("Roofline: every MNIST layer sits under the memory-port ceiling",
                 color=INK, fontsize=11, loc="left")
    ax.set_ylim(0.3, 600)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "roofline.png"), dpi=160, facecolor=SURFACE)
    plt.close(fig)

    # ------------------------------------------------------------------
    # 3. K tile depth
    # ------------------------------------------------------------------
    md += ["## 3. K tile depth (buffer depth per bank)", "",
           "The OpenRAM macro is 256 words deep, so K_TILE up to 256 uses the same 16 macros: "
           "the only cost is an 8-bit -> 9-bit read address.", "",
           "| K_TILE | fc cycles | fc K tiles | Total cycles/image | SRAM macros |", "|---|---|---|---|---|"]
    for kt in (32, 64, 128, 256):
        r = mnist(ArchCfg(k_tile=kt))
        tot = sum(x["cycles"] for x in r.values())
        macros = 16 * max(1, -(-kt // 256))
        md.append(f"| {kt}{' (RTL)' if kt == 128 else ''} | {r['fc']['cycles']:,} | {r['fc']['tiles']} | {tot:,} | {macros} |")
    md.append("")

    # ------------------------------------------------------------------
    # 4. Batch size
    # ------------------------------------------------------------------
    batches = [1, 2, 4, 8, 16, 32]
    md += ["## 4. Batch size (N grows with batch; weights are reused across images)", "",
           "| Batch | conv1 util | conv2 util | fc util | Cycles/image | Images/s @50 MHz |",
           "|---|---|---|---|---|---|"]
    bres = {}
    for b in batches:
        r = mnist(cfg, b)
        bres[b] = r
        tot = sum(x["cycles"] for x in r.values())
        md.append(f"| {b} | {100 * r['conv1']['util']:.1f}% | {100 * r['conv2']['util']:.1f}% | "
                  f"{100 * r['fc']['util']:.1f}% | {tot / b:,.0f} | {CLK_MHZ * 1e6 * b / tot:,.0f} |")
    md.append("")

    fig, ax = plt.subplots(figsize=(6.4, 3.8), facecolor=SURFACE)
    style_axes(ax)
    for name in ("conv1", "conv2", "fc"):
        col, mk = LAYER_STYLE[name]
        ys = [100 * bres[b][name]["util"] for b in batches]
        ax.plot(batches, ys, color=col, linewidth=2, marker=mk, markersize=7,
                markeredgecolor=SURFACE, markeredgewidth=1.5)
        ax.annotate(name, (batches[-1], ys[-1]), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=9, color=INK)
    ax.set_xscale("log", base=2)
    ax.set_xticks(batches)
    ax.set_xticklabels([str(b) for b in batches])
    ax.set_xlim(0.8, 50)
    ax.set_xlabel("Batch size (images per layer call)", color=INK, fontsize=10)
    ax.set_ylabel("Array utilization (%)", color=INK, fontsize=10)
    ax.set_title("Batching turns the fc matrix-vector product into a matrix product",
                 color=INK, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "batch_sweep.png"), dpi=160, facecolor=SURFACE)
    plt.close(fig)

    with open(RES, "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
