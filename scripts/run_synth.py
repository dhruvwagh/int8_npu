#!/usr/bin/env python3
"""
Milestone 10: ASIC synthesis (Yosys -> sky130 HD) and pre-layout STA (OpenROAD).

  python3 scripts/run_synth.py                    # synth + STA at 50 MHz
  python3 scripts/run_synth.py --sta-only --period 15 --tag _p15
  python3 scripts/run_synth.py --sweep 20 15 12.5 10   # STA-only sweep on one netlist

Tools (see docs/ENGINEERING_LOG.md, M10): Yosys on PATH, OpenROAD from
~/tools/mamba/openroad (override with $OPENROAD).
"""

import argparse
import os
import re
import subprocess
import sys
import time

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OPENROAD = os.environ.get("OPENROAD", os.path.expanduser("~/tools/mamba/openroad/bin/openroad"))


def sh(cmd, env_extra, log):
    env = dict(os.environ, **env_extra)
    t0 = time.time()
    with open(log, "w") as f:
        rc = subprocess.run(cmd, cwd=REPO_ROOT, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
    print(f"  {' '.join(cmd[:1])} finished in {time.time() - t0:.0f} s (log: {os.path.relpath(log, REPO_ROOT)})")
    return rc


def synth(period_ns, tag):
    print(f"== Yosys synthesis (ABC delay target {period_ns} ns)")
    rc = sh(["yosys", "-q", "-c", "pd/synth/synth.tcl"],
            {"CLK_PERIOD_PS": str(int(period_ns * 1000)), "SYNTH_TAG": tag},
            os.path.join(REPO_ROOT, f"pd/reports/synth{tag}.log"))
    if rc != 0:
        sys.exit("[FAIL] synthesis")


SRAM = "sky130_sram_1kbyte_1rw1r_32x256_8"


def make_corner_sram_libs():
    """OpenRAM ships TT only; give each STA corner its own library name."""
    src = os.path.join(REPO_ROOT, "pd/sram", f"{SRAM}_TT_1p8V_25C.lib")
    text = open(src).read()
    for c in ("ss", "tt", "ff"):
        dst = os.path.join(REPO_ROOT, "pd/sram", f"{SRAM}_TT_as_{c}.lib")
        with open(dst, "w") as f:
            f.write(text.replace(f"library ({SRAM}_TT_1p8V_25C_lib)", f"library ({SRAM}_TT_1p8V_25C_lib_{c})", 1))


def sta(period_ns, net_tag, rpt_tag, resume=False):
    print(f"== OpenROAD repair + STA at {period_ns} ns")
    make_corner_sram_libs()
    rc = sh([OPENROAD, "-no_splash", "-exit", "pd/sta/sta.tcl"],
            {"CLK_PERIOD": str(period_ns), "SYNTH_TAG": net_tag, "RPT_TAG": rpt_tag,
             "RESUME": "1" if resume else "0"},
            os.path.join(REPO_ROOT, f"pd/reports/sta{rpt_tag}.log"))
    if rc != 0:
        sys.exit("[FAIL] STA")
    summ = os.path.join(REPO_ROOT, f"pd/reports/sta{rpt_tag}_summary.txt")
    print(open(summ).read())
    # Last "ss setup WNS" line is the final result
    vals = re.findall(r"^\s*ss\s+setup WNS\s+(-?[\d.]+)", open(summ).read(), re.M)
    return float(vals[-1]) if vals else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", type=float, default=20.0, help="clock period in ns")
    ap.add_argument("--tag", default="")
    ap.add_argument("--sta-only", action="store_true")
    ap.add_argument("--resume", action="store_true", help="reuse the placed+repaired DB")
    ap.add_argument("--sweep", type=float, nargs="*", help="STA-only period sweep (ns)")
    args = ap.parse_args()

    os.makedirs(os.path.join(REPO_ROOT, "pd/reports"), exist_ok=True)
    os.makedirs(os.path.join(REPO_ROOT, "pd/netlist"), exist_ok=True)
    if args.sweep:
        rows = []
        for p in args.sweep:
            wns = sta(p, args.tag, f"{args.tag}_p{p:g}", resume=True)
            rows.append((p, wns))
        print("period(ns)  MHz    ss setup WNS(ns)")
        for p, w in rows:
            print(f"{p:9.2f}  {1000 / p:6.1f}  {w:8.3f}")
        return
    if not args.sta_only:
        synth(args.period, args.tag)
    sta(args.period, args.tag, args.tag, resume=args.resume)


if __name__ == "__main__":
    main()
