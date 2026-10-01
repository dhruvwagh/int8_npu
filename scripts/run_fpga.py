#!/usr/bin/env python3
"""
Milestone 11: FPGA build of npu_fpga_top for Lattice ECP5 (open-source flow).

  python3 scripts/run_fpga.py --arr 8                   # synth + place & route
  python3 scripts/run_fpga.py --arr 16 --synth-only     # resource estimate only

Yosys synth_ecp5 (PE multipliers in LUTs, epilogue multipliers in DSPs), then
nextpnr-ecp5 on an LFE5U-85F (CABGA381). Reports go to pd/reports/fpga_*.
Tools: yosys on PATH; nextpnr-ecp5 from ~/tools/oss-cad-suite (or $NEXTPNR).
"""

import argparse
import json
import os
import subprocess
import sys
import time

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
NEXTPNR = os.environ.get("NEXTPNR", os.path.expanduser("~/tools/oss-cad-suite/bin/nextpnr-ecp5"))


def sh(cmd, env_extra, log):
    t0 = time.time()
    with open(log, "w") as f:
        rc = subprocess.run(cmd, cwd=REPO_ROOT, env=dict(os.environ, **env_extra),
                            stdout=f, stderr=subprocess.STDOUT).returncode
    print(f"  {os.path.basename(cmd[0])} finished in {time.time() - t0:.0f} s, rc={rc} "
          f"(log: {os.path.relpath(log, REPO_ROOT)})")
    return rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arr", type=int, default=8)
    ap.add_argument("--freq", type=float, default=50.0, help="target clock (MHz)")
    ap.add_argument("--synth-only", action="store_true")
    ap.add_argument("--skip-synth", action="store_true")
    args = ap.parse_args()

    tag = f"_arr{args.arr}"
    rep = os.path.join(REPO_ROOT, "pd/reports")
    os.makedirs(rep, exist_ok=True)
    if not args.skip_synth:
        print(f"== Yosys synth_ecp5, {args.arr}x{args.arr} array")
        if sh(["yosys", "-q", "-c", "pd/fpga/synth_ecp5.tcl"],
              {"FPGA_ARR": str(args.arr), "FPGA_TAG": tag},
              os.path.join(rep, f"fpga_synth{tag}.log")) != 0:
            sys.exit("[FAIL] synthesis")
    print(open(os.path.join(rep, f"fpga_synth{tag}.txt")).read().split("Number of cells")[-1][:600])
    if args.synth_only:
        return

    print(f"== nextpnr-ecp5, LFE5U-85F CABGA381, target {args.freq} MHz")
    report = os.path.join(rep, f"fpga_pnr{tag}.json")
    rc = sh([NEXTPNR, "--85k", "--package", "CABGA381", "--json", f"pd/fpga/npu_fpga{tag}.json",
             "--freq", str(args.freq), "--report", report, "--timing-allow-fail",
             "--lpf-allow-unconstrained", "--seed", "1"], {},
            os.path.join(rep, f"fpga_pnr{tag}.log"))
    if rc != 0:
        sys.exit("[FAIL] place and route")
    r = json.load(open(report))
    print("Fmax:")
    for clk, v in r.get("fmax", {}).items():
        print(f"  {clk}: achieved {v['achieved']:.1f} MHz (constraint {v['constraint']:.1f} MHz)")
    print("Utilization:")
    for k, v in r.get("utilization", {}).items():
        if v["used"]:
            print(f"  {k:16s} {v['used']:6d} / {v['available']:6d}  ({100 * v['used'] / v['available']:.1f}%)")


if __name__ == "__main__":
    main()
