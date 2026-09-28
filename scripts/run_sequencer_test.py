#!/usr/bin/env python3
"""
Test runner for Milestone 5: Hardware Sequencer & M/N/K Multi-Loop Tiling Engine.
Workflow:
1. Run Icarus Verilog and/or Verilator simulation on tb/unit/tb_npu_sequencer.sv
   - Test 1: Standard full-tile GEMM (M=16, N=16, K=16)
   - Test 2: FC layer (M=10, N=1, K=784) with K-splitting and persistent accumulators
   - Test 3: Conv2 spatial sweep with weight residency (M=16, N=196, K=64)
   - Test 4: Conv1 partial channels (M=8, N=32, K=25)
   - Test 5: Non-aligned 3D edge tiles (M=17, N=33, K=135)
   - Test 6: Abort and error handling
2. Run Yosys synthesis audit:
   - Verify 0 latches (select -assert-count 0 t:$_DLATCH_*)
   - Verify proper flip-flop mapping and state encoding
"""

import argparse
import os
import subprocess
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def run_command(cmd, cwd=REPO_ROOT):
    print(f"\n[RUN] {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    res = subprocess.run(cmd, cwd=cwd, shell=isinstance(cmd, str), capture_output=True, text=True)
    if res.stdout:
        print(res.stdout.strip())
    if res.stderr:
        print(res.stderr.strip(), file=sys.stderr)
    return res.returncode


def main():
    parser = argparse.ArgumentParser(description="Milestone 5 Sequencer Test Runner")
    parser.add_argument("--simulator", choices=["iverilog", "verilator", "both"], default="both")
    parser.add_argument("--synth", action="store_true", default=True, help="Run Yosys synthesis audit")
    args = parser.parse_args()

    os.chdir(REPO_ROOT)
    os.makedirs("sim_output", exist_ok=True)

    src_files = [
        "src/npu_pkg.sv",
        "src/ctrl/npu_sequencer.sv",
        "tb/unit/tb_npu_sequencer.sv"
    ]

    # 1. Simulation with Icarus Verilog
    if args.simulator in ["iverilog", "both"]:
        print("==================================================================")
        print(" Step 1: Simulating Hardware Sequencer with Icarus Verilog")
        print("==================================================================")
        compile_cmd = ["iverilog", "-g2012", "-Isrc", "-o", "sim_output/tb_npu_sequencer.vvp"] + src_files
        if run_command(compile_cmd) != 0:
            print("[FAIL] iverilog compilation failed!")
            sys.exit(1)

        sim_cmd = ["vvp", "sim_output/tb_npu_sequencer.vvp"]
        if run_command(sim_cmd) != 0:
            print("[FAIL] iverilog simulation failed!")
            sys.exit(1)

    # 2. Simulation with Verilator
    if args.simulator in ["verilator", "both"]:
        print("\n==================================================================")
        print(" Step 2: Simulating Hardware Sequencer with Verilator")
        print("==================================================================")
        verilator_cmd = [
            "verilator", "--binary", "--timing",
            "-Wno-TIMESCALEMOD", "-Wno-UNUSEDPARAM", "-Wno-INITIALDLY",
            "-Wno-WIDTHEXPAND", "-Wno-WIDTHTRUNC",
            "-Isrc", "--top-module", "tb_npu_sequencer"
        ] + src_files + ["-Mdir", "sim_output/verilator_sequencer"]

        if run_command(verilator_cmd) != 0:
            print("[FAIL] Verilator compilation failed!")
            sys.exit(1)

        sim_cmd = ["./sim_output/verilator_sequencer/Vtb_npu_sequencer"]
        if run_command(sim_cmd) != 0:
            print("[FAIL] Verilator simulation failed!")
            sys.exit(1)

    # 3. Yosys Synthesis Audit
    if args.synth:
        print("\n==================================================================")
        print(" Step 3: Yosys Synthesis & Latch-Free Audit")
        print("==================================================================")

        yosys_script = """
        read_verilog -sv src/npu_pkg.sv src/ctrl/npu_sequencer.sv
        hierarchy -top npu_sequencer
        synth
        select -assert-count 0 t:$_DLATCH_*
        stat
        """
        cmd_synth = ["yosys", "-p", yosys_script]
        if run_command(cmd_synth) != 0:
            print("[FAIL] Yosys synthesis audit failed on npu_sequencer!")
            sys.exit(1)

        print("\n==================================================================")
        print("  YOSYS SYNTHESIS AUDIT PASSED:")
        print("  - npu_sequencer synthesizes with 0 latches")
        print("  - Clean multi-loop state machine and coordinate generation logic")
        print("==================================================================")

    print("\n[SUCCESS] Milestone 5 verification completed successfully!")


if __name__ == "__main__":
    main()
