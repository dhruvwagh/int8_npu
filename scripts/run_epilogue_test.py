#!/usr/bin/env python3
"""
Test runner for Milestone 6: Requantization Epilogue Unit & Output Staging Buffer.
Workflow:
1. Generate 1,332 bit-exact test vectors using model/golden_epilogue.py
2. Run Icarus Verilog and/or Verilator simulation on tb/unit/tb_npu_epilogue.sv
   - Test 1: 1,332 golden vectors (saturation clamp [-128, 127], round half away from zero, ReLU, bias)
   - Test 2: 16-lane parallel SIMD execution
   - Test 3: Full 16x16 tile drain into npu_obuf
   - Test 4: Double-buffered concurrent drain & DMA readback
3. Run Yosys synthesis audit:
   - Verify npu_obuf banks synthesize as register files (no SRAM macros, 0 latches)
   - Verify npu_epilogue synthesizes with 0 latches
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
    parser = argparse.ArgumentParser(description="Milestone 6 Epilogue & Output Buffer Test Runner")
    parser.add_argument("--simulator", choices=["iverilog", "verilator", "both"], default="both")
    parser.add_argument("--synth", action="store_true", default=True, help="Run Yosys synthesis audit")
    args = parser.parse_args()

    os.chdir(REPO_ROOT)
    os.makedirs("sim_output", exist_ok=True)
    os.makedirs("tb/vectors", exist_ok=True)

    # 1. Vector generation
    print("==================================================================")
    print(" Step 1: Generating Epilogue Golden Vectors from Python Model")
    print("==================================================================")
    from model.golden_epilogue import export_epilogue_vectors

    vec_path = os.path.join(REPO_ROOT, "tb", "vectors", "epilogue_vectors.txt")
    export_epilogue_vectors(vec_path, num_random=1000, seed=123)

    src_files = [
        "src/buffer/npu_sram.sv",
        "src/epilogue/npu_epilogue.sv",
        "src/epilogue/npu_obuf.sv",
        "tb/unit/tb_npu_epilogue.sv"
    ]

    # 2. Simulation with Icarus Verilog
    if args.simulator in ["iverilog", "both"]:
        print("\n==================================================================")
        print(" Step 2a: Simulating Epilogue with Icarus Verilog")
        print("==================================================================")
        compile_cmd = ["iverilog", "-g2012", "-Isrc", "-o", "sim_output/tb_npu_epilogue.vvp"] + src_files
        if run_command(compile_cmd) != 0:
            print("[FAIL] iverilog compilation failed!")
            sys.exit(1)

        sim_cmd = ["vvp", "sim_output/tb_npu_epilogue.vvp"]
        if run_command(sim_cmd) != 0:
            print("[FAIL] iverilog simulation failed!")
            sys.exit(1)

    # 3. Simulation with Verilator
    if args.simulator in ["verilator", "both"]:
        print("\n==================================================================")
        print(" Step 2b: Simulating Epilogue with Verilator")
        print("==================================================================")
        verilator_cmd = [
            "verilator", "--binary", "--timing",
            "-Wno-TIMESCALEMOD", "-Wno-UNUSEDPARAM", "-Wno-INITIALDLY",
            "-Wno-WIDTHEXPAND", "-Wno-WIDTHTRUNC", "-Wno-IMPLICITSTATIC",
            "-Isrc", "--top-module", "tb_npu_epilogue"
        ] + src_files + ["-Mdir", "sim_output/verilator_epilogue"]

        if run_command(verilator_cmd) != 0:
            print("[FAIL] Verilator compilation failed!")
            sys.exit(1)

        sim_cmd = ["./sim_output/verilator_epilogue/Vtb_npu_epilogue"]
        if run_command(sim_cmd) != 0:
            print("[FAIL] Verilator simulation failed!")
            sys.exit(1)

    # 4. Yosys Synthesis Audit
    if args.synth:
        print("\n==================================================================")
        print(" Step 3: Yosys Synthesis Audit (Area & Latch-Free Check)")
        print("==================================================================")

        # Audit npu_obuf
        yosys_script_obuf = """
        read_liberty -lib pd/sram/sky130_sram_1kbyte_1rw1r_32x256_8_TT_1p8V_25C.lib
        read_verilog -sv src/buffer/npu_sram.sv src/epilogue/npu_obuf.sv
        hierarchy -top npu_obuf
        synth -flatten
        select -assert-count 0 t:sky130_sram_1kbyte_1rw1r_32x256_8
        select -assert-count 0 t:$_DLATCH_*
        stat
        """
        cmd_obuf = ["yosys", "-p", yosys_script_obuf]
        if run_command(cmd_obuf) != 0:
            print("[FAIL] Yosys synthesis audit failed on npu_obuf!")
            sys.exit(1)

        # Audit npu_epilogue
        yosys_script_epi = """
        read_verilog -sv src/epilogue/npu_epilogue.sv
        hierarchy -top npu_epilogue
        synth
        select -assert-count 0 t:$_DLATCH_*
        stat
        """
        cmd_epi = ["yosys", "-p", yosys_script_epi]
        if run_command(cmd_epi) != 0:
            print("[FAIL] Yosys synthesis audit failed on npu_epilogue!")
            sys.exit(1)

        print("\n==================================================================")
        print("  YOSYS SYNTHESIS AUDIT PASSED:")
        print("  - npu_obuf: register-file banks (USE_MACRO=0), no SRAM macros, 0 latches")
        print("  - npu_epilogue: 0 latches, fully synthesized parallel SIMD datapath")
        print("==================================================================")

    print("\n[SUCCESS] Milestone 6 verification completed successfully!")


if __name__ == "__main__":
    main()
