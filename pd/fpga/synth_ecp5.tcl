# ============================================================================
# File: synth_ecp5.tcl
# Description: Yosys synthesis of npu_fpga_top for Lattice ECP5
#   yosys -c pd/fpga/synth_ecp5.tcl        (run from the repo root)
# Environment: FPGA_ARR (array size, default 16), FPGA_TAG (output suffix),
#              FPGA_SYNTH_OPTS (extra synth_ecp5 flags)
# ============================================================================
yosys -import
set tag  [expr {[info exists ::env(FPGA_TAG)] ? $::env(FPGA_TAG) : ""}]
set opts [expr {[info exists ::env(FPGA_SYNTH_OPTS)] ? $::env(FPGA_SYNTH_OPTS) : ""}]
set arr  [expr {[info exists ::env(FPGA_ARR)] ? $::env(FPGA_ARR) : 16}]

set fp [open src/npu.f r]
set files [split [string trim [read $fp]] "\n"]
close $fp
read_verilog -sv -DFPGA {*}$files src/fpga/npu_fpga_top.sv
chparam -set ARR $arr npu_fpga_top

# The ECP5-85F has 156 MULT18X18D blocks. The 16 wide epilogue multipliers
# (33x32, 4 blocks each) need them most; the 256 small 8x8 PE multipliers are
# built from LUTs instead (Booth-encoded) before DSP inference runs.
synth_ecp5 -top npu_fpga_top -abc9 {*}$opts -run begin:coarse
yosys proc
booth {$paramod*npu_pe*}
synth_ecp5 -top npu_fpga_top -abc9 {*}$opts -run coarse: -json pd/fpga/npu_fpga${tag}.json
tee -o pd/reports/fpga_synth${tag}.txt stat
