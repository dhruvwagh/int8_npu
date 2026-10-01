# ============================================================================
# File: synth_xilinx.tcl
# Description: Resource estimate of the full 16x16 npu_fpga_top for an AMD
#              7-series device (synthesis only; no open-source P&R used here)
#   yosys -c pd/fpga/synth_xilinx.tcl
# ============================================================================
yosys -import
set fp [open src/npu.f r]
set files [split [string trim [read $fp]] "\n"]
close $fp
read_verilog -sv -DFPGA {*}$files src/fpga/npu_fpga_top.sv
chparam -set ARR 16 npu_fpga_top
synth_xilinx -top npu_fpga_top -family xc7
tee -o pd/reports/fpga_synth_xc7_arr16.txt stat
