# ============================================================================
# File: npu_top.sdc
# Description: Timing constraints for npu_top (pre-layout)
#   CLK_PERIOD (ns) is set by the calling script before this file is read.
# Assumptions, stated so the numbers can be judged:
#   - Ideal clock (no CTS). Clock uncertainty covers expected skew (+ jitter
#     for setup).
#   - I/O ports belong to a synchronous SoC fabric; 30% of the period is
#     consumed outside the NPU on every input and output.
#   - rst_n is asynchronous and synchronized externally; its buffer tree is
#     built during CTS/PnR, so its timing arcs are excluded here.
# ============================================================================

if {![info exists CLK_PERIOD]} { set CLK_PERIOD 20.0 }

create_clock -name clk -period $CLK_PERIOD [get_ports clk]
# Setup: skew + jitter margin. Hold: skew only (jitter is cycle-to-cycle and
# cancels on same-edge hold checks).
set_clock_uncertainty -setup 0.50 [get_clocks clk]
set_clock_uncertainty -hold  0.10 [get_clocks clk]
set_clock_transition  0.15 [get_clocks clk]

set data_inputs [delete_from_list [all_inputs] [get_ports {clk rst_n}]]
set_input_delay  [expr {0.3 * $CLK_PERIOD}] -clock clk $data_inputs
set_output_delay [expr {0.3 * $CLK_PERIOD}] -clock clk [all_outputs]

set_driving_cell -lib_cell sky130_fd_sc_hd__buf_4 -pin X $data_inputs
set_load 0.02 [all_outputs]

set_false_path -from [get_ports rst_n]
set_ideal_network [get_ports rst_n]
