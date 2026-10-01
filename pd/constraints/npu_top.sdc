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

# Layer configuration used by the epilogue (RQ_MULT, RQ_SHIFT, FLAGS) is
# write-protected while the NPU is busy (npu_csr), so it is static whenever
# the datapath samples it. Its first use is a drain, at least CAP_DELAY (18)
# cycles after start. Allow 4 cycles; 4 is conservative.
# (Yosys autoname names each flop after the signal it drives: rq_mult,
# rq_shift_q, and the FLAGS bits bias_en / relu_en / signed_act.)
set cfg_regs [get_cells -quiet {u_csr/*rq_mult* u_csr/*rq_shift_q* u_csr/*flags_q*
                                u_csr/*bias_en* u_csr/*relu_en* u_csr/*signed_act*}]
if {[llength $cfg_regs] > 0} {
  set_multicycle_path 4 -setup -from $cfg_regs
  set_multicycle_path 3 -hold  -from $cfg_regs
}
puts "SDC: [llength $cfg_regs] static configuration registers get 4-cycle setup paths"
