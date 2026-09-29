# ============================================================================
# File: sta.tcl
# Description: Placement-aware design repair + multi-corner STA with OpenROAD
#   openroad -exit pd/sta/sta.tcl        (run from the repo root)
# Environment:
#   CLK_PERIOD  ns (default 20.0)
#   SYNTH_TAG   netlist suffix (default "")
#   RPT_TAG     report suffix (default SYNTH_TAG)
#   UTIL        floorplan core utilization, percent (default 50)
# Flow:
#   1. Link the Yosys netlist against sky130 HD + OpenRAM macro LEF/liberty.
#   2. Rough floorplan + global placement (standard cells AND SRAM macros).
#      No legalization, CTS or routing: the placement exists only so that
#      (a) the resizer can build real buffer trees for high-fanout nets and
#      (b) wire parasitics are estimated from placed pin locations.
#   3. repair_design (max fanout/slew/cap), repair_timing (setup, then hold).
#   4. STA at three corners with placement-estimated parasitics:
#        ss_100C_1v60 (setup-critical), tt_025C_1v80, ff_n40C_1v95 (hold)
#      The OpenRAM macro is characterised at TT only; its TT timing is used in
#      every corner (stated limitation).
# ============================================================================

set CLK_PERIOD [expr {[info exists ::env(CLK_PERIOD)] ? $::env(CLK_PERIOD) : 20.0}]
set tag        [expr {[info exists ::env(SYNTH_TAG)] ? $::env(SYNTH_TAG) : ""}]
set rpt_tag    [expr {[info exists ::env(RPT_TAG)] ? $::env(RPT_TAG) : $tag}]
set util       [expr {[info exists ::env(UTIL)] ? $::env(UTIL) : 50}]
set rpt        pd/reports/sta${rpt_tag}

# Per-corner renamed copies of the TT SRAM liberty (made by scripts/run_synth.py);
# OpenSTA rejects loading one library name into several corners.
set sram_lib_fmt pd/sram/sky130_sram_1kbyte_1rw1r_32x256_8_TT_as_%s.lib

set db pd/netlist/npu_top${tag}_placed.odb
set resume [expr {[info exists ::env(RESUME)] && $::env(RESUME) && [file exists $db]}]

if {$resume} {
  read_db $db
} else {
  read_lef pd/lib/sky130_fd_sc_hd__nom.tlef
  read_lef pd/lib/sky130_fd_sc_hd.lef
  read_lef pd/sram/sky130_sram_1kbyte_1rw1r_32x256_8.lef
}

define_corners ss tt ff
read_liberty -corner ss pd/lib/sky130_fd_sc_hd__ss_100C_1v60.lib
read_liberty -corner tt pd/lib/sky130_fd_sc_hd__tt_025C_1v80.lib
read_liberty -corner ff pd/lib/sky130_fd_sc_hd__ff_n40C_1v95.lib
foreach c {ss tt ff} { read_liberty -corner $c [format $sram_lib_fmt $c] }

if {!$resume} {
  read_verilog pd/netlist/npu_top${tag}.v
  link_design npu_top
}
read_sdc pd/constraints/npu_top.sdc
set_dont_use {*lpflow* *probe* *clkdlybuf* *dlygate* *dlymetal*}
set_wire_rc -signal -layer met2
set_wire_rc -clock  -layer met5

proc summary {f label} {
  puts $f "== $label"
  foreach c {ss tt ff} {
    set co [sta::find_corner $c]
    # STA internals are in seconds; format_time converts to the user unit (ns)
    puts $f [format "  %s  setup WNS %9s ns  TNS %12s ns   hold WNS %8s ns  TNS %10s ns" $c \
      [sta::format_time [sta::worst_slack_corner $co max] 3] \
      [sta::format_time [sta::total_negative_slack_corner_cmd $co max] 3] \
      [sta::format_time [sta::worst_slack_corner $co min] 3] \
      [sta::format_time [sta::total_negative_slack_corner_cmd $co min] 3]]
  }
  flush $f
}

# --------------------------------------------------------------------------
# Rough placement for buffering and wire estimation
# --------------------------------------------------------------------------
set f [open ${rpt}_summary.txt [expr {$resume ? "a" : "w"}]]
if {!$resume} {
  puts $f "npu_top STA, clock period $CLK_PERIOD ns ([format %.1f [expr {1000.0 / $CLK_PERIOD}]] MHz)"
  puts $f "Global placement at ${util}% utilization, placement-estimated parasitics (met2 RC),"
  puts $f "ideal clock: setup uncertainty 0.5 ns, hold uncertainty 0.1 ns."
  puts $f ""
  initialize_floorplan -utilization $util -aspect_ratio 1.0 -core_space 20 -sites unithd
  make_tracks
  place_pins -hor_layers met3 -ver_layers met2
  global_placement -density 0.60
  estimate_parasitics -placement
  summary $f "after global placement, before repair"

  # ------------------------------------------------------------------------
  # Repair design rules (buffer trees for fanout, slew, cap)
  # ------------------------------------------------------------------------
  set_max_fanout 16 [current_design]
  repair_design
  write_db $db
} else {
  puts $f "\n(resumed from $db at clock period $CLK_PERIOD ns)"
}
estimate_parasitics -placement
summary $f "after repair_design (fanout/slew/cap)"
report_check_types -max_slew -max_capacitance -max_fanout -violators > ${rpt}_post_drv_repair.txt
report_checks -path_delay max -corner ss -group_count 10 -format full_clock_expanded \
  -fields {slew cap fanout} -digits 3 > ${rpt}_setup_ss_before_timing_repair.txt

repair_timing -setup
estimate_parasitics -placement
summary $f "after repair_timing -setup"
report_checks -path_delay max -corner ss -group_count 10 -format full_clock_expanded \
  -fields {slew cap fanout} -digits 3 > ${rpt}_setup_ss.txt

repair_timing -hold -hold_margin 0.05
estimate_parasitics -placement
summary $f "final (after hold repair)"
close $f

# --------------------------------------------------------------------------
# Final reports
# --------------------------------------------------------------------------
report_checks -path_delay max -corner ss -group_count 10 -format full_clock_expanded \
  -fields {slew cap fanout} -digits 3 > ${rpt}_setup_ss_final.txt
report_checks -path_delay max -corner tt -group_count 3 -digits 3 > ${rpt}_setup_tt.txt
report_checks -path_delay min -corner ff -group_count 5 -digits 3 > ${rpt}_hold_ff.txt
report_check_types -max_slew -max_capacitance -max_fanout -violators > ${rpt}_post_repair_drv.txt
report_design_area > ${rpt}_area.txt
write_verilog pd/netlist/npu_top${rpt_tag}_repaired.v
