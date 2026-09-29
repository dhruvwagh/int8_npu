# ============================================================================
# File: synth.tcl
# Description: Yosys synthesis of npu_top to SkyWater 130 nm HD standard cells
#   yosys -c pd/synth/synth.tcl     (run from the repo root)
# Environment:
#   CLK_PERIOD_PS  delay target for ABC (default 20000 = 50 MHz)
#   SYNTH_TAG      suffix for output files (default "")
# Outputs:
#   pd/netlist/npu_top<tag>.v               gate-level netlist (hierarchical)
#   pd/reports/synth_area<tag>.txt          per-module cell/area report
# Hierarchy is kept so the area report breaks down by module; the 256 PEs
# share one synthesized npu_pe.
# ============================================================================

yosys -import

set period [expr {[info exists ::env(CLK_PERIOD_PS)] ? $::env(CLK_PERIOD_PS) : 20000}]
set tag    [expr {[info exists ::env(SYNTH_TAG)] ? $::env(SYNTH_TAG) : ""}]
set lib    pd/lib/sky130_fd_sc_hd__tt_025C_1v80.lib
set sram   pd/sram/sky130_sram_1kbyte_1rw1r_32x256_8_TT_1p8V_25C.lib

# Cells OpenLane also excludes: low-power/probe cells and delay cells
set dont_use {*lpflow* *probe* *clkdlybuf* *dlygate* *dlymetal*}

read_liberty -lib $lib
read_liberty -lib $sram

set fp [open src/npu.f r]
set files [split [string trim [read $fp]] "\n"]
close $fp
read_verilog -sv -DSYNTHESIS {*}$files

hierarchy -check -top npu_top
synth -top npu_top
opt_clean -purge

# Technology mapping
set du_args {}
foreach d $dont_use { lappend du_args -dont_use $d }
dfflibmap -liberty $lib {*}$du_args
# Delay-driven mapping, then fanout buffering (max fanout 16) and gate sizing
# against the delay target, as in OpenLane's delay strategies. Buffering is
# done here because pre-placement repair in OpenROAD cannot build buffer trees.
set abc_script "+strash;&get,-n;&st;&dch;&nf,{D};&put;buffer,-N,16;upsize,{D};dnsize,{D};stime,-p"
abc -liberty $lib -D $period {*}$du_args -script $abc_script
opt_clean -purge

hilomap -singleton -hicell sky130_fd_sc_hd__conb_1 HI -locell sky130_fd_sc_hd__conb_1 LO
setundef -zero
splitnets
opt_clean -purge

tee -o pd/reports/synth_area${tag}.txt stat -liberty $lib -liberty $sram -top npu_top
write_verilog -noattr -noexpr -nohex -nodec pd/netlist/npu_top${tag}.v

# Sanity checks (after writing outputs so a failure keeps the artifacts)
tee -o pd/reports/synth_check${tag}.txt check
select -assert-count 0 {t:$_DLATCH_*}
select -assert-none {t:$_*}
