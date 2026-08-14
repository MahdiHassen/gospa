# =============================================================================
# run_synth.tcl -- Vivado out-of-context synth + place & route for gospa.sv
# GoSPA Project -- Team 19, ECE 720 (Spring 2026)
#
# Gives the FPGA numbers: post-route Fmax, critical path, and resource
# utilisation (LUT / FF / DSP / BRAM). Runs out-of-context (the design's big
# host interface isn't pinned out -- we time the core logic register-to-register,
# which is what sets Fmax).
#
# Run:   cd rtl/synth/vivado && vivado -mode batch -source run_synth.tcl
# Edit the knobs below. Defaults: Kria KV260 (K26 SOM), a KV260-fitting config.
#
# Notes
#  * Vivado reads SystemVerilog natively -- no sv2v needed.
#  * SRAM/FIFO memories infer as BRAM; the accumulator array infers as FF (its
#    reset-all loop blocks BRAM), so N_CID (= E*E, set by H) drives the FF count.
#    Fmax / critical path are ~independent of config; utilisation scales with it.
#  * To scale up resources for the "full" number, raise H and/or target a bigger
#    part (e.g. xcu250-figd2104-2L-e = 3.4M FF).
# Outputs: timing_route.rpt, critical_paths.rpt, utilization_route.rpt
# =============================================================================

# ---- board / clock ----------------------------------------------------------
set PART      "xck26-sfvc784-2LV-c"   ;# Kria KV260 (K26 / ZU5EV). Bigger: xcu250-figd2104-2L-e
set PERIOD_NS 1.000                    ;# 1 ns = 1 GHz (aggressive: forces the tool to push Fmax).
set RUN_IMPL  1                        ;# 1 = place & route (real Fmax); 0 = synth-only (quick)
set TOP       "gospa"

# ---- design config (passed as -generic; KV260-fitting) ----------------------
# Paper array (N_PE=8, N_MULTS=4) with a small map so the FF accumulators fit.
array set CFG {
    H        8
    F        3
    S        1
    N_PE     8
    N_MULTS  4
    N_ROWS   8
    N_NZ_MAX 64
    FIFO_D   64
    DATA_W   16
    ACC_W    32
}

# ---- read RTL (script lives in rtl/synth/vivado -> ../.. is rtl/) ------------
set SCRIPT_DIR [file dirname [file normalize [info script]]]
set RTL_DIR    [file normalize $SCRIPT_DIR/../..]
set rtl_files [concat \
    [glob -nocomplain $RTL_DIR/common/*.sv] \
    [glob -nocomplain $RTL_DIR/common/arith/*.sv] \
    [glob -nocomplain $RTL_DIR/apu/stage1/*.sv] \
    [glob -nocomplain $RTL_DIR/apu/stage2/*.sv] \
    [glob -nocomplain $RTL_DIR/apu/*.sv] \
    [glob -nocomplain $RTL_DIR/pe/*.sv] \
    [glob -nocomplain $RTL_DIR/*.sv]]
# Vivado (unlike Verilator) rejects `input logic` ports under
# `default_nettype none` -- it wants an explicit `wire` net type. Rather than
# edit a dozen of the team's RTL files, neutralise the directive in throwaway
# copies (read by module name, so location is irrelevant) and read those.
set BUILD_RTL [file join $SCRIPT_DIR build_rtl]
file delete -force $BUILD_RTL
file mkdir $BUILD_RTL
set read_list {}
foreach f $rtl_files {
    set fh [open $f r]; set txt [read $fh]; close $fh
    regsub -all {`default_nettype\s+none} $txt {`default_nettype wire} txt
    set dst [file join $BUILD_RTL [file tail $f]]
    set oh [open $dst w]; puts -nonewline $oh $txt; close $oh
    lappend read_list $dst
}
puts "== reading [llength $read_list] RTL files (default_nettype neutralised) =="
foreach f $read_list { read_verilog -sv $f }

# ---- synthesize (out-of-context, with the config generics) ------------------
set gen {}
foreach {k v} [array get CFG] { lappend gen -generic $k=$v }
puts "== synth config: [array get CFG] =="
synth_design -top $TOP -part $PART -mode out_of_context -flatten_hierarchy rebuilt {*}$gen

create_clock -name clk -period $PERIOD_NS [get_ports clk]

# ---- helper: print a timing headline from the current (synth/route) netlist --
proc timing_headline {label period} {
    set path [lindex [get_timing_paths -delay_type max -max_paths 1] 0]
    set wns  [get_property SLACK $path]
    set ach  [expr {$period - $wns}]
    puts "----------------------------------------------------------------------"
    puts " $label"
    puts "   target period : $period ns  ([format %.1f [expr 1000.0/$period]] MHz)"
    puts "   WNS (slack)   : $wns ns   -> [expr {$wns >= 0 ? {MET} : {VIOLATED}}]"
    puts "   achievable    : ~[format %.3f $ach] ns  (Fmax ~ [format %.1f [expr 1000.0/$ach]] MHz)"
    puts "   crit path     : [get_property STARTPOINT_PIN $path]"
    puts "               -> : [get_property ENDPOINT_PIN $path]"
    puts "----------------------------------------------------------------------"
}

timing_headline "POST-SYNTH (estimate)" $PERIOD_NS

# ---- place & route for the REAL numbers -------------------------------------
if {$RUN_IMPL} {
    opt_design
    place_design
    phys_opt_design
    route_design

    report_timing_summary -delay_type max -max_paths 10 -file timing_route.rpt
    report_timing -delay_type max -max_paths 15 -path_type full_clock_expanded \
        -input_pins -file critical_paths.rpt
    report_utilization -file utilization_route.rpt

    puts "\n====================== POST-ROUTE (real) ======================"
    timing_headline "POST-ROUTE (signoff)" $PERIOD_NS
    puts "  resource utilisation -> see utilization_route.rpt (LUT/FF/DSP/BRAM)"
    puts "  critical path detail -> critical_paths.rpt (top path first)"
    puts "==============================================================="
} else {
    report_timing_summary -delay_type max -max_paths 10 -file timing_synth.rpt
    report_utilization -file utilization_synth.rpt
    puts "  (synth-only) reports: timing_synth.rpt, utilization_synth.rpt"
}
