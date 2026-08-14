# =============================================================================
# run_sweep.tcl -- synthesize ONE gospa config and append its utilisation to CSV
# GoSPA Project -- Team 19, ECE 720 (Spring 2026)
#
# Invoked once per config by run_sweep.sh:
#   vivado -mode batch -notrace -source run_sweep.tcl -tclargs <N_PE> <N_MULTS> <FIFO_D> [RUN_IMPL]
#
# The FIFO_D argument sets BOTH FIFO depths together (FIFO-A = FIFO_D and
# FIFO-B = FIFOB_D = FIFO_D), so the sweep axis is a single "FIFO depth" knob.
#
# RUN_IMPL (optional, default 0): 0 = synth only (fast; utilisation is accurate
# post-synthesis), 1 = full place & route (matches RESULTS.md, slower).
# Appends one row to sweep_results.csv.  H=8, F=3, S=1 fixed (small, KV260-fitting).
# =============================================================================

if {[llength $argv] < 3} {
    puts "usage: -tclargs N_PE N_MULTS FIFO_D \[RUN_IMPL\]"
    exit 1
}
lassign $argv NPE NMULTS FIFOD RUN_IMPL
if {$RUN_IMPL eq ""} { set RUN_IMPL 0 }

set PART       "xck26-sfvc784-2LV-c"
set SCRIPT_DIR [file dirname [file normalize [info script]]]
set RTL_DIR    [file normalize $SCRIPT_DIR/../..]
set CSV        [file join $SCRIPT_DIR sweep_results.csv]

# ---- read RTL, neutralising `default_nettype none` (same shim as run_synth) --
set BUILD [file join $SCRIPT_DIR build_rtl]
file delete -force $BUILD
file mkdir $BUILD
set files [concat \
    [glob -nocomplain $RTL_DIR/common/*.sv] \
    [glob -nocomplain $RTL_DIR/common/arith/*.sv] \
    [glob -nocomplain $RTL_DIR/apu/stage1/*.sv] \
    [glob -nocomplain $RTL_DIR/apu/stage2/*.sv] \
    [glob -nocomplain $RTL_DIR/apu/*.sv] \
    [glob -nocomplain $RTL_DIR/pe/*.sv] \
    [glob -nocomplain $RTL_DIR/*.sv]]
foreach f $files {
    set fh [open $f r]; set txt [read $fh]; close $fh
    regsub -all {`default_nettype\s+none} $txt {`default_nettype wire} txt
    set dst [file join $BUILD [file tail $f]]
    set oh [open $dst w]; puts -nonewline $oh $txt; close $oh
    read_verilog -sv $dst
}

# ---- synthesize the config (FIFO_D drives BOTH FIFO-A and FIFO-B) ----------
synth_design -top gospa -part $PART -mode out_of_context -flatten_hierarchy rebuilt \
    -generic H=8 -generic F=3 -generic S=1 -generic N_ROWS=8 -generic N_NZ_MAX=64 \
    -generic N_PE=$NPE -generic N_MULTS=$NMULTS \
    -generic FIFO_D=$FIFOD -generic FIFOB_D=$FIFOD

if {$RUN_IMPL} {
    opt_design
    place_design
    route_design
}

# ---- parse utilisation -----------------------------------------------------
set rpt [report_utilization -return_string]
set lut 0; set ff 0; set dsp 0; set bram 0
regexp {CLB LUTs\*?\s*\|\s*(\d+)}    $rpt -> lut
regexp {CLB Registers\s*\|\s*(\d+)}  $rpt -> ff
regexp {DSPs\s*\|\s*(\d+)}           $rpt -> dsp
regexp {Block RAM Tile\s*\|\s*(\d+)} $rpt -> bram

# ---- append one CSV row (header written by run_sweep.sh) -------------------
set out [open $CSV a]
puts $out "$NPE,$NMULTS,$FIFOD,$lut,$ff,$dsp,$bram"
close $out
puts "SWEEP_ROW: N_PE=$NPE N_MULTS=$NMULTS FIFO_D=$FIFOD -> LUT=$lut FF=$ff DSP=$dsp BRAM=$bram"
