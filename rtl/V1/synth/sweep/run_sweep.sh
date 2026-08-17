#!/usr/bin/env bash
# =============================================================================
# run_sweep.sh -- drive the V1 gospa resource sweep across N_PE / N_MULTS / FIFO_D
# GoSPA Project -- Team 19, ECE 720 (Spring 2026)
#
# V1 counterpart of rtl/V2/synth/sweep/run_sweep.sh. Same config points, so the
# two sweeps can be plotted against each other. Runs one Vivado synth per config
# (each independent), appending a row to sweep_results.csv. Then run:
#   python3 plot_sweep.py
#
#   bash run_sweep.sh            # synth-only (fast)
#   RUN_IMPL=1 bash run_sweep.sh # full place & route (slower)
#
# NOTE: this truncates sweep_results.csv before the first run.
# =============================================================================
set -e
cd "$(dirname "$0")"

RUN_IMPL="${RUN_IMPL:-0}"
CSV=sweep_results.csv
echo "N_PE,N_MULTS,FIFO_D,LUT,FF,DSP,BRAM" > "$CSV"

# Baseline held while sweeping each axis: N_PE=8, N_MULTS=4, FIFO_D=64.
CONFIGS=(
  # N_PE sweep (N_MULTS=4, FIFO_D=64)
  "2 4 64"  "4 4 64"  "8 4 64"  "16 4 64"
  # N_MULTS sweep (N_PE=8, FIFO_D=64)   -- 8 4 64 already above
  "8 1 64"  "8 2 64"  "8 8 64"
  # FIFO_D sweep (N_PE=8, N_MULTS=4)    -- 8 4 64 already above
  "8 4 16"  "8 4 32"  "8 4 128"
)

for cfg in "${CONFIGS[@]}"; do
  read -r npe nmults fifod <<< "$cfg"
  echo "=================================================="
  echo "  V1 synth: N_PE=$npe  N_MULTS=$nmults  FIFO_D=$fifod  (RUN_IMPL=$RUN_IMPL)"
  echo "=================================================="
  vivado -mode batch -notrace -source run_sweep.tcl -tclargs "$npe" "$nmults" "$fifod" "$RUN_IMPL"
done

echo ""
echo "V1 sweep complete -> $CSV"
rm -rf build_rtl .Xil vivado*.jou vivado*.log 2>/dev/null || true
