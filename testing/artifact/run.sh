#!/usr/bin/env bash
# ============================================================================
# GoSPA report artifact runner — regenerates the report's RTL sim results.
#
#   ./run.sh smoke      functional suite (~2 min)
#   ./run.sh mobilenet  MobileNetV2 conv1 golden + perf tests (~2 min)
#   ./run.sh alexnet    AlexNet conv3/4/5 tiled layers (~30-45 min)
#   ./run.sh all        everything above except alexnet
#
# Results land in artifact/results/*.csv; each run is golden-checked in the
# simulation itself — a FAIL aborts the artifact.
# Requirements: Verilator >= 5.0, python venv with cocotb 2.x, numpy, torch.
# ============================================================================
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
RES="$HERE/results"
mkdir -p "$RES"

log() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

run_smoke() {
  log "Functional suite (testing/gospa)"
  make -C "$ROOT/testing/gospa" clean >/dev/null; make -C "$ROOT/testing/gospa" MODULE=test_gospa | grep "TESTS="
}

run_mobilenet() {
  local out="$RES/mobilenet_conv1.csv"
  : > "$out"
  echo "arch,layer,cycles,useful_macs,util_pct,fps_100MHz" >> "$out"

  log "MobileNetV2 conv1 (real 80x80 apple input, golden-checked)"
  local v2log="$RES/_v2_apple.log"
  make -C "$ROOT/testing/gospa" apple STAGE1_BATCH=16 FILL_W=16 | tee "$v2log" \
    | grep -E "total cycles|TESTS="
  python3 - "$out" "$v2log" <<'PY'
import re, sys
out, log = sys.argv[1], sys.argv[2]
cyc = int(re.search(r"total cycles.*= (\d+)", open(log).read()).group(1))
macs = 1027158                       # conv1 useful MACs
with open(out, "a") as fh:
    fh.write(f"V2,mobilenet_conv1,{cyc},{macs},"
             f"{100.0*macs/(cyc*32):.2f},{1e8/cyc:.1f}\n")
PY

  log "MobileNetV2 conv1, 8x4 @ 100 MHz"
  column -s, -t "$out"
}

run_mobilenet_e2e() {
  log "Full MobileNetV2 network, 52 layers, 32x4 (real tensors — slow)"
  N_PE="${N_PE:-32}" N_MULTS=4 \
    MN_ROWS="$RES/mobilenet_e2e_${N_PE:-32}x4_rows.csv" \
    MN_OUT="$RES/mobilenet_e2e_${N_PE:-32}x4.txt" \
    python3 "$ROOT/testing/gospa/run_mobilenet_all.py" | tail -8
}

run_alexnet() {
  log "AlexNet conv3/4/5 (tiled, real tensors — slow)"
  for L in 3 4 5; do
    make -C "$ROOT/testing/gospa" clean >/dev/null
    LAYER=$L ROWS_CSV="$RES/alexnet_rows.csv" \
      make -C "$ROOT/testing/gospa" MODULE=test_gospa_alexnet \
           H=16 F=3 S=1 N_ROWS=16 N_NZ_MAX=256 | grep -E "TESTS=|FAIL"
  done
  echo "rows appended to $RES/alexnet_rows.csv"
}

case "${1:-all}" in
  smoke)         run_smoke ;;
  mobilenet)     run_mobilenet ;;
  mobilenet-e2e) run_mobilenet_e2e ;;
  alexnet)       run_alexnet ;;
  all)           run_smoke; run_mobilenet ;;
  *) echo "usage: $0 {smoke|mobilenet|mobilenet-e2e|alexnet|all}"; exit 1 ;;
esac
