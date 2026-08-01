#!/usr/bin/env python3
"""
get_rtl_metrics.py -- one-stop RTL metrics for the gospa accelerator:
FPS, multiplier utilization, and latency, measured (not modeled) on

  A. density classes da = dw = 0.1 .. 1.0 (synthetic 32x32, Cin=3, Cout=32
     pool -- the test_gospa_dseg workload), and
  B. a representative MobileNetV2 layer subset on the real apple-80 tensors
     (default: conv1, a dw layer, and pw layers across spatial sizes --
     covering all three mapping regimes without the full-network runtime).

Config: N_PE=8 x N_MULTS=4 with the proven fast settings (STAGE1_BATCH=16,
FILL_W=16, S2_BEATS=16, DRAIN_W=64) unless overridden via env (N_PE=...,
FCLK_MHZ=...).

Usage (from testing/gospa, venv active):
    python get_rtl_metrics.py
Writes gospa_rtl_metrics.txt.
"""
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "ref")))
sys.path.insert(0, _HERE)

import mobilenet_layers                                   # noqa: E402
from run_mobilenet_all import (                           # noqa: E402
    PERF_VARS, layer_config, next_pow2)

N_PE = int(os.environ.get("N_PE", "8"))
N_MULTS = int(os.environ.get("N_MULTS", "4"))
FCLK_MHZ = float(os.environ.get("FCLK_MHZ", "100"))
LAYERS = [int(s) for s in
          os.environ.get("METRIC_LAYERS", "0,1,5,33,44,51").split(",")]
OUT = os.path.join(_HERE, "gospa_rtl_metrics.txt")

LANES = N_PE * N_MULTS
FCLK = FCLK_MHZ * 1e6


def _make(vars_, env_extra):
    env = dict(os.environ, **env_extra)
    subprocess.run(["make", "clean"], cwd=_HERE, env=env, capture_output=True)
    args = (["make", "MODULE=" + env_extra["MODULE"], "SIM=verilator"]
            + [f"{k}={v}" for k, v in vars_.items()])
    r = subprocess.run(args, cwd=_HERE, env=env, capture_output=True,
                       text=True)
    ok = r.returncode == 0 and "FAIL=0" in r.stdout
    return ok, r.stdout


def density_section():
    csv = os.path.join(_HERE, "metrics_dseg_rows.csv")
    if os.path.exists(csv):
        os.remove(csv)
    vars_ = dict(H=32, F=3, S=1, N_PE=N_PE, N_MULTS=N_MULTS, N_ROWS=32,
                 N_NZ_MAX=1024, FIFO_D=2048, FIFOB_D=2048, **PERF_VARS)
    ok, out = _make(vars_, dict(MODULE="test_gospa_dseg", DSEG_CSV=csv,
                                FIFOB_D="2048"))
    assert ok, "density sweep FAILED:\n" + "\n".join(out.splitlines()[-20:])

    rows = []
    with open(csv) as fh:
        for ln in fh:
            p = ln.strip().split(",")
            rows.append((float(p[1]), int(p[2]), int(p[10])))
    lines = [
        "A. DENSITY CLASSES (da = dw; synthetic 32x32, Cin=3, Cout=32 pool)",
        f"{'dens':>5} {'cycles':>8} {'useMACs':>8} {'util%':>6} "
        f"{'MAC/cyc':>8} {'lat_us':>7} {'wl/s':>9}",
        "-" * 56,
    ]
    for d, tot, macs in rows:
        lines.append(
            f"{d:>5.1f} {tot:>8} {macs:>8} "
            f"{100.0 * macs / (tot * LANES):>6.1f} {macs / tot:>8.2f} "
            f"{tot / FCLK * 1e6:>7.1f} {FCLK / tot:>9,.0f}")
    return lines


def layer_section():
    meta = {m["idx"]: m for m in mobilenet_layers.list_layers()}
    mobilenet_layers.ensure_extracted(LAYERS)
    csv = os.path.join(_HERE, "metrics_layer_rows.csv")
    if os.path.exists(csv):
        os.remove(csv)

    lines = [
        "B. MOBILENETV2 LAYERS (real apple-80 tensors, golden-checked)",
        f"{'#':>3} {'type':<9} {'shape':<14} {'mode':<20} {'useMACs':>9} "
        f"{'cycles':>8} {'util%':>6} {'lat_ms':>7} {'GMAC/s':>7}",
        "-" * 92,
    ]
    for idx in LAYERS:
        m = meta[idx]
        tag, v = layer_config(m, N_PE, N_MULTS)
        ok, out = _make(v, dict(MODULE="test_gospa_mobilenet",
                                LAYER_IDX=str(idx), ROWS_CSV=csv))
        assert ok, f"layer {idx} FAILED:\n" + "\n".join(out.splitlines()[-20:])
        with open(csv) as fh:
            p = fh.readlines()[-1].strip().split(",")
        useful, total = int(p[7]), int(p[9])
        lines.append(
            f"{idx:>3} {m['type']:<9} "
            f"{str(m['cin']) + '->' + str(m['cout']) + ' @' + str(m['H']):<14} "
            f"{tag:<20} {useful:>9} {total:>8} "
            f"{100.0 * useful / (total * LANES):>6.1f} "
            f"{total / FCLK * 1e3:>7.3f} {useful * FCLK / total / 1e9:>7.2f}")
    return lines


def main():
    hdr = [
        "GOSPA RTL METRICS",
        f"N_PE={N_PE} x N_MULTS={N_MULTS} ({LANES} multipliers)  "
        f"f_clk={FCLK_MHZ:.0f} MHz  "
        + " ".join(f"{k}={v}" for k, v in PERF_VARS.items()),
        f"peak = {LANES * FCLK / 1e9:.2f} GMAC/s; util% = useful MACs / "
        f"(cycles x {LANES}); wl/s = full workload passes per second",
        "",
    ]
    lines = hdr + density_section() + [""] + layer_section()
    with open(OUT, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
