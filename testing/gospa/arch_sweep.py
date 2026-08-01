#!/usr/bin/env python3
"""
arch_sweep.py -- architecture sweep for best MobileNetV2 FPS: vary
(N_PE, N_MULTS) and FIFO-B depth, measure a 3-layer real-tensor proxy
(conv1 + the two largest pointwise layers -- pw dominates MobileNet
runtime), rank by proxy cycles, then check FIFO sensitivity on the winner.

Depthwise layers are EXCLUDED from the scaling sweep: the dw mosaic mapping
is currently hardwired to a 3x3 tile grid / 8 PEs, so its cycles do not
scale with N_PE (noted in the report; dw is ~11% of network cycles).

S2_BEATS is scaled as 64 // N_MULTS so the router pop width stays 64 pairs
per cycle across configs.

Usage (from testing/gospa, venv active):
    python arch_sweep.py
Writes gospa_arch_sweep.txt. Run the full network at the winner with:
    N_PE=<pe> N_MULTS=<m> python run_mobilenet_all.py
"""
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "ref")))
sys.path.insert(0, _HERE)

import mobilenet_layers                                   # noqa: E402
from run_mobilenet_all import layer_config                # noqa: E402

FCLK_MHZ = float(os.environ.get("FCLK_MHZ", "100"))
FCLK = FCLK_MHZ * 1e6
PROXY = [0, 44, 51]
GRID = [(4, 4), (8, 4), (16, 4), (32, 4), (8, 8), (16, 8)]
FIFOB_GRID = [256, 1024, 4096]
OUT = os.path.join(_HERE, "gospa_arch_sweep.txt")
CSV = os.path.join(_HERE, "arch_sweep_rows.csv")


def run_layer(idx, m, n_pe, n_mults, fifob=None):
    tag, v = layer_config(m, n_pe, n_mults)
    v["S2_BEATS"] = max(4, 64 // n_mults)
    if fifob:
        v["FIFOB_D"] = fifob
    env = dict(os.environ, LAYER_IDX=str(idx), ROWS_CSV=CSV)
    subprocess.run(["make", "clean"], cwd=_HERE, env=env, capture_output=True)
    args = (["make", "MODULE=test_gospa_mobilenet", "SIM=verilator"]
            + [f"{k}={val}" for k, val in v.items()])
    r = subprocess.run(args, cwd=_HERE, env=env, capture_output=True,
                       text=True)
    assert r.returncode == 0 and "FAIL=0" in r.stdout, (
        f"L{idx} @ PE{n_pe}xM{n_mults} FAILED:\n"
        + "\n".join(r.stdout.splitlines()[-20:]))
    with open(CSV) as fh:
        p = fh.readlines()[-1].strip().split(",")
    return int(p[7]), int(p[9])          # useful MACs, total cycles


def main():
    meta = {m["idx"]: m for m in mobilenet_layers.list_layers()}
    mobilenet_layers.ensure_extracted(PROXY)
    if os.path.exists(CSV):
        os.remove(CSV)

    lines = [
        "GOSPA ARCH SWEEP -- MobileNetV2 3-layer real-tensor proxy "
        "(conv1 + pw 960->160 + pw 320->1280)",
        f"f_clk={FCLK_MHZ:.0f} MHz; S2_BEATS=64/N_MULTS; dw excluded "
        "(mosaic mapping fixed at 8 PEs -- see report)",
        "",
        f"{'PExM':>7} {'mults':>6} {'proxyCyc':>9} {'lat_ms':>7} "
        f"{'util%':>6} {'GMAC/s':>7} {'speedup':>8}",
        "-" * 56,
    ]
    results = []
    base_cyc = None
    for (pe, mm) in GRID:
        tot_c = tot_u = 0
        for idx in PROXY:
            u, c = run_layer(idx, meta[idx], pe, mm)
            tot_u += u
            tot_c += c
        lanes = pe * mm
        if (pe, mm) == (8, 4):
            base_cyc = tot_c
        results.append((pe, mm, tot_c, tot_u))
        print(f"  PE{pe}xM{mm}: {tot_c} cyc, "
              f"util {100.0 * tot_u / (tot_c * lanes):.1f}%", flush=True)
    for (pe, mm, tot_c, tot_u) in results:
        lanes = pe * mm
        lines.append(
            f"{f'{pe}x{mm}':>7} {lanes:>6} {tot_c:>9} "
            f"{tot_c / FCLK * 1e3:>7.3f} "
            f"{100.0 * tot_u / (tot_c * lanes):>6.1f} "
            f"{tot_u * FCLK / tot_c / 1e9:>7.2f} "
            f"{(base_cyc or tot_c) / tot_c:>7.2f}x")

    best = min(results, key=lambda r: r[2])
    pe, mm = best[0], best[1]
    lines += ["", f"winner: PE{pe}xM{mm} -- FIFO-B depth sensitivity:"]
    for fb in FIFOB_GRID:
        tot_c = tot_u = 0
        for idx in PROXY:
            u, c = run_layer(idx, meta[idx], pe, mm, fifob=fb)
            tot_u += u
            tot_c += c
        lines.append(f"  FIFOB_D={fb:>5}: {tot_c:>9} cyc  "
                     f"util {100.0 * tot_u / (tot_c * pe * mm):>5.1f}%")
        print(lines[-1], flush=True)

    with open(OUT, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {OUT}\nrun full network at winner: "
          f"N_PE={pe} N_MULTS={mm} python run_mobilenet_all.py")


if __name__ == "__main__":
    main()
