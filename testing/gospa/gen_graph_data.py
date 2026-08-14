#!/usr/bin/env python3
"""
gen_graph_data.py -- fresh RTL-measured datasets for poster graphs, all
golden-checked (test_gospa_dseg pipelined flow, H=32 F=3 S=1, 32-out-ch
pool, 3 input channels). Four sweeps:

  1. router_width.csv   : S2_BEATS 1..16 at densities {0.3, 0.6, 1.0}
  2. scanner_width.csv  : STAGE1_BATCH 1..64 at the same densities
  3. drain_width.csv    : DRAIN_W 1..64 at the same densities
  4. sparsity_asym.csv  : dw varied @ da=1.0, and da varied @ dw=1.0
                          (the two sparsity types behave differently)

Tidy columns: sweep, param, da, dw, cycles, useful_macs, util_pct, plus the
wall-clock segment shares (walk/fs/chew/swap/pre/drain %).

Usage (from testing/gospa, venv active):  python gen_graph_data.py
Writes CSVs to ../../poster/data/ and a summary to gospa_graphdata.txt.
"""
import csv
import os
import subprocess

_HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.abspath(os.path.join(_HERE, "..", "..", "poster", "data"))
TMPCSV = os.path.join(_HERE, "_dseg_tmp.csv")

BASE = dict(H=32, F=3, S=1, N_PE=8, N_MULTS=4, N_ROWS=32, N_NZ_MAX=1024,
            FIFO_D=2048, FIFOB_D=2048, STAGE1_BATCH=16, FILL_W=16,
            S2_BEATS=16, DRAIN_W=64)
DENS3 = "0.3,0.6,1.0"
LANES = BASE["N_PE"] * BASE["N_MULTS"]


def run(cfg_over, env_over, tag):
    if os.path.exists(TMPCSV):
        os.remove(TMPCSV)
    v = dict(BASE, **cfg_over)
    env = dict(os.environ, DSEG_CSV=TMPCSV, DSEG_TAG=tag,
               FIFOB_D=str(v["FIFOB_D"]), **env_over)
    subprocess.run(["make", "clean"], cwd=_HERE, env=env, capture_output=True)
    args = (["make", "MODULE=test_gospa_dseg", "SIM=verilator"]
            + [f"{k}={val}" for k, val in v.items()])
    r = subprocess.run(args, cwd=_HERE, env=env, capture_output=True,
                       text=True)
    assert r.returncode == 0 and "FAIL=0" in r.stdout, (
        f"{tag} FAILED:\n" + "\n".join(r.stdout.splitlines()[-15:]))
    rows = []
    with open(TMPCSV) as fh:
        for ln in fh:
            p = ln.strip().split(",")
            # tag,dens,tot,pre,walk,fs,idle,swap,drain,walk_busy,macs
            tot = int(p[2])
            segs = [int(x) for x in p[3:9]]      # pre,walk,fs,idle,swap,drain
            rows.append(dict(dens=float(p[1]), cycles=tot, macs=int(p[10]),
                             pre=segs[0], walk=segs[1], fs=segs[2],
                             chew=segs[3], swap=segs[4], drain=segs[5]))
    print(f"  {tag}: {len(rows)} points", flush=True)
    return rows


def emit(path, rows):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["sweep", "param", "da", "dw", "cycles", "useful_macs",
                    "util_pct", "walk_pct", "fs_pct", "chew_pct", "swap_pct",
                    "pre_pct", "drain_pct"])
        for r in rows:
            tot = r["cycles"]
            w.writerow([r["sweep"], r["param"], r["da"], r["dw"], tot,
                        r["macs"], round(100.0 * r["macs"] / (tot * LANES), 2)]
                       + [round(100.0 * r[k] / tot, 2)
                          for k in ("walk", "fs", "chew", "swap", "pre",
                                    "drain")])
    print("wrote", path, flush=True)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    summary = []

    # 1. router width -------------------------------------------------------
    out = []
    for s2 in (1, 2, 4, 8, 16):
        for r in run({"S2_BEATS": s2}, {"DENS": DENS3}, f"router{s2}"):
            out.append(dict(sweep="router_width", param=s2, da=r["dens"],
                            dw=r["dens"], **r))
    emit(os.path.join(OUTDIR, "router_width.csv"), out)
    summary.append(("router_width", len(out)))

    # 2. scanner width ------------------------------------------------------
    out = []
    for sb in (1, 4, 16, 64):
        for r in run({"STAGE1_BATCH": sb}, {"DENS": DENS3}, f"scan{sb}"):
            out.append(dict(sweep="scanner_width", param=sb, da=r["dens"],
                            dw=r["dens"], **r))
    emit(os.path.join(OUTDIR, "scanner_width.csv"), out)
    summary.append(("scanner_width", len(out)))

    # 3. drain width --------------------------------------------------------
    out = []
    for dwid in (1, 8, 64):
        for r in run({"DRAIN_W": dwid}, {"DENS": DENS3}, f"drain{dwid}"):
            out.append(dict(sweep="drain_width", param=dwid, da=r["dens"],
                            dw=r["dens"], **r))
    emit(os.path.join(OUTDIR, "drain_width.csv"), out)
    summary.append(("drain_width", len(out)))

    # 4. asymmetric sparsity (base build, two env-only runs) ---------------
    out = []
    for r in run({}, {"DA_FIX": "1.0"}, "dwsweep"):
        out.append(dict(sweep="dw_at_da1", param=r["dens"], da=1.0,
                        dw=r["dens"], **r))
    for r in run({}, {"DW_FIX": "1.0"}, "dasweep"):
        out.append(dict(sweep="da_at_dw1", param=r["dens"], da=r["dens"],
                        dw=1.0, **r))
    emit(os.path.join(OUTDIR, "sparsity_asym.csv"), out)
    summary.append(("sparsity_asym", len(out)))

    with open(os.path.join(_HERE, "gospa_graphdata.txt"), "w") as fh:
        fh.write("RTL graph datasets (golden-checked) -> poster/data/\n")
        for name, n in summary:
            fh.write(f"  {name}.csv : {n} rows\n")
    print("all sweeps done")


if __name__ == "__main__":
    main()
