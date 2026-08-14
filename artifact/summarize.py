#!/usr/bin/env python3
"""Merge the V1 perf CSV and the V2 dseg CSV into one comparison table.

Usage: summarize.py <v1_vs_v2_density.csv> <v2_dseg_tmp.csv>
Appends the V2 rows to the first CSV and prints the merged table.
V1 rows:   arch,dens,cycles,useful_macs,util_pct        (header line first)
V2 (dseg): tag,dens,tot,pre,walk,fs,idle,swap,drain,walk_busy,macs
"""
import sys

LANES = 32          # 8 PEs x 4 lanes on both sides of the comparison


def main(v1_path, v2_path):
    rows = {}                                    # (arch, dens) -> (cyc, macs)
    for ln in open(v1_path):
        p = ln.strip().split(",")
        if len(p) == 5 and p[0] != "arch":
            rows[(p[0], float(p[1]))] = (int(p[2]), int(p[3]))
    for ln in open(v2_path):
        p = ln.strip().split(",")
        if len(p) == 11:
            rows[(p[0], float(p[1]))] = (int(p[2]), int(p[10]))

    with open(v1_path, "a") as fh:
        for (arch, d), (cyc, macs) in sorted(rows.items()):
            if arch != "V1":
                fh.write(f"{arch},{d},{cyc},{macs},"
                         f"{100.0 * macs / (cyc * LANES):.2f}\n")

    archs = sorted({a for (a, _) in rows})
    dens = sorted({d for (_, d) in rows})
    print(f"\n{'':>6}" + "".join(f"{a + ' cyc':>12}{a + ' util':>10}"
                                 for a in archs))
    for d in dens:
        line = f"{d:>6.1f}"
        for a in archs:
            if (a, d) in rows:
                cyc, macs = rows[(a, d)]
                line += f"{cyc:>12,}{100.0 * macs / (cyc * LANES):>9.1f}%"
            else:
                line += f"{'--':>12}{'--':>10}"
        print(line)
    v1d = {d: c for (a, d), (c, _) in rows.items() if a == "V1"}
    for a in archs:
        if a == "V1":
            continue
        sp = [v1d[d] / rows[(a, d)][0] for d in dens
              if d in v1d and (a, d) in rows]
        if sp:
            print(f"\n{a} speedup over V1: "
                  + ", ".join(f"{s:.2f}x" for s in sp)
                  + f"  (geomean {_geomean(sp):.2f}x)")


def _geomean(xs):
    p = 1.0
    for x in xs:
        p *= x
    return p ** (1.0 / len(xs))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
