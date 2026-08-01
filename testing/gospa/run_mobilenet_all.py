#!/usr/bin/env python3
"""
run_mobilenet_all.py -- run every quantized conv layer of MobileNetV2 (real
apple-80 tensors) through the gospa RTL and aggregate a network-level
utilization report.

Layers are grouped by (H, F, S): each group is one Verilator build running
test_gospa_mobilenet with LAYER_IDX=<comma list>. The linear classifier is
not mapped (conv datapath only) and is excluded.

Usage (from testing/gospa, venv active):
    python run_mobilenet_all.py
Writes gospa_mobilenet_rows.csv (per layer) and gospa_mobilenet_network.txt.
"""
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "ref")))

import mobilenet_layers                                   # noqa: E402

N_PE = int(os.environ.get("N_PE", "8"))
N_MULTS = int(os.environ.get("N_MULTS", "4"))
ROWS = os.path.join(_HERE, "gospa_mobilenet_rows.csv")
OUT = os.path.join(_HERE, "gospa_mobilenet_network.txt")


def next_pow2(n):
    p = 1
    while p < n:
        p *= 2
    return p


# Performance config used for every build (the proven "fast" settings).
PERF_VARS = dict(STAGE1_BATCH=16, FILL_W=16, S2_BEATS=16, DRAIN_W=64)


def pw_batch_f(cin):
    """Virtual F for CH_PID: batch = F^2 input channels. Largest batch that
    still leaves >= 3 rounds to amortize/hide the batch preamble."""
    best = 3
    for f in (4, 5, 6, 7, 8):
        if cin // 3 >= f * f:
            best = f
    return best


def dw_mosaic_cfg(h, s):
    """(H_comp, DW_COLW) for the 3x3-tile depthwise mosaic."""
    ts = h + 2
    if s == 2 and ts % 2:
        ts += 1
    min_rows = 2 * ts + h
    if s == 1:
        e = next_pow2(min_rows - 2)
        h_comp = e + 2
    else:
        e = next_pow2((min_rows - 2) // 2 + 1)
        h_comp = 2 * e + 1
    return h_comp, e.bit_length() - 1


def layer_config(m, n_pe=N_PE, n_mults=N_MULTS):
    """Make-variable dict + mode tag for one layer meta row."""
    v = dict(N_PE=n_pe, N_MULTS=n_mults, FIFOB_D=0, CH_PID=0, DW_COLW=0,
             **PERF_VARS)
    if m["type"] == "pw1x1" and m["cin"] >= 19:
        w, f = m["H"], pw_batch_f(m["cin"])
        v.update(H=max(w * w, f * f), F=f, S=1, N_ROWS=f * f, CH_PID=1,
                 N_NZ_MAX=next_pow2(max(16, f * f * w * w)), FIFO_D=4096)
        tag = f"chpid W={w} batch={f * f}"
    elif m["type"].startswith("dw"):
        h_comp, colw = dw_mosaic_cfg(m["H"], m["S"])
        v.update(H=h_comp, F=3, S=m["S"], N_ROWS=h_comp, DW_COLW=colw,
                 N_NZ_MAX=next_pow2(max(16, 8 * m["H"] * m["H"])),
                 FIFO_D=16384 if m["H"] >= 40 else 4096)
        tag = f"dw-mosaic H={m['H']} S={m['S']}"
    else:
        v.update(H=m["H"], F=m["F"], S=m["S"], N_ROWS=m["H"],
                 N_NZ_MAX=next_pow2(max(16, m["H"] * m["H"])), FIFO_D=4096)
        tag = f"spatial H={m['H']} F={m['F']} S={m['S']}"
    if not v["FIFOB_D"]:
        v["FIFOB_D"] = v["FIFO_D"]
    return tag, v


def run_layers(idxs, extra_env=None, cfg_override=None, cwd=_HERE):
    """Build + run one group of same-geometry layers; returns (rc, stdout)."""
    env = dict(os.environ, **(extra_env or {}))
    subprocess.run(["make", "clean"], cwd=cwd, env=env, capture_output=True)
    args = ["make", "MODULE=test_gospa_mobilenet", "SIM=verilator"]
    for k, val in (cfg_override or {}).items():
        args.append(f"{k}={val}")
    r = subprocess.run(args, cwd=cwd, env=env, capture_output=True, text=True)
    return r.returncode, r.stdout


def main():
    meta = mobilenet_layers.list_layers()
    print(f"{len(meta)} conv layers to run (linear classifier excluded)")
    mobilenet_layers.ensure_extracted([m["idx"] for m in meta])

    # Per-layer mode selection:
    #   pw1x1, Cin >= 19 -> channel-as-PID (9 input channels per round)
    #   pw1x1, Cin <  19 -> spatial F=1 flow (batch preamble would not amortize)
    #   depthwise        -> 3x3-tile mosaic + CID-band demux
    #   conv             -> spatial flow
    def dw_cfg(h, s):
        ts = h + 2
        if s == 2 and ts % 2:
            ts += 1
        min_rows = 2 * ts + h
        if s == 1:
            e = next_pow2(min_rows - 2)
            h_comp = e + 2
        else:
            e = next_pow2((min_rows - 2) // 2 + 1)
            h_comp = 2 * e + 1
        return h_comp, e.bit_length() - 1

    def pw_batch_f(cin):
        """Virtual F for CH_PID: batch = F^2 input channels. Largest batch
        that still leaves >= 3 rounds to amortize/hide the batch preamble."""
        best = 3
        for f in (4, 5, 6, 7, 8):
            if cin // 3 >= f * f:
                best = f
        return best

    groups = {}
    for m in meta:
        if m["type"] == "pw1x1" and m["cin"] >= 19:
            key = ("chpid", m["H"], pw_batch_f(m["cin"]))
        elif m["type"].startswith("dw"):
            key = ("dw", m["H"], m["S"])
        else:
            key = (m["H"], m["F"], m["S"])
        groups.setdefault(key, []).append(m["idx"])

    if os.path.exists(ROWS):
        os.remove(ROWS)

    for gi, (key, idxs) in enumerate(sorted(groups.items(),
                                            key=lambda kv: min(kv[1]))):
        colw = 0
        if key[0] == "chpid":
            w, f = key[1], key[2]
            # H = pixel-count CID space, but at least F^2 so the scanner's
            # row coordinate (clog2(H) bits) covers the channel index.
            h, s, nrows, chpid = max(w * w, f * f), 1, f * f, 1
            nz = next_pow2(max(16, f * f * w * w))
            fifod = 4096
            tag = f"CH_PID pw W={w} batch={f * f}"
        elif key[0] == "dw":
            hr, s = key[1], key[2]
            h, colw = dw_cfg(hr, s)
            f, nrows, chpid = 3, h, 0
            nz = next_pow2(max(16, 8 * hr * hr))
            fifod = 16384 if hr >= 40 else 4096
            tag = f"DW mosaic H={hr} S={s} (comp={h}, E=2^{colw})"
        else:
            h, f, s = key
            nrows, chpid = h, 0
            nz = next_pow2(max(16, h * h))
            fifod = 4096
            tag = f"H={h} F={f} S={s}"
        print(f"[{gi + 1}/{len(groups)}] {tag} N_NZ_MAX={nz}: layers {idxs}",
              flush=True)
        env = dict(os.environ,
                   LAYER_IDX=",".join(str(i) for i in idxs), ROWS_CSV=ROWS)
        subprocess.run(["make", "clean"], cwd=_HERE, env=env,
                       capture_output=True)
        r = subprocess.run(
            ["make", "MODULE=test_gospa_mobilenet", "SIM=verilator",
             f"H={h}", f"F={f}", f"S={s}", f"N_PE={N_PE}",
             f"N_MULTS={N_MULTS}", f"N_ROWS={nrows}", f"N_NZ_MAX={nz}",
             f"FIFO_D={fifod}", "STAGE1_BATCH=16", "FILL_W=16",
             f"S2_BEATS={max(4, 64 // N_MULTS)}",
             f"CH_PID={chpid}", f"DW_COLW={colw}", "DRAIN_W=64"],
            cwd=_HERE, env=env, capture_output=True, text=True)
        tail = "\n".join(r.stdout.splitlines()[-25:])
        if r.returncode != 0 or "FAIL=0" not in r.stdout:
            print(tail)
            sys.exit(f"group H={h} F={f} S={s} FAILED (rc={r.returncode})")
        for ln in r.stdout.splitlines():
            if "PASS" in ln and "pipeUtil" in ln:
                print("   " + ln.split("cocotb.gospa")[-1].strip(), flush=True)

    # ---- aggregate -------------------------------------------------------
    rows = []
    with open(ROWS) as fh:
        for ln in fh:
            p = ln.strip().split(",")
            rows.append(dict(
                idx=int(p[0]), type=p[1], H=int(p[2]), F=int(p[3]),
                S=int(p[4]), cin=int(p[5]), cout=int(p[6]), useful=int(p[7]),
                exec_m=int(p[8]), total=int(p[9]), pre=int(p[10]),
                load=int(p[11]), pipe=int(p[12]), drain=int(p[13])))
    rows.sort(key=lambda r: r["idx"])
    lanes = N_PE * N_MULTS

    hdr = (f"{'#':>3} {'type':<9} {'HxH':>5} {'Cin':>4} {'Cout':>5} "
           f"{'useMACs':>10} {'cycles':>9} {'pre%':>5} {'load%':>6} "
           f"{'pipe%':>6} {'drn%':>5} {'pipeU%':>7} {'e2eU%':>6}")
    lines = [
        "GOSPA MOBILENETV2 NETWORK PERF -- all conv layers, real apple-80 tensors",
        f"N_PE={N_PE} N_MULTS={N_MULTS} ({lanes} multipliers)  STAGE1_BATCH=4 "
        f"FILL_W=16 S2_BEATS=4  (linear classifier not mapped)",
        "dw layers: 1-active-PE mapping (depthwise gap); all layers golden-checked",
        "",
        hdr,
        "-" * len(hdr),
    ]
    tot_u = tot_c = 0
    by_type = {}
    for r in rows:
        tot_u += r["useful"]
        tot_c += r["total"]
        tt = "dw" if r["type"].startswith("dw") else (
            "pw" if r["type"] == "pw1x1" else "conv")
        a = by_type.setdefault(tt, [0, 0])
        a[0] += r["useful"]
        a[1] += r["total"]
        pu = 100.0 * r["useful"] / (r["pipe"] * lanes) if r["pipe"] else 0.0
        eu = 100.0 * r["useful"] / (r["total"] * lanes) if r["total"] else 0.0
        lines.append(
            f"{r['idx']:>3} {r['type']:<9} {r['H']:>5} {r['cin']:>4} "
            f"{r['cout']:>5} {r['useful']:>10} {r['total']:>9} "
            f"{100.0 * r['pre'] / r['total']:>5.1f} "
            f"{100.0 * r['load'] / r['total']:>6.1f} "
            f"{100.0 * r['pipe'] / r['total']:>6.1f} "
            f"{100.0 * r['drain'] / r['total']:>5.1f} {pu:>7.1f} {eu:>6.1f}")
    lines += ["-" * len(hdr)]
    for tt, (u, c) in sorted(by_type.items()):
        lines.append(f"  {tt:<5}: useful={u:>11}  cycles={c:>9}  "
                     f"e2e util={100.0 * u / (c * lanes):>5.1f}%  "
                     f"({100.0 * c / tot_c:.0f}% of runtime)")
    ms = tot_c * 10e-9 * 1e3                     # 100 MHz sim clock
    lines += [
        "",
        f"NETWORK: useful MACs={tot_u}  cycles={tot_c}  "
        f"util={100.0 * tot_u / (tot_c * lanes):.1f}%  "
        f"latency={ms:.2f} ms @100MHz  ({1e3 / ms:.1f} fps)",
    ]
    with open(OUT, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
