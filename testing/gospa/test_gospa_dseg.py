"""
test_gospa_dseg.py -- exact cycle attribution for the density sweep: where
does every cycle of a low-density pass go?

Per channel round, wall-clock segments are timestamped:
    walk  : s2_start -> s2_done (router streaming the shared beat stream)
    fs    : next channel's fill+scan (+ parallel weight load), after the walk
    idle  : wait_pes_idle tail (backlog finishing)
    swap  : weight-bank swap
plus per-segment PE-busy (consumed beats from the FIFO-B monitor), so the
multicast dw-bound (PE busy only its WSP fraction of the walk) is measured
directly. pre/drain accounted as before.

One config per build (S2_BEATS x STAGE1_BATCH); writes
gospa_dseg_S<S2_BEATS>B<STAGE1_BATCH>.txt and appends DSEG_CSV rows.
"""
import os
import random
import sys

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import Event

_TEST_DIR = os.path.dirname(__file__)
sys.path.insert(0, os.path.abspath(os.path.join(_TEST_DIR, "..", "..", "sw")))
sys.path.insert(0, os.path.abspath(os.path.join(_TEST_DIR, "..", "ref")))
sys.path.insert(0, _TEST_DIR)

import functional as fm                                   # noqa: E402
fm._VERBOSE = False

import gospa_tb as tb                                     # noqa: E402
from gospa_tb import make_activation, rand_kernels        # noqa: E402
from test_gospa_apple import (                            # noqa: E402
    SysCounters, _fifob_monitor, _delta, _load_group_weights)

S1B = int(os.environ.get("STAGE1_BATCH", "4"))
CSV = os.environ.get(
    "DSEG_CSV", os.path.join(_TEST_DIR, "gospa_dseg_rows.csv"))
# DENS="0.1,0.3,1.0" runs a subset of the diagonal (selective checks).
GRID = ([float(s) for s in os.environ["DENS"].split(",")]
        if os.environ.get("DENS") else
        [round(0.1 * i, 1) for i in range(1, 11)])
N_CHAN = 3
N_OUT = 32


async def _run_group(dut, mon, acts, kers, gch, seg):
    def sp(c):
        return [fm.kernel_to_sparse(kers[c][k])[1] for k in gch]

    t = mon.snap()
    await tb.reset(dut)
    await tb.fill_activation_csr(dut, acts[0])
    await tb.run_scan(dut)
    await _load_group_weights(dut, sp(0))
    await tb.swap_weights(dut)
    t2 = mon.snap()
    seg["pre"] += t2[0] - t[0]
    t = t2

    macs = 0
    for c in range(N_CHAN):
        c0 = mon.snap()
        await tb.kick_stage2(dut)
        c1 = mon.snap()
        tload = None
        if c + 1 < N_CHAN:
            tload = cocotb.start_soon(_load_group_weights(dut, sp(c + 1)))
            await tb.fill_activation_csr(dut, acts[c + 1])
            await tb.run_scan(dut)
        if tload is not None:
            await tload
        c2 = mon.snap()
        await tb.wait_pes_idle(dut)
        c3 = mon.snap()
        if c + 1 < N_CHAN:
            await tb.swap_weights(dut)
        c4 = mon.snap()
        seg["walk"] += c1[0] - c0[0]
        seg["fs"] += c2[0] - c1[0]
        seg["idle"] += c3[0] - c2[0]
        seg["swap"] += c4[0] - c3[0]
        seg["walk_busy"] += sum(_delta(c0, c1)[1])   # consumed beats in walk
        seg["fs_busy"] += sum(_delta(c1, c2)[1])
        macs += sum(_delta(c0, c4)[4])
    seg["macs"] += macs

    got = await tb.drain_all(dut)   # drain accounted as the density-level residual
    return got


@cocotb.test()
async def test_density_segments(dut):
    cocotb.start_soon(Clock(dut.clk, tb.CLK_NS, unit="ns").start())
    mon = SysCounters(tb.N_PE)
    stop = Event()
    cocotb.start_soon(_fifob_monitor(dut, mon, stop))
    rng = random.Random(0xD5EE9)
    lanes = tb.N_PE * tb.N_MULTS
    tag = f"S{tb.S2_BEATS}B{S1B}"

    hdr = (f"{'dens':>5} {'totCyc':>7} {'walk%':>6} {'fs%':>5} {'idle%':>6} "
           f"{'swap%':>6} {'pre%':>5} {'drn%':>5} {'wBusy%':>7} {'e2eU%':>6}")
    lines = [
        f"GOSPA DENSITY SEGMENT ATTRIBUTION -- cfg {tag} "
        f"(S2_BEATS={tb.S2_BEATS}, STAGE1_BATCH={S1B}, "
        f"FIFOB_D={os.environ.get('FIFOB_D', '?')} DRAIN_W={tb.DRAIN_W})",
        f"H={tb.H} F={tb.F} S={tb.S} E={tb.E}  N_PE={tb.N_PE} x M={tb.N_MULTS}"
        f"  pool={N_OUT} out-ch, {N_CHAN} in-ch, {N_OUT // tb.N_PE} passes",
        "walk = router streaming the shared beat stream; wBusy% = mean PE",
        "duty inside the walk (the multicast dw-bound, ~= weight density)",
        "", hdr, "-" * len(hdr),
    ]
    with open(CSV, "a") as fh:
        for d in GRID:
            kers = [rand_kernels(rng, N_OUT, d) for _ in range(N_CHAN)]
            acts = [make_activation(rng, d, pad=0) for _ in range(N_CHAN)]
            golden = []
            for k in range(N_OUT):
                acc = [[0] * tb.E for _ in range(tb.E)]
                for c in range(N_CHAN):
                    part = fm.conv2d_reference(acts[c], kers[c][k], tb.S)
                    for i in range(tb.E):
                        for j in range(tb.E):
                            acc[i][j] += part[i][j]
                golden.append(acc)

            seg = dict(pre=0, walk=0, fs=0, idle=0, swap=0, drain=0,
                       walk_busy=0, fs_busy=0, macs=0)
            d0 = mon.snap()
            for g in range(N_OUT // tb.N_PE):
                gch = list(range(g * tb.N_PE, (g + 1) * tb.N_PE))
                got = await _run_group(dut, mon, acts, kers, gch, seg)
                for p, k in enumerate(gch):
                    out = [[got[p].get(r * tb.E + col, 0)
                            for col in range(tb.E)] for r in range(tb.E)]
                    assert out == golden[k], f"d={d} out-ch {k} mismatch"
            tot = mon.snap()[0] - d0[0]
            seg["drain"] = tot - (seg["pre"] + seg["walk"] + seg["fs"]
                                  + seg["idle"] + seg["swap"])

            wb = (100.0 * seg["walk_busy"]
                  / (seg["walk"] * tb.N_PE)) if seg["walk"] else 0.0
            lines.append(
                f"{d:>5.1f} {tot:>7} "
                f"{100.0 * seg['walk'] / tot:>6.1f} "
                f"{100.0 * seg['fs'] / tot:>5.1f} "
                f"{100.0 * seg['idle'] / tot:>6.1f} "
                f"{100.0 * seg['swap'] / tot:>6.1f} "
                f"{100.0 * seg['pre'] / tot:>5.1f} "
                f"{100.0 * seg['drain'] / tot:>5.1f} "
                f"{wb:>7.1f} "
                f"{100.0 * seg['macs'] / (tot * lanes):>6.1f}")
            fh.write(f"{tag},{d},{tot},{seg['pre']},{seg['walk']},{seg['fs']},"
                     f"{seg['idle']},{seg['swap']},{seg['drain']},"
                     f"{seg['walk_busy']},{seg['macs']}\n")
            dut._log.info(lines[-1])
    stop.set()

    path = os.path.join(_TEST_DIR, f"gospa_dseg_{tag}.txt")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    dut._log.info(f"wrote {path}")
