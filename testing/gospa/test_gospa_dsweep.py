"""
test_gospa_dsweep.py -- equal-density sweep (da = dw = 0.1 .. 1.0) on the
pipelined flow, comparing NAIVE vs WSP-SIMILARITY output-channel scheduling.

A physical mapping co-schedules output channels with similar weight counts in
the same pass (the pass ends when the busiest PE finishes, so similar WSPs =
balanced work). Per density point: one pool of 32 output channels x Cin=3
random kernels; run 4 group passes of 8 twice from the same pool --
  seq    : channels in index order (naive)
  sorted : channels ordered by total weight count (compiler-style)
Golden-checked per group. DRAIN_W-wide parallel drains.

One FIFOB_D per build; appends CSV rows and writes gospa_dsweep_B<depth>.txt.
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
    SysCounters, _fifob_monitor, _delta, _load_group_weights, fifo_a_lanes)

FIFOB_D = int(os.environ.get("FIFOB_D", str(tb.FIFO_D)))
CSV = os.environ.get(
    "DSWEEP_CSV", os.path.join(_TEST_DIR, "gospa_dsweep_rows.csv"))
GRID = [round(0.1 * i, 1) for i in range(1, 11)]
N_CHAN = 3
N_OUT = 32


async def _run_group(dut, mon, acts, kers, gch):
    """One pipelined 3-channel pass for 8 output channels; returns
    (pre, pipe, drain cycles, per-PE {cid: acc})."""
    def sp(c):
        return [fm.kernel_to_sparse(kers[c][k])[1] for k in gch]

    t = mon.snap()
    await tb.reset(dut)
    await tb.fill_activation_csr(dut, acts[0])
    await tb.run_scan(dut)
    await _load_group_weights(dut, sp(0))
    await tb.swap_weights(dut)
    t2 = mon.snap(); pre = _delta(t, t2); t = t2

    for c in range(N_CHAN):
        await tb.kick_stage2(dut)
        tload = None
        if c + 1 < N_CHAN:
            tload = cocotb.start_soon(_load_group_weights(dut, sp(c + 1)))
            await tb.fill_activation_csr(dut, acts[c + 1])
            await tb.run_scan(dut)
        if tload is not None:
            await tload
        await tb.wait_pes_idle(dut)
        if c + 1 < N_CHAN:
            await tb.swap_weights(dut)
    t2 = mon.snap(); pipe = _delta(t, t2); t = t2

    got = await tb.drain_all(dut)
    t2 = mon.snap(); drain = _delta(t, t2)
    return pre[0], pipe[0], drain[0], sum(pipe[4]), got


async def _run_sched(dut, mon, acts, kers, golden, order):
    pre = pipe = drn = macs = 0
    for g in range(N_OUT // tb.N_PE):
        gch = [order[g * tb.N_PE + p] for p in range(tb.N_PE)]
        p0, p1, p2, m, got = await _run_group(dut, mon, acts, kers, gch)
        pre += p0
        pipe += p1
        drn += p2
        macs += m
        for p, k in enumerate(gch):
            out = [[got[p].get(r * tb.E + col, 0) for col in range(tb.E)]
                   for r in range(tb.E)]
            assert out == golden[k], f"out-ch {k} mismatch"
    return pre, pipe, drn, macs


@cocotb.test()
async def test_density_sweep(dut):
    cocotb.start_soon(Clock(dut.clk, tb.CLK_NS, unit="ns").start())
    mon = SysCounters(tb.N_PE)
    stop = Event()
    cocotb.start_soon(_fifob_monitor(dut, mon, stop))
    rng = random.Random(0xD5EE9)
    lanes = tb.N_PE * tb.N_MULTS

    hdr = (f"{'dens':>5} {'useMACs':>8} "
           f"{'seqCyc':>7} {'seqP%':>6} {'seqE%':>6} "
           f"{'srtCyc':>7} {'srtP%':>6} {'srtE%':>6} {'gain':>6}")
    lines = [
        f"GOSPA DENSITY SWEEP -- naive vs WSP-similarity scheduling, "
        f"FIFOB_D={FIFOB_D} DRAIN_W={tb.DRAIN_W}",
        f"H={tb.H} F={tb.F} S={tb.S} E={tb.E}  N_PE={tb.N_PE} x M={tb.N_MULTS}"
        f"  S2_BEATS={tb.S2_BEATS}  pool={N_OUT} out-ch, {N_CHAN} in-ch, "
        f"{N_OUT // tb.N_PE} passes",
        "seq = channels in index order; srt = grouped by weight count",
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

            wcnt = [sum(len(fm.kernel_to_sparse(kers[c][k])[1])
                        for c in range(N_CHAN)) for k in range(N_OUT)]
            seq_order = list(range(N_OUT))
            srt_order = sorted(range(N_OUT), key=lambda k: -wcnt[k])

            res = {}
            for name, order in (("seq", seq_order), ("srt", srt_order)):
                pre, pipe, drn, macs = await _run_sched(
                    dut, mon, acts, kers, golden, order)
                res[name] = (pre + pipe + drn, pipe, macs)
                fh.write(f"{FIFOB_D},{name},{d},{pre},{pipe},{drn},{macs}\n")

            (stot, spipe, macs), (ttot, tpipe, _) = res["seq"], res["srt"]
            lines.append(
                f"{d:>5.1f} {macs:>8} "
                f"{stot:>7} {100.0 * macs / (spipe * lanes):>6.1f} "
                f"{100.0 * macs / (stot * lanes):>6.1f} "
                f"{ttot:>7} {100.0 * macs / (tpipe * lanes):>6.1f} "
                f"{100.0 * macs / (ttot * lanes):>6.1f} "
                f"{stot / ttot:>5.2f}x")
            dut._log.info(lines[-1])
    stop.set()

    path = os.path.join(_TEST_DIR, f"gospa_dsweep_B{FIFOB_D}.txt")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    dut._log.info(f"wrote {path}")
