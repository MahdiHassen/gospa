"""
test_gospa_alexnet.py -- one AlexNet conv layer (real pretrained weights +
real apple-image activations, integer-quantized) run on the goSPA RTL with
16x16 input tiling, golden-checked against numpy integer conv end to end.

Mapping: build H=16 (one tile), F/S per layer. The padded global activation
map is cut into 16x16 tiles on an E_tile*S grid; each (output-channel group,
tile) is an accumulation pass over ALL input channels using the pipelined
round flow (kick router -> prefetch next channel's tile + weights into the
shadow bank -> wait backlog -> swap), then a drain. Tile outputs assemble
into the global output map and must match golden EXACTLY.

Env: LAYER=1..5, ROWS_CSV for the metrics row.
Run via run_alexnet_all.py (one build per layer geometry).
"""
import os
import sys

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import Event

_TEST_DIR = os.path.dirname(__file__)
sys.path.insert(0, os.path.abspath(os.path.join(_TEST_DIR, "..", "..", "sw")))
sys.path.insert(0, os.path.abspath(os.path.join(_TEST_DIR, "..", "ref")))
sys.path.insert(0, _TEST_DIR)

import numpy as np                                        # noqa: E402
import functional as fm                                   # noqa: E402
fm._VERBOSE = False

import gospa_tb as tb                                     # noqa: E402
import alexnet_layers as ax                               # noqa: E402
from test_gospa_apple import (                            # noqa: E402
    SysCounters, _fifob_monitor, _delta, _load_group_weights)

LAYER = int(os.environ.get("LAYER", "3"))
ROWS_CSV = os.environ.get(
    "ROWS_CSV", os.path.join(_TEST_DIR, "gospa_alexnet_rows.csv"))


def useful_macs(acts_p, weights, stride, f):
    """Exact useful MAC count: per input channel, non-zero-activation count
    at each kernel offset over the output grid, dotted with the per-offset
    non-zero weight counts over all output channels."""
    cin = acts_p.shape[0]
    nz = (acts_p != 0).astype(np.int64)
    win = np.lib.stride_tricks.sliding_window_view(
        nz, (f, f), axis=(1, 2))[:, ::stride, ::stride]     # (Cin,E,E,F,F)
    pairs = win.sum(axis=(1, 2))                            # (Cin,F,F)
    whit = (weights != 0).sum(axis=0)                       # (Cin,F,F)
    return int((pairs * whit).sum())


@cocotb.test()
async def test_alexnet_layer(dut):
    L = ax.get_layer(LAYER)
    acts, weights = L["acts"], L["weights"]
    stride, pad = L["stride"], L["pad"]
    cin, cout = acts.shape[0], weights.shape[0]
    f = weights.shape[2]
    ht = tb.H                                        # tile size = build H
    assert tb.F == f and tb.S == stride, (
        f"build F={f} S={stride} required (got {tb.F}/{tb.S})")

    e_tile = (ht - f) // stride + 1
    tstep = e_tile * stride
    hp = acts.shape[1] + 2 * pad
    e_glob = (hp - f) // stride + 1
    ntiles = -(-e_glob // e_tile)                    # ceil
    need = (ntiles - 1) * tstep + ht                 # padded rows required
    acts_p = np.zeros((cin, need, need), dtype=np.int64)
    acts_p[:, pad:pad + acts.shape[1], pad:pad + acts.shape[1]] = acts

    golden = ax.golden_conv(acts_p[:, :hp, :hp], weights, stride)
    total_useful = useful_macs(acts_p[:, :hp, :hp], weights, stride, f)
    da = float((acts != 0).mean())
    dw = float((weights != 0).mean())
    dut._log.info(
        f"alexnet conv{LAYER}: Cin={cin} Cout={cout} F={f} S={stride} "
        f"E={e_glob} tiles={ntiles}x{ntiles} da={da:.3f} dw={dw:.3f} "
        f"useful={total_useful}")

    cocotb.start_soon(Clock(dut.clk, tb.CLK_NS, unit="ns").start())
    mon = SysCounters(tb.N_PE)
    stop = Event()
    cocotb.start_soon(_fifob_monitor(dut, mon, stop))

    def sp(gch, c):
        out = []
        stuffed = 0
        for k in gch:
            _, s = fm.kernel_to_sparse(weights[k][c].tolist())
            if not s:
                s = [(0, 0)]
                stuffed += 1
            out.append(s)
        return out, stuffed

    out_map = np.zeros((cout, e_glob, e_glob), dtype=np.int64)
    n_stuffed = 0
    t0 = mon.snap()

    for g0 in range(0, cout, tb.N_PE):
        gch = list(range(g0, min(cout, g0 + tb.N_PE)))
        for tr in range(ntiles):
            for tc in range(ntiles):
                r0, c0 = tr * tstep, tc * tstep
                tiles = [acts_p[c, r0:r0 + ht, c0:c0 + ht].tolist()
                         for c in range(cin)]

                await tb.reset(dut)
                await tb.fill_activation_csr(dut, tiles[0])
                await tb.run_scan(dut)
                s0, st = sp(gch, 0)
                n_stuffed += st
                await _load_group_weights(dut, s0)
                await tb.swap_weights(dut)

                for c in range(cin):
                    await tb.kick_stage2(dut)
                    tload = None
                    if c + 1 < cin:
                        s1, st = sp(gch, c + 1)
                        n_stuffed += st
                        tload = cocotb.start_soon(
                            _load_group_weights(dut, s1))
                        await tb.fill_activation_csr(dut, tiles[c + 1])
                        await tb.run_scan(dut)
                    if tload is not None:
                        await tload
                    await tb.wait_pes_idle(dut)
                    if c + 1 < cin:
                        await tb.swap_weights(dut)

                got = await tb.drain_all(dut)
                for p, k in enumerate(gch):
                    for i in range(e_tile):
                        gr = tr * e_tile + i
                        if gr >= e_glob:
                            continue
                        for j in range(e_tile):
                            gc = tc * e_tile + j
                            if gc < e_glob:
                                out_map[k, gr, gc] = got[p].get(
                                    i * tb.E + j, 0)
    stop.set()

    bad = int((out_map != golden).sum())
    assert bad == 0, f"conv{LAYER}: {bad} output values mismatch"
    total = mon.snap()[0] - t0[0]
    macs = sum(mon.exec_macs)
    lanes = tb.N_PE * tb.N_MULTS
    util = 100.0 * total_useful / (total * lanes)

    with open(ROWS_CSV, "a") as fh:
        fh.write(f"{LAYER},{cin},{cout},{f},{stride},{e_glob},"
                 f"{da:.4f},{dw:.4f},{total_useful},{macs},{total},"
                 f"{util:.2f},{n_stuffed}\n")
    dut._log.info(
        f"conv{LAYER} PASS: cycles={total} useful={total_useful} "
        f"measuredMACs={macs} util={util:.1f}% stuffed={n_stuffed}")
