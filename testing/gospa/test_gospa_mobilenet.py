"""
test_gospa_mobilenet.py -- real MobileNetV2 layers on the V2 RTL, one or more
layers per run (comma-separated LAYER_IDX; all must share this build's
H/F/S geometry), real quantized tensors from the apple-80 forward pass.

Activations are each layer's REAL inputs (int_repr - zero_point, cached by
testing/ref/mobilenet_layers.py); weights are the real int8 tensors. Golden
per output channel is conv2d_reference on the same tensors (valid-region,
padless -- same convention as the conv1 apple test).

Flows:
  groups == 1 (conv / pw1x1): pipelined group flow -- groups of N_PE output
    channels, inner loop over Cin with kick_stage2 + hidden fill/scan of the
    next input channel, one drain per group.
  depthwise (groups == Cout): each output channel depends ONLY on its own
    input channel, but the machine holds one resident channel multicast to
    all PEs -- so only ONE PE can do real work per pass. Honest mapping: one
    pass per channel. The utilisation collapse is the measured cost of the
    missing depthwise mapping, not a bug.

Each layer appends one machine-readable line to ROWS_CSV (default
gospa_mobilenet_rows.csv) for the network-level aggregator
(run_mobilenet_all.py).
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

import functional as fm                                   # noqa: E402
fm._VERBOSE = False

import gospa_tb as tb                                     # noqa: E402
import mobilenet_layers                                   # noqa: E402
from test_gospa_apple import (                            # noqa: E402
    SysCounters, _fifob_monitor, _delta, _load_group_weights,
    fifo_a_lanes, stuffed_sparse)

LAYER_IDXS = [int(s) for s in os.environ.get("LAYER_IDX", "2").split(",")]
ROWS_CSV = os.environ.get(
    "ROWS_CSV", os.path.join(_TEST_DIR, "gospa_mobilenet_rows.csv"))


def _wsp_hits(lane_len_c, wsp):
    return sum(lane_len_c[p] for p in range(tb.F * tb.F) if wsp[p])


async def _run_layer(dut, mon, idx):
    """Run one layer; returns a result-row dict. DUT is reset per pass."""
    L = mobilenet_layers.get_layer(idx)
    acts, weights = L["acts"], L["weights"]
    cin, cout = len(acts), len(weights)
    depthwise = L["groups"] == cout and L["groups"] > 1

    if tb.CH_PID:
        w_real = len(acts[0])
        assert L["type"] == "pw1x1" and tb.H >= w_real * w_real, (
            f"CH_PID build (H={tb.H}) needs a pw1x1 layer with <= {tb.H} "
            f"pixels (layer {idx} is {L['type']} {w_real}x{w_real})")
    elif tb.DW_COLW > 0:
        assert depthwise and tb.F == len(weights[0][0]) \
            and tb.S == L["stride"], (
            f"DW_COLW build needs a depthwise F={tb.F} S={tb.S} layer "
            f"(layer {idx} is {L['type']})")
    else:
        assert tb.H == len(acts[0]) and tb.F == len(weights[0][0]) \
            and tb.S == L["stride"], (
            f"layer {idx} needs H={len(acts[0])} F={len(weights[0][0])} "
            f"S={L['stride']} (build is H={tb.H} F={tb.F} S={tb.S})")

    if not tb.CH_PID:
        lanes = [fifo_a_lanes(acts[c]) for c in range(cin)]
        lane_len = [[len(l) for l in lanes[c]] for c in range(cin)]
    phases = []
    t = mon.snap()
    useful = exec_m = mismatches = 0

    dut._log.info(f"layer {idx}: {L['type']} Cin={cin} Cout={cout} "
                  f"H={tb.H} F={tb.F} S={tb.S} depthwise={depthwise} "
                  f"ch_pid={tb.CH_PID}")

    if tb.CH_PID:
        # ---- pointwise channel-batch flow: N_PID input channels per round --
        CB = tb.F * tb.F                       # channels per batch
        flats = [[v for row in acts[c] for v in row] for c in range(cin)]
        nnz = [sum(1 for v in fl if v) for fl in flats]
        batches = [list(range(b, min(cin, b + CB))) for b in range(0, cin, CB)]

        def _bsp(gch, bat):
            """Per-PE (pid, w) list for one batch: pid j <-> channel bat[j]."""
            out = []
            for k in gch:
                sp = [(j, weights[k][c][0][0]) for j, c in enumerate(bat)
                      if weights[k][c][0][0] != 0]
                out.append(sp if sp else [(0, 0)])
            return out

        for g0 in range(0, cout, tb.N_PE):
            gch = list(range(g0, min(cout, g0 + tb.N_PE)))
            for k in gch:
                for bat in batches:
                    stf = all(weights[k][c][0][0] == 0 for c in bat)
                    for j, c in enumerate(bat):
                        if weights[k][c][0][0] != 0:
                            useful += nnz[c]
                            exec_m += nnz[c]
                        elif stf and j == 0:
                            exec_m += nnz[c]        # stuffed zero weight
            await tb.reset(dut)
            await tb.fill_activation_csr(dut, [flats[c] for c in batches[0]])
            await tb.run_scan(dut, n_rows=len(batches[0]))
            await _load_group_weights(dut, _bsp(gch, batches[0]))
            await tb.swap_weights(dut)
            t2 = mon.snap(); phases.append(("pre", _delta(t, t2))); t = t2

            for b in range(len(batches)):
                await tb.kick_stage2(dut)
                tload = None
                if b + 1 < len(batches):
                    tload = cocotb.start_soon(
                        _load_group_weights(dut, _bsp(gch, batches[b + 1])))
                    await tb.fill_activation_csr(
                        dut, [flats[c] for c in batches[b + 1]])
                    await tb.run_scan(dut, n_rows=len(batches[b + 1]))
                if tload is not None:
                    await tload
                await tb.wait_pes_idle(dut)
                if b + 1 < len(batches):
                    await tb.swap_weights(dut)
                t2 = mon.snap(); phases.append(("pipe", _delta(t, t2))); t = t2

            got = await tb.drain_all(dut)
            t2 = mon.snap(); phases.append(("drain", _delta(t, t2))); t = t2

            n_px = len(flats[0])
            for p, k in enumerate(gch):
                golden = [sum(flats[c][px] * weights[k][c][0][0]
                              for c in range(cin)) for px in range(n_px)]
                out = [got[p].get(px, 0) for px in range(n_px)]
                if out != golden:
                    mismatches += 1
                    dut._log.error(f"L{idx} out-ch {k} mismatch (CH_PID)")
    elif depthwise and tb.DW_COLW > 0:
        # ---- depthwise mosaic: 8 channels tiled 3x3 into one composite -----
        # map (zero gaps between tiles), one traditional-conv pass; the
        # router demuxes beats to PEs by 2-D CID band, drains walk windows.
        W = len(acts[0])
        TS = W + (tb.F - 1)
        if tb.S == 2 and TS % 2:
            TS += 1                              # keep tile origins stride-aligned
        e_tile = (W - tb.F) // tb.S + 1
        assert tb.E == (1 << tb.DW_COLW), (
            f"build E={tb.E} must equal 2^DW_COLW={1 << tb.DW_COLW}")
        assert 2 * TS + W <= tb.H, f"3x3 tiles need {2 * TS + W} rows > H={tb.H}"

        def origin(t):
            return ((t // 3) * TS, (t % 3) * TS)

        def out_band(o):
            return o // tb.S                      # origins are S-aligned

        # The mosaic grid is 3x3 (8 channel tiles + 1 spare), so at most 8
        # PEs get work per pass regardless of N_PE (extra PEs idle -- the
        # dw mapping does not scale with the array; see report).
        tile_cap = min(tb.N_PE, 8)
        for b0 in range(0, cout, tile_cap):
            batch = list(range(b0, min(cout, b0 + tile_cap)))
            comp = [[0] * tb.H for _ in range(tb.H)]
            for p, c in enumerate(batch):
                orr, occ = origin(p)
                for i in range(W):
                    for j in range(W):
                        comp[orr + i][occ + j] = acts[c][i][j]

            # model: pairs of the composite, ownership by band
            comp_lanes = fifo_a_lanes(comp)
            bands = []
            for p in range(tb.N_PE):
                if p < len(batch):
                    orr, occ = origin(p)
                    bands.append((out_band(orr), out_band(orr) + e_tile,
                                  out_band(occ), out_band(occ) + e_tile))
                else:
                    bands.append((0, 0, 0, 0))
            sps = []
            for p in range(tb.N_PE):
                if p < len(batch):
                    wsp_st, sp = stuffed_sparse(weights[batch[p]][0])
                    wsp_re, _ = fm.kernel_to_sparse(weights[batch[p]][0])
                else:
                    wsp_st = wsp_re = [0] * (tb.F * tb.F)
                    sp = []
                sps.append(sp)
                for pid in range(tb.F * tb.F):
                    for (_a, cid) in comp_lanes[pid]:
                        rr, cc = cid >> tb.DW_COLW, cid & (tb.E - 1)
                        r0, r1, c0, c1 = bands[p]
                        if r0 <= rr < r1 and c0 <= cc < c1:
                            if wsp_re[pid]:
                                useful += 1
                                exec_m += 1
                            elif wsp_st[pid]:
                                exec_m += 1

            await tb.reset(dut)
            dut.dw_en.value = 1
            for sig, vals in (("band_r0", [b[0] for b in bands]),
                              ("band_r1", [b[1] for b in bands]),
                              ("band_c0", [b[2] for b in bands]),
                              ("band_c1", [b[3] for b in bands]),
                              ("drain_r0", [b[0] for b in bands]),
                              ("drain_c0", [b[2] for b in bands])):
                packed = 0
                for p, v in enumerate(vals):
                    packed |= v << (p * tb.CID_W)
                getattr(dut, sig).value = packed
            dut.drain_rlen.value = e_tile
            dut.drain_clen.value = e_tile

            await tb.fill_activation_csr(dut, comp)
            await tb.run_scan(dut)
            await _load_group_weights(dut, sps)
            await tb.swap_weights(dut)
            t2 = mon.snap(); phases.append(("pre", _delta(t, t2))); t = t2

            await tb.kick_stage2(dut)
            await tb.wait_pes_idle(dut)
            t2 = mon.snap(); phases.append(("pipe", _delta(t, t2))); t = t2

            got = await tb.drain_all(dut)
            t2 = mon.snap(); phases.append(("drain", _delta(t, t2))); t = t2

            for p, c in enumerate(batch):
                golden = fm.conv2d_reference(acts[c], weights[c][0], tb.S)
                r0, _, c0, _ = bands[p]
                ok = True
                for i in range(e_tile):
                    for j in range(e_tile):
                        cid = ((r0 + i) << tb.DW_COLW) | (c0 + j)
                        if got[p].get(cid, 0) != golden[i][j]:
                            ok = False
                if not ok:
                    mismatches += 1
                    dut._log.error(f"L{idx} dw-mosaic channel {c} mismatch")
    elif depthwise:
        for c in range(cout):
            kern = weights[c][0]
            wsp_st, sparse = stuffed_sparse(kern)
            wsp_re, _ = fm.kernel_to_sparse(kern)
            useful += _wsp_hits(lane_len[c], wsp_re)
            exec_m += _wsp_hits(lane_len[c], wsp_st)

            await tb.reset(dut)
            await tb.fill_activation_csr(dut, acts[c])
            await tb.run_scan(dut)
            t2 = mon.snap(); phases.append(("pre", _delta(t, t2))); t = t2

            await _load_group_weights(dut, [sparse] + [[]] * (tb.N_PE - 1))
            await tb.arm_pe_array(dut)
            t2 = mon.snap(); phases.append(("load", _delta(t, t2))); t = t2

            await tb.kick_stage2(dut)
            await tb.wait_pes_idle(dut)
            t2 = mon.snap(); phases.append(("pipe", _delta(t, t2))); t = t2

            got = await tb.drain_all(dut)
            t2 = mon.snap(); phases.append(("drain", _delta(t, t2))); t = t2

            golden = fm.conv2d_reference(acts[c], kern, tb.S)
            out = [[got[0].get(r * tb.E + col, 0) for col in range(tb.E)]
                   for r in range(tb.E)]
            if out != golden:
                mismatches += 1
                dut._log.error(f"L{idx} dw channel {c} mismatch")
    else:
        for g0 in range(0, cout, tb.N_PE):
            gch = list(range(g0, min(cout, g0 + tb.N_PE)))
            for k in gch:
                for c in range(cin):
                    useful += _wsp_hits(
                        lane_len[c], fm.kernel_to_sparse(weights[k][c])[0])
                    exec_m += _wsp_hits(
                        lane_len[c], stuffed_sparse(weights[k][c])[0])

            def _sp(c):
                return [stuffed_sparse(weights[k][c])[1] for k in gch]

            await tb.reset(dut)
            await tb.fill_activation_csr(dut, acts[0])
            await tb.run_scan(dut)
            await _load_group_weights(dut, _sp(0))
            await tb.swap_weights(dut)
            t2 = mon.snap(); phases.append(("pre", _delta(t, t2))); t = t2

            for c in range(cin):
                # Round c streams with its weights active; meanwhile the NEXT
                # round's weights fill the shadow bank and the next channel's
                # CSR fills FIFO-A -- all hidden behind the FIFO-B backlog.
                await tb.kick_stage2(dut)
                tload = None
                if c + 1 < cin:
                    tload = cocotb.start_soon(
                        _load_group_weights(dut, _sp(c + 1)))
                    await tb.fill_activation_csr(dut, acts[c + 1])
                    await tb.run_scan(dut)
                if tload is not None:
                    await tload
                await tb.wait_pes_idle(dut)
                if c + 1 < cin:
                    await tb.swap_weights(dut)
                t2 = mon.snap(); phases.append(("pipe", _delta(t, t2))); t = t2

            got = await tb.drain_all(dut)
            t2 = mon.snap(); phases.append(("drain", _delta(t, t2))); t = t2

            for p, k in enumerate(gch):
                golden = [[0] * tb.E for _ in range(tb.E)]
                for c in range(cin):
                    part = fm.conv2d_reference(acts[c], weights[k][c], tb.S)
                    for i in range(tb.E):
                        for j in range(tb.E):
                            golden[i][j] += part[i][j]
                out = [[got[p].get(r * tb.E + col, 0) for col in range(tb.E)]
                       for r in range(tb.E)]
                if out != golden:
                    mismatches += 1
                    dut._log.error(f"L{idx} out-ch {k} mismatch")

    assert mismatches == 0, f"L{idx}: {mismatches} output channels wrong"
    meas_exec = sum(sum(d[4]) for _, d in phases)
    assert meas_exec == exec_m, (
        f"L{idx}: executed MACs {meas_exec} != model {exec_m}")

    seg = {}
    for nm in ("pre", "load", "pipe", "drain"):
        seg[nm] = sum(d[0] for pnm, d in phases if pnm == nm)
    total = sum(d[0] for _, d in phases)
    return dict(idx=idx, type=L["type"], cin=cin, cout=cout,
                useful=useful, exec=exec_m, total=total, **seg)


@cocotb.test()
async def test_mobilenet_layers(dut):
    cocotb.start_soon(Clock(dut.clk, tb.CLK_NS, unit="ns").start())
    mon = SysCounters(tb.N_PE)
    stop = Event()
    cocotb.start_soon(_fifob_monitor(dut, mon, stop))

    mobilenet_layers.ensure_extracted(LAYER_IDXS)
    rows = []
    for idx in LAYER_IDXS:
        rows.append(await _run_layer(dut, mon, idx))
    stop.set()

    n_lanes = tb.N_PE * tb.N_MULTS
    with open(ROWS_CSV, "a") as fh:
        for r in rows:
            fh.write(f"{r['idx']},{r['type']},{tb.H},{tb.F},{tb.S},"
                     f"{r['cin']},{r['cout']},{r['useful']},{r['exec']},"
                     f"{r['total']},{r['pre']},{r['load']},{r['pipe']},"
                     f"{r['drain']}\n")

    for r in rows:
        up = 100.0 * r["useful"] / (r["pipe"] * n_lanes) if r["pipe"] else 0.0
        ue = 100.0 * r["useful"] / (r["total"] * n_lanes) if r["total"] else 0.0
        dut._log.info(
            f"L{r['idx']:>2} {r['type']:<9} {r['cin']:>4}->{r['cout']:<4} "
            f"useful={r['useful']:>9} cyc={r['total']:>8} "
            f"pipeUtil={up:5.1f}% e2eUtil={ue:5.1f}%  PASS")
