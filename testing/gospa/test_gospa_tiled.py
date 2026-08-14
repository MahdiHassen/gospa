"""
test_gospa_tiled.py -- the ENTIRE first layer on a small, fixed gospa,
driven by the mini-compiler's schedule (sw/gospa_compile.py).

The RTL is built for one H x H tile (H=32 or 64 -- much smaller than the
80x80 layer). compile_layer() lowers MobileNetV2's first conv (3 input
channels -> 32 output channels, real 80x80 apple activations) onto it:
spatial tiles with the F-S halo, input channels innermost so pe_acc
accumulates across channels, one drain per tile, weight reload per pass.
The gospa is REUSED across tiles (rst_n between tiles clears pe_acc).

Every stitched output channel is checked against the PyTorch-derived
golden of the full 80x80 conv, and utilization / overhead / transfer
numbers land in gospa_tiled_perf_H<H>.txt for comparison against the
monolithic H=80 run (gospa_apple_perf.txt).

Run (fits the two candidate sizes):
    make clean && make MODULE=test_gospa_tiled SIM=verilator \
         H=32 F=3 S=2 N_PE=8 N_MULTS=4 N_ROWS=32 N_NZ_MAX=1024 FIFO_D=2048
    make clean && make MODULE=test_gospa_tiled SIM=verilator \
         H=64 F=3 S=2 N_PE=8 N_MULTS=4 N_ROWS=64 N_NZ_MAX=4096 FIFO_D=2048
"""
import os
import sys

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import Event

sys.path.insert(0, os.path.dirname(__file__))
import test_gospa as tg                                   # noqa: E402
import test_gospa_apple as ta                             # noqa: E402
import functional as fm                                   # noqa: E402
import apple_input                                        # noqa: E402
from gospa_compile import HW, compile_layer               # noqa: E402

N_CHAN_IN = 3
G_IMG = apple_input.RES        # the layer's real input size (80)


def _tile_matrix(chan, r0, c0, h):
    """h x h local window of an 80x80 channel, zero-padded past the edge."""
    tile = [[0] * h for _ in range(h)]
    for r in range(r0, min(r0 + h, G_IMG)):
        row = chan[r]
        for c in range(c0, min(c0 + h, G_IMG)):
            tile[r - r0][c - c0] = row[c]
    return tile


@cocotb.test()
async def test_tiled_first_conv_perf(dut):
    assert tg.F == 3 and tg.S == 2 and tg.N_CHAN == 32, "MobileNet first-conv shape"
    assert tg.H < G_IMG, f"tile H={tg.H} must be smaller than the {G_IMG}x{G_IMG} layer"

    cocotb.start_soon(Clock(dut.clk, tg.CLK_NS, units="ns").start())

    # -- compile the layer onto this fixed HW --------------------------------
    hw = HW(H=tg.H, F=tg.F, S=tg.S, N_PE=tg.N_PE, N_MULTS=tg.N_MULTS,
            N_NZ_MAX=tg.N_NZ_MAX, N_ROWS=tg.N_ROWS,
            FIFO_D=int(os.environ.get("FIFO_D", "2048")))
    sched, rep = compile_layer(hw, C_in=N_CHAN_IN, C_out=tg.N_CHAN,
                               H_img=G_IMG, W_img=G_IMG)
    assert rep["fits_N_ROWS"] and rep["fits_N_NZ_MAX(dense)"] and rep["fits_FIFO_D(dense)"], rep
    Eh, Ew = rep["out_size"]

    chans = apple_input.load_channels()
    kernel_sets = [tg._load_kernels_for_channel(c)[:tg.N_CHAN]
                   for c in range(N_CHAN_IN)]
    state = [tg._compute_per_channel_state(ks) for ks in kernel_sets]

    golden = []
    for k_idx in range(tg.N_CHAN):
        acc = [[0] * Ew for _ in range(Eh)]
        for c in range(N_CHAN_IN):
            part = fm.conv2d_reference(chans[c], kernel_sets[c][k_idx], tg.S)
            for i in range(Eh):
                for j in range(Ew):
                    acc[i][j] += part[i][j]
        golden.append(acc)

    dut._log.info(
        f"tiled schedule: HW tile {tg.H}x{tg.H} (E={tg.E}) -> "
        f"{rep['spatial_tiles']} tiles x {N_CHAN_IN} ch = {rep['passes']} passes, "
        f"{rep['weight_loads']} weight loads, {rep['drains']} drains")

    # -- drive the schedule ---------------------------------------------------
    mon = ta.SysCounters(tg.N_PE)
    stop = Event()
    cocotb.start_soon(ta._fifob_monitor(dut, mon, stop))

    phase_cyc = {}                  # phase-type -> cycles
    s2 = {"consumed": [0] * tg.N_PE, "stalled": [0] * tg.N_PE,
          "starved": [0] * tg.N_PE}
    total_macs = 0
    exp_consumed = [0] * tg.N_PE
    n_weights = 0
    act_words = 0                   # CSR entries streamed (incl. halo refetch)
    tile_rows = []                  # per-tile: (r0, c0, nnz, s2cyc, macs)
    stitched = [[[0] * Ew for _ in range(Eh)] for _ in range(tg.N_CHAN)]

    t = mon.snap()

    def bump(name, t_old):
        t_new = mon.snap()
        d = ta._delta(t_old, t_new)
        phase_cyc[name] = phase_cyc.get(name, 0) + d[0]
        return t_new, d

    await tg.reset(dut)
    t, _ = bump("reset", t)

    cur_tile = {"nnz": 0, "s2": 0, "macs": 0, "r0": 0, "c0": 0}
    for p in sched:
        per_k_wsp, per_k_sw, chunks, union = state[p.in_ch]
        matrix = _tile_matrix(chans[p.in_ch], p.in_r0, p.in_c0, tg.H)
        nnz = sum(1 for row in matrix for v in row if v != 0)
        cur_tile["nnz"] += nnz
        cur_tile["r0"], cur_tile["c0"] = p.in_r0, p.in_c0
        act_words += nnz

        # exact expected stream + MACs for this pass (functional model)
        fifo_b = ta.route_per_pe(matrix, union)
        for pe in range(len(chunks)):
            exp_consumed[pe] += len(fifo_b[pe])
            m = sum(ta.macs_per_pe_lane(fifo_b[pe], per_k_wsp, chunks[pe]))
            total_macs += m
            cur_tile["macs"] += m

        if p.reload_w:
            await tg._load_pe_weights_for(dut, per_k_sw, chunks)
            await tg._arm_pe_array_for(dut, per_k_sw, chunks)
            await tg._load_apu_wsps_for(dut, union)
            n_weights += sum(len(s) for s in per_k_sw)
            t, _ = bump("load_arm", t)

        await tg.fill_activation_csr(dut, matrix)
        t, _ = bump("act_fill", t)
        await tg.run_scan(dut)
        t, _ = bump("scan", t)
        await tg.run_stage2(dut)
        t, d = bump("stage2", t)
        cur_tile["s2"] += d[0]
        for pe in range(tg.N_PE):
            s2["consumed"][pe] += d[1][pe]
            s2["stalled"][pe] += d[2][pe]
            s2["starved"][pe] += d[3][pe]

        if p.drain:
            got = await tg.drain_all(dut)
            t, _ = bump("drain", t)
            for pe in range(tg.N_PE):
                for lane in range(tg.N_MULTS):
                    k_idx = chunks[pe][lane]
                    for r in range(p.out_h):
                        for cc in range(p.out_c):
                            stitched[k_idx][p.out_r0 + r][p.out_c0 + cc] = \
                                got[pe][lane].get(r * tg.E + cc, 0)
            tile_rows.append((cur_tile["r0"], cur_tile["c0"], cur_tile["nnz"],
                              cur_tile["s2"], cur_tile["macs"]))
            cur_tile = {"nnz": 0, "s2": 0, "macs": 0, "r0": 0, "c0": 0}
            # reuse the same gospa for the next tile: reset clears pe_acc
            await tg.reset(dut)
            t, _ = bump("reset", t)
    stop.set()

    # -- verify ----------------------------------------------------------------
    mismatches = sum(1 for k in range(tg.N_CHAN) if stitched[k] != golden[k])
    assert mismatches == 0, f"{mismatches}/{tg.N_CHAN} stitched channels wrong"

    meas_consumed = list(s2["consumed"])
    assert meas_consumed == exp_consumed, (
        f"consumed beats {meas_consumed} != routed streams {exp_consumed}")

    # -- report ------------------------------------------------------------------
    total_cycles = sum(phase_cyc.values())
    s2_cycles = phase_cyc["stage2"]
    n_lanes = tg.N_PE * tg.N_MULTS

    def util(cyc):
        return 100.0 * total_macs / (cyc * n_lanes) if cyc else 0.0

    cons, stal, starv = (sum(s2[k]) for k in ("consumed", "stalled", "starved"))
    beats = cons + stal + starv
    ENTRY_BITS = tg.DATA_W + tg.PID_W + tg.CID_W
    act_bits = act_words * (tg.DATA_W + tg.IDX_W) \
        + rep["passes"] * (tg.N_ROWS + 1) * tg.PTR_W
    wgt_bits = n_weights * (tg.DATA_W + tg.PID_W)
    fifob_bits = sum(exp_consumed) * ENTRY_BITS
    drain_bits = rep["drains"] * n_lanes * tg.N_CID * tg.ACC_W

    lines = [
        f"GOSPA TILED LAYER PERF -- MobileNetV2 first conv on a {tg.H}x{tg.H} gospa (reused)",
        f"layer: 80x80x3 -> 32x{Eh}x{Ew}   HW: E={tg.E} N_CID={tg.N_CID} "
        f"N_PE={tg.N_PE} N_MULTS={tg.N_MULTS} N_NZ_MAX={tg.N_NZ_MAX}",
        f"schedule: {rep['spatial_tiles']} tiles x {N_CHAN_IN} channels = "
        f"{rep['passes']} passes, {rep['weight_loads']} weight loads, {rep['drains']} drains",
        f"useful MACs = {total_macs}   act words streamed = {act_words} "
        f"(halo refetch x{act_words / sum(sum(1 for row in ch for v in row if v) for ch in chans):.2f})",
        "",
        f"{'tile(r0,c0)':<12} {'nnz(3ch)':>9} {'s2 cyc':>7} {'macs':>8} {'s2util%':>8}",
    ]
    for (r0, c0, nnz3, s2c, m) in tile_rows:
        lines.append(f"({r0:>3},{c0:>3})   {nnz3:>9} {s2c:>7} {m:>8} "
                     f"{100.0 * m / (s2c * n_lanes):>8.1f}")
    lines += ["", f"{'phase':<10} {'cycles':>8} {'share%':>7}"]
    for nm in ("load_arm", "act_fill", "scan", "stage2", "drain", "reset"):
        cyc = phase_cyc.get(nm, 0)
        lines.append(f"{nm:<10} {cyc:>8} {100.0 * cyc / total_cycles:>7.1f}")
    lines += [
        f"{'TOTAL':<10} {total_cycles:>8}",
        "",
        "multiplier utilization = MACs / (window x 32 lanes):",
        f"  over stage2 windows only : {util(s2_cycles):6.1f}%   ({s2_cycles} cyc)",
        f"  over scan+stage2 compute : {util(phase_cyc['scan'] + s2_cycles):6.1f}%",
        f"  end-to-end (all passes)  : {util(total_cycles):6.1f}%   ({total_cycles} cyc)",
        "",
        "stage2 beat split: consumed %.1f%%  stalled %.2f%%  starved %.2f%%"
        % (100.0 * cons / beats, 100.0 * stal / beats, 100.0 * starv / beats),
        "",
        "data transfer (bits):",
        f"  activation CSR fills: {act_bits}  ({act_bits / total_macs:.2f} b/MAC)",
        f"  weight fills        : {wgt_bits}  ({wgt_bits / total_macs:.2f} b/MAC)",
        f"  FIFO-B beats        : {fifob_bits}  ({fifob_bits / total_macs:.2f} b/MAC)",
        f"  drain readout       : {drain_bits}  ({drain_bits / total_macs:.2f} b/MAC)",
        "",
        f"golden check: PASS -- all {tg.N_CHAN} stitched channels match the full-layer reference",
    ]
    report = os.path.join(os.path.dirname(__file__), f"gospa_tiled_perf_H{tg.H}.txt")
    with open(report, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    for ln in lines:
        dut._log.info(ln)
