"""
test_gospa_apple.py -- MobileNetV2 first conv on the real apple input, measured
end to end on the V2 "1 weight x 4 activations" architecture with the batched
A->B router (S2_BEATS) and wide CSR fill (FILL_W).

Pipelined channel flow (per group of N_PE output channels):

    reset
    fill+scan ch0 into FIFO-A                       (preamble, PEs idle)
    for each input channel c:
        load weights(c) + arm                       (boundary, PEs idle ~90cy)
        kick_stage2: router drains FIFO-A -> FIFO-B at S2_BEATS beats/cycle,
                     returns at s2_done (FIFO-A empty, PEs chewing backlog)
        fill+scan ch c+1                            (hidden under PE backlog)
        wait_pes_idle: backlog consumed
    drain

With S2_BEATS=4 the router outruns the PEs 4x, so FIFO-B holds the decoupling
backlog and the APU front end (fill+scan of the next channel) runs while the
PEs work. The weight swap is the only remaining serialized PE-idle window per
channel (beats carry no channel tag, so channels must not mix in FIFO-B).

Empty per-channel kernels (9 of 96 in the real weights) are "stuffed" with a
single zero-valued weight at PID 0 so a re-arm never leaves a stale WSP.

Per-PE, per-cycle FIFO-B accounting (sampled on internal fifob_valid/ready):
  consumed = valid && ready       (PE admitted an M-wide beat)
  stalled  = valid && !ready      (pe_fetch SKIP: weight-switch bubble)
  starved  = !valid               (no beat: WSP-filtered / boundary / supply)
plus executed MACs = sum of popcount(lane_valid) over admitted beats. Useful
MACs (excluding stuffed zero-weight work) come exactly from the functional
model.

Run (canonical, same array cost as the V1 baseline: 32 multipliers):
    make apple                       # N_PE=8  -> 4 group passes
    make apple N_PE=32               # 128 mults, all 32 channels in one pass
"""
import os
import random
import sys

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ReadOnly, Event

_TEST_DIR = os.path.dirname(__file__)
sys.path.insert(0, os.path.abspath(os.path.join(_TEST_DIR, "..", "..", "sw")))
sys.path.insert(0, os.path.abspath(os.path.join(_TEST_DIR, "..", "ref")))
sys.path.insert(0, _TEST_DIR)

import functional as fm                                   # noqa: E402
fm._VERBOSE = False

import gospa_tb as tb                                     # noqa: E402
import apple_input                                        # noqa: E402

N_OUT     = 32     # output channels of the first conv
N_CHAN_IN = 3      # RGB input channels

S1_BATCH  = int(os.environ.get("STAGE1_BATCH", "1"))      # report only

# EQUAL_NW > 0: replace the real MobileNet kernels with synthetic ones that
# ALL have exactly EQUAL_NW non-zero weights (balanced density, random PIDs /
# values; real apple activations unchanged). Isolates the load-imbalance
# ceiling: balanced kernels -> every PE owns the same amount of work.
EQUAL_NW  = int(os.environ.get("EQUAL_NW", "0"))

# GROUP_SORT=1: assign output channels to group passes sorted by their beat
# count (densest kernels grouped together) instead of sequentially. Each
# pipe window is set by the busiest PE in the group, so packing the dense
# kernels into the same passes shortens the sparse groups' windows.
GROUP_SORT = int(os.environ.get("GROUP_SORT", "0"))

IDX_W = tb.IDX_W
PTR_W = tb.PTR_W
# One FIFO-B entry is an M-wide beat: {pid, lane_valid[M], act[M], cid[M]}.
BEAT_W = tb.PID_W + tb.N_MULTS + tb.N_MULTS * tb.DATA_W + tb.N_MULTS * tb.CID_W


# ---------------------------------------------------------------------------
# Golden-side accounting (exact, from the functional model)
# ---------------------------------------------------------------------------
def fifo_a_lanes(matrix):
    """FIFO-A occupancy per PID for one input channel (same chain the APU
    runs): fifo_a[pid] = list of (axy, cid)."""
    values, col_idx, row_ptr = fm.dense_to_csr(matrix)
    pos = fm.csr_to_positional(values, col_idx, row_ptr)
    pairs = []
    for (axy, x, y) in pos:
        a, px, py, cx, cy = fm.axy_to_pcid(axy, x, y, tb.S)
        pairs.extend(fm.pcid_to_cid_pid(a, px, py, cx, cy, tb.F, tb.H, tb.S))
    pairs = fm.zero_act_filter(pairs)
    return fm.route_to_fifo_a(pairs, tb.F)


def stuffed_sparse(kernel):
    """(wsp, sparse) for one kernel; an all-zero kernel becomes a single
    zero-valued weight at PID 0 so the PE's WSP is always freshly written."""
    wsp, sparse = fm.kernel_to_sparse(kernel)
    if not sparse:
        wsp = [1] + [0] * (tb.F * tb.F - 1)
        sparse = [(0, 0)]
    return wsp, sparse


def ceil_div(a, b):
    return -(-a // b)


def equal_kernel(rng, n_nz):
    """FxF kernel with exactly n_nz non-zero taps at random PIDs."""
    k = [[0] * tb.F for _ in range(tb.F)]
    for p in rng.sample(range(tb.F * tb.F), n_nz):
        v = rng.randint(1, 50)
        k[p // tb.F][p % tb.F] = v if rng.random() < 0.5 else -v
    return k


def golden_from(chans, kers):
    """Per-output-channel golden = sum over input channels of conv2d."""
    out = []
    for k in range(N_OUT):
        acc = [[0] * tb.E for _ in range(tb.E)]
        for c in range(N_CHAN_IN):
            part = fm.conv2d_reference(chans[c], kers[c][k], tb.S)
            for i in range(tb.E):
                for j in range(tb.E):
                    acc[i][j] += part[i][j]
        out.append(acc)
    return out


# ---------------------------------------------------------------------------
# Per-cycle monitor on the internal FIFO-B interfaces
# ---------------------------------------------------------------------------
class SysCounters:
    def __init__(self, n_pe):
        self.cycles = 0
        self.consumed = [0] * n_pe
        self.stalled = [0] * n_pe
        self.starved = [0] * n_pe
        self.exec_macs = [0] * n_pe            # popcount(lane_valid) on consume

    def snap(self):
        return (self.cycles, list(self.consumed), list(self.stalled),
                list(self.starved), list(self.exec_macs))


async def _fifob_monitor(dut, c, stop):
    lane_mask = (1 << tb.N_MULTS) - 1
    while True:
        await RisingEdge(dut.clk)
        if stop.is_set():
            break
        await ReadOnly()
        c.cycles += 1
        v = int(dut.fifob_valid.value)
        r = int(dut.fifob_ready.value)
        lv = int(dut.fifob_lane_valid.value)
        for pe in range(tb.N_PE):
            if (v >> pe) & 1:
                if (r >> pe) & 1:
                    c.consumed[pe] += 1
                    c.exec_macs[pe] += bin((lv >> (pe * tb.N_MULTS)) & lane_mask).count("1")
                else:
                    c.stalled[pe] += 1
            else:
                c.starved[pe] += 1


def _delta(a, b):
    return tuple(b[0] - a[0] if i == 0 else [y - x for x, y in zip(a[i], b[i])]
                 for i in range(5))


async def _load_group_weights(dut, sparse_list):
    """PE-parallel weight fill: one PID per cycle across all PEs."""
    await tb.load_pe_sparse(dut, sparse_list)


# ---------------------------------------------------------------------------
# The test
# ---------------------------------------------------------------------------
@cocotb.test()
async def test_apple_first_conv_perf(dut):
    assert tb.F == 3 and tb.S == 2 and tb.H == apple_input.RES, (
        f"run with H={apple_input.RES} F=3 S=2 (got H={tb.H} F={tb.F} S={tb.S})")
    assert N_OUT % tb.N_PE == 0, f"N_PE={tb.N_PE} must divide {N_OUT} output channels"
    n_groups = N_OUT // tb.N_PE

    cocotb.start_soon(Clock(dut.clk, tb.CLK_NS, unit="ns").start())

    # -- real inputs + real (or balanced synthetic) weights -----------------
    chans = apple_input.load_channels()                   # 3 x (80x80) signed
    if EQUAL_NW > 0:
        rng = random.Random(0xE0DA)
        kers = [[equal_kernel(rng, EQUAL_NW) for _ in range(N_OUT)]
                for _ in range(N_CHAN_IN)]
        golden = golden_from(chans, kers)
    else:
        kers = apple_input.first_conv_kernels(N_OUT)      # [in_ch][out_ch] FxF
        golden = apple_input.golden_output(N_OUT)         # 32 x (ExE), 3ch sum

    # per (in_ch, out_ch): stuffed wsp/sparse; per in_ch: FIFO-A occupancy
    wsp_sp = [[stuffed_sparse(kers[c][k]) for k in range(N_OUT)]
              for c in range(N_CHAN_IN)]
    lanes = [fifo_a_lanes(chans[c]) for c in range(N_CHAN_IN)]
    lane_len = [[len(l) for l in lanes[c]] for c in range(N_CHAN_IN)]

    # exact expected per-output-channel stream lengths / MACs
    n_pairs = [sum(lane_len[c]) for c in range(N_CHAN_IN)]
    exp_beats = [[0] * N_OUT for _ in range(N_CHAN_IN)]   # ideal M-wide beats
    exp_exec = [[0] * N_OUT for _ in range(N_CHAN_IN)]    # MACs incl stuffed
    exp_useful = [[0] * N_OUT for _ in range(N_CHAN_IN)]  # real-weight MACs
    for c in range(N_CHAN_IN):
        for k in range(N_OUT):
            wsp_st, _ = wsp_sp[c][k]
            wsp_real, _ = fm.kernel_to_sparse(kers[c][k])
            for pid in range(tb.F * tb.F):
                if wsp_st[pid]:
                    exp_beats[c][k] += ceil_div(lane_len[c][pid], tb.N_MULTS)
                    exp_exec[c][k] += lane_len[c][pid]
                if wsp_real[pid]:
                    exp_useful[c][k] += lane_len[c][pid]
    total_useful = sum(sum(r) for r in exp_useful)
    total_exec = sum(sum(r) for r in exp_exec)
    n_weights = sum(len(fm.kernel_to_sparse(kers[c][k])[1])
                    for c in range(N_CHAN_IN) for k in range(N_OUT))
    n_stuffed = sum(1 for c in range(N_CHAN_IN) for k in range(N_OUT)
                    if not fm.kernel_to_sparse(kers[c][k])[1])
    nnz = [sum(1 for row in ch for v in row if v != 0) for ch in chans]

    # Channel -> (group, PE) assignment: sequential, or sorted by beat count.
    tot_beats = [sum(exp_beats[c][k] for c in range(N_CHAN_IN))
                 for k in range(N_OUT)]
    if GROUP_SORT:
        order = sorted(range(N_OUT), key=lambda k: -tot_beats[k])
    else:
        order = list(range(N_OUT))

    dut._log.info(
        f"apple first conv (V2 pipelined): H={tb.H} F={tb.F} S={tb.S} E={tb.E} "
        f"N_PE={tb.N_PE} x M={tb.N_MULTS} S2_BEATS={tb.S2_BEATS} -> "
        f"{n_groups} group passes; nnz/ch={nnz} pairs/ch={n_pairs}; "
        f"useful MACs={total_useful} (+{total_exec - total_useful} stuffed); "
        f"GROUP_SORT={GROUP_SORT} order={order}")

    # -- drive the RTL ------------------------------------------------------
    mon = SysCounters(tb.N_PE)
    stop = Event()
    cocotb.start_soon(_fifob_monitor(dut, mon, stop))

    phases = []                            # (name, delta-of-snaps)
    anatomy = []                           # (g, c, router, fill+scan, tail-wait)
    got_all = [None] * N_OUT               # per out-ch {cid: acc}
    t = mon.snap()

    for g in range(n_groups):
        gch = [order[g * tb.N_PE + p] for p in range(tb.N_PE)]   # PE p -> out-ch
        await tb.reset(dut)
        t2 = mon.snap(); phases.append((f"g{g}_reset", _delta(t, t2))); t = t2

        # Preamble: channel 0's CSR into FIFO-A + weights into the shadow
        # bank, then swap -- nothing routed yet.
        await tb.fill_activation_csr(dut, chans[0])
        await tb.run_scan(dut)
        await _load_group_weights(dut, [wsp_sp[0][k][1] for k in gch])
        await tb.swap_weights(dut)
        t2 = mon.snap(); phases.append((f"g{g}_pre", _delta(t, t2))); t = t2

        for c in range(N_CHAN_IN):
            # Router sprints ahead; PEs consume the FIFO-B backlog behind it.
            r_cyc = await tb.kick_stage2(dut)

            # FIFO-A is free: fill+scan the NEXT channel under the backlog,
            # while the next channel's weights stream into the shadow bank.
            fs_cyc = 0
            tload = None
            if c + 1 < N_CHAN_IN:
                s0 = mon.cycles
                tload = cocotb.start_soon(_load_group_weights(
                    dut, [wsp_sp[c + 1][k][1] for k in gch]))
                await tb.fill_activation_csr(dut, chans[c + 1])
                await tb.run_scan(dut)
                fs_cyc = mon.cycles - s0
            if tload is not None:
                await tload

            w_cyc = await tb.wait_pes_idle(dut)
            if c + 1 < N_CHAN_IN:
                await tb.swap_weights(dut)
            anatomy.append((g, c, r_cyc, fs_cyc, w_cyc))
            t2 = mon.snap(); phases.append((f"g{g}c{c}_pipe", _delta(t, t2))); t = t2

        got = await tb.drain_all(dut)
        for p in range(tb.N_PE):
            got_all[gch[p]] = got[p]
        t2 = mon.snap(); phases.append((f"g{g}_drain", _delta(t, t2))); t = t2

    stop.set()

    # -- verify every output channel against PyTorch-derived golden ---------
    mismatches = 0
    for k in range(N_OUT):
        out = [[got_all[k].get(r * tb.E + col, 0) for col in range(tb.E)]
               for r in range(tb.E)]
        if out != golden[k]:
            mismatches += 1
            dut._log.error(f"out-ch {k} (group {k // tb.N_PE} PE {k % tb.N_PE}) mismatch")
    assert mismatches == 0, f"{mismatches}/{N_OUT} output channels wrong"

    # cross-check: monitored executed MACs == functional-model pair hits
    meas_exec = sum(sum(d[4]) for _, d in phases)
    assert meas_exec == total_exec, (
        f"executed MACs {meas_exec} != model {total_exec}")
    meas_beats = sum(sum(d[1]) for _, d in phases)
    ideal_beats = sum(sum(r) for r in exp_beats)
    assert meas_beats >= ideal_beats

    # -- report -------------------------------------------------------------
    def agg(suffix):
        """Sum phase deltas whose name ends with `suffix`; returns (n, sums)."""
        out = [0, [0] * tb.N_PE, [0] * tb.N_PE, [0] * tb.N_PE, [0] * tb.N_PE]
        n = 0
        for nm, d in phases:
            if nm.endswith(suffix):
                n += 1
                out[0] += d[0]
                for i in range(1, 5):
                    out[i] = [a + b for a, b in zip(out[i], d[i])]
        return n, out

    total_cycles = sum(d[0] for _, d in phases)
    pipe_cycles = agg("pipe")[1][0]
    n_lanes = tb.N_PE * tb.N_MULTS

    def util(cyc):
        return 100.0 * total_useful / (cyc * n_lanes) if cyc else 0.0

    act_fill_bits = n_groups * (sum(nnz) * (tb.DATA_W + IDX_W)
                                + N_CHAN_IN * (tb.N_ROWS + 1) * PTR_W)
    wgt_bits = (n_weights + n_stuffed) * (tb.DATA_W + tb.PID_W)
    fifob_bits = meas_beats * BEAT_W
    drain_bits = N_OUT * tb.N_CID * tb.ACC_W

    weights_note = (f"SYNTHETIC balanced kernels: all {EQUAL_NW}/9 non-zero "
                    f"(EQUAL_NW)" if EQUAL_NW > 0 else
                    "real MobileNetV2 first-conv weights")
    lines = [
        "GOSPA APPLE LAYER PERF -- V2 arch, batched router + pipelined channels",
        f"MobileNetV2 first conv shape, real 80x80 apple input; {weights_note}",
        f"H={tb.H} F={tb.F} S={tb.S} E={tb.E}  N_PE={tb.N_PE} N_MULTS={tb.N_MULTS} "
        f"STAGE1_BATCH={S1_BATCH} FILL_W={tb.FILL_W} S2_BEATS={tb.S2_BEATS}",
        f"{n_groups} group passes x {N_CHAN_IN} input channels "
        f"({tb.N_PE} output channels/group, {n_lanes} multipliers)"
        + ("; GROUP_SORT: channels grouped by beat count" if GROUP_SORT else ""),
        f"nnz per channel  = {nnz}  (density {[round(z / (tb.H * tb.H), 3) for z in nnz]})",
        f"(pid,cid) pairs  = {n_pairs} per channel",
        f"weights loaded   = {n_weights} words (+{n_stuffed} stuffed zeros for "
        f"empty kernels)",
        f"useful MACs      = {total_useful}   executed MACs = {total_exec}",
        f"FIFO-B beats     = {meas_beats} measured vs {ideal_beats} ideal "
        f"(packing efficiency {100.0 * ideal_beats / meas_beats:.1f}%)",
        "",
        f"{'segment (total)':<16} {'n':>3} {'cycles':>8} {'consume/PE':>11} "
        f"{'stall/PE':>9} {'starve/PE':>10}",
        "-" * 62,
    ]
    for suffix in ("reset", "pre", "load", "pipe", "drain"):
        n, d = agg(suffix)
        lines.append(f"{suffix:<16} {n:>3} {d[0]:>8} {sum(d[1]) / tb.N_PE:>11.0f} "
                     f"{sum(d[2]) / tb.N_PE:>9.1f} {sum(d[3]) / tb.N_PE:>10.0f}")
    lines += [
        "-" * 62,
        f"total cycles (all groups, reset->drain done) = {total_cycles}",
        "",
        f"multiplier utilization = useful MACs / (window x {n_lanes} lanes):",
        f"  over pipe windows (router+PE+hidden fill/scan): {util(pipe_cycles):6.1f}%"
        f"   ({pipe_cycles} cyc)",
        f"  end-to-end incl pre/load/drain               : {util(total_cycles):6.1f}%"
        f"   ({total_cycles} cyc)",
        "",
        "pipe anatomy per channel (cycles):",
        f"{'g':>2} {'c':>2} {'router':>7} {'fill+scan':>10} {'tail-wait':>10} "
        f"{'pipe-total':>11}   note",
        "-" * 62,
    ]
    for (g, c, r_cyc, fs_cyc, w_cyc) in anatomy:
        pipe_tot = next(d[0] for nm, d in phases if nm == f"g{g}c{c}_pipe")
        note = "last ch: nothing to prefetch" if fs_cyc == 0 else \
            ("fill+scan fully hidden" if w_cyc > 4 else "fill+scan exposed")
        lines.append(f"{g:>2} {c:>2} {r_cyc:>7} {fs_cyc:>10} {w_cyc:>10} "
                     f"{pipe_tot:>11}   {note}")
    lines += [
        "-" * 62,
        "  router    = s2_start -> s2_done (FIFO-A drained at S2_BEATS beats/cyc)",
        "  fill+scan = NEXT channel's CSR fill + scan, running under PE backlog",
        "  tail-wait = remaining PE backlog after fill+scan returned",
        "",
        "loss split inside the pipe windows, per group pass:",
        f"{'g':>2} {'PE':>3} {'out':>4} {'consumed':>9} {'stalled':>8} {'starved':>8} "
        f"{'useMACs':>8} {'util%':>6} {'partial%':>8} {'waste%':>7} {'starve%':>8}",
    ]
    per_group = []
    for g in range(n_groups):
        acc = [0, [0] * tb.N_PE, [0] * tb.N_PE, [0] * tb.N_PE, [0] * tb.N_PE]
        for nm, d in phases:
            if nm.startswith(f"g{g}") and nm.endswith("pipe"):
                acc[0] += d[0]
                for i in range(1, 5):
                    acc[i] = [a + b for a, b in zip(acc[i], d[i])]
        per_group.append(acc)
    for g in range(n_groups):
        cyc, cons, stal, starv, ex = per_group[g]
        for pe in range(tb.N_PE):
            k = order[g * tb.N_PE + pe]
            w = cons[pe] + stal[pe] + starv[pe]
            useful = sum(exp_useful[c][k] for c in range(N_CHAN_IN))
            lanes_tot = w * tb.N_MULTS
            partial = cons[pe] * tb.N_MULTS - ex[pe]      # invalid lanes in beats
            waste = ex[pe] - useful                        # stuffed zero-weight MACs
            lines.append(
                f"{g:>2} {pe:>3} {k:>4} {cons[pe]:>9} {stal[pe]:>8} {starv[pe]:>8} "
                f"{useful:>8} {100.0 * useful / lanes_tot:>6.1f} "
                f"{100.0 * partial / lanes_tot:>8.1f} {100.0 * waste / lanes_tot:>7.1f} "
                f"{100.0 * starv[pe] / w:>8.1f}")
    lines += [
        "",
        "  util%    = useful MACs / (window x M)      (weight-sparsity work only)",
        "  partial% = admitted-beat lanes with no activation (tail of a PID run)",
        "  waste%   = zero-stuffed kernel MACs (empty per-channel kernels)",
        "  starve%  = no beat for this PE: WSP filtering (sparse kernels receive",
        "             fewer beats) + the tail of the pipe window.",
        "",
        "data transfer (bits):",
        f"  activation CSR fill : {act_fill_bits}  ({act_fill_bits / total_useful:.1f} b/MAC)"
        f"  [{n_groups}x refill of each channel]",
        f"  weight fills        : {wgt_bits}  ({wgt_bits / total_useful:.2f} b/MAC)",
        f"  FIFO-B beats        : {fifob_bits}  ({fifob_bits / total_useful:.1f} b/MAC)",
        f"  drain (32 x E^2 acc): {drain_bits}  ({drain_bits / total_useful:.1f} b/MAC)",
        "",
        f"golden check: PASS -- all {N_OUT} output channels match PyTorch-derived "
        "reference",
        "V1 baseline for comparison: testing/gospa/v1_baseline/gospa_apple_perf_v1.txt",
    ]
    report = os.path.join(_TEST_DIR, "gospa_apple_perf.txt")
    with open(report, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    dut._log.info("report written to %s" % report)
    for ln in lines:
        dut._log.info(ln)
