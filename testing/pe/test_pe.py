"""
test_pe.py -- cocotb tests for pe.sv (GoSPA Processing Element)
GoSPA Project -- Team 19, ECE 720 (Spring 2026)

Run with `make pe` (see testing/pe/Makefile for targets and knobs).

Verification: the PE is checked against TRUE DENSE CONVOLUTION
(functional.conv2d_reference), NOT against functional.pe_process. The team's
pe_process uses a single-step Curr/Next slide that multiplies by the wrong
weight whenever a non-zero weight PID receives no activation (sparse inputs);
it disagrees with dense conv in ~14% of random sparse cases. pe.sv handles that
SKIP case correctly, so dense conv is the right golden.

For each (activation, kernel): route it through the real functional front end
(Stage 1 + Stage 2 for a single PE) to get the PID-ordered FIFO-B stream and the
sparse weight list, drive both into the DUT, drain the accumulators, and compare
the E x E result against conv2d_reference.
"""

import os
import sys
import random

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ReadOnly, Event

_SW_DIR  = os.path.join(os.path.dirname(__file__), "..", "..", "sw")
_REF_DIR = os.path.join(os.path.dirname(__file__), "..", "ref")
sys.path.insert(0, os.path.abspath(_SW_DIR))
sys.path.insert(0, os.path.abspath(_REF_DIR))
import functional as fm                       # noqa: E402
fm._VERBOSE = False

# -- config (matches -G/-P overrides; PE is elaborated with NUM_PID, NUM_CID) -----
H          = int(os.environ.get("H", "8"))
F          = int(os.environ.get("F", "3"))
S          = int(os.environ.get("S", "1"))
NUM_MULTS  = int(os.environ.get("NUM_MULTS", "4"))   # V2 default: 4 lanes/PE (synth config)
DATA_WIDTH = 16
ACC_WIDTH  = 32

E       = (H - F) // S + 1
NUM_PID = F * F
NUM_CID = E * E
CLK_NS  = 2      # clock period in ns -> 500 MHz (simulation only)
# Override the clock the perf report quotes GMAC/s at (e.g. the period the
# synthesis flow actually closes on). Default: the simulated clock.
FCLK_MHZ    = os.environ.get("FCLK_MHZ")
RPTR_WIDTH  = max(1, (NUM_PID + 1 - 1).bit_length())
CID_WIDTH   = 1 if NUM_CID < 2 else (NUM_CID - 1).bit_length()


def _signed(v, bits):
    return v - (1 << bits) if (v >> (bits - 1)) & 1 else v


def _mask(v, bits):
    return v & ((1 << bits) - 1)


# ---------------------------------------------------------------------------
# Golden + routing, straight from the functional model
# ---------------------------------------------------------------------------
def route_single_pe(act, ker):
    """Run the real front end for one kernel -> (fifo_b, sparse_weights)."""
    values, col_idx, row_ptr = fm.dense_to_csr(act)
    wsp, sw = fm.kernel_to_sparse(ker)
    pos = fm.csr_to_positional(values, col_idx, row_ptr)
    pairs = []
    for (axy, x, y) in pos:
        a, px, py, cx, cy = fm.axy_to_pcid(axy, x, y, S)
        pairs.extend(fm.pcid_to_cid_pid(a, px, py, cx, cy, F, H, S))
    pairs = fm.zero_act_filter(pairs)
    fifo_a = fm.route_to_fifo_a(pairs, F)
    fifo_b = fm.broadcast_to_fifo_b(fifo_a, [wsp])[0]   # [(axy, cid, pid), ...]
    return fifo_b, sw


def rand_matrix(R, C, density, rng, lo=-9, hi=9):
    return [[(rng.randint(lo, hi) if rng.random() < density else 0)
             for _ in range(C)] for _ in range(R)]


# ---------------------------------------------------------------------------
# DUT driver
# ---------------------------------------------------------------------------
async def reset(dut):
    dut.rst_n.value         = 0
    dut.wfill_we.value      = 0
    dut.wfill_lane.value    = 0
    dut.wfill_pid.value     = 0
    dut.wfill_val.value     = 0
    dut.wload_done.value    = 0
    dut.b_valid.value       = 0
    dut.b_act.value         = 0
    dut.b_pid.value         = 0
    dut.b_cid.value         = 0
    dut.drain_start.value   = 0
    dut.out_ready.value     = 0
    for _ in range(4):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


async def load_weights(dut, sw):
    """Single-channel load into lane 0 of an NUM_MULTS-wide V2 PE.

    The PE derives each lane's slot, weight count, and WSP from the fill
    stream, so we just append sw[i] to lane 0 and pulse wload_done. Other
    lanes get no fills -> count 0, WSP 0 -> they IDLE on every activation.
    """
    # 1) weight SRAM fill (lane 0): append one weight per cycle in PID order.
    for (pid, val) in sw:
        dut.wfill_we.value   = 1
        dut.wfill_lane.value = 0
        dut.wfill_pid.value  = pid
        dut.wfill_val.value  = _mask(val, DATA_WIDTH)
        await RisingEdge(dut.clk)
    dut.wfill_we.value = 0

    # 2) arm: pulse wload_done. The PE arms the next cycle with an empty window
    # (no warm sequence); each lane's first hit fetches its weight on demand.
    dut.wload_done.value = 1
    await RisingEdge(dut.clk)
    dut.wload_done.value = 0
    await RisingEdge(dut.clk)


async def stream_fifo_b(dut, fifo_b):
    """Feed (axy, cid, pid) honoring b_ready (which stalls during weight skips)."""
    for (axy, cid, pid) in fifo_b:
        dut.b_valid.value = 1
        dut.b_act.value   = _mask(axy, DATA_WIDTH)
        dut.b_pid.value   = pid
        dut.b_cid.value   = cid
        while True:
            await RisingEdge(dut.clk)
            if dut.b_ready.value == 1:
                break
        dut.b_valid.value = 0
    dut.b_valid.value = 0


async def drain(dut, timeout=20000):
    """Pulse drain_start, collect lane-0's NUM_CID accumulator beats -> {cid: acc}.

    out_valid/cid/acc are now packed [NUM_MULTS-1:0]/[NUM_MULTS-1:0][...]; we
    just inspect lane 0 (LSB) for the single-channel test.
    """
    dut.drain_start.value = 1
    await RisingEdge(dut.clk)
    dut.drain_start.value = 0
    dut.out_ready.value = (1 << NUM_MULTS) - 1
    got = {}
    cid_mask = (1 << CID_WIDTH) - 1
    acc_mask = (1 << ACC_WIDTH) - 1
    guard = 0
    while len(got) < NUM_CID and guard < timeout:
        await ReadOnly()
        ov = int(dut.out_valid.value)
        if ov & 1:                                          # lane 0 valid
            cid0 = int(dut.out_cid.value) & cid_mask        # lane 0 -> LSBs
            acc0_u = int(dut.out_acc.value) & acc_mask
            got[cid0] = _signed(acc0_u, ACC_WIDTH)
        await RisingEdge(dut.clk)
        guard += 1
    dut.out_ready.value = 0
    return got


async def run_case(dut, act, ker, name):
    fifo_b, sw = route_single_pe(act, ker)
    golden = fm.conv2d_reference(act, ker, S)          # E x E, true dense conv

    await reset(dut)
    await load_weights(dut, sw)
    await stream_fifo_b(dut, fifo_b)
    got = await drain(dut)

    # Reshape {cid: acc} -> E x E and compare against dense conv.
    out = [[got.get(r * E + c, 0) for c in range(E)] for r in range(E)]
    assert out == golden, (
        f"[{name}] H={H} F={F} S={S}\n"
        f"  weights(sparse) = {sw}\n"
        f"  fifo_b          = {fifo_b}\n"
        f"  expected (dense)= {golden}\n"
        f"  got (PE)        = {out}"
    )
    nz = sum(1 for r in golden for v in r if v != 0)
    dut._log.info(f"[{name}] PASS  H={H} F={F} S={S}  ({len(fifo_b)} acts, "
                  f"{len(sw)} weights, {nz} nonzero outputs)")


# ===========================================================================
# Performance measurement
# ===========================================================================

def _popcount(v):
    return bin(v & ((1 << NUM_MULTS) - 1)).count("1")


def _safe_int(sig):
    """Read a DUT signal as int; return None if the handle is absent or the
    value is unresolved (x/z)."""
    if sig is None:
        return None
    try:
        return int(sig.value)
    except Exception:
        return None


class PerfCounters:
    """Cycle counters for one PE run. All counts are in clk cycles."""
    __slots__ = ("cycles", "offered", "consumed", "stalled",
                 "stall_fetch", "macs", "load_cycles", "drain_cycles")

    def __init__(self):
        self.cycles = 0          # every sampled cycle the monitor is alive
        self.offered = 0         # b_valid==1 (PE was presented an activation)
        self.consumed = 0        # b_valid && b_ready (activation admitted)
        self.stalled = 0         # b_valid && !b_ready (bubble)
        self.stall_fetch = 0     # of the stalls, a lane was awaiting a weight fetch
        self.macs = 0            # sum of per-lane useful MACs (popcount mac_en_q)
        self.load_cycles = 0     # fixed overhead: weight-load + arm
        self.drain_cycles = 0    # fixed overhead: accumulator drain-out

    # --- derived metrics ---------------------------------------------------
    @property
    def stall_other(self):
        """Stalled cycles not attributed to a lane fetch -- must be 0 if
        need_fetch is readable (b_ready = !need_fetch)."""
        return self.stalled - self.stall_fetch

    @property
    def overall_lane_util(self):
        """Useful MACs / (offered x NUM_MULTS): fraction of multiplier-slots
        doing useful work, counting fetch-stall bubbles as idle (a stalled
        cycle is a cycle the multipliers produced nothing). offered is every
        clk in the streaming phase (excludes weight-load and drain)."""
        d = self.offered * NUM_MULTS
        return self.macs / d if d else 0.0


async def _mac_cycle_monitor(dut, perf, stop):
    """Count total cycles + useful MACs from mac_fire (a 1-cycle pulse per
    accumulate). NOTE: mac_en_q is now the multi-cycle "MAC in flight" drain
    gate (the multiplier is deeply pipelined), so it would over-count by the
    pipeline fill/drain tail -- mac_fire is the exact per-MAC event."""
    while True:
        await RisingEdge(dut.clk)
        # Check stop in the writable region so the monitor never returns parked
        # in ReadOnly (which would block the next case's driver writes).
        if stop.is_set():
            break
        await ReadOnly()            # sample mac_fire in the settled region
        perf.cycles += 1
        mf = _safe_int(dut.mac_fire)
        if mf:
            perf.macs += _popcount(mf)


async def stream_fifo_b_perf(dut, fifo_b, perf, sig_need_fetch):
    """stream_fifo_b + per-cycle handshake accounting (offered/consumed/stalled
    + stall cause).

    The PE has a single stall cause: b_ready is low iff some wsp-hit lane still
    needs to fetch its weight (need_fetch). That signal is asserted on every
    stalled cycle, so a per-cycle check attributes them all -- stall_other must
    end at 0 (verified in run_case_perf)."""
    for (axy, cid, pid) in fifo_b:
        dut.b_valid.value = 1
        dut.b_act.value   = _mask(axy, DATA_WIDTH)
        dut.b_pid.value   = pid
        dut.b_cid.value   = cid
        while True:
            await RisingEdge(dut.clk)
            perf.offered += 1
            if dut.b_ready.value == 1:
                perf.consumed += 1
                break
            perf.stalled += 1
            if _safe_int(sig_need_fetch):
                perf.stall_fetch += 1
        dut.b_valid.value = 0
    dut.b_valid.value = 0


async def run_case_perf(dut, act, kernels, name, sig_need_fetch):
    """Drive NUM_MULTS real kernels into one V2 PE, measure the streaming phase,
    and verify every lane against dense conv. Returns (PerfCounters, n_acts)."""
    fifo_b, per_lane_sw, per_lane_wsp = route_v2_one_pe(act, kernels)
    goldens = [fm.conv2d_reference(act, k, S) for k in kernels]

    await reset(dut)

    # Start the monitor before the weight load so perf.cycles captures the
    # fixed load+arm overhead too (no MACs fire before running, so this does
    # not perturb the MAC count).
    perf = PerfCounters()
    stop = Event()
    mon = cocotb.start_soon(_mac_cycle_monitor(dut, perf, stop))

    await load_weights_multi(dut, per_lane_sw, per_lane_wsp)
    perf.load_cycles = perf.cycles

    await stream_fifo_b_perf(dut, fifo_b, perf, sig_need_fetch)

    c_pre_drain = perf.cycles
    got = await drain_all_lanes(dut)
    perf.drain_cycles = perf.cycles - c_pre_drain
    stop.set()
    await mon

    assert perf.stall_other == 0, (
        f"[{name}] {perf.stall_other} stalled cycles unattributed "
        f"(need_fetch probe unreadable?)")

    for k, golden in enumerate(goldens):
        out = [[got[k].get(r * E + c, 0) for c in range(E)] for r in range(E)]
        assert out == golden, (
            f"[{name}] lane {k}  H={H} F={F} S={S}\n"
            f"  weights(sparse) = {per_lane_sw[k]}\n"
            f"  expected (dense)= {golden}\n"
            f"  got (PE)        = {out}")
    # Cross-check the monitor against the routed stream: useful MACs == the
    # (activation, lane) pairs whose WSP hits the activation's PID.
    expected_macs = sum(
        1
        for (_axy, _cid, pid) in fifo_b
        for k in range(len(per_lane_wsp))
        if per_lane_wsp[k][pid]
    )
    assert perf.macs == expected_macs, (
        f"[{name}] monitor MACs {perf.macs} != expected {expected_macs}")
    return perf, len(fifo_b)


@cocotb.test()
async def test_perf(dut):
    """Measure PE throughput / stall-rate / lane-utilization across a spread of
    activation & weight densities. Reports per-case and aggregate numbers so an
    optimization can target the dominant loss (stalls vs union under-util).

    Runs in the normal suite; isolate + sweep it with e.g.
    `TESTCASE=test_perf NUM_MULTS=4 make pe` (honors the H/F/S/NUM_MULTS knobs).
    """
    cocotb.start_soon(Clock(dut.clk, CLK_NS, units="ns").start())
    rng = random.Random(0xBEEF)

    # Resolve the internal stall-cause probe once (None if absent -> the
    # stall_other assert in run_case_perf catches an unusable probe). It lives
    # inside the fetch submodule (u_fetch), reached hierarchically.
    fetch = getattr(dut, "u_fetch", None)
    sig_need_fetch = getattr(fetch, "need_fetch", None)

    # (name, activation density, weight density). Names read actXX_wgtYY =
    # XX% activation density, YY% weight density.
    # Note: b_ready stalls only on lanes that can't MAC the current beat, so a
    # same-PID run hides the fetch latency; "roofline" sustains ~1.0. Stalls
    # appear when a lane must jump to a new weight (sparse-activation PID jumps).
    grid = [
        # dense reference
        ("roofline",     1.0, 1.0),   # both dense: peak / roofline baseline

        # equal-density diagonal sweep: 90/90 down to 10/10, descending from roofline
        ("act90_wgt90",  0.9, 0.9),
        ("act80_wgt80",  0.8, 0.8),
        ("act70_wgt70",  0.7, 0.7),
        ("act60_wgt60",  0.6, 0.6),
        ("act50_wgt50d", 0.5, 0.5),
        ("act40_wgt40",  0.4, 0.4),
        ("act30_wgt30d", 0.3, 0.3),
        ("act20_wgt20",  0.2, 0.2),
        ("act10_wgt10",  0.1, 0.1),

        None,   # blank-line separator: diagonal sweep above, original cases below

        # weight sweep @ dense activation: isolates union-gating (no act stalls)
        ("act100_wgt70", 1.0, 0.7),
        ("act100_wgt50", 1.0, 0.5),
        ("act100_wgt30", 1.0, 0.3),
        ("act100_wgt10", 1.0, 0.1),

        # activation sweep @ dense weight: isolates reload stalls (short PID runs)
        ("act80_wgt100", 0.8, 1.0),
        ("act60_wgt100", 0.6, 1.0),
        ("act30_wgt100", 0.3, 1.0),
        ("act10_wgt100", 0.1, 1.0),   # sparse act, dense wgt: reload-stall stress

        # mixed densities: the realistic middle of the space
        ("act80_wgt70",  0.8, 0.7),
        ("act60_wgt50",  0.6, 0.5),
        ("act50_wgt50",  0.5, 0.5),
        ("act30_wgt70",  0.3, 0.7),
        ("act30_wgt30",  0.3, 0.3),
        ("act10_wgt50",  0.1, 0.5),
        ("act10_wgt20",  0.1, 0.2),   # both sparse: low-utilization corner
    ]

    rows = []
    # density 1.0 must have no structural zeros (rand_matrix draws 0 ~1/19 of
    # the time) so the dense row is a true roofline.
    def _fill_nonzero(mat):
        return [[(v if v != 0 else 1) for v in row] for row in mat]

    for entry in grid:
        if entry is None:
            rows.append(None)                    # section break -> blank line
            continue
        (nm, da, dw) = entry
        act = rand_matrix(H, H, da, rng)
        if da >= 1.0:
            act = _fill_nonzero(act)
        # one distinct kernel per lane -> real union-gating, not a 1-of-M artifact
        kernels = []
        for _ in range(NUM_MULTS):
            ker = rand_matrix(F, F, dw, rng)
            if dw >= 1.0:
                ker = _fill_nonzero(ker)
            if all(v == 0 for r in ker for v in r):
                ker[0][0] = 3
            kernels.append(ker)
        perf, nacts = await run_case_perf(dut, act, kernels, nm, sig_need_fetch)
        rows.append((nm, perf, nacts))

    # Clock-derived absolute rate: MAC/s = mac_per_cycle * f_clk. FCLK_MHZ
    # overrides the quoted clock (e.g. the synthesized period); default = sim.
    if FCLK_MHZ is not None:
        f_clk_hz  = float(FCLK_MHZ) * 1e6
        f_clk_src = "specified"
    else:
        f_clk_hz  = 1e9 / CLK_NS
        f_clk_src = "simulated"
    # Cycles = offered, the sparsity-aware streaming phase (admitted acts + reload stalls);
    # matches perf_pe.PEStats.pe_cycles. The dense baseline is the roofline case
    # (dense act + dense wgt, same H/F/S); speedup = how much sparsity shortens
    # that window. Weight-load + drain are fixed overhead (amortized per layer),
    # reported separately, not folded into the speedup.
    dense_cycles = next((r[1].offered for r in rows if r and r[0] == "roofline"), 0)
    peak_gops = NUM_MULTS * 2 * f_clk_hz / 1e9

    # GOP/s over the offered (streaming) cycles (1 MAC = 2 ops); peak is NUM_MULTS*2*f_clk.
    def _gops(p):
        return p.macs * 2 * f_clk_hz / 1e9 / p.offered if p.offered else 0.0

    hdr = (f"{'case':<12} {'minCyc':>8} {'computeCyc':>10} {'latency':>8} "
           f"{'speedup':>8} {'GOPS/s':>8} {'multUtil%':>10}")
    lines = [
        "PE PERF  H=%d F=%d S=%d NUM_MULTS=%d  f_clk=%.0f MHz (%s)  "
        "dense=%d cyc  peak=%.3f GOPS/s"
        % (H, F, S, NUM_MULTS, f_clk_hz / 1e6, f_clk_src, dense_cycles, peak_gops),
        hdr,
        "-" * len(hdr),
    ]
    for row in rows:
        if row is None:
            lines.append("")                     # separator between case groups
            continue
        (nm, p, nacts) = row
        speedup = dense_cycles / p.offered if p.offered else 0.0
        lines.append(f"{nm:<12} {p.consumed:>8} {p.offered:>10} {p.cycles:>8} "
                     f"{speedup:>8.2f} {_gops(p):>8.3f} "
                     f"{p.overall_lane_util * 100:>10.1f}")
    lines.append("-" * len(hdr))
    lines += [
        "legend:",
        "  minCyc     = theoretical min num of cycles. 1 activation/cycle with no stalls ",
        "  computeCyc = actual cycles it takes to process input -> minCyc + reload stalls",
        "  latency    = end-to-end: computeCyc + weight-load + warm-up + drain",
        "  speedup    = baseline computeCyc / computeCyc of particular case",
        "  GOPS/s     = rate over offered cycles, 1 MAC = 2 ops",
        "  multUtil%  = MACs / (computeCyc x NUM_MULTS)",
    ]

    report_path = os.path.join(os.path.dirname(__file__), "pe_perf.txt")
    with open(report_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    dut._log.info("PE perf report written to %s" % report_path)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
@cocotb.test()
async def test_dense(dut):
    cocotb.start_soon(Clock(dut.clk, CLK_NS, units="ns").start())
    rng = random.Random(1)
    await run_case(dut, rand_matrix(H, H, 1.0, rng), rand_matrix(F, F, 1.0, rng), "dense")


@cocotb.test()
async def test_single_weight(dut):
    """Only one non-zero weight (exercises the no-Next path)."""
    cocotb.start_soon(Clock(dut.clk, CLK_NS, units="ns").start())
    rng = random.Random(2)
    ker = [[0] * F for _ in range(F)]
    ker[rng.randrange(F)][rng.randrange(F)] = 5
    await run_case(dut, rand_matrix(H, H, 0.6, rng), ker, "single_weight")


@cocotb.test()
async def test_sparse_act_dense_wgt(dut):
    """The case that breaks pe_process: sparse activations + dense kernel
    => many weight PIDs get skipped. The PE must still match dense conv."""
    cocotb.start_soon(Clock(dut.clk, CLK_NS, units="ns").start())
    rng = random.Random(3)
    for i in range(8):
        act = rand_matrix(H, H, rng.choice([0.1, 0.15, 0.25]), rng)
        ker = rand_matrix(F, F, 1.0, rng)               # fully dense kernel
        if all(v == 0 for r in ker for v in r):
            continue
        await run_case(dut, act, ker, f"sparse_act_dense_wgt[{i}]")


@cocotb.test()
async def test_random_mix(dut):
    cocotb.start_soon(Clock(dut.clk, CLK_NS, units="ns").start())
    rng = random.Random(4)
    for i in range(12):
        da = rng.choice([0.2, 0.4, 0.6, 0.9])
        dw = rng.choice([0.3, 0.5, 0.8, 1.0])
        act = rand_matrix(H, H, da, rng)
        ker = rand_matrix(F, F, dw, rng)
        if all(v == 0 for r in ker for v in r):
            ker[0][0] = 3
        await run_case(dut, act, ker, f"random[{i},da={da},dw={dw}]")


# ===========================================================================
# Multi-lane V2 with real MobileNet weights
# ===========================================================================
async def load_weights_multi(dut, per_lane_sw, per_lane_wsp=None):
    """Load up to NUM_MULTS distinct kernels into one V2 PE.

    per_lane_sw[k] : list[(pid, val)] for lane k (in PID order), [] if empty.
    The PE derives each lane's slot, count, and WSP from the fill stream, so
    per_lane_wsp is no longer driven (kept for call-site compatibility).
    """
    assert len(per_lane_sw) <= NUM_MULTS

    # 1) per-lane weight SRAM writes: append one weight per cycle, PID order.
    for k in range(len(per_lane_sw)):
        for (pid, val) in per_lane_sw[k]:
            dut.wfill_we.value   = 1
            dut.wfill_lane.value = k
            dut.wfill_pid.value  = pid
            dut.wfill_val.value  = _mask(val, DATA_WIDTH)
            await RisingEdge(dut.clk)
    dut.wfill_we.value = 0

    # 2) arm. The PE arms the next cycle with an empty window (no warm sequence);
    # each lane's first hit fetches its weight on demand.
    dut.wload_done.value = 1
    await RisingEdge(dut.clk)
    dut.wload_done.value = 0
    await RisingEdge(dut.clk)


async def drain_all_lanes(dut, timeout=20000):
    """Drain NUM_MULTS lanes in parallel. Returns list[NUM_MULTS] of {cid: acc}."""
    dut.drain_start.value = 1
    await RisingEdge(dut.clk)
    dut.drain_start.value = 0
    dut.out_ready.value = (1 << NUM_MULTS) - 1

    got = [dict() for _ in range(NUM_MULTS)]
    cid_mask = (1 << CID_WIDTH) - 1
    acc_mask = (1 << ACC_WIDTH) - 1
    guard = 0
    while guard < timeout and any(len(g) < NUM_CID for g in got):
        await ReadOnly()
        ov = int(dut.out_valid.value)
        oc = int(dut.out_cid.value)
        oa = int(dut.out_acc.value)
        for k in range(NUM_MULTS):
            if (ov >> k) & 1:
                cid = (oc >> (k * CID_WIDTH)) & cid_mask
                acc = _signed((oa >> (k * ACC_WIDTH)) & acc_mask, ACC_WIDTH)
                got[k][cid] = acc
        await RisingEdge(dut.clk)
        guard += 1
    dut.out_ready.value = 0
    return got


def route_v2_one_pe(act, kernels):
    """Functional-model APU Stage 1 + Stage 2 for one V2 PE holding `kernels`."""
    per_lane_wsp = []
    per_lane_sw  = []
    for ker in kernels:
        w, s = fm.kernel_to_sparse(ker)
        per_lane_wsp.append(w)
        per_lane_sw.append(s)
    union = fm.wsp_union(per_lane_wsp)

    values, col_idx, row_ptr = fm.dense_to_csr(act)
    pos = fm.csr_to_positional(values, col_idx, row_ptr)
    pairs = []
    for (axy, x, y) in pos:
        a, px, py, cx, cy = fm.axy_to_pcid(axy, x, y, S)
        pairs.extend(fm.pcid_to_cid_pid(a, px, py, cx, cy, F, H, S))
    pairs = fm.zero_act_filter(pairs)
    fifo_a = fm.route_to_fifo_a(pairs, F)
    fifo_b = fm.broadcast_to_fifo_b(fifo_a, [union])[0]
    return fifo_b, per_lane_sw, per_lane_wsp


def _make_sparse_activation(rng, density=0.5):
    return [[(rng.randint(1, 50) if rng.random() < density else 0)
             * (1 if rng.random() < 0.5 else -1) for _ in range(H)]
            for _ in range(H)]


def _load_mobilenet_kernels(n):
    """Load `n` red-channel kernels from MobileNetV2's first conv.
    Falls back to synthetic INT8 kernels when F != 3 (so the test still
    runs on other layer configs)."""
    if F == 3:
        try:
            from mobilenet import get_first_conv
            _, conv0 = get_first_conv()
            iw = conv0.weight().int_repr().numpy()
            return [[[int(v) for v in row] for row in iw[f, 0]]
                    for f in range(min(n, iw.shape[0]))]
        except Exception as exc:
            cocotb.log.warning(f"falling back to synthetic kernels: {exc}")
    rng = random.Random(0xA17 ^ n ^ F)
    return [[[rng.randint(-50, 50) for _ in range(F)] for _ in range(F)]
            for _ in range(n)]


@cocotb.test()
async def test_v2_mobilenet_multilane(dut):
    """V2 PE end-to-end with NUM_MULTS distinct kernels packed into one PE.

    Each lane gets its own MobileNet red-channel filter; the FIFO-B is the
    union-WSP-gated stream from fm.goSPA_route. Every lane's E x E output
    is checked against conv2d_reference(act, that lane's kernel)."""
    cocotb.start_soon(Clock(dut.clk, CLK_NS, units="ns").start())

    rng = random.Random(0xBEEF + H * 17 + F * 31)
    act = _make_sparse_activation(rng, density=0.5)
    kernels = _load_mobilenet_kernels(NUM_MULTS)
    assert len(kernels) == NUM_MULTS, f"need {NUM_MULTS} kernels, got {len(kernels)}"

    golden = [fm.conv2d_reference(act, ker, S) for ker in kernels]
    fifo_b, per_lane_sw, per_lane_wsp = route_v2_one_pe(act, kernels)
    n_nz = sum(1 for row in act for v in row if v != 0)
    dut._log.info(
        f"V2 PE: NUM_MULTS={NUM_MULTS} kernels, H={H} F={F} S={S} E={E}, "
        f"{n_nz} input non-zeros, {len(fifo_b)} FIFO-B entries, "
        f"per-lane #weights={[len(s) for s in per_lane_sw]}"
    )

    await reset(dut)
    await load_weights_multi(dut, per_lane_sw, per_lane_wsp)
    await stream_fifo_b(dut, fifo_b)
    got = await drain_all_lanes(dut)

    mismatches = 0
    for k in range(NUM_MULTS):
        out_k = [[got[k].get(r * E + c, 0) for c in range(E)] for r in range(E)]
        if out_k != golden[k]:
            mismatches += 1
            dut._log.error(
                f"lane#{k} mismatch: kernel={kernels[k]}\n"
                f"  golden[:3] = {golden[k][:3]}\n"
                f"  got   [:3] = {out_k[:3]}"
            )
    if mismatches:
        raise AssertionError(
            f"{mismatches}/{NUM_MULTS} lanes mismatched against conv2d_reference")

    dut._log.info(f"PASS -- all {NUM_MULTS} lanes match conv2d_reference for their kernel")
