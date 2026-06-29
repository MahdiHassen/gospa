"""
test_pe.py -- cocotb tests for pe.sv (GoSPA Processing Element)
GoSPA Project -- Team 19, ECE 720 (Spring 2026)

Run (default Icarus):
    make MODULE=test_pe
    make MODULE=test_pe H=8 F=3 S=2
    make sweep_pe

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
from cocotb.triggers import RisingEdge, ReadOnly

_SW_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "sw")
sys.path.insert(0, os.path.abspath(_SW_DIR))
import functional as fm                       # noqa: E402
fm._VERBOSE = False

# -- config (matches -G/-P overrides; PE is elaborated with N_PID, N_CID) -----
H       = int(os.environ.get("H", "8"))
F       = int(os.environ.get("F", "3"))
S       = int(os.environ.get("S", "1"))
N_MULTS = int(os.environ.get("N_MULTS", "1"))   # V1-equivalent default: 1 lane
DATA_W  = 16
ACC_W   = 32

E       = (H - F) // S + 1
N_PID   = F * F
N_CID   = E * E
CLK_NS  = 10
WPTR_W  = max(1, (N_PID + 1 - 1).bit_length())
CID_W   = 1 if N_CID < 2 else (N_CID - 1).bit_length()


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
    dut.wfill_slot.value    = 0
    dut.wfill_pid.value     = 0
    dut.wfill_val.value     = 0
    dut.wsp_we.value        = 0
    dut.wsp_lane.value      = 0
    dut.wsp_data.value      = 0
    dut.wload_count.value   = 0
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


def _pack_per_lane_counts(counts):
    """Pack a list[N_MULTS] of per-lane weight counts into the wload_count bus."""
    val = 0
    for k, c in enumerate(counts):
        val |= (c & ((1 << WPTR_W) - 1)) << (k * WPTR_W)
    return val


def _wsp_from_sw(sw):
    """Derive per-lane WSP (N_PID-bit MSB-first by PID)? -- here LSB=PID 0
    because the RTL stores wsp_q[k] indexed by PID directly: wsp_q[k][p]."""
    val = 0
    for (pid, _) in sw:
        val |= 1 << pid
    return val


async def load_weights(dut, sw):
    """Single-channel load into lane 0 of an N_MULTS-wide V2 PE.

    Steps:
      1) Write sw[i] -> SRAM at (lane=0, slot=i) using {wfill_we/lane/slot/pid/val}.
      2) Write lane 0's WSP via {wsp_we/lane/data}; other lanes' WSPs left 0
         so they IDLE on every activation.
      3) Drive per-lane counts (count[0]=len(sw), others=0), then pulse
         wload_done. The PE issues 2*N_MULTS SRAM reads to pre-load every
         lane's Curr/Next before entering S_RUN.
    """
    # 1) weight SRAM fill (lane 0)
    for slot, (pid, val) in enumerate(sw):
        dut.wfill_we.value   = 1
        dut.wfill_lane.value = 0
        dut.wfill_slot.value = slot
        dut.wfill_pid.value  = pid
        dut.wfill_val.value  = _mask(val, DATA_W)
        await RisingEdge(dut.clk)
    dut.wfill_we.value = 0

    # 2) per-lane WSP -- lane 0 only.
    dut.wsp_we.value   = 1
    dut.wsp_lane.value = 0
    dut.wsp_data.value = _wsp_from_sw(sw)
    await RisingEdge(dut.clk)
    dut.wsp_we.value = 0

    # 3) arm with per-lane counts (only lane 0 has weights).
    counts = [0] * N_MULTS
    counts[0] = len(sw)
    dut.wload_count.value = _pack_per_lane_counts(counts)
    dut.wload_done.value  = 1
    await RisingEdge(dut.clk)
    dut.wload_done.value = 0
    # Warm-up: 2*N_MULTS+1 cycles to seed every lane's Curr+Next from SRAM.
    for _ in range(2 * N_MULTS + 2):
        await RisingEdge(dut.clk)


async def stream_fifo_b(dut, fifo_b):
    """Feed (axy, cid, pid) honoring b_ready (which stalls during weight skips)."""
    for (axy, cid, pid) in fifo_b:
        dut.b_valid.value = 1
        dut.b_act.value   = _mask(axy, DATA_W)
        dut.b_pid.value   = pid
        dut.b_cid.value   = cid
        while True:
            await RisingEdge(dut.clk)
            if dut.b_ready.value == 1:
                break
        dut.b_valid.value = 0
    dut.b_valid.value = 0


async def drain(dut, timeout=20000):
    """Pulse drain_start, collect lane-0's N_CID accumulator beats -> {cid: acc}.

    out_valid/cid/acc are now packed [N_MULTS-1:0]/[N_MULTS-1:0][...]; we
    just inspect lane 0 (LSB) for the single-channel test.
    """
    dut.drain_start.value = 1
    await RisingEdge(dut.clk)
    dut.drain_start.value = 0
    dut.out_ready.value = (1 << N_MULTS) - 1
    got = {}
    cid_mask = (1 << CID_W) - 1
    acc_mask = (1 << ACC_W) - 1
    guard = 0
    while len(got) < N_CID and guard < timeout:
        await ReadOnly()
        ov = int(dut.out_valid.value)
        if ov & 1:                                          # lane 0 valid
            cid0 = int(dut.out_cid.value) & cid_mask        # lane 0 -> LSBs
            acc0_u = int(dut.out_acc.value) & acc_mask
            got[cid0] = _signed(acc0_u, ACC_W)
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
