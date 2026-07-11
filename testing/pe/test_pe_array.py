"""
test_pe_array.py -- cocotb tests for pe_array.sv (NUM_PE GoSPA PEs)
GoSPA Project -- Team 19, ECE 720 (Spring 2026)

Run with `make array` (see testing/pe/Makefile for targets and knobs).

Verification: one input channel, NUM_PE output channels (one kernel per PE).
Route the activation through the real functional front end (Stage 1 + Stage 2
with each PE's WSP) to get each PE's FIFO-B stream, drive them all in parallel
(PEs stall independently on weight skips), drain every PE, and check each PE's
E x E output against dense convolution conv2d_reference(act, kernel_k).
"""

import os
import sys
import random

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ReadOnly
from cocotb.utils import get_sim_time


def _check_resolvable(sig, name):
    """Fail loudly (with signal name + time) if a sampled port carries X/Z,
    instead of the opaque ValueError raised by int(x_value) downstream. Gate-
    level sim exposes X that RTL sim masks via X-optimism; this pinpoints it."""
    val = sig.value
    if not val.is_resolvable:
        t = get_sim_time("ns")
        raise AssertionError(
            f"{name} has unresolvable X/Z: '{val.binstr}' @ {t}ns")

_SW_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "sw")
sys.path.insert(0, os.path.abspath(_SW_DIR))
import functional as fm                       # noqa: E402
fm._VERBOSE = False

H         = int(os.environ.get("H", "8"))
F         = int(os.environ.get("F", "3"))
S         = int(os.environ.get("S", "1"))
NUM_PE    = int(os.environ.get("NUM_PE", "4"))
NUM_MULTS = int(os.environ.get("NUM_MULTS", "1"))   # lanes per PE; each PE here uses lane 0
DATA_WIDTH = 16
ACC_WIDTH  = 32

E       = (H - F) // S + 1
NUM_PID = F * F
NUM_CID = E * E
CLK_NS  = 10


def _rtl_clog2(n):
    return 0 if n <= 1 else (n - 1).bit_length()


PID_WIDTH   = 1 if NUM_PID < 2 else _rtl_clog2(NUM_PID)
CID_WIDTH   = 1 if NUM_CID < 2 else _rtl_clog2(NUM_CID)
FIFOB_WIDTH = DATA_WIDTH + PID_WIDTH + CID_WIDTH
RPTR_WIDTH  = max(1, (NUM_PID + 1 - 1).bit_length())   # mirror RTL $clog2(NUM_PID+1)


def _signed(v, bits):
    return v - (1 << bits) if (v >> (bits - 1)) & 1 else v


def _mask(v, bits):
    return v & ((1 << bits) - 1)


def rand_matrix(R, C, density, rng, lo=-9, hi=9):
    return [[(rng.randint(lo, hi) if rng.random() < density else 0)
             for _ in range(C)] for _ in range(R)]


def rand_kernel(density, rng):
    k = rand_matrix(F, F, density, rng)
    if all(v == 0 for r in k for v in r):
        k[0][0] = rng.randint(1, 9)
    return k


# ---------------------------------------------------------------------------
# Routing: one activation -> NUM_PE FIFO-B streams + per-PE sparse weights
# ---------------------------------------------------------------------------
def route_array(act, kernels):
    values, col_idx, row_ptr = fm.dense_to_csr(act)
    wsps, sws = [], []
    for ker in kernels:
        wsp, sw = fm.kernel_to_sparse(ker)
        wsps.append(wsp)
        sws.append(sw)
    pos = fm.csr_to_positional(values, col_idx, row_ptr)
    pairs = []
    for (a, x, y) in pos:
        a2, px, py, cx, cy = fm.axy_to_pcid(a, x, y, S)
        pairs.extend(fm.pcid_to_cid_pid(a2, px, py, cx, cy, F, H, S))
    pairs = fm.zero_act_filter(pairs)
    fifo_a = fm.route_to_fifo_a(pairs, F)
    fifo_b_list = fm.broadcast_to_fifo_b(fifo_a, wsps)     # one per PE
    return fifo_b_list, sws


# ---------------------------------------------------------------------------
# DUT driver
# ---------------------------------------------------------------------------
async def reset(dut):
    dut.rst_n.value         = 0
    dut.wfill_we.value      = 0
    dut.wfill_pe.value      = 0
    dut.wfill_lane.value    = 0
    dut.wfill_pid.value     = 0
    dut.wfill_val.value     = 0
    dut.wload_done.value    = 0
    dut.fifob_valid.value   = 0
    dut.fifob_data.value    = 0
    dut.drain_start.value   = 0
    dut.out_ready.value     = 0
    for _ in range(4):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


async def load_weights(dut, sw_list):
    """Append each PE's weights into lane 0 of its on-chip SRAM (one weight per
    cycle, PID order), then pulse wload_done to arm all PEs together. The PE
    derives each lane's slot, weight count, and WSP from the fill stream, so no
    slot/count/WSP ports are driven. Every PE here uses only lane 0.

    RTL sim only: sram.sv's `initial` zeroing defines unwritten slots, so a lane
    with no fills reads 0 (have_curr stays 0 -> IDLE). Gate-level sim, which
    lacks that zeroing, would need explicit clearing of the unused slots."""
    for k in range(NUM_PE):
        for (pid, val) in sw_list[k]:
            dut.wfill_we.value   = 1
            dut.wfill_pe.value   = k
            dut.wfill_lane.value = 0
            dut.wfill_pid.value  = pid
            dut.wfill_val.value  = _mask(val, DATA_WIDTH)
            await RisingEdge(dut.clk)
    dut.wfill_we.value = 0

    dut.wload_done.value = 1
    await RisingEdge(dut.clk)
    dut.wload_done.value = 0
    # Warm-up: 2*NUM_MULTS+2 cycles to seed every lane's Curr+Next from SRAM.
    for _ in range(2 * NUM_MULTS + 2):
        await RisingEdge(dut.clk)


async def stream_fifob(dut, fifo_b_list, timeout=200000):
    """Present every PE's head; pop the lanes that handshake (ready) each cycle."""
    queues = [list(fb) for fb in fifo_b_list]
    guard = 0
    while any(queues) and guard < timeout:
        vbits = 0
        dbits = 0
        for k in range(NUM_PE):
            if queues[k]:
                axy, cid, pid = queues[k][0]
                vbits |= (1 << k)
                field = ((_mask(axy, DATA_WIDTH) << (PID_WIDTH + CID_WIDTH))
                         | (pid << CID_WIDTH) | cid)
                dbits |= field << (k * FIFOB_WIDTH)
        dut.fifob_valid.value = vbits
        dut.fifob_data.value  = dbits
        await ReadOnly()
        _check_resolvable(dut.fifob_ready, "fifob_ready")
        rdy = int(dut.fifob_ready.value)
        for k in range(NUM_PE):
            if (vbits >> k) & 1 and (rdy >> k) & 1:
                queues[k].pop(0)
        await RisingEdge(dut.clk)
        guard += 1
    dut.fifob_valid.value = 0


async def drain(dut, timeout=50000):
    """Drain lane 0 of every PE (each PE here uses only lane 0). out_valid/
    out_cid/out_acc are packed [NUM_PE-1:0][NUM_MULTS-1:0][...]; lane 0 of PE k
    sits at bit/field offset k*NUM_MULTS (+0 for lane 0)."""
    dut.drain_start.value = 1
    await RisingEdge(dut.clk)
    dut.drain_start.value = 0
    # out_ready is a plain bit per (pe,lane); set lane-0 bit of every PE.
    ready_val = 0
    for k in range(NUM_PE):
        ready_val |= 1 << (k * NUM_MULTS)
    dut.out_ready.value = ready_val

    got = [dict() for _ in range(NUM_PE)]
    cid_mask = (1 << CID_WIDTH) - 1
    acc_mask = (1 << ACC_WIDTH) - 1
    guard = 0
    while guard < timeout and any(len(g) < NUM_CID for g in got):
        await ReadOnly()
        _check_resolvable(dut.out_valid, "out_valid")
        vbits = int(dut.out_valid.value)
        oc    = int(dut.out_cid.value)
        oa    = int(dut.out_acc.value)
        for k in range(NUM_PE):
            lane_bit = k * NUM_MULTS
            if (vbits >> lane_bit) & 1:
                cid = (oc >> (lane_bit * CID_WIDTH)) & cid_mask
                acc = _signed((oa >> (lane_bit * ACC_WIDTH)) & acc_mask, ACC_WIDTH)
                got[k][cid] = acc
        await RisingEdge(dut.clk)
        guard += 1
    dut.out_ready.value = 0
    return got


async def run_case(dut, act, kernels, name):
    fifo_b_list, sw_list = route_array(act, kernels)
    golden = [fm.conv2d_reference(act, ker, S) for ker in kernels]

    await reset(dut)
    await load_weights(dut, sw_list)
    await stream_fifob(dut, fifo_b_list)
    got = await drain(dut)

    for k in range(NUM_PE):
        out_k = [[got[k].get(r * E + c, 0) for c in range(E)] for r in range(E)]
        assert out_k == golden[k], (
            f"[{name}] PE#{k} (H={H} F={F} S={S} NUM_PE={NUM_PE})\n"
            f"  weights = {sw_list[k]}\n"
            f"  expected= {golden[k]}\n"
            f"  got     = {out_k}"
        )
    depths = [len(fb) for fb in fifo_b_list]
    dut._log.info(f"[{name}] PASS  {NUM_PE} channels OK  "
                  f"(FIFO-B depths {depths}, H={H} F={F} S={S})")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
@cocotb.test()
async def test_dense_channels(dut):
    cocotb.start_soon(Clock(dut.clk, CLK_NS, units="ns").start())
    rng = random.Random(10)
    act = rand_matrix(H, H, 1.0, rng)
    kernels = [rand_kernel(1.0, rng) for _ in range(NUM_PE)]
    await run_case(dut, act, kernels, "dense_channels")


@cocotb.test()
async def test_sparse_act_dense_wgt(dut):
    """Sparse activations + dense kernels => weight skips on every PE."""
    cocotb.start_soon(Clock(dut.clk, CLK_NS, units="ns").start())
    rng = random.Random(11)
    for i in range(4):
        act = rand_matrix(H, H, rng.choice([0.1, 0.2]), rng)
        kernels = [rand_kernel(1.0, rng) for _ in range(NUM_PE)]
        await run_case(dut, act, kernels, f"sparse_act_dense_wgt[{i}]")


@cocotb.test()
async def test_mixed_sparsity(dut):
    """Per-channel kernel sparsity differs => unequal FIFO-B depths / PEs
    finish at different times (load imbalance)."""
    cocotb.start_soon(Clock(dut.clk, CLK_NS, units="ns").start())
    rng = random.Random(12)
    for i in range(4):
        act = rand_matrix(H, H, rng.choice([0.3, 0.5, 0.8]), rng)
        kernels = [rand_kernel(rng.choice([0.2, 0.5, 1.0]), rng) for _ in range(NUM_PE)]
        await run_case(dut, act, kernels, f"mixed_sparsity[{i}]")
