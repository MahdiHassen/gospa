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
H      = int(os.environ.get("H", "8"))
F      = int(os.environ.get("F", "3"))
S      = int(os.environ.get("S", "1"))
DATA_W = 16
ACC_W  = 32

E      = (H - F) // S + 1
N_PID  = F * F
N_CID  = E * E
CLK_NS = 10


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
    dut.rst_n.value       = 0
    dut.wload_en.value    = 0
    dut.wload_pid.value   = 0
    dut.wload_val.value   = 0
    dut.wload_done.value  = 0
    dut.b_valid.value     = 0
    dut.b_act.value       = 0
    dut.b_pid.value       = 0
    dut.b_cid.value       = 0
    dut.drain_start.value = 0
    dut.out_ready.value   = 0
    for _ in range(4):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


async def load_weights(dut, sw):
    """Preload sparse weights (pid, val) in PID order, then arm the PE."""
    for (pid, val) in sw:
        dut.wload_en.value  = 1
        dut.wload_pid.value = pid
        dut.wload_val.value = _mask(val, DATA_W)
        await RisingEdge(dut.clk)
    dut.wload_en.value = 0
    await RisingEdge(dut.clk)          # separate cycle: no overlap with wload_done
    dut.wload_done.value = 1
    await RisingEdge(dut.clk)
    dut.wload_done.value = 0


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
    """Pulse drain_start, collect N_CID accumulator beats -> {cid: acc}."""
    dut.drain_start.value = 1
    await RisingEdge(dut.clk)
    dut.drain_start.value = 0
    dut.out_ready.value = 1
    got = {}
    guard = 0
    while len(got) < N_CID and guard < timeout:
        await ReadOnly()
        if dut.out_valid.value == 1:
            got[int(dut.out_cid.value)] = _signed(int(dut.out_acc.value), ACC_W)
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
