"""
test_apu.py -- cocotb tests for apu.sv (full APU = Stage 1 + Stage 2)
GoSPA Project -- Team 19, ECE 720 (Spring 2026)

End-to-end APU RTL <-> SW cosim.

Drives a dense H x H activation matrix as CSR into apu.sv, sets per-PE WSPs,
pulses s2_start, drains the FIFO-B bank concurrently, and checks each PE's
collected (a_xy, pid, cid) stream against the team functional model:

    csr_to_positional -> zero_act_filter -> axy_to_pcid
                      -> pcid_to_cid_pid  -> route_to_fifo_a
                      -> broadcast_to_fifo_b

The TB depends on the same valid/ready handshakes and FIFO backpressure
semantics that are unit-tested in test_apu_stage1.py and tb_routing.py.

Run:
    make MODULE=test_apu SIM=icarus                       # default H=8 F=3 S=1 N_PE=4
    make MODULE=test_apu SIM=icarus H=8 F=3 S=2 N_PE=8
    make sweep_apu                                        # many configs
"""

import os
import sys
import random

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ReadOnly

_SW_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "sw")
sys.path.insert(0, os.path.abspath(_SW_DIR))
import functional as fm                                  # noqa: E402
fm._VERBOSE = False

# -- Config (must match Makefile -P/-G overrides) ---------------------------
H      = int(os.environ.get("H", "8"))
F      = int(os.environ.get("F", "3"))
S      = int(os.environ.get("S", "1"))
N_PE   = int(os.environ.get("N_PE", "4"))
DATA_W = 16


def _rtl_clog2(n):
    return 0 if n <= 1 else (n - 1).bit_length()


E       = (H - F) // S + 1
N_PID   = F * F
CID_W   = 1 if (E * E) < 2 else _rtl_clog2(E * E)
PID_W   = 1 if N_PID    < 2 else _rtl_clog2(N_PID)
FIFOB_W = DATA_W + PID_W + CID_W

CLK_NS = 10


# ---------------------------------------------------------------------------
# Golden model: Stage 1 + Stage 2 from functional.py
# ---------------------------------------------------------------------------
def golden_fifo_b(matrix, wsps):
    """Return list[N_PE] of [(axy, cid, pid), ...] -- expected FIFO-B contents."""
    values, col_idx, row_ptr = fm.dense_to_csr(matrix)
    stream = fm.csr_to_positional(values, col_idx, row_ptr)
    stream = fm.zero_act_filter(stream)
    pairs = []
    for (axy, x, y) in stream:
        axy, px, py, cx, cy = fm.axy_to_pcid(axy, x, y, S)
        pairs.extend(fm.pcid_to_cid_pid(axy, px, py, cx, cy, F, H, S))
    fifo_a = fm.route_to_fifo_a(pairs, F)
    fifo_b = fm.broadcast_to_fifo_b(fifo_a, wsps)
    # Mask to RTL widths so signed/unsigned comparisons line up.
    act_mask = (1 << DATA_W) - 1
    cid_mask = (1 << CID_W)  - 1
    return [[(a & act_mask, c & cid_mask, p) for (a, c, p) in fbk] for fbk in fifo_b]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _signed(v, bits):
    return v - (1 << bits) if (v >> (bits - 1)) & 1 else v


def _rand_matrix(rng, density):
    m = []
    for _ in range(H):
        row = []
        for _ in range(H):
            if rng.random() < density:
                v = rng.randint(1, 100)
                row.append(v if rng.random() < 0.5 else -v)
            else:
                row.append(0)
        m.append(row)
    return m


def _rand_wsps(rng):
    """Random per-PE WSPs as list[N_PE][N_PID] (index = PID)."""
    return [[rng.randint(0, 1) for _ in range(N_PID)] for _ in range(N_PE)]


def _pack_wsp(wsps):
    """list[N_PE][N_PID] -> packed bus. MSB-first by PID, matches RTL wsp[k][N_PID-1-pid]."""
    val = 0
    for k in range(N_PE):
        for p in range(N_PID):
            if wsps[k][p]:
                val |= 1 << (k * N_PID + (N_PID - 1 - p))
    return val


# ---------------------------------------------------------------------------
# Driver / monitor
# ---------------------------------------------------------------------------
async def reset(dut):
    dut.rst_n.value          = 0
    dut.row_ptr_valid.value  = 0
    dut.row_ptr_data.value   = 0
    dut.entry_valid.value    = 0
    dut.entry_value.value    = 0
    dut.entry_col.value      = 0
    dut.s2_start.value       = 0
    dut.wsp.value            = 0
    dut.fifob_rd_ready.value = 0
    for _ in range(4):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


async def _send_row_ptr(dut, ptr):
    dut.row_ptr_valid.value = 1
    dut.row_ptr_data.value  = ptr
    while True:
        await RisingEdge(dut.clk)
        if dut.row_ptr_ready.value == 1:
            break
    dut.row_ptr_valid.value = 0


async def _send_entry(dut, value, col):
    dut.entry_valid.value = 1
    dut.entry_value.value = value & ((1 << DATA_W) - 1)
    dut.entry_col.value   = col
    while True:
        await RisingEdge(dut.clk)
        if dut.entry_ready.value == 1:
            break
    dut.entry_valid.value = 0


async def feed_matrix(dut, matrix):
    values, col_idx, row_ptr = fm.dense_to_csr(matrix)

    async def drive_rptr():
        for ptr in row_ptr:
            await _send_row_ptr(dut, ptr)

    async def drive_entries():
        for v, c in zip(values, col_idx):
            await _send_entry(dut, v, c)

    rt = cocotb.start_soon(drive_rptr())
    et = cocotb.start_soon(drive_entries())
    await rt
    await et
    # let the last accepted entry land in FIFO-A
    for _ in range(5):
        await RisingEdge(dut.clk)


async def run_stage2_and_drain(dut, timeout=200000):
    """Pulse s2_start, drain FIFO-B concurrently, return list[N_PE] of (a,p,c).

    fifob_rd_ready is held high throughout so the FIFO-B's never act as a queue;
    that gives the simplest functional check. (A separate test exercises FIFO-B
    backpressure.)
    """
    collected = [[] for _ in range(N_PE)]
    dut.fifob_rd_ready.value = (1 << N_PE) - 1

    # Single-cycle start pulse.
    await RisingEdge(dut.clk)
    dut.s2_start.value = 1
    await RisingEdge(dut.clk)
    dut.s2_start.value = 0

    guard = 0
    seen_done = False
    drain_extra = 0
    while guard < timeout:
        await ReadOnly()
        vbits = int(dut.fifob_rd_valid.value)
        if vbits != 0:
            binstr = dut.fifob_rd_data.value.binstr   # MSB-first, len = N_PE*FIFOB_W
            L = len(binstr)
            for k in range(N_PE):
                if (vbits >> k) & 1:
                    field = binstr[L - (k + 1) * FIFOB_W : L - k * FIFOB_W]
                    payload = int(field, 2)
                    cid = payload & ((1 << CID_W) - 1)
                    pid = (payload >> CID_W) & ((1 << PID_W) - 1)
                    axy = (payload >> (CID_W + PID_W)) & ((1 << DATA_W) - 1)
                    collected[k].append((axy, pid, cid))
        # Sample done before the next edge -- it's a 1-cycle pulse.
        if int(dut.s2_done.value) == 1:
            seen_done = True
        await RisingEdge(dut.clk)
        if seen_done and vbits == 0:
            drain_extra += 1
            if drain_extra >= 2:
                break
        elif vbits == 0 and not seen_done:
            # Nothing to drain yet, keep waiting.
            pass
        guard += 1

    dut.fifob_rd_ready.value = 0
    await RisingEdge(dut.clk)
    return collected


def check_against_golden(dut, got, matrix, wsps, name):
    expect = golden_fifo_b(matrix, wsps)
    for k in range(N_PE):
        got_ac = [(a, c, p) for (a, p, c) in got[k]]    # reorder to model's (a,c,p)
        assert got_ac == expect[k], (
            f"[{name}] FIFO-B[{k}] mismatch:\n"
            f"    got    = {got_ac}\n"
            f"    expect = {expect[k]}"
        )
    total = sum(len(s) for s in got)
    dut._log.info(f"[{name}] PASS -- {total} entries across {N_PE} FIFO-B's "
                  f"(H={H} F={F} S={S} N_PE={N_PE} E={E})")


async def run_case(dut, matrix, wsps, name):
    await reset(dut)
    dut.wsp.value = _pack_wsp(wsps)
    await feed_matrix(dut, matrix)
    got = await run_stage2_and_drain(dut)
    check_against_golden(dut, got, matrix, wsps, name)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
@cocotb.test()
async def test_empty(dut):
    """All-zero matrix; every FIFO-B should end up empty."""
    cocotb.start_soon(Clock(dut.clk, CLK_NS, units="ns").start())
    wsps = [[1] * N_PID for _ in range(N_PE)]
    await run_case(dut, [[0] * H for _ in range(H)], wsps, "empty")


@cocotb.test()
async def test_single_nonzero_all_wsp(dut):
    """One non-zero, all WSPs = 1 -> every PE sees every matching PID."""
    cocotb.start_soon(Clock(dut.clk, CLK_NS, units="ns").start())
    m = [[0] * H for _ in range(H)]
    m[H // 2][H // 2] = 7
    wsps = [[1] * N_PID for _ in range(N_PE)]
    await run_case(dut, m, wsps, "single_nonzero_all_wsp")


@cocotb.test()
async def test_dense_disjoint_wsps(dut):
    """Dense matrix with disjoint per-PE WSPs (each PE owns ~1 PID)."""
    cocotb.start_soon(Clock(dut.clk, CLK_NS, units="ns").start())
    m = [[((r * H + c) % 50) - 25 or 1 for c in range(H)] for r in range(H)]
    wsps = [[1 if p == k % N_PID else 0 for p in range(N_PID)] for k in range(N_PE)]
    await run_case(dut, m, wsps, "dense_disjoint_wsps")


@cocotb.test()
async def test_random(dut):
    """Several randomized matrices x randomized WSPs."""
    cocotb.start_soon(Clock(dut.clk, CLK_NS, units="ns").start())
    rng = random.Random(0xA9)
    for i, density in enumerate([0.1, 0.25, 0.5, 0.9]):
        m = _rand_matrix(rng, density)
        wsps = _rand_wsps(rng)
        await run_case(dut, m, wsps, f"random[d={density}]")


@cocotb.test()
async def test_paper_toy(dut):
    """Paper's running example (only meaningful at F=2, H=3, S=1)."""
    cocotb.start_soon(Clock(dut.clk, CLK_NS, units="ns").start())
    if (H, F, S) != (3, 2, 1):
        dut._log.info(f"[paper_toy] skipped (config H={H} F={F} S={S} != 3,2,1)")
        return
    m = [[-1, 3, 1],
         [0,  2, 0],
         [0,  0, -2]]
    wsps = [[1, 0, 1, 0], [0, 1, 1, 0]][:N_PE]
    # If we have more PEs than the toy example covers, pad with all-zero WSPs.
    while len(wsps) < N_PE:
        wsps.append([0] * N_PID)
    await run_case(dut, m, wsps, "paper_toy")
