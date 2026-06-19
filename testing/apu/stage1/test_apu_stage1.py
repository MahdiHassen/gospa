"""
test_apu_stage1.py -- cocotb tests for apu_stage1.sv (APU Stage 1, full chain)
GoSPA Project -- Team 19, ECE 720 (Spring 2026)

Run (full design includes idgen.sv which crashes old Verilator -- use Icarus):
    make MODULE=test_apu_stage1 SIM=icarus
    make MODULE=test_apu_stage1 SIM=icarus H=8 F=3 S=2
    make sweep_s1                              # several configs

Stage-1 RTL <-> SW co-simulation
--------------------------------
Drives a dense H x H activation matrix into apu_stage1 as CSR, drains all
F^2 FIFO-A read ports, and checks the result against the *team functional
model* (sw/functional.py) used as the golden reference:

    csr_to_positional -> zero_act_filter -> axy_to_pcid
                      -> pcid_to_cid_pid  -> route_to_fifo_a

NOTE: Stage 1 does NOT apply WSP (neither does functional.py's route_to_fifo_a).
WSP gating happens in Stage 2 (broadcast_to_fifo_b). So FIFO-A here holds every
geometrically-valid (Axy, CID) pair, PID-binned -- exactly what the golden emits.

FIFO-A[p] payload is { a_xy[DATA_W-1:0], cid[CID_W-1:0] }; PID is implicit (= p).
"""

import os
import sys
import math
import random

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ReadOnly

# --- import the team functional model as the golden reference --------------
_SW_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "sw")
sys.path.insert(0, os.path.abspath(_SW_DIR))
import functional as fm                      # noqa: E402
fm._VERBOSE = False                          # silence the model's hot-loop prints

# --- elaborated config (must match the -P/-G overrides in the Makefile) ----
H      = int(os.environ.get("H", "8"))
F      = int(os.environ.get("F", "3"))
S      = int(os.environ.get("S", "1"))
DATA_W = 16                                  # apu_stage1 / csr_decode default

E      = (H - F) // S + 1
G      = (F + S - 1) // S                    # ceil(F/S)
N_PID  = F * F


def _rtl_clog2(n):
    """$clog2(n): bits to index n values. $clog2(1)=0, $clog2(4)=2."""
    return 0 if n <= 1 else (n - 1).bit_length()


CID_W   = 1 if (E * E) < 2 else _rtl_clog2(E * E)   # mirrors apu_stage1.sv
FIFOA_W = DATA_W + CID_W
CLK_NS  = 10


# ---------------------------------------------------------------------------
# Golden model -- Stage 1 front end straight out of functional.py
# ---------------------------------------------------------------------------
def golden_fifo_a(matrix):
    """Return list[F*F] of [(axy, cid), ...] -- the expected FIFO-A contents."""
    values, col_idx, row_ptr = fm.dense_to_csr(matrix)
    stream = fm.csr_to_positional(values, col_idx, row_ptr)
    stream = fm.zero_act_filter(stream)
    pairs = []
    for (axy, x, y) in stream:
        axy, px, py, cx, cy = fm.axy_to_pcid(axy, x, y, S)
        pairs.extend(fm.pcid_to_cid_pid(axy, px, py, cx, cy, F, H, S))
    return fm.route_to_fifo_a(pairs, F)


# ---------------------------------------------------------------------------
# Small helpers
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
    dut.fifoa_rd_ready.value = 0
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
    """Stream a dense matrix in as CSR (row_ptr + entry streams, concurrently)."""
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
    # let the last accepted entry propagate into FIFO-A
    for _ in range(5):
        await RisingEdge(dut.clk)


async def drain_fifo_a(dut, timeout=200000):
    """Drain every FIFO-A slot in parallel; return list[F*F] of [(axy, cid), ...]."""
    collected = [[] for _ in range(N_PID)]
    dut.fifoa_rd_ready.value = (1 << N_PID) - 1     # pop from all slots at once
    guard = 0
    while guard < timeout:
        await ReadOnly()
        vbits = int(dut.fifoa_rd_valid.value)
        if vbits == 0:
            break
        # rd_data of an *empty* slot is x (fifo.sv: stale when empty), so parse
        # only the slots flagged valid -- slice the binary string per slot
        # rather than int()-ing the whole packed vector (which trips on x).
        binstr = dut.fifoa_rd_data.value.binstr     # MSB-first, len = N_PID*FIFOA_W
        L = len(binstr)
        for p in range(N_PID):
            if (vbits >> p) & 1:
                field = binstr[L - (p + 1) * FIFOA_W : L - p * FIFOA_W]
                payload = int(field, 2)
                cid = payload & ((1 << CID_W) - 1)
                axy = _signed((payload >> CID_W) & ((1 << DATA_W) - 1), DATA_W)
                collected[p].append((axy, cid))
        await RisingEdge(dut.clk)
        guard += 1
    await RisingEdge(dut.clk)
    dut.fifoa_rd_ready.value = 0
    await RisingEdge(dut.clk)
    return collected


def check_against_golden(dut, got, matrix, name):
    golden = golden_fifo_a(matrix)
    for p in range(N_PID):
        assert got[p] == golden[p], (
            f"[{name}] FIFO-A[{p}] mismatch (PID={p}):\n"
            f"    got    = {got[p]}\n"
            f"    expect = {golden[p]}"
        )
    total = sum(len(s) for s in got)
    dut._log.info(f"[{name}] PASS -- {total} entries across {N_PID} FIFO-A slots "
                  f"(H={H} F={F} S={S} E={E} G={G})")


async def run_case(dut, matrix, name):
    await reset(dut)
    await feed_matrix(dut, matrix)
    got = await drain_fifo_a(dut)
    check_against_golden(dut, got, matrix, name)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
@cocotb.test()
async def test_empty(dut):
    """All-zero activation map -> every FIFO-A empty."""
    cocotb.start_soon(Clock(dut.clk, CLK_NS, units="ns").start())
    await run_case(dut, [[0] * H for _ in range(H)], "empty")


@cocotb.test()
async def test_single_nonzero(dut):
    """One non-zero near the centre."""
    cocotb.start_soon(Clock(dut.clk, CLK_NS, units="ns").start())
    m = [[0] * H for _ in range(H)]
    m[H // 2][H // 2] = 7
    await run_case(dut, m, "single_nonzero")


@cocotb.test()
async def test_dense(dut):
    """Fully dense map (stresses FIFO-A occupancy and the join logic)."""
    cocotb.start_soon(Clock(dut.clk, CLK_NS, units="ns").start())
    m = [[((r * H + c) % 50) - 25 or 1 for c in range(H)] for r in range(H)]
    await run_case(dut, m, "dense")


@cocotb.test()
async def test_random(dut):
    """Several randomized sparse maps (with negative values)."""
    cocotb.start_soon(Clock(dut.clk, CLK_NS, units="ns").start())
    rng = random.Random(0xA9)
    for i in range(6):
        density = [0.1, 0.25, 0.4, 0.5, 0.75, 0.9][i]
        await run_case(dut, _rand_matrix(rng, density), f"random[d={density}]")


@cocotb.test()
async def test_paper_toy(dut):
    """The reference-doc toy example (only meaningful at F=2,H=3,S=1)."""
    cocotb.start_soon(Clock(dut.clk, CLK_NS, units="ns").start())
    if (H, F, S) != (3, 2, 1):
        dut._log.info(f"[paper_toy] skipped (config H={H} F={F} S={S} != 3,2,1)")
        return
    # W=[[0,3],[-3,0]], A below -> dense conv output [[9,-3],[6,0]] (per doc).
    m = [[-1, 3, 1],
         [0,  2, 0],
         [0,  0, -2]]
    await run_case(dut, m, "paper_toy")
