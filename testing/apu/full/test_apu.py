"""
test_apu.py -- cocotb tests for apu.sv (full APU = Activation SRAM + Stage 1 + Stage 2)
GoSPA Project -- Team 19, ECE 720 (Spring 2026)

End-to-end APU RTL <-> SW cosim with the new pipelined front-end.

For each test the TB:
  1. Writes a dense H x H activation matrix into the on-chip activation
     SRAM via the fill port (one entry/cycle, addr = row*H + col).
  2. Pulses scan_start with n_rows=H, base_y=0 to walk the SRAM. The
     scanner emits (val, x, y) one per cycle into apu_stage1.
  3. After scan_done, pulses s2_start; routing drains FIFO-A and
     multicasts WSP-gated entries into the per-PE FIFO-B bank.
  4. Drains FIFO-B concurrently with Stage 2 and checks each PE's stream
     against the functional model:

         csr_to_positional -> zero_act_filter -> axy_to_pcid
                           -> pcid_to_cid_pid  -> route_to_fifo_a
                           -> broadcast_to_fifo_b

Run:
    make MODULE=test_apu                                  # default H=8 F=3 S=1 N_PE=4
    make MODULE=test_apu H=8 F=3 S=2 N_PE=8
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
H        = int(os.environ.get("H", "8"))
F        = int(os.environ.get("F", "3"))
S        = int(os.environ.get("S", "1"))
N_PE     = int(os.environ.get("N_PE", "4"))
N_ROWS   = int(os.environ.get("N_ROWS",   str(H)))     # rows the scanner walks
N_NZ_MAX = int(os.environ.get("N_NZ_MAX", str(H * H))) # entry SRAM depth
DATA_W   = 16


def _rtl_clog2(n):
    return 0 if n <= 1 else (n - 1).bit_length()


E         = (H - F) // S + 1
N_PID     = F * F
CID_W     = 1 if (E * E) < 2 else _rtl_clog2(E * E)
PID_W     = 1 if N_PID   < 2 else _rtl_clog2(N_PID)
IDX_W     = 1 if H       < 2 else _rtl_clog2(H)
PTR_W     = 1 if (N_NZ_MAX + 1) < 2 else _rtl_clog2(N_NZ_MAX + 1)
ENT_AW    = 1 if N_NZ_MAX       < 2 else _rtl_clog2(N_NZ_MAX)
RPTR_AW   = 1 if (N_ROWS + 1)   < 2 else _rtl_clog2(N_ROWS + 1)
FIFOB_W   = DATA_W + PID_W + CID_W

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


def _pack_one_wsp(wsp_per_pe):
    """Pack one PE's WSP (list[N_PID], index = PID) MSB-first to match RTL."""
    val = 0
    for p in range(N_PID):
        if wsp_per_pe[p]:
            val |= 1 << (N_PID - 1 - p)
    return val


# ---------------------------------------------------------------------------
# Driver / monitor
# ---------------------------------------------------------------------------
async def reset(dut):
    dut.rst_n.value             = 0
    dut.fill_entry_we.value     = 0
    dut.fill_entry_addr.value   = 0
    dut.fill_entry_value.value  = 0
    dut.fill_entry_col.value    = 0
    dut.fill_rptr_we.value      = 0
    dut.fill_rptr_addr.value    = 0
    dut.fill_rptr_data.value    = 0
    dut.scan_start.value        = 0
    dut.scan_n_rows.value       = 0
    dut.scan_base_row.value     = 0
    dut.s2_start.value          = 0
    dut.wsp_we.value            = 0
    dut.wsp_waddr.value         = 0
    dut.wsp_wdata.value         = 0
    dut.fifob_rd_ready.value    = 0
    for _ in range(4):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


async def load_wsps(dut, wsps):
    """Write each PE's WSP into the on-chip wsp_file (one per cycle)."""
    dut.wsp_we.value = 1
    for k in range(N_PE):
        dut.wsp_waddr.value = k
        dut.wsp_wdata.value = _pack_one_wsp(wsps[k])
        await RisingEdge(dut.clk)
    dut.wsp_we.value    = 0
    dut.wsp_waddr.value = 0
    dut.wsp_wdata.value = 0
    await RisingEdge(dut.clk)


async def fill_sram(dut, matrix):
    """Encode `matrix` as CSR and stream it into both fill ports.

    Writes:
      - row_ptr SRAM:  N_ROWS + 1 pointers
      - entry SRAM:    N non-zero {value, col} pairs (row-major)

    One write per cycle on each port; the two streams can be sequenced
    serially because Stage 2 won't start until scan_done.
    """
    values, col_idx, row_ptr = fm.dense_to_csr(matrix)
    if len(values) > N_NZ_MAX:
        raise ValueError(f"matrix has {len(values)} non-zeros but entry SRAM "
                         f"only holds {N_NZ_MAX}")

    # Fill row_ptr SRAM (N_ROWS+1 entries; pad upper slots with the last value
    # so any out-of-range scan still terminates cleanly).
    dut.fill_rptr_we.value = 1
    last_ptr = row_ptr[-1]
    for r in range(N_ROWS + 1):
        ptr = row_ptr[r] if r < len(row_ptr) else last_ptr
        dut.fill_rptr_addr.value = r
        dut.fill_rptr_data.value = ptr
        await RisingEdge(dut.clk)
    dut.fill_rptr_we.value   = 0
    dut.fill_rptr_addr.value = 0
    dut.fill_rptr_data.value = 0

    # Fill entry SRAM ({value, col} per non-zero).
    val_mask = (1 << DATA_W) - 1
    dut.fill_entry_we.value = 1
    for k in range(len(values)):
        dut.fill_entry_addr.value  = k
        dut.fill_entry_value.value = values[k] & val_mask
        dut.fill_entry_col.value   = col_idx[k]
        await RisingEdge(dut.clk)
    dut.fill_entry_we.value    = 0
    dut.fill_entry_addr.value  = 0
    dut.fill_entry_value.value = 0
    dut.fill_entry_col.value   = 0
    await RisingEdge(dut.clk)


async def trigger_scan(dut, n_rows=None, base_row=0, timeout=200000):
    """Pulse scan_start, wait for scan_done, then drain the inner pipeline.

    Returns the cycle count from start to done.
    """
    if n_rows is None:
        n_rows = H
    dut.scan_n_rows.value = n_rows
    dut.scan_base_row.value = base_row
    await RisingEdge(dut.clk)
    dut.scan_start.value = 1
    await RisingEdge(dut.clk)
    dut.scan_start.value = 0

    cycles = 0
    while cycles < timeout:
        await ReadOnly()
        if int(dut.scan_done.value) == 1:
            break
        await RisingEdge(dut.clk)
        cycles += 1
    await RisingEdge(dut.clk)

    # Combinational pipeline + 1-cycle FIFO-A write -- give it a few cycles
    # to settle so Stage 2 sees the final FIFO-A contents.
    for _ in range(6):
        await RisingEdge(dut.clk)
    return cycles


async def feed_matrix(dut, matrix):
    """Fill the SRAM with `matrix` and scan all H rows."""
    await fill_sram(dut, matrix)
    await trigger_scan(dut, n_rows=H, base_row=0)


async def run_stage2_and_drain(dut, timeout=200000):
    """Pulse s2_start, drain FIFO-B concurrently, return list[N_PE] of (a,p,c)."""
    collected = [[] for _ in range(N_PE)]
    dut.fifob_rd_ready.value = (1 << N_PE) - 1

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
            binstr = dut.fifob_rd_data.value.binstr   # MSB-first
            L = len(binstr)
            for k in range(N_PE):
                if (vbits >> k) & 1:
                    field = binstr[L - (k + 1) * FIFOB_W : L - k * FIFOB_W]
                    payload = int(field, 2)
                    cid = payload & ((1 << CID_W) - 1)
                    pid = (payload >> CID_W) & ((1 << PID_W) - 1)
                    axy = (payload >> (CID_W + PID_W)) & ((1 << DATA_W) - 1)
                    collected[k].append((axy, pid, cid))
        if int(dut.s2_done.value) == 1:
            seen_done = True
        await RisingEdge(dut.clk)
        if seen_done and vbits == 0:
            drain_extra += 1
            if drain_extra >= 2:
                break
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
                  f"(H={H} F={F} S={S} N_PE={N_PE} N_ROWS={N_ROWS} E={E})")


async def run_case(dut, matrix, wsps, name):
    await reset(dut)
    await load_wsps(dut, wsps)
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
    for density in [0.1, 0.25, 0.5, 0.9]:
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
    while len(wsps) < N_PE:
        wsps.append([0] * N_PID)
    await run_case(dut, m, wsps, "paper_toy")
