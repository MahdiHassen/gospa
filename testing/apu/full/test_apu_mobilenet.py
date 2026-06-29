"""
test_apu_mobilenet.py -- RTL APU vs functional model on MobileNetV2's first
conv (red channel), V2 mapping (N_PE=8 PEs each holding N_MULTS=4 kernels).

GoSPA Project -- Team 19, ECE 720 (Spring 2026)

What this test exercises end-to-end
-----------------------------------
1. Pulls the real INT8 weights from MobileNetV2's first conv via
   testing/ref/mobilenet.py and slices out the red channel (in_ch=0) of all
   32 output filters.
2. Bundles 32 kernels into 8 PE chunks of 4 (V2). For each chunk the host
   computes the UNION WSP -- the per-PE bit pattern that gates routing.
3. Loads each PE's union WSP into the APU's on-chip wsp_file via the
   write port (one PE per cycle).
4. Generates a synthetic sparse INT8 activation (raw H x H, padded to
   (H+2) x (H+2) to mimic padding=1) and encodes it as CSR into the
   activation SRAM via the entry + row_ptr fill ports.
5. Pulses scan_start; after scan_done, pulses s2_start; drains FIFO-B and
   collects (Axy, PID, CID) per PE.
6. Compares those streams against fm.goSPA_route(..., interpretation="v2")
   -- the same routing the functional model produces for the same inputs.

Run
---
    make MODULE=test_apu_mobilenet H=34 F=3 S=2 N_PE=8 N_ROWS=34 N_NZ_MAX=2048
    # or just `make mobilenet` for the canonical config + VCD off
    WAVES=1 make mobilenet                 # also dumps a VCD trace
"""

import os
import sys
import random

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ReadOnly

# ---------------------------------------------------------------------------
# Reach the sw/ model and the testing/ref/ MobileNet loader.
# ---------------------------------------------------------------------------
_TEST_DIR = os.path.dirname(__file__)
_SW_DIR   = os.path.abspath(os.path.join(_TEST_DIR, "..", "..", "..", "sw"))
_REF_DIR  = os.path.abspath(os.path.join(_TEST_DIR, "..", "..", "..", "testing", "ref"))
sys.path.insert(0, _SW_DIR)
sys.path.insert(0, _REF_DIR)
import functional as fm                              # noqa: E402
fm._VERBOSE = False
from mobilenet import get_first_conv                 # noqa: E402

# ---------------------------------------------------------------------------
# Config (must match Makefile -G/-P overrides)
# ---------------------------------------------------------------------------
H        = int(os.environ.get("H", "34"))            # padded activation map
F        = int(os.environ.get("F", "3"))             # MobileNet first-conv kernel
S        = int(os.environ.get("S", "2"))             # MobileNet first-conv stride
N_PE     = int(os.environ.get("N_PE", "8"))
N_MULTS  = int(os.environ.get("N_MULTS", "4"))       # host-side; no RTL counterpart
N_ROWS   = int(os.environ.get("N_ROWS",   str(H)))
N_NZ_MAX = int(os.environ.get("N_NZ_MAX", "2048"))
DATA_W   = 16


def _clog2(n):
    return 0 if n <= 1 else (n - 1).bit_length()


E       = (H - F) // S + 1
N_PID   = F * F
CID_W   = 1 if (E * E)        < 2 else _clog2(E * E)
PID_W   = 1 if N_PID          < 2 else _clog2(N_PID)
IDX_W   = 1 if H              < 2 else _clog2(H)
PTR_W   = 1 if (N_NZ_MAX + 1) < 2 else _clog2(N_NZ_MAX + 1)
ENT_AW  = 1 if N_NZ_MAX       < 2 else _clog2(N_NZ_MAX)
RPTR_AW = 1 if (N_ROWS + 1)   < 2 else _clog2(N_ROWS + 1)
FIFOB_W = DATA_W + PID_W + CID_W
CLK_NS  = 10

# ---------------------------------------------------------------------------
# Load MobileNetV2 first-conv weights ONCE (module-level so cocotb startup
# pays the cost; subsequent tests reuse).
# ---------------------------------------------------------------------------
_, _conv0 = get_first_conv()
_int_w = _conv0.weight().int_repr().numpy()              # (32, 3, 3, 3) int8
RED_KERNELS = [[[int(v) for v in row] for row in _int_w[f, 0]]
               for f in range(_int_w.shape[0])]
N_KERNELS = N_PE * N_MULTS
assert N_KERNELS <= len(RED_KERNELS), \
    f"need {N_KERNELS} kernels but only {len(RED_KERNELS)} available"
KERNELS = RED_KERNELS[:N_KERNELS]

# Build per-PE union WSPs the same way fm.goSPA_route does for v2.
_per_kernel_wsps = [fm.kernel_to_sparse(k)[0] for k in KERNELS]      # list[N_PID] of {0,1}
_pe_chunks = []
for pe in range(N_PE):
    chunk = list(range(pe * N_MULTS, min((pe + 1) * N_MULTS, len(KERNELS))))
    if chunk:
        _pe_chunks.append(chunk)
UNION_WSPS = [fm.wsp_union([_per_kernel_wsps[k] for k in c]) for c in _pe_chunks]


# ---------------------------------------------------------------------------
# Helpers (shared shape with test_apu.py)
# ---------------------------------------------------------------------------
def _pack_one_wsp(wsp_per_pe):
    val = 0
    for p in range(N_PID):
        if wsp_per_pe[p]:
            val |= 1 << (N_PID - 1 - p)
    return val


def _signed(v, bits):
    return v - (1 << bits) if (v >> (bits - 1)) & 1 else v


def make_padded_activation(rng, sparsity=0.5):
    """raw (H-2) x (H-2) random INT8, zero-padded by 1 on every side -> H x H."""
    h_raw = H - 2
    pad = 1
    m = [[0] * H for _ in range(H)]
    for r in range(h_raw):
        for c in range(h_raw):
            if rng.random() < sparsity:
                continue
            v = rng.randint(1, 127)
            if rng.random() < 0.5:
                v = -v
            m[r + pad][c + pad] = v
    return m


# ---------------------------------------------------------------------------
# DUT drivers
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
    """Write up to N_PE per-PE WSPs into wsp_file (one PE per cycle).
    `wsps` is list[<=N_PE][N_PID]; missing PE entries default to all-zero."""
    dut.wsp_we.value = 1
    for k in range(N_PE):
        dut.wsp_waddr.value = k
        if k < len(wsps):
            dut.wsp_wdata.value = _pack_one_wsp(wsps[k])
        else:
            dut.wsp_wdata.value = 0
        await RisingEdge(dut.clk)
    dut.wsp_we.value    = 0
    dut.wsp_waddr.value = 0
    dut.wsp_wdata.value = 0
    await RisingEdge(dut.clk)


async def fill_sram_csr(dut, matrix):
    """Encode `matrix` as CSR and stream into the activation SRAM + row_ptr flops."""
    values, col_idx, row_ptr = fm.dense_to_csr(matrix)
    if len(values) > N_NZ_MAX:
        raise ValueError(f"matrix has {len(values)} non-zeros but entry SRAM "
                         f"holds {N_NZ_MAX}")

    # row_ptr flop array
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

    # Activation SRAM entries
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
    # Drain the inner combinational pipeline into FIFO-A.
    for _ in range(8):
        await RisingEdge(dut.clk)


async def run_stage2_and_drain(dut, timeout=400000):
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
            binstr = dut.fifob_rd_data.value.binstr
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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
@cocotb.test()
async def test_mobilenet_red_channel(dut):
    """Drive the MobileNet first-conv red channel through the APU and compare
    every per-PE FIFO-B against fm.goSPA_route (V2)."""
    cocotb.start_soon(Clock(dut.clk, CLK_NS, units="ns").start())
    rng = random.Random(0xA1B2)

    dut._log.info(
        f"config: H={H} (raw {H-2}+2 padding) F={F} S={S} E={E} "
        f"N_PE={N_PE} N_MULTS={N_MULTS} N_KERNELS={N_KERNELS} "
        f"N_ROWS={N_ROWS} N_NZ_MAX={N_NZ_MAX}"
    )

    await reset(dut)

    # 1) Load per-PE union WSPs (V2 mapping).
    await load_wsps(dut, UNION_WSPS)
    dut._log.info(f"loaded {len(UNION_WSPS)} per-PE union WSPs from "
                  f"MobileNetV2 first-conv red-channel kernels")

    # 2) Build a synthetic sparse activation, pad to H x H.
    matrix = make_padded_activation(rng, sparsity=0.5)
    n_nz = sum(1 for row in matrix for v in row if v != 0)
    dut._log.info(f"activation: {H}x{H} ({n_nz} non-zeros, ~{n_nz/(H*H):.0%} density)")

    # 3) Stream into the activation SRAM as CSR and scan.
    await fill_sram_csr(dut, matrix)
    await trigger_scan(dut, n_rows=H, base_row=0)

    # 4) Run Stage 2 and drain FIFO-B.
    got = await run_stage2_and_drain(dut)

    # 5) Golden: functional model's V2 route on the same inputs.
    r = fm.goSPA_route(matrix, KERNELS, S,
                       num_pes=N_PE, num_mults=N_MULTS, interpretation="v2")
    expected = r.fifo_b_list
    act_mask = (1 << DATA_W) - 1
    cid_mask = (1 << CID_W) - 1
    expected_masked = [
        [(a & act_mask, c & cid_mask, p) for (a, c, p) in fb]
        for fb in expected
    ]

    # 6) Compare per-PE.
    mismatches = 0
    for k in range(N_PE):
        got_ac = [(a, c, p) for (a, p, c) in got[k]]
        if got_ac != expected_masked[k]:
            mismatches += 1
            # Find first divergence for diagnostics.
            n = min(len(got_ac), len(expected_masked[k]))
            for i in range(n):
                if got_ac[i] != expected_masked[k][i]:
                    dut._log.error(
                        f"PE#{k} first mismatch at index {i}: "
                        f"got={got_ac[i]} expect={expected_masked[k][i]}"
                    )
                    break
            else:
                dut._log.error(
                    f"PE#{k} length mismatch: got {len(got_ac)} entries, "
                    f"expected {len(expected_masked[k])}"
                )

    if mismatches:
        raise AssertionError(f"{mismatches}/{N_PE} PEs mismatched against goSPA_route v2")

    totals = [len(s) for s in got]
    dut._log.info(
        f"PASS -- per-PE FIFO-B counts {totals}, total {sum(totals)} entries, "
        f"all match functional v2 golden"
    )
