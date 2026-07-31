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
3. Drives each PE's union WSP straight onto the APU's per-PE wsp input
   (models the PE array's exported WSP; LSB-first by PID).
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
from cocotb.utils import get_sim_time

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
NUM_MULTS = int(os.environ.get("NUM_MULTS", str(N_MULTS)))   # FIFO-B beat width
CLK_NS  = 10


def _field(binstr, lsb, width):
    """Extract unsigned field [lsb +: width] from a packed-vector binstr."""
    L = len(binstr)
    return int(binstr[L - lsb - width : L - lsb], 2)

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
def _pack_wsps(wsps):
    """Pack the full wsp[N_PE][N_PID] input, LSB-first by PID (matches pe.wsp).
    `wsps` is list[<=N_PE][N_PID]; missing PE entries default to all-zero."""
    val = 0
    for k in range(min(len(wsps), N_PE)):
        for p in range(N_PID):
            if wsps[k][p]:
                val |= 1 << (k * N_PID + p)
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
    dut.wsp.value               = 0
    dut.fifob_rd_ready.value    = 0
    for _ in range(4):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


async def load_wsps(dut, wsps):
    """Drive the per-PE WSP input directly (models the PE array's wsp export).
    `wsps` is list[<=N_PE][N_PID]; missing PE entries default to all-zero."""
    dut.wsp.value = _pack_wsps(wsps)
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
            pid_s = str(dut.fifob_rd_pid.value)
            lv_s  = str(dut.fifob_rd_lane_valid.value)
            act_s = str(dut.fifob_rd_act.value)
            cid_s = str(dut.fifob_rd_cid.value)
            for k in range(N_PE):
                if (vbits >> k) & 1:
                    pid = _field(pid_s, k * PID_W, PID_W)
                    for i in range(NUM_MULTS):
                        lane = k * NUM_MULTS + i
                        if _field(lv_s, lane, 1):
                            axy = _field(act_s, lane * DATA_W, DATA_W)
                            cid = _field(cid_s, lane * CID_W, CID_W)
                            collected[k].append((axy, pid, cid))
        if int(dut.s2_done.value) == 1:
            seen_done = True
        await RisingEdge(dut.clk)
        # Beats take a few cycles to surface through FIFO-B's show-ahead after
        # s2_done; wait for a run of consecutive idle cycles before stopping.
        if seen_done and vbits == 0:
            drain_extra += 1
            if drain_extra >= 8:
                break
        else:
            drain_extra = 0
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


# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------
def _cyc(t_start_ns, t_end_ns):
    return int(round((t_end_ns - t_start_ns) / CLK_NS))


@cocotb.test()
async def test_mobilenet_perf(dut):
    """Run the same MobileNet workload but instrument each phase and print
    a cycle-count summary. Useful for tracking RTL throughput as the design
    evolves and for cross-checking against sw/perf_model.py."""
    cocotb.start_soon(Clock(dut.clk, CLK_NS, units="ns").start())
    rng = random.Random(0xBEEF)

    await reset(dut)

    # --- Phase 1: WSP load -------------------------------------------------
    t0 = get_sim_time(unit="ns")
    await load_wsps(dut, UNION_WSPS)
    t_wsp = get_sim_time(unit="ns")

    # --- Phase 2: SRAM fill (row_ptr flops + entry SRAM, CSR-encoded) ------
    matrix = make_padded_activation(rng, sparsity=0.5)
    n_nz = sum(1 for row in matrix for v in row if v != 0)

    t_fill_a = get_sim_time(unit="ns")
    await fill_sram_csr(dut, matrix)
    t_fill_b = get_sim_time(unit="ns")

    # --- Phase 3: Scan + Stage 1 (CSR -> FIFO-A) ---------------------------
    t_scan_a = get_sim_time(unit="ns")
    await trigger_scan(dut, n_rows=H, base_row=0)
    t_scan_b = get_sim_time(unit="ns")

    # --- Phase 4: Stage 2 drain (FIFO-A -> FIFO-B via routing) -------------
    t_s2_a = get_sim_time(unit="ns")
    got = await run_stage2_and_drain(dut)
    t_s2_b = get_sim_time(unit="ns")

    # ---------------------------- Reporting --------------------------------
    # Workload stats from the functional model golden (free reference data).
    r = fm.goSPA_route(matrix, KERNELS, S,
                       num_pes=N_PE, num_mults=N_MULTS, interpretation="v2")
    n_pairs = r.n_pairs                       # NZ after Stage 1 (CID,PID) expansion
    per_pid = [0] * N_PID
    # Re-derive per-PID FIFO-A occupancy for theoretical Stage 2 bound.
    for fb in r.fifo_b_list:
        # Each fb entry is (a, c, p). Stage 2 drains FIFO-A per PID; FIFO-A
        # holds the (a, c) tuples for each PID, then routing fans out.
        # The per-PID FIFO-A count equals (entries in fb at that PID) /
        # (number of selected PEs for that PID), but here we just want a
        # theoretical lower bound for the drain phase, which is the max
        # over PIDs of (FIFO-A[p] entries).
        pass
    # Easier: re-route through Stage 1 only and use route_to_fifo_a output.
    values, col_idx, row_ptr = fm.dense_to_csr(matrix)
    stream = fm.csr_to_positional(values, col_idx, row_ptr)
    stream = fm.zero_act_filter(stream)
    pairs = []
    for (axy, x, y) in stream:
        a, px, py, cx, cy = fm.axy_to_pcid(axy, x, y, S)
        pairs.extend(fm.pcid_to_cid_pid(a, px, py, cx, cy, F, H, S))
    fifo_a_lens = [len(slot) for slot in fm.route_to_fifo_a(pairs, F)]
    s2_lower_bound = sum(fifo_a_lens)         # routing pops each FIFO-A entry once

    # Dense baseline: scan walks H*H entries unconditionally; Stage 2 drains
    # H*H * G^2 (every cell hits up to G^2 output positions). This is the
    # naive cost without sparsity exploitation.
    G = (F + S - 1) // S
    dense_scan_baseline = H * H
    dense_s2_baseline   = H * H * G * G       # rough; ignores edge masking

    fill_cyc = _cyc(t_fill_a, t_fill_b)
    wsp_cyc  = _cyc(t0,        t_wsp)
    scan_cyc = _cyc(t_scan_a,  t_scan_b)
    s2_cyc   = _cyc(t_s2_a,    t_s2_b)
    total    = _cyc(t0,        t_s2_b)
    total_fb = sum(len(s) for s in got)

    log = dut._log.info
    log("============================================================")
    log("  RTL APU performance -- MobileNetV2 first conv, red channel")
    log("============================================================")
    log(f"  Layer cfg : H={H} F={F} S={S}  -> E={E}  N_PID={N_PID}  G={G}")
    log(f"  Mapping   : {N_PE} PEs x {N_MULTS} kernels/PE = {N_KERNELS} channels (V2)")
    log(f"  Activation: {H}x{H} ({n_nz} non-zeros, {n_nz/(H*H):.1%} density)")
    log(f"  After S1  : {n_pairs} (CID,PID) pairs in FIFO-A "
        f"({n_pairs/n_nz:.2f}x expansion)")
    log(f"  After S2  : {total_fb} entries across {N_PE} FIFO-Bs "
        f"(~{total_fb/N_PE:.0f}/PE)")
    log("")
    log(f"  Phase                                  Cycles    Notes")
    log(f"  -------------------------------------- ------    -------------------------")
    log(f"  WSP load           ({N_PE} PE writes)  {wsp_cyc:>6d}    ~1 cyc/PE write")
    log(f"  SRAM fill (CSR)    ({n_nz} NZ + {N_ROWS+1} ptrs)  "
        f"{fill_cyc:>6d}    ~1 cyc/word")
    log(f"  Scan / Stage 1      (scan_start -> done) {scan_cyc:>6d}    "
        f"{n_nz/scan_cyc:.2f} NZ/cyc, {n_pairs/scan_cyc:.2f} pairs/cyc")
    log(f"  Stage 2 drain       (s2_start  -> done)  {s2_cyc:>6d}    "
        f"{total_fb/s2_cyc:.2f} FIFO-B push/cyc")
    log(f"  --------------------------------------------------------")
    log(f"  TOTAL (host writes -> last PE pop)      {total:>6d}")
    log("")
    log(f"  Latency  (first scan -> last drain)   = {_cyc(t_scan_a, t_s2_b)} cyc")
    log(f"  Compute throughput (sparse / dense baseline):")
    log(f"     Scan stage : {scan_cyc} vs dense {dense_scan_baseline} "
        f"-> {dense_scan_baseline/scan_cyc:.2f}x speedup")
    log(f"     S2   stage : {s2_cyc} vs dense {dense_s2_baseline}   "
        f"-> {dense_s2_baseline/s2_cyc:.2f}x speedup")
    log(f"  Stage 2 efficiency vs FIFO-A drain LB = {s2_lower_bound}/{s2_cyc} "
        f"= {s2_lower_bound/s2_cyc:.2f}x (1.0 = perfect)")
    log("============================================================")
