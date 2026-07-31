"""
gospa_tb.py -- shared cocotb driver/monitor helpers for the goSPA top-level DUT.

"""

import os
import math
import random

import cocotb
from cocotb.triggers import RisingEdge, ReadOnly


# -- Config (matches Makefile -G/-P overrides) -----------------------------
H        = int(os.environ.get("H", "10"))
F        = int(os.environ.get("F", "3"))
S        = int(os.environ.get("S", "2"))
N_PE     = int(os.environ.get("N_PE", "4"))          # = output channels
N_MULTS  = int(os.environ.get("N_MULTS", "4"))       # activation lanes / beat
N_ROWS   = int(os.environ.get("N_ROWS",   str(H)))
N_NZ_MAX = int(os.environ.get("N_NZ_MAX", "1024"))
FIFO_D   = int(os.environ.get("FIFO_D", "64"))
DATA_W   = 16
ACC_W    = 32
CLK_NS   = 10                                         # 100 MHz sim clock

# Input-tile size (single-bank tiling): tile_h*tile_w <= FIFO_D keeps FIFO-A
# from overflowing for any H. Default = largest square that fits FIFO_D.
TILE_H   = int(os.environ.get("TILE_H", "0")) or max(1, min(H, math.isqrt(FIFO_D)))
TILE_W   = int(os.environ.get("TILE_W", "0")) or max(1, min(H, math.isqrt(FIFO_D)))


def _clog2(n):
    return 0 if n <= 1 else (n - 1).bit_length()


E       = (H - F) // S + 1
N_PID   = F * F
N_CID   = E * E
CID_W   = max(1, _clog2(N_CID))
PID_W   = max(1, _clog2(N_PID))


# -- Data generation -------------------------------------------------------
def load_mobilenet_kernels(in_ch=0):
    """N_PE MobileNetV2 first-conv kernels for input channel `in_ch` (F==3 +
    torch only), else synthetic INT8. Used by the correctness tests."""
    if F == 3:
        try:
            from mobilenet import get_first_conv
            _, conv0 = get_first_conv()
            iw = conv0.weight().int_repr().numpy()       # (32, 3, 3, 3) int8
            n  = min(N_PE, iw.shape[0])
            ks = [[[int(v) for v in row] for row in iw[f, in_ch]]
                  for f in range(n)]
        except Exception as exc:
            cocotb.log.warning(f"falling back to synthetic kernels: {exc}")
            ks = []
    else:
        ks = []
    if not ks:
        rng = random.Random(0xA17 ^ F ^ (in_ch * 911))
        ks = [[[rng.randint(-50, 50) for _ in range(F)] for _ in range(F)]
              for _ in range(N_PE)]
    while len(ks) < N_PE:
        ks.append([[0] * F for _ in range(F)])
    return ks


def rand_kernel(rng, density):
    """One FxF INT8 kernel; each tap non-zero with prob `density`. Guarantees
    at least one non-zero so the PE always has a weight to load."""
    k = [[(rng.randint(1, 50) * (1 if rng.random() < 0.5 else -1)
           if rng.random() < density else 0)
          for _ in range(F)] for _ in range(F)]
    if all(v == 0 for row in k for v in row):
        k[0][0] = 3
    return k


def rand_kernels(rng, n, density):
    return [rand_kernel(rng, density) for _ in range(n)]


def make_activation(rng, density, pad=1):
    """Random INT8 (H-2*pad)x(H-2*pad) core (each cell non-zero with prob
    `density`), zero-padded by `pad` -> HxH. pad=0 for AlexNet conv1 (no
    padding); pad=1 matches the MobileNet-shaped correctness tests."""
    m = [[0] * H for _ in range(H)]
    core = H - 2 * pad
    for r in range(core):
        for c in range(core):
            if rng.random() < density:
                v = rng.randint(1, 80)
                m[r + pad][c + pad] = v if rng.random() < 0.5 else -v
    return m


# -- Small utilities -------------------------------------------------------
def _signed(v, bits):
    return v - (1 << bits) if (v >> (bits - 1)) & 1 else v


def _mask(v, bits):
    return v & ((1 << bits) - 1)


def _popcount(v):
    return bin(v).count("1")


def _try_handle(dut, name):
    """Return an internal-signal handle if the sim exposes it, else None, so
    perf/idle logic degrades gracefully instead of crashing."""
    try:
        h = getattr(dut, name)
        int(h.value)
        return h
    except Exception:
        return None


# -- DUT drivers -----------------------------------------------------------
async def reset(dut):
    dut.rst_n.value            = 0
    dut.fill_entry_we.value    = 0
    dut.fill_entry_addr.value  = 0
    dut.fill_entry_value.value = 0
    dut.fill_entry_col.value   = 0
    dut.fill_rptr_we.value     = 0
    dut.fill_rptr_addr.value   = 0
    dut.fill_rptr_data.value   = 0
    dut.scan_start.value       = 0
    dut.scan_n_rows.value      = 0
    dut.scan_base_row.value    = 0
    dut.s2_start.value         = 0
    dut.pe_wfill_we.value      = 0
    dut.pe_wfill_pe.value      = 0
    dut.pe_wfill_pid.value     = 0
    dut.pe_wfill_val.value     = 0
    dut.pe_wload_done.value    = 0
    dut.drain_start.value      = 0
    dut.out_ready.value        = 0
    for _ in range(4):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


async def load_pe_weights(dut, kernels):
    """Stream each PE's kernel into its bank, one weight/cycle in PID order.
    Slot and WSP are derived inside the PE; WSP reaches the router on arm."""
    import functional as fm
    dut.pe_wfill_we.value = 1
    for pe in range(N_PE):
        _, sparse = fm.kernel_to_sparse(kernels[pe])     # (pid, w) in PID order
        for (pid, val) in sparse:
            dut.pe_wfill_pe.value  = pe
            dut.pe_wfill_pid.value = pid
            dut.pe_wfill_val.value = _mask(val, DATA_W)
            await RisingEdge(dut.clk)
    dut.pe_wfill_we.value = 0
    await RisingEdge(dut.clk)


async def arm_pe_array(dut):
    """Pulse wload_done, then wait for each PE to seed Curr/Next from its bank."""
    dut.pe_wload_done.value = 1
    await RisingEdge(dut.clk)
    dut.pe_wload_done.value = 0
    for _ in range(2 * N_MULTS + 4):
        await RisingEdge(dut.clk)


async def fill_activation_csr(dut, matrix):
    """Encode `matrix` as CSR and stream it into the activation SRAM."""
    import functional as fm
    values, col_idx, row_ptr = fm.dense_to_csr(matrix)
    if len(values) > N_NZ_MAX:
        raise ValueError(f"{len(values)} non-zeros > N_NZ_MAX={N_NZ_MAX}")

    dut.fill_rptr_we.value = 1
    last_ptr = row_ptr[-1]
    for r in range(N_ROWS + 1):
        dut.fill_rptr_addr.value = r
        dut.fill_rptr_data.value = row_ptr[r] if r < len(row_ptr) else last_ptr
        await RisingEdge(dut.clk)
    dut.fill_rptr_we.value = 0

    val_mask = (1 << DATA_W) - 1
    dut.fill_entry_we.value = 1
    for k in range(len(values)):
        dut.fill_entry_addr.value  = k
        dut.fill_entry_value.value = values[k] & val_mask
        dut.fill_entry_col.value   = col_idx[k]
        await RisingEdge(dut.clk)
    dut.fill_entry_we.value = 0
    await RisingEdge(dut.clk)


async def run_scan(dut, n_rows=None, base_row=0, timeout=200000):
    """Pulse scan_start, wait scan_done, settle. Returns scan cycle count."""
    if n_rows is None:
        n_rows = H
    dut.scan_n_rows.value   = n_rows
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
    for _ in range(6):
        await RisingEdge(dut.clk)
    return cycles


async def run_stage2(dut, timeout=400000, perf=None):
    """Pulse s2_start, wait s2_done, then let the PEs drain FIFO-B.

    If `perf` (a dict) is given, accumulate useful MACs, the PE-busy window
    (cycles with >=1 beat accepted) and the total stage2+PE window from the
    internal beat handshake -- perf_model's util math, measured.
    """
    fv = _try_handle(dut, "fifob_valid")
    fr = _try_handle(dut, "fifob_ready")
    lv = _try_handle(dut, "fifob_lane_valid")
    measure = perf is not None and None not in (fv, fr, lv)

    await RisingEdge(dut.clk)
    dut.s2_start.value = 1
    await RisingEdge(dut.clk)
    dut.s2_start.value = 0

    cycles = 0
    macs   = 0
    busy   = 0
    last_active = -1                                      # cycle of last accepted beat
    seen_done = False
    idle      = 0
    lane_mask = (1 << N_MULTS) - 1
    while cycles < timeout:
        await ReadOnly()
        active = (fv is not None) and (int(fv.value) != 0)
        if measure:
            vb  = int(fv.value)
            rb  = int(fr.value)
            lvb = int(lv.value)
            if vb & rb:
                busy += 1
                last_active = cycles
                for pe in range(N_PE):
                    if (vb >> pe) & 1 and (rb >> pe) & 1:
                        macs += _popcount((lvb >> (pe * N_MULTS)) & lane_mask)
        if int(dut.s2_done.value) == 1:
            seen_done = True
        await RisingEdge(dut.clk)
        cycles += 1
        if seen_done:
            if fv is None:
                idle += 1
                if idle >= 32:
                    break
            elif not active:
                idle += 1
                if idle >= 16:
                    break
            else:
                idle = 0

    if measure:
        perf["macs"] = macs
        perf["busy_cycles"] = busy
        # Streaming window: s2_start -> last accepted beat (excludes the trailing
        # idle-confirmation tail). Falls back to full count if nothing surfaced.
        perf["active_cycles"] = (last_active + 1) if last_active >= 0 else cycles
        perf["s2_cycles"] = cycles
    return cycles


def _tile_block(matrix, r0, c0, th, tw):
    """Local (th_actual x H) dense block covering input tile [r0,r0+th) x
    [c0,c0+tw). Global column index is preserved so idgen sees global coords.
    Returns (rows, n_nz, th_actual)."""
    h = len(matrix)
    th_a = min(th, h - r0)
    rows, n_nz = [], 0
    for lr in range(th_a):
        row = [0] * h
        for c in range(c0, min(c0 + tw, h)):
            v = matrix[r0 + lr][c]
            if v:
                row[c] = v
                n_nz += 1
        rows.append(row)
    return rows, n_nz, th_a


async def run_tiled_channel(dut, matrix, tile_h=None, tile_w=None, perf=None):
    """Process one input channel in 2-D input tiles, each a normal
    fill->scan->route pass into the SAME persistent PE banks. FIFO-A never
    holds more than tile_h*tile_w entries, so any H fits a fixed FIFO_D.
    No drain here -- the caller drains once after all channels/passes."""
    th = tile_h or TILE_H
    tw = tile_w or TILE_W
    scan_c = active_c = macs = 0
    for r0 in range(0, len(matrix), th):
        for c0 in range(0, len(matrix), tw):
            block, n_nz, th_a = _tile_block(matrix, r0, c0, th, tw)
            if n_nz == 0:                       # empty tile: nothing to route
                continue
            await fill_activation_csr(dut, block)
            scan_c += await run_scan(dut, n_rows=th_a, base_row=r0)
            p = {}
            await run_stage2(dut, perf=p)
            active_c += p.get("active_cycles", 0)
            macs     += p.get("macs", 0)
    if perf is not None:
        perf["scan_cycles"]   = scan_c
        perf["active_cycles"] = active_c
        perf["macs"]          = macs


async def drain_all(dut, timeout=50000, perf=None):
    """Pulse drain_start, collect per-PE {cid: signed_acc}. If `perf` is given,
    record perf['drain_cycles'] = cycles from drain_start to drain_done."""
    dut.drain_start.value = 1
    await RisingEdge(dut.clk)
    dut.drain_start.value = 0
    dut.out_ready.value = (1 << N_PE) - 1

    got = [dict() for _ in range(N_PE)]
    cid_mask = (1 << CID_W) - 1
    acc_mask = (1 << ACC_W) - 1

    guard = 0
    seen_done = False
    while guard < timeout:
        await ReadOnly()
        ov = int(dut.out_valid.value)
        oc = int(dut.out_cid.value)
        oa = int(dut.out_acc.value)
        for pe in range(N_PE):
            if (ov >> pe) & 1:
                cid = (oc >> (pe * CID_W)) & cid_mask
                acc = _signed((oa >> (pe * ACC_W)) & acc_mask, ACC_W)
                got[pe][cid] = acc
        if int(dut.drain_done.value) == 1:
            seen_done = True
        await RisingEdge(dut.clk)
        guard += 1
        if seen_done:
            break
    dut.out_ready.value = 0
    await RisingEdge(dut.clk)
    if perf is not None:
        perf["drain_cycles"] = guard
    return got


def compare(dut, got, golden, name):
    """Assert every PE's ExE output map matches `golden[pe]`."""
    mismatches = 0
    for pe in range(N_PE):
        out_map = [[got[pe].get(r * E + c, 0) for c in range(E)] for r in range(E)]
        if out_map != golden[pe]:
            mismatches += 1
            dut._log.error(
                f"[{name}] PE#{pe} (channel {pe}) mismatch:\n"
                f"  golden[:2] = {golden[pe][:2]}\n"
                f"  got   [:2] = {out_map[:2]}")
    if mismatches:
        raise AssertionError(f"[{name}] {mismatches}/{N_PE} output channels mismatched")
