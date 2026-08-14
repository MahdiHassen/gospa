"""
test_gospa.py -- End-to-end goSPA accelerator cosim.

Drives a synthetic sparse activation through the full goSPA RTL
(APU + N_PE V2 PEs) using MobileNetV2's first conv (red-channel) as the
weight set, then checks every output channel of every PE against the
functional model's conv2d_reference.

Run:
    make MODULE=test_gospa SIM=verilator     # default H=10 F=3 S=2 N_PE=4 N_MULTS=4
    make mobilenet                            # same, but matches the APU's mobilenet shape
    WAVES=1 make mobilenet                    # + VCD dump for waveform viewing
"""

import os
import sys
import random

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ReadOnly

_TEST_DIR = os.path.dirname(__file__)
_SW_DIR   = os.path.abspath(os.path.join(_TEST_DIR, "..", "sw_v1"))
_REF_DIR  = os.path.abspath(os.path.join(_TEST_DIR, "..", "ref"))
sys.path.insert(0, _SW_DIR)
sys.path.insert(0, _REF_DIR)
import functional as fm                                  # noqa: E402
fm._VERBOSE = False


# -- Config (matches Makefile -G/-P overrides) -----------------------------
H        = int(os.environ.get("H", "10"))
F        = int(os.environ.get("F", "3"))
S        = int(os.environ.get("S", "2"))
N_PE     = int(os.environ.get("N_PE", "4"))
N_MULTS  = int(os.environ.get("N_MULTS", "4"))
N_ROWS   = int(os.environ.get("N_ROWS",   str(H)))
N_NZ_MAX = int(os.environ.get("N_NZ_MAX", "1024"))
DATA_W   = 16
ACC_W    = 32


def _clog2(n):
    return 0 if n <= 1 else (n - 1).bit_length()


E         = (H - F) // S + 1
N_PID     = F * F
N_CID     = E * E
N_CHAN    = N_PE * N_MULTS                # total output channels
CID_W     = max(1, _clog2(N_CID))
PID_W     = max(1, _clog2(N_PID))
IDX_W     = max(1, _clog2(H))
PESEL_W   = max(1, _clog2(N_PE))
LANE_W    = max(1, _clog2(N_MULTS))
WPTR_W    = max(1, _clog2(N_PID + 1))
ENT_AW    = max(1, _clog2(N_NZ_MAX))
RPTR_AW   = max(1, _clog2(N_ROWS + 1))
PTR_W     = max(1, _clog2(N_NZ_MAX + 1))
FIFOB_W   = DATA_W + PID_W + CID_W
CLK_NS    = 10


# -- Load MobileNetV2 weights ONCE ------------------------------------------
def _load_red_kernels():
    """Pull `n` red-channel 3x3 kernels from MobileNetV2's first conv. If F
    is not 3 or torch isn't available, fall back to synthetic INT8 kernels."""
    if F == 3:
        try:
            from mobilenet import get_first_conv
            _, conv0 = get_first_conv()
            iw = conv0.weight().int_repr().numpy()      # (32, 3, 3, 3) int8
            return [[[int(v) for v in row] for row in iw[f, 0]]
                    for f in range(iw.shape[0])]
        except Exception as exc:
            cocotb.log.warning(f"falling back to synthetic kernels: {exc}")
    rng = random.Random(0xA17 ^ F)
    return [[[rng.randint(-50, 50) for _ in range(F)] for _ in range(F)]
            for _ in range(N_CHAN)]


RED_KERNELS = _load_red_kernels()
KERNELS = RED_KERNELS[:N_CHAN]

# Per-PE chunks (V2 mapping: lanes 0..N_MULTS-1 of PE pe = kernels pe*N_MULTS..)
PE_CHUNKS = [list(range(pe * N_MULTS, min((pe + 1) * N_MULTS, len(KERNELS))))
             for pe in range(N_PE)]
PE_CHUNKS = [c for c in PE_CHUNKS if c]

# Per-kernel WSP + sparse weight list (cached for both fill paths).
PER_KERNEL_WSP = [fm.kernel_to_sparse(k)[0] for k in KERNELS]
PER_KERNEL_SW  = [fm.kernel_to_sparse(k)[1] for k in KERNELS]

# Per-PE union WSP (used by the APU's routing) and per-(PE,lane) WSP / weight
# list (used by each PE's internal storage).
PER_PE_UNION_WSP = [fm.wsp_union([PER_KERNEL_WSP[k] for k in chunk])
                    for chunk in PE_CHUNKS]


# -- Helpers ---------------------------------------------------------------
def _signed(v, bits):
    return v - (1 << bits) if (v >> (bits - 1)) & 1 else v


def _mask(v, bits):
    return v & ((1 << bits) - 1)


def _pack_one_wsp(wsp_per_pid):
    """Pack one WSP as N_PID bits, MSB-first by PID (matches apu_stage2.routing)."""
    val = 0
    for p in range(N_PID):
        if wsp_per_pid[p]:
            val |= 1 << (N_PID - 1 - p)
    return val


def _pack_pe_lsb_wsp(wsp_per_pid):
    """Pack one WSP as N_PID bits LSB-first (matches pe.sv's wsp_q[k][pid])."""
    val = 0
    for p in range(N_PID):
        if wsp_per_pid[p]:
            val |= 1 << p
    return val


def _pack_count_grid(pe_chunks):
    """Pack `wload_count[N_PE][N_MULTS][WPTR_W]` packed bus, lane k of PE p
    in bits [(p*N_MULTS + k)*WPTR_W +: WPTR_W]."""
    val = 0
    for pe in range(N_PE):
        chunk = pe_chunks[pe] if pe < len(pe_chunks) else []
        for lane in range(N_MULTS):
            c = len(PER_KERNEL_SW[chunk[lane]]) if lane < len(chunk) else 0
            val |= (c & ((1 << WPTR_W) - 1)) << ((pe * N_MULTS + lane) * WPTR_W)
    return val


def _make_padded_activation(rng, sparsity=0.5):
    """Raw (H-2)x(H-2) random INT8, zero-padded by 1 -> HxH."""
    h_raw = H - 2
    pad   = 1
    m = [[0] * H for _ in range(H)]
    for r in range(h_raw):
        for c in range(h_raw):
            if rng.random() < sparsity:
                continue
            v = rng.randint(1, 80)
            if rng.random() < 0.5:
                v = -v
            m[r + pad][c + pad] = v
    return m


# -- DUT drivers -----------------------------------------------------------
async def reset(dut):
    dut.rst_n.value             = 0
    dut.fill_entry_we.value     = 0
    dut.fill_entry_addr.value   = 0
    dut.fill_entry_value.value  = 0
    dut.fill_entry_col.value    = 0
    dut.fill_rptr_we.value      = 0
    dut.fill_rptr_addr.value    = 0
    dut.fill_rptr_data.value    = 0
    dut.apu_wsp_we.value        = 0
    dut.apu_wsp_waddr.value     = 0
    dut.apu_wsp_wdata.value     = 0
    dut.scan_start.value        = 0
    dut.scan_n_rows.value       = 0
    dut.scan_base_row.value     = 0
    dut.s2_start.value          = 0
    dut.pe_wfill_we.value       = 0
    dut.pe_wfill_pe.value       = 0
    dut.pe_wfill_lane.value     = 0
    dut.pe_wfill_slot.value     = 0
    dut.pe_wfill_pid.value      = 0
    dut.pe_wfill_val.value      = 0
    dut.pe_wsp_we.value         = 0
    dut.pe_wsp_pe.value         = 0
    dut.pe_wsp_lane.value       = 0
    dut.pe_wsp_data.value       = 0
    dut.pe_wload_count.value    = 0
    dut.pe_wload_done.value     = 0
    dut.drain_start.value       = 0
    dut.out_ready.value         = 0
    for _ in range(4):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


async def load_apu_wsps(dut):
    """Load per-PE UNION WSPs into the APU's wsp_file (one PE / cycle)."""
    dut.apu_wsp_we.value = 1
    for pe in range(N_PE):
        dut.apu_wsp_waddr.value = pe
        if pe < len(PER_PE_UNION_WSP):
            dut.apu_wsp_wdata.value = _pack_one_wsp(PER_PE_UNION_WSP[pe])
        else:
            dut.apu_wsp_wdata.value = 0
        await RisingEdge(dut.clk)
    dut.apu_wsp_we.value = 0
    dut.apu_wsp_waddr.value = 0
    dut.apu_wsp_wdata.value = 0
    await RisingEdge(dut.clk)


async def load_pe_weights(dut):
    """Write per-(PE, lane) sparse weights into the PE SRAMs."""
    dut.pe_wfill_we.value = 1
    for pe in range(N_PE):
        chunk = PE_CHUNKS[pe] if pe < len(PE_CHUNKS) else []
        for lane in range(min(N_MULTS, len(chunk))):
            k_idx = chunk[lane]
            for slot, (pid, val) in enumerate(PER_KERNEL_SW[k_idx]):
                dut.pe_wfill_pe.value   = pe
                dut.pe_wfill_lane.value = lane
                dut.pe_wfill_slot.value = slot
                dut.pe_wfill_pid.value  = pid
                dut.pe_wfill_val.value  = _mask(val, DATA_W)
                await RisingEdge(dut.clk)
    dut.pe_wfill_we.value = 0
    await RisingEdge(dut.clk)


async def load_pe_wsps(dut):
    """Write per-(PE, lane) WSPs into each PE's WSP register file."""
    dut.pe_wsp_we.value = 1
    for pe in range(N_PE):
        chunk = PE_CHUNKS[pe] if pe < len(PE_CHUNKS) else []
        for lane in range(N_MULTS):
            dut.pe_wsp_pe.value   = pe
            dut.pe_wsp_lane.value = lane
            if lane < len(chunk):
                dut.pe_wsp_data.value = _pack_pe_lsb_wsp(PER_KERNEL_WSP[chunk[lane]])
            else:
                dut.pe_wsp_data.value = 0
            await RisingEdge(dut.clk)
    dut.pe_wsp_we.value = 0
    await RisingEdge(dut.clk)


async def arm_pe_array(dut):
    dut.pe_wload_count.value = _pack_count_grid(PE_CHUNKS)
    dut.pe_wload_done.value  = 1
    await RisingEdge(dut.clk)
    dut.pe_wload_done.value = 0
    # Warm-up: each PE needs 2*N_MULTS+1 cycles to seed Curr/Next from SRAM.
    for _ in range(2 * N_MULTS + 4):
        await RisingEdge(dut.clk)


async def fill_activation_csr(dut, matrix):
    """Encode `matrix` as CSR and stream into the APU's activation SRAM."""
    values, col_idx, row_ptr = fm.dense_to_csr(matrix)
    if len(values) > N_NZ_MAX:
        raise ValueError(f"matrix has {len(values)} non-zeros > N_NZ_MAX={N_NZ_MAX}")

    # row_ptr flop array
    dut.fill_rptr_we.value = 1
    last_ptr = row_ptr[-1]
    for r in range(N_ROWS + 1):
        ptr = row_ptr[r] if r < len(row_ptr) else last_ptr
        dut.fill_rptr_addr.value = r
        dut.fill_rptr_data.value = ptr
        await RisingEdge(dut.clk)
    dut.fill_rptr_we.value = 0

    # entry SRAM
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
    for _ in range(8):
        await RisingEdge(dut.clk)


async def run_stage2(dut, timeout=400000):
    await RisingEdge(dut.clk)
    dut.s2_start.value = 1
    await RisingEdge(dut.clk)
    dut.s2_start.value = 0
    cycles = 0
    while cycles < timeout:
        await ReadOnly()
        if int(dut.s2_done.value) == 1:
            break
        await RisingEdge(dut.clk)
        cycles += 1
    await RisingEdge(dut.clk)
    # Wait until every PE has actually consumed its FIFO-B entries. Stage 2
    # finishing only means routing is done pushing; PEs may still be MAC'ing.
    # Without this, back-to-back passes mix unconsumed entries with new ones.
    idle = 0
    while idle < 16 and cycles < timeout:
        await ReadOnly()
        if int(dut.fifob_valid.value) == 0:
            idle += 1
        else:
            idle = 0
        await RisingEdge(dut.clk)
        cycles += 1


async def drain_all(dut, timeout=50000):
    """Pulse drain_start and collect per-(PE, lane) accumulator beats.

    Returns: list[N_PE][N_MULTS] of {cid: signed_acc}."""
    dut.drain_start.value = 1
    await RisingEdge(dut.clk)
    dut.drain_start.value = 0
    dut.out_ready.value = (1 << (N_PE * N_MULTS)) - 1

    got = [[dict() for _ in range(N_MULTS)] for _ in range(N_PE)]
    cid_mask = (1 << CID_W) - 1
    acc_mask = (1 << ACC_W) - 1

    guard = 0
    while guard < timeout and any(
            len(got[pe][lane]) < N_CID
            for pe in range(N_PE) for lane in range(N_MULTS)):
        await ReadOnly()
        ov = int(dut.out_valid.value)
        oc = int(dut.out_cid.value)
        oa = int(dut.out_acc.value)
        for pe in range(N_PE):
            for lane in range(N_MULTS):
                bit_idx = pe * N_MULTS + lane
                if (ov >> bit_idx) & 1:
                    cid = (oc >> (bit_idx * CID_W)) & cid_mask
                    acc = _signed((oa >> (bit_idx * ACC_W)) & acc_mask, ACC_W)
                    got[pe][lane][cid] = acc
        await RisingEdge(dut.clk)
        guard += 1
    dut.out_ready.value = 0
    return got


# ===========================================================================
# Tests
# ===========================================================================
@cocotb.test()
async def test_mobilenet_end_to_end(dut):
    """One input channel of MobileNetV2's first conv (red channel), V2
    mapping. Every output channel is checked against conv2d_reference."""
    cocotb.start_soon(Clock(dut.clk, CLK_NS, units="ns").start())
    rng = random.Random(0xC0FFEE)

    dut._log.info(
        f"gospa cfg: H={H} F={F} S={S} -> E={E}  "
        f"N_PE={N_PE} x N_MULTS={N_MULTS} = {N_CHAN} output channels  "
        f"({len(KERNELS)} MobileNet kernels loaded)"
    )

    matrix = _make_padded_activation(rng, sparsity=0.5)
    n_nz   = sum(1 for row in matrix for v in row if v != 0)
    dut._log.info(f"activation: {H}x{H}, {n_nz} non-zeros (~{n_nz/(H*H):.0%} density)")

    # 1) Functional-model golden per-output-channel (one E x E map each).
    golden = [fm.conv2d_reference(matrix, ker, S) for ker in KERNELS]

    # 2) Drive the RTL end-to-end.
    await reset(dut)
    await load_apu_wsps(dut)
    await load_pe_weights(dut)
    await load_pe_wsps(dut)
    await arm_pe_array(dut)
    await fill_activation_csr(dut, matrix)
    await run_scan(dut)
    await run_stage2(dut)
    got = await drain_all(dut)

    # 3) Compare per-channel.
    mismatches = 0
    for pe in range(N_PE):
        chunk = PE_CHUNKS[pe] if pe < len(PE_CHUNKS) else []
        for lane in range(N_MULTS):
            if lane >= len(chunk):
                continue
            k_idx = chunk[lane]
            out_lane = [[got[pe][lane].get(r * E + c, 0) for c in range(E)]
                        for r in range(E)]
            if out_lane != golden[k_idx]:
                mismatches += 1
                dut._log.error(
                    f"PE#{pe} lane#{lane} (channel #{k_idx}) mismatch:\n"
                    f"  golden[:3] = {golden[k_idx][:3]}\n"
                    f"  got   [:3] = {out_lane[:3]}"
                )
    if mismatches:
        raise AssertionError(
            f"{mismatches} / {N_CHAN} output channels mismatched")

    dut._log.info(
        f"PASS -- all {N_CHAN} output channels match conv2d_reference  "
        f"(MobileNetV2 first conv red channel)"
    )


# ============================================================================
# Partial-sum accumulation across multiple inputs + a mid-stream weight reload
# ============================================================================
def _load_kernels_for_channel(in_ch):
    """N_CHAN MobileNet first-conv kernels sliced at input channel `in_ch`."""
    if F == 3:
        try:
            from mobilenet import get_first_conv
            _, conv0 = get_first_conv()
            iw = conv0.weight().int_repr().numpy()      # (32, 3, 3, 3) int8
            return [[[int(v) for v in row] for row in iw[f, in_ch]]
                    for f in range(min(N_CHAN, iw.shape[0]))]
        except Exception as exc:
            cocotb.log.warning(f"falling back to synthetic kernels: {exc}")
    rng = random.Random(0x533 ^ F ^ (in_ch * 911))
    return [[[rng.randint(-50, 50) for _ in range(F)] for _ in range(F)]
            for _ in range(N_CHAN)]


def _compute_per_channel_state(kernel_set):
    """Pre-compute everything load_*_helpers need for a given kernel set."""
    per_kernel_wsp = [fm.kernel_to_sparse(k)[0] for k in kernel_set]
    per_kernel_sw  = [fm.kernel_to_sparse(k)[1] for k in kernel_set]
    pe_chunks = [list(range(pe * N_MULTS,
                            min((pe + 1) * N_MULTS, len(kernel_set))))
                 for pe in range(N_PE)]
    pe_chunks = [c for c in pe_chunks if c]
    union_wsp = [fm.wsp_union([per_kernel_wsp[k] for k in chunk])
                 for chunk in pe_chunks]
    return per_kernel_wsp, per_kernel_sw, pe_chunks, union_wsp


async def _load_apu_wsps_for(dut, union_wsp):
    dut.apu_wsp_we.value = 1
    for pe in range(N_PE):
        dut.apu_wsp_waddr.value = pe
        dut.apu_wsp_wdata.value = (_pack_one_wsp(union_wsp[pe])
                                    if pe < len(union_wsp) else 0)
        await RisingEdge(dut.clk)
    dut.apu_wsp_we.value = 0
    await RisingEdge(dut.clk)


async def _load_pe_weights_for(dut, per_kernel_sw, pe_chunks):
    dut.pe_wfill_we.value = 1
    for pe in range(N_PE):
        chunk = pe_chunks[pe] if pe < len(pe_chunks) else []
        for lane in range(min(N_MULTS, len(chunk))):
            for slot, (pid, val) in enumerate(per_kernel_sw[chunk[lane]]):
                dut.pe_wfill_pe.value   = pe
                dut.pe_wfill_lane.value = lane
                dut.pe_wfill_slot.value = slot
                dut.pe_wfill_pid.value  = pid
                dut.pe_wfill_val.value  = _mask(val, DATA_W)
                await RisingEdge(dut.clk)
    dut.pe_wfill_we.value = 0
    await RisingEdge(dut.clk)


async def _load_pe_wsps_for(dut, per_kernel_wsp, pe_chunks):
    dut.pe_wsp_we.value = 1
    for pe in range(N_PE):
        chunk = pe_chunks[pe] if pe < len(pe_chunks) else []
        for lane in range(N_MULTS):
            dut.pe_wsp_pe.value   = pe
            dut.pe_wsp_lane.value = lane
            dut.pe_wsp_data.value = (_pack_pe_lsb_wsp(per_kernel_wsp[chunk[lane]])
                                      if lane < len(chunk) else 0)
            await RisingEdge(dut.clk)
    dut.pe_wsp_we.value = 0
    await RisingEdge(dut.clk)


async def _arm_pe_array_for(dut, per_kernel_sw, pe_chunks):
    val = 0
    for pe in range(N_PE):
        chunk = pe_chunks[pe] if pe < len(pe_chunks) else []
        for lane in range(N_MULTS):
            c = len(per_kernel_sw[chunk[lane]]) if lane < len(chunk) else 0
            val |= (c & ((1 << WPTR_W) - 1)) << ((pe * N_MULTS + lane) * WPTR_W)
    dut.pe_wload_count.value = val
    dut.pe_wload_done.value  = 1
    await RisingEdge(dut.clk)
    dut.pe_wload_done.value = 0
    for _ in range(2 * N_MULTS + 4):
        await RisingEdge(dut.clk)


async def _run_one_input_channel(dut, matrix, per_kernel_wsp, per_kernel_sw,
                                 pe_chunks, union_wsp, name):
    """Run one input channel: re-arm the PE so its Curr/Next slide window
    resets to slot 0/1, load APU's union WSP, stream activation through
    Stage 1 + 2 (PE accumulators absorb it). No drain.

    The re-arm is essential: each pass slides every lane's Curr/Next cursor
    forward across the kernel weights, so without a re-arm the window
    starts the next pass already at the end of the weight list and most
    lanes retire on the first activation. pe_acc partial sums persist
    across re-arms (clear is tied to 0), so accumulators stay intact."""
    await _load_apu_wsps_for(dut, union_wsp)
    await _arm_pe_array_for(dut, per_kernel_sw, pe_chunks)
    await fill_activation_csr(dut, matrix)
    await run_scan(dut)
    await run_stage2(dut)
    dut._log.info(f"[{name}] absorbed channel into PE accumulators")


@cocotb.test()
async def test_back_to_back_partial_sums(dut):
    """Two inputs with the RED kernels followed by two inputs with the GREEN
    kernels, all accumulated into the same per-(PE,lane) pe_acc banks. After
    the four channels, drain and compare against the sum of four
    conv2d_reference results per output channel.

    This exercises:
      - Activation SRAM refill between input channels.
      - APU per-PE union WSP rewrite between channels (red vs green kernels
        have different sparsity patterns).
      - PE weight SRAM rewrite + re-arm mid-stream (red -> green).
      - pe_acc persistence across all of the above (clear is tied 0)."""
    cocotb.start_soon(Clock(dut.clk, CLK_NS, units="ns").start())
    rng = random.Random(0xACC0)

    red_kernels   = _load_kernels_for_channel(0)[:N_CHAN]
    green_kernels = _load_kernels_for_channel(1)[:N_CHAN]
    if any(len(k) != F for k in red_kernels + green_kernels):
        raise RuntimeError("Unexpected kernel shape")

    # Pre-compute helper state for both kernel sets.
    red_wsp_per_k,   red_sw_per_k,   red_chunks,   red_union   = \
        _compute_per_channel_state(red_kernels)
    green_wsp_per_k, green_sw_per_k, green_chunks, green_union = \
        _compute_per_channel_state(green_kernels)

    # Four distinct activations -- two red, two green.
    act_red_1   = _make_padded_activation(rng, sparsity=0.5)
    act_red_2   = _make_padded_activation(rng, sparsity=0.5)
    act_green_1 = _make_padded_activation(rng, sparsity=0.5)
    act_green_2 = _make_padded_activation(rng, sparsity=0.5)

    # Functional model golden: per-channel output is the SUM of all 4 conv
    # contributions (red weights with red activations + green weights with
    # green activations).
    def _add(a, b):
        return [[a[r][c] + b[r][c] for c in range(E)] for r in range(E)]

    golden = []
    for k_idx in range(N_CHAN):
        partial = [[0] * E for _ in range(E)]
        partial = _add(partial, fm.conv2d_reference(act_red_1,   red_kernels[k_idx],   S))
        partial = _add(partial, fm.conv2d_reference(act_red_2,   red_kernels[k_idx],   S))
        partial = _add(partial, fm.conv2d_reference(act_green_1, green_kernels[k_idx], S))
        partial = _add(partial, fm.conv2d_reference(act_green_2, green_kernels[k_idx], S))
        golden.append(partial)

    dut._log.info(
        f"back-to-back partial sums: H={H} F={F} S={S} -> E={E}, "
        f"N_PE={N_PE} x N_MULTS={N_MULTS} = {N_CHAN} output channels  "
        f"(2 red inputs + reload weights + 2 green inputs)"
    )

    # ---- RTL drive ---------------------------------------------------------
    await reset(dut)

    # First arm: RED kernel weights into the PE array.
    await _load_pe_weights_for(dut, red_sw_per_k, red_chunks)
    await _load_pe_wsps_for(dut, red_wsp_per_k, red_chunks)
    await _arm_pe_array_for(dut, red_sw_per_k, red_chunks)

    # Two red input channels, back-to-back. Same kernels, different activations.
    await _run_one_input_channel(dut, act_red_1, red_wsp_per_k, red_sw_per_k,
                                 red_chunks, red_union, "red_1")
    await _run_one_input_channel(dut, act_red_2, red_wsp_per_k, red_sw_per_k,
                                 red_chunks, red_union, "red_2")

    # ---- Mid-stream weight swap: RED -> GREEN ------------------------------
    # Overwrite the PE weight SRAMs and per-lane WSPs, then pulse wload_done
    # to re-warm Curr/Next. pe_acc banks keep their red partials.
    await _load_pe_weights_for(dut, green_sw_per_k, green_chunks)
    await _load_pe_wsps_for(dut, green_wsp_per_k, green_chunks)
    await _arm_pe_array_for(dut, green_sw_per_k, green_chunks)

    # Two green input channels.
    await _run_one_input_channel(dut, act_green_1, green_wsp_per_k, green_sw_per_k,
                                 green_chunks, green_union, "green_1")
    await _run_one_input_channel(dut, act_green_2, green_wsp_per_k, green_sw_per_k,
                                 green_chunks, green_union, "green_2")

    # ---- Drain + compare ---------------------------------------------------
    got = await drain_all(dut)

    mismatches = 0
    for pe in range(N_PE):
        chunk = red_chunks[pe] if pe < len(red_chunks) else []
        for lane in range(N_MULTS):
            if lane >= len(chunk):
                continue
            k_idx = chunk[lane]
            out_lane = [[got[pe][lane].get(r * E + c, 0) for c in range(E)]
                        for r in range(E)]
            if out_lane != golden[k_idx]:
                mismatches += 1
                dut._log.error(
                    f"PE#{pe} lane#{lane} (channel #{k_idx}) mismatch:\n"
                    f"  golden[:3] = {golden[k_idx][:3]}\n"
                    f"  got   [:3] = {out_lane[:3]}"
                )
    if mismatches:
        raise AssertionError(
            f"{mismatches} / {N_CHAN} output channels mismatched after "
            f"4-pass partial-sum accumulation"
        )

    dut._log.info(
        f"PASS -- all {N_CHAN} channels' partial-sum accumulators match "
        f"sum(conv2d_reference) over (act_red_1 + act_red_2) with red weights "
        f"+ (act_green_1 + act_green_2) with green weights"
    )


@cocotb.test()
async def test_two_pass_same_kernel_accum(dut):
    """Bisect probe: just 2 red activations with red kernels, accumulate,
    drain, compare. If this passes but the 4-pass test fails the bug is in
    the mid-stream weight reload, not in the back-to-back pass path."""
    cocotb.start_soon(Clock(dut.clk, CLK_NS, units="ns").start())
    rng = random.Random(0xACC1)

    kernels = _load_kernels_for_channel(0)[:N_CHAN]
    wsp_per_k, sw_per_k, chunks, union = _compute_per_channel_state(kernels)

    act_1 = _make_padded_activation(rng, sparsity=0.5)
    act_2 = _make_padded_activation(rng, sparsity=0.5)

    def _add(a, b):
        return [[a[r][c] + b[r][c] for c in range(E)] for r in range(E)]

    golden = []
    for k_idx in range(N_CHAN):
        partial = fm.conv2d_reference(act_1, kernels[k_idx], S)
        partial = _add(partial, fm.conv2d_reference(act_2, kernels[k_idx], S))
        golden.append(partial)

    await reset(dut)
    await _load_pe_weights_for(dut, sw_per_k, chunks)
    await _load_pe_wsps_for(dut, wsp_per_k, chunks)
    await _arm_pe_array_for(dut, sw_per_k, chunks)
    await _run_one_input_channel(dut, act_1, wsp_per_k, sw_per_k, chunks, union, "pass_1")
    await _run_one_input_channel(dut, act_2, wsp_per_k, sw_per_k, chunks, union, "pass_2")
    got = await drain_all(dut)

    mismatches = 0
    for pe in range(N_PE):
        chunk = chunks[pe] if pe < len(chunks) else []
        for lane in range(N_MULTS):
            if lane >= len(chunk):
                continue
            k_idx = chunk[lane]
            out_lane = [[got[pe][lane].get(r * E + c, 0) for c in range(E)]
                        for r in range(E)]
            if out_lane != golden[k_idx]:
                mismatches += 1
                if mismatches <= 2:
                    dut._log.error(
                        f"PE#{pe} lane#{lane} (channel #{k_idx}) mismatch:\n"
                        f"  golden[:3] = {golden[k_idx][:3]}\n"
                        f"  got   [:3] = {out_lane[:3]}"
                    )
    if mismatches:
        raise AssertionError(
            f"{mismatches} / {N_CHAN} channels mismatched after 2-pass accum (same kernel)"
        )
    dut._log.info(
        f"PASS -- 2-pass same-kernel accumulation OK for all {N_CHAN} channels"
    )
