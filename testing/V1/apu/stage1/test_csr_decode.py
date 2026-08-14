"""
test_csr_decode.py -- cocotb tests for csr_decode.sv
GoSPA Project -- Team 19, ECE 720 (Spring 2026)

Run: make MODULE=test_csr_decode

Golden model: inline CSR iteration -- no external dependency.
CSR format:
    row_ptr[r]   = start index in values[] for row r
    row_ptr[r+1] = end   index in values[] for row r (exclusive)
    values[k], col_idx[k] = k-th non-zero
    x = row index, y = col_idx[k]
"""

import random
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, FallingEdge, Timer


# ---------------------------------------------------------------------------
# Golden model
# ---------------------------------------------------------------------------

def csr_golden(H, row_ptr, values, col_idx):
    """
    Returns list of (value, x, y) in row-major order.
    Mirrors exactly what csr_decode.sv must output.
    """
    out = []
    for r in range(H):
        for k in range(row_ptr[r], row_ptr[r + 1]):
            out.append((values[k], r, col_idx[k]))
    return out


# ---------------------------------------------------------------------------
# Driver helpers
# ---------------------------------------------------------------------------

async def reset(dut):
    dut.rst_n.value         = 0
    dut.row_ptr_valid.value = 0
    dut.row_ptr_data.value  = 0
    dut.entry_valid.value   = 0
    dut.entry_value.value   = 0
    dut.entry_col.value     = 0
    dut.out_ready.value     = 1
    for _ in range(4):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


async def send_row_ptr(dut, ptr):
    """Send one row_ptr value, respecting ready."""
    dut.row_ptr_valid.value = 1
    dut.row_ptr_data.value  = ptr
    while True:
        await RisingEdge(dut.clk)
        if dut.row_ptr_ready.value == 1:
            break
    dut.row_ptr_valid.value = 0


async def send_entry(dut, value, col):
    """Send one (value, col) entry, respecting ready."""
    dut.entry_valid.value = 1
    dut.entry_value.value = value
    dut.entry_col.value   = col
    while True:
        await RisingEdge(dut.clk)
        if dut.entry_ready.value == 1:
            break
    dut.entry_valid.value = 0


async def collect_outputs(dut, n_expected, timeout_cycles=2000):
    """
    Collect n_expected (value, x, y) tuples from the output stream.
    Returns list in order received.
    """
    results = []
    cycles  = 0
    dut.out_ready.value = 1
    while len(results) < n_expected and cycles < timeout_cycles:
        await RisingEdge(dut.clk)
        cycles += 1
        if dut.out_valid.value == 1 and dut.out_ready.value == 1:
            results.append((
                int(dut.out_value.value),
                int(dut.out_x.value),
                int(dut.out_y.value),
            ))
    return results


async def run_csr_test(dut, H, row_ptr, values, col_idx, test_name,
                       out_ready_fn=None):
    """
    Drive CSR data into dut, collect outputs, compare against golden.
    out_ready_fn: optional callable(cycle)->int for backpressure injection.
    """
    golden = csr_golden(H, row_ptr, values, col_idx)
    n      = len(golden)

    # Launch row_ptr and entry drivers concurrently with output collector
    results = []
    cycles  = [0]

    async def drive_row_ptrs():
        for ptr in row_ptr:
            await send_row_ptr(dut, ptr)

    async def drive_entries():
        for val, col in zip(values, col_idx):
            await send_entry(dut, val, col)

    async def collect():
        cyc = 0
        while len(results) < n and cyc < 5000:
            await RisingEdge(dut.clk)
            cyc += 1
            if out_ready_fn is not None:
                dut.out_ready.value = out_ready_fn(cyc)
            if dut.out_valid.value == 1 and int(dut.out_ready.value) == 1:
                results.append((
                    int(dut.out_value.value),
                    int(dut.out_x.value),
                    int(dut.out_y.value),
                ))
        cycles[0] = cyc

    dut.out_ready.value = 1

    rptr_task  = cocotb.start_soon(drive_row_ptrs())
    entry_task = cocotb.start_soon(drive_entries())
    await collect()

    await rptr_task
    await entry_task

    assert len(results) == len(golden), \
        f"[{test_name}] expected {len(golden)} outputs, got {len(results)}"
    for i, (got, exp) in enumerate(zip(results, golden)):
        assert got == exp, \
            f"[{test_name}] output[{i}]: got (val={got[0]:#06x}, x={got[1]}, y={got[2]})" \
            f" expected (val={exp[0]:#06x}, x={exp[1]}, y={exp[2]})"

    dut._log.info(f"[{test_name}] PASS -- {n} entries verified in {cycles[0]} cycles")


# ---------------------------------------------------------------------------
# Test 1: single non-zero in row 0
# ---------------------------------------------------------------------------
@cocotb.test()
async def test_single_entry(dut):
    """One non-zero in the first row, rest empty."""
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    await reset(dut)

    H       = 4   # use small H (DUT is parameterized to H=32, but data fits)
    # Build a 4-row CSR (we send only 5 row_ptrs: rows 0..3)
    row_ptr = [0, 1, 1, 1, 1]  # only row 0 has one entry
    values  = [0xABCD]
    col_idx = [2]

    await run_csr_test(dut, 4, row_ptr, values, col_idx, "single_entry")


# ---------------------------------------------------------------------------
# Test 2: multiple non-zeros in one row
# ---------------------------------------------------------------------------
@cocotb.test()
async def test_multi_in_one_row(dut):
    """Three non-zeros in row 1, rest empty."""
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    await reset(dut)

    row_ptr = [0, 0, 3, 3, 3]   # row 1 has 3 entries
    values  = [0x0001, 0x0002, 0x0003]
    col_idx = [0, 5, 10]

    await run_csr_test(dut, 4, row_ptr, values, col_idx, "multi_in_one_row")


# ---------------------------------------------------------------------------
# Test 3: empty rows (row_ptr[r] == row_ptr[r+1]) skipped silently
# ---------------------------------------------------------------------------
@cocotb.test()
async def test_empty_rows_skipped(dut):
    """Rows with no entries must produce no output and not stall."""
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    await reset(dut)

    row_ptr = [0, 0, 0, 2, 2]   # rows 0,1,3 empty; row 2 has 2 entries
    values  = [0x1111, 0x2222]
    col_idx = [3, 7]

    await run_csr_test(dut, 4, row_ptr, values, col_idx, "empty_rows_skipped")


# ---------------------------------------------------------------------------
# Test 4: all rows empty (sparse matrix with no non-zeros)
# ---------------------------------------------------------------------------
@cocotb.test()
async def test_all_empty(dut):
    """Completely empty matrix -- no output expected."""
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    await reset(dut)

    H       = 4
    row_ptr = [0, 0, 0, 0, 0]
    values  = []
    col_idx = []

    golden = csr_golden(H, row_ptr, values, col_idx)
    assert golden == [], "Golden should be empty"

    # Drive row_ptrs only -- no entries
    for ptr in row_ptr:
        await send_row_ptr(dut, ptr)

    # Wait a few cycles and verify nothing came out
    dut.out_ready.value = 1
    captured = []
    for _ in range(20):
        await RisingEdge(dut.clk)
        if dut.out_valid.value == 1:
            captured.append((int(dut.out_value.value),
                             int(dut.out_x.value),
                             int(dut.out_y.value)))

    assert captured == [], f"Expected no output, got {captured}"
    dut._log.info("[all_empty] PASS")


# ---------------------------------------------------------------------------
# Test 5: entries spread across all rows
# ---------------------------------------------------------------------------
@cocotb.test()
async def test_all_rows_populated(dut):
    """One non-zero per row for 4 rows."""
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    await reset(dut)

    row_ptr = [0, 1, 2, 3, 4]
    values  = [0x0A, 0x0B, 0x0C, 0x0D]
    col_idx = [1, 2, 3, 0]

    await run_csr_test(dut, 4, row_ptr, values, col_idx, "all_rows_populated")


# ---------------------------------------------------------------------------
# Test 6: backpressure mid-stream
# ---------------------------------------------------------------------------
@cocotb.test()
async def test_backpressure(dut):
    """Assert out_ready=0 for some cycles; data must not be lost or corrupted."""
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    await reset(dut)

    row_ptr = [0, 3, 3, 5, 5]
    values  = [0x0001, 0x0002, 0x0003, 0x0004, 0x0005]
    col_idx = [0, 4, 8, 2, 6]

    # Toggle out_ready: stall every 3rd cycle
    def bp(cyc):
        return 0 if (cyc % 3 == 0) else 1

    await run_csr_test(dut, 4, row_ptr, values, col_idx, "backpressure",
                       out_ready_fn=bp)


# ---------------------------------------------------------------------------
# Test 7: randomized 8x8 sparse matrix
# ---------------------------------------------------------------------------
@cocotb.test()
async def test_random_8x8(dut):
    """Random 20% density 8x8 matrix -- verifies output matches golden."""
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    await reset(dut)

    random.seed(42)
    H       = 8
    density = 0.20   # ~20% non-zeros

    # Build dense matrix, then convert to CSR
    vals_flat = []
    cols_flat = []
    rptr      = [0]

    for r in range(H):
        row_nnz = 0
        for c in range(H):
            if random.random() < density:
                v = random.randint(1, 0xFFFF)
                vals_flat.append(v)
                cols_flat.append(c)
                row_nnz += 1
        rptr.append(rptr[-1] + row_nnz)

    await run_csr_test(dut, H, rptr, vals_flat, cols_flat, "random_8x8")


# ---------------------------------------------------------------------------
# Test 8: reset mid-stream recovery
# ---------------------------------------------------------------------------
@cocotb.test()
async def test_reset_mid_stream(dut):
    """Assert rst_n=0 mid-stream, then bring up and run a clean matrix."""
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    await reset(dut)

    # Start driving row_ptrs and an entry, then reset abruptly
    dut.row_ptr_valid.value = 1
    dut.row_ptr_data.value  = 0
    await RisingEdge(dut.clk)
    dut.row_ptr_valid.value = 0

    # Interrupt with reset
    dut.rst_n.value = 0
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)

    # Now run a clean simple matrix
    row_ptr = [0, 1, 1, 1, 1]
    values  = [0x5A5A]
    col_idx = [3]

    await run_csr_test(dut, 4, row_ptr, values, col_idx, "reset_mid_stream")
