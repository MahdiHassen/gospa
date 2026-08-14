"""
test_zero_act.py -- cocotb tests for zero_act.sv
GoSPA Project -- Team 19, ECE 720 (Spring 2026)

Run: make MODULE=test_zero_act
"""

import cocotb
from cocotb.triggers import RisingEdge, Timer
from cocotb.clock import Clock


async def init(dut):
    """Drive all inputs to safe defaults (no clock needed -- purely combinational)."""
    dut.in_valid.value  = 0
    dut.in_value.value  = 0
    dut.in_x.value      = 0
    dut.in_y.value      = 0
    dut.out_ready.value = 1
    await Timer(1, units="ns")


# ---------------------------------------------------------------------------
# Test 1: zero value is filtered -- out_valid must be 0
# ---------------------------------------------------------------------------
@cocotb.test()
async def test_zero_filtered(dut):
    """Zero-valued activation must not appear downstream."""
    await init(dut)

    dut.in_valid.value  = 1
    dut.in_value.value  = 0          # zero!
    dut.in_x.value      = 5
    dut.in_y.value      = 3
    dut.out_ready.value = 1
    await Timer(1, units="ns")

    assert dut.out_valid.value == 0, \
        f"Expected out_valid=0 for zero value, got {dut.out_valid.value}"


# ---------------------------------------------------------------------------
# Test 2: non-zero value passes through unchanged
# ---------------------------------------------------------------------------
@cocotb.test()
async def test_nonzero_passthrough(dut):
    """Non-zero activation passes through with all fields intact."""
    await init(dut)

    dut.in_valid.value  = 1
    dut.in_value.value  = 0xABCD
    dut.in_x.value      = 7
    dut.in_y.value      = 15
    dut.out_ready.value = 1
    await Timer(1, units="ns")

    assert dut.out_valid.value  == 1,      "out_valid should be 1 for non-zero"
    assert dut.out_value.value  == 0xABCD, f"out_value mismatch: {dut.out_value.value}"
    assert dut.out_x.value      == 7,      f"out_x mismatch: {dut.out_x.value}"
    assert dut.out_y.value      == 15,     f"out_y mismatch: {dut.out_y.value}"


# ---------------------------------------------------------------------------
# Test 3: in_valid=0 means no output regardless of value
# ---------------------------------------------------------------------------
@cocotb.test()
async def test_invalid_suppressed(dut):
    """When in_valid=0 there should be no output even if value is non-zero."""
    await init(dut)

    dut.in_valid.value  = 0
    dut.in_value.value  = 0x1234
    dut.out_ready.value = 1
    await Timer(1, units="ns")

    assert dut.out_valid.value == 0, \
        f"out_valid should be 0 when in_valid=0, got {dut.out_valid.value}"


# ---------------------------------------------------------------------------
# Test 4: backpressure -- out_ready=0 propagates to in_ready
# ---------------------------------------------------------------------------
@cocotb.test()
async def test_backpressure(dut):
    """out_ready=0 must set in_ready=0 (backpressure pass-through)."""
    await init(dut)

    dut.in_valid.value  = 1
    dut.in_value.value  = 0x00FF
    dut.out_ready.value = 0          # downstream not ready
    await Timer(1, units="ns")

    assert dut.in_ready.value == 0, \
        f"in_ready should be 0 when out_ready=0, got {dut.in_ready.value}"


# ---------------------------------------------------------------------------
# Test 5: backpressure released -- in_ready follows out_ready
# ---------------------------------------------------------------------------
@cocotb.test()
async def test_backpressure_release(dut):
    """When out_ready goes 1, in_ready must also go 1."""
    await init(dut)

    dut.out_ready.value = 0
    await Timer(1, units="ns")
    assert dut.in_ready.value == 0

    dut.out_ready.value = 1
    await Timer(1, units="ns")
    assert dut.in_ready.value == 1, \
        f"in_ready should follow out_ready; got {dut.in_ready.value}"


# ---------------------------------------------------------------------------
# Test 6: value=1 (minimum non-zero) passes through
# ---------------------------------------------------------------------------
@cocotb.test()
async def test_min_nonzero(dut):
    """Value of 1 (the smallest non-zero) must pass the filter."""
    await init(dut)

    dut.in_valid.value  = 1
    dut.in_value.value  = 1
    dut.in_x.value      = 0
    dut.in_y.value      = 0
    dut.out_ready.value = 1
    await Timer(1, units="ns")

    assert dut.out_valid.value == 1, "Value=1 should pass the filter"
    assert dut.out_value.value == 1


# ---------------------------------------------------------------------------
# Test 7: max value (0xFFFF for DATA_W=16) passes through
# ---------------------------------------------------------------------------
@cocotb.test()
async def test_max_value(dut):
    """Maximum 16-bit value passes through unchanged."""
    await init(dut)

    dut.in_valid.value  = 1
    dut.in_value.value  = 0xFFFF
    dut.in_x.value      = 31
    dut.in_y.value      = 31
    dut.out_ready.value = 1
    await Timer(1, units="ns")

    assert dut.out_valid.value  == 1
    assert dut.out_value.value  == 0xFFFF
    assert dut.out_x.value      == 31
    assert dut.out_y.value      == 31


# ---------------------------------------------------------------------------
# Test 8: sweep of zero vs non-zero with a clock (verifies no latency)
# ---------------------------------------------------------------------------
@cocotb.test()
async def test_combinational_no_latency(dut):
    """
    Module is purely combinational -- output must respond within 1 ns,
    not require a clock edge.
    """
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start()) if hasattr(dut, "clk") else None
    await init(dut)

    cases = [
        (0,      0, 0, False),
        (0x0001, 1, 2, True),
        (0x0000, 3, 4, False),
        (0xBEEF, 5, 6, True),
    ]

    for val, x, y, expect_valid in cases:
        dut.in_valid.value  = 1
        dut.in_value.value  = val
        dut.in_x.value      = x
        dut.in_y.value      = y
        dut.out_ready.value = 1
        await Timer(1, units="ns")   # combinational settle

        got = int(dut.out_valid.value)
        exp = 1 if expect_valid else 0
        assert got == exp, \
            f"val=0x{val:04X}: expected out_valid={exp}, got {got}"
        if expect_valid:
            assert int(dut.out_value.value) == val
            assert int(dut.out_x.value)     == x
            assert int(dut.out_y.value)     == y
