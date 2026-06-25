"""
cocotb testbench for sram.

Verifies the synchronous SRAM model against a Python reference memory:
  * Port A write / read-back (whole-word writes).
  * Write-first behaviour: a read issued on the same cycle as a write to the
    same address returns the freshly written data.
  * Read latency: 1 cycle when OUTPUT_REG=0, 2 cycles when OUTPUT_REG=1.
  * a_rdata_vld / b_rdata_vld qualifiers track the enable, delayed by latency.
  * Port B read-only, independent address (dual-port only).
  * Reset clears the (registered) outputs.

Run:
    make SIM=verilator DATA_WIDTH=32 ADDR_WIDTH=8 USE_DUAL_PORT=1 OUTPUT_REG=1
"""

import os
import random
import cocotb
from cocotb.triggers import RisingEdge, FallingEdge, ClockCycles
from cocotb.clock import Clock


DATA_WIDTH    = int(os.environ.get("DATA_WIDTH", "32"))
ADDR_WIDTH    = int(os.environ.get("ADDR_WIDTH", "8"))
USE_DUAL_PORT = int(os.environ.get("USE_DUAL_PORT", "1"))
OUTPUT_REG    = int(os.environ.get("OUTPUT_REG", "1"))

DEPTH    = 1 << ADDR_WIDTH
DATA_MAX = (1 << DATA_WIDTH) - 1

# Read latency in clock cycles: the read-data path is one FF, plus one
# more FF when the optional output register is enabled.
LATENCY = 2 if OUTPUT_REG else 1

CLK_NS = 10


def rnd_data():
    return random.randint(0, DATA_MAX)


async def reset_dut(dut):
    """Pulse rst_n low; idle all enables."""
    dut.a_en.value = 0
    dut.a_we.value = 0
    dut.a_addr.value = 0
    dut.a_wdata.value = 0
    dut.b_en.value = 0
    dut.b_addr.value = 0
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 3)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


async def start_clock(dut):
    cocotb.start_soon(Clock(dut.clk, CLK_NS, unit="ns").start())


# ---------------------------------------------------------------------------
@cocotb.test()
async def reset_clears_outputs(dut):
    """After reset, the registered read outputs and valids are 0."""
    await start_clock(dut)
    await reset_dut(dut)

    if OUTPUT_REG:
        assert int(dut.a_rdata.value) == 0, "a_rdata not cleared by reset"
        assert int(dut.a_rdata_vld.value) == 0, "a_rdata_vld not cleared by reset"
        if USE_DUAL_PORT:
            assert int(dut.b_rdata.value) == 0, "b_rdata not cleared by reset"
            assert int(dut.b_rdata_vld.value) == 0, "b_rdata_vld not cleared by reset"

    dut._log.info("PASS: reset clears registered outputs")


# ---------------------------------------------------------------------------
@cocotb.test()
async def write_then_read_back(dut):
    """Write known words, then read them back through port A with correct latency."""
    await start_clock(dut)
    await reset_dut(dut)

    ref = {}
    n = min(DEPTH, 64)
    addrs = random.sample(range(DEPTH), n)

    # --- write phase ---
    for addr in addrs:
        data = rnd_data()
        ref[addr] = data
        dut.a_en.value = 1
        dut.a_we.value = 1
        dut.a_addr.value = addr
        dut.a_wdata.value = data
        await RisingEdge(dut.clk)
    dut.a_we.value = 0

    for addr in addrs:
        dut.a_en.value = 1
        dut.a_we.value = 0
        dut.a_addr.value = addr
        await RisingEdge(dut.clk)
        # wait remaining latency, sampling after the data has propagated
        await ClockCycles(dut.clk, LATENCY - 1)
        await FallingEdge(dut.clk)  # settle, read on stable half-cycle

        got = int(dut.a_rdata.value)
        exp = ref[addr]
        assert got == exp, f"read addr 0x{addr:x}: got 0x{got:x} expected 0x{exp:x}"
        assert int(dut.a_rdata_vld.value) == 1, f"a_rdata_vld low for read addr 0x{addr:x}"

    dut.a_en.value = 0
    dut._log.info(f"PASS: wrote/read {n} words with latency={LATENCY}")


# ---------------------------------------------------------------------------
@cocotb.test()
async def write_first_same_cycle(dut):
    """A read on the same cycle as a write to the same address returns new data."""
    await start_clock(dut)
    await reset_dut(dut)

    addr = random.randrange(DEPTH)
    old = rnd_data()
    new = (old ^ DATA_MAX) & DATA_MAX  # guaranteed different

    # seed old value
    dut.a_en.value = 1
    dut.a_we.value = 1
    dut.a_addr.value = addr
    dut.a_wdata.value = old
    await RisingEdge(dut.clk)

    # write-first: write new value while reading same address this cycle
    dut.a_we.value = 1
    dut.a_wdata.value = new
    dut.a_addr.value = addr
    await RisingEdge(dut.clk)
    dut.a_we.value = 0

    # the read captured on that write cycle should be the NEW data
    await ClockCycles(dut.clk, LATENCY - 1)
    await FallingEdge(dut.clk)
    got = int(dut.a_rdata.value)
    assert got == new, f"write-first failed: got 0x{got:x} expected new 0x{new:x}"

    dut.a_en.value = 0
    dut._log.info("PASS: write-first returns freshly written data")


# ---------------------------------------------------------------------------
@cocotb.test()
async def port_b_read(dut):
    """Port B reads independently of port A (dual-port configs only)."""
    if not USE_DUAL_PORT:
        dut._log.info("SKIP: single-port config, no port B")
        return

    await start_clock(dut)
    await reset_dut(dut)

    # write words via port A
    ref = {}
    n = min(DEPTH, 16)
    addrs = random.sample(range(DEPTH), n)
    for addr in addrs:
        data = rnd_data()
        ref[addr] = data
        dut.a_en.value = 1
        dut.a_we.value = 1
        dut.a_addr.value = addr
        dut.a_wdata.value = data
        await RisingEdge(dut.clk)
    dut.a_en.value = 0
    dut.a_we.value = 0

    # read them back via port B
    for addr in addrs:
        dut.b_en.value = 1
        dut.b_addr.value = addr
        await RisingEdge(dut.clk)
        await ClockCycles(dut.clk, LATENCY - 1)
        await FallingEdge(dut.clk)
        got = int(dut.b_rdata.value)
        exp = ref[addr]
        assert got == exp, f"port B read addr 0x{addr:x}: got 0x{got:x} expected 0x{exp:x}"
        assert int(dut.b_rdata_vld.value) == 1, f"b_rdata_vld low for addr 0x{addr:x}"

    dut.b_en.value = 0
    dut._log.info(f"PASS: port B read {n} words")


# ---------------------------------------------------------------------------
@cocotb.test()
async def disable_clears_valid(dut):
    """With the port enable low, the read-data valid qualifier is deasserted."""
    await start_clock(dut)
    await reset_dut(dut)

    # one enabled access to raise valid
    dut.a_en.value = 1
    dut.a_we.value = 0
    dut.a_addr.value = 0
    await RisingEdge(dut.clk)

    # now drop enable and let the pipeline drain
    dut.a_en.value = 0
    await ClockCycles(dut.clk, LATENCY + 1)
    await FallingEdge(dut.clk)
    assert int(dut.a_rdata_vld.value) == 0, "a_rdata_vld stuck high after a_en low"

    dut._log.info("PASS: a_en=0 deasserts a_rdata_vld")
