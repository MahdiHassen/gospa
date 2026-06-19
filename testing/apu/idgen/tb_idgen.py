"""
Cocotb Idgen Testbench

For a given (H, F, S), it sweeps every activation coordinate (x, y),
drives the bundle, collects all (CID, PID) pairs the valid units emit,
and checks that set against a ground-truth convolution reference.

This TB verifies a single configuration per simulator run. Use the Makefile to sweep
configs.

The Makefile passes H/F/S/PIPE into the Verilog via -P/-G overrides.

PIPE selects the DUT timing: PIPE=0 (default) the outputs are
combinational; PIPE=1 they are registered, so the TB runs a clock and
samples on the capturing edge.
"""

import os
import math
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import Timer, RisingEdge


# ---- read the config the bundle was elaborated with ----
H = int(os.environ.get("H", "8"))
F = int(os.environ.get("F", "3"))
S = int(os.environ.get("S", "1"))
ACT_WIDTH = int(os.environ.get("ACT_WIDTH", "8"))
PIPE = int(os.environ.get("PIPE", "0"))   # 0 = combinational DUT, 1 = registered outputs
E = (H - F) // S + 1
G = math.ceil(F / S)          
NUM_UNIT = G * G
CLK_PERIOD_NS = 10


async def reset_dut(dut):
    """Bring the DUT to a known idle state; start a clock only if PIPE=1."""
    dut.in_valid.value = 0
    dut.rst_n.value = 1
    dut.clk.value = 0
    if PIPE:
        cocotb.start_soon(Clock(dut.clk, CLK_PERIOD_NS, unit="ns").start())
        dut.rst_n.value = 0
        await RisingEdge(dut.clk)
        dut.rst_n.value = 1
    else:
        await Timer(1, unit="ns")


async def settle(dut):
    """Advance so the outputs reflect the currently-driven inputs.

    PIPE=0: outputs are combinational, a short delay is enough.
    PIPE=1: outputs are registered, so a rising edge captures the inputs;
            sample shortly after the edge.
    """
    if PIPE:
        await RisingEdge(dut.clk)
        await Timer(1, unit="ns")
    else:
        await Timer(2, unit="ns")


def act_value(x, y):
    """Deterministic stand-in for the activation a_xy at coordinate (x,y).

    The DUT carries a_xy straight through each unit, so we drive a
    known value per (x,y) and check it comes back out unchanged on every
    emitted pair. Masked to ACT_WIDTH bits to match the port width.
    """
    return ((x * 31 + y * 7 + 1) & ((1 << ACT_WIDTH) - 1))


def reference_pairs(H, F, S):
    """Ground-truth set of (x, y, a_xy, CID, PID) from a plain convolution."""
    E = (H - F) // S + 1
    truth = set()
    for ox in range(E):
        for oy in range(E):
            for fx in range(F):
                for fy in range(F):
                    x = ox * S + fx
                    y = oy * S + fy
                    cid = ox * E + oy
                    pid = fx * F + fy
                    truth.add((x, y, act_value(x, y), cid, pid))
    return truth


def slice_vec(packed, idx, width):
    """Extract element idx (width bits) from a packed cocotb value int."""
    return (packed >> (idx * width)) & ((1 << width) - 1)


@cocotb.test()
async def sweep_all_coords(dut):
    """Drive every (x,y); compare emitted pairs to convolution reference."""
    CID_WIDTH = 1 if E * E < 2 else (E * E - 1).bit_length()
    PID_WIDTH = 1 if F * F < 2 else (F * F - 1).bit_length()

    await reset_dut(dut)

    got = set()

    for x in range(H):
        for y in range(H):
            Px = x % S
            Py = y % S
            Cx = x // S
            Cy = y // S

            dut.Px.value = Px
            dut.Py.value = Py
            dut.Cx.value = Cx
            dut.Cy.value = Cy
            dut.a_xy_in.value = act_value(x, y)
            dut.in_valid.value = 1
            await settle(dut)

            valid_vec = int(dut.valid.value)
            axy_packed = int(dut.a_xy_out.value)
            cid_packed = int(dut.cid.value)
            pid_packed = int(dut.pid.value)

            for u in range(NUM_UNIT):
                if (valid_vec >> u) & 1:
                    axy = slice_vec(axy_packed, u, ACT_WIDTH)
                    cid = slice_vec(cid_packed, u, CID_WIDTH)
                    pid = slice_vec(pid_packed, u, PID_WIDTH)
                    got.add((x, y, axy, cid, pid))

    dut.in_valid.value = 0
    await Timer(1, unit="ns")

    truth = reference_pairs(H, F, S)

    missing = truth - got
    extra = got - truth

    dut._log.info(f"config H={H} F={F} S={S} PIPE={PIPE} E={E} G={G}")
    dut._log.info(f"reference pairs={len(truth)}  emitted pairs={len(got)}")

    if missing:
        dut._log.error(f"MISSING {len(missing)} pairs, e.g. {sorted(missing)[:5]}")
    if extra:
        dut._log.error(f"EXTRA {len(extra)} pairs, e.g. {sorted(extra)[:5]}")

    assert not missing, f"{len(missing)} reference pairs not produced"
    assert not extra, f"{len(extra)} wrong pairs produced"

    dut._log.info("PASS: emitted pair set exactly matches convolution reference")


@cocotb.test()
async def invalid_suppresses_output(dut):
    """With in_valid=0, no unit should report valid."""
    await reset_dut(dut)
    dut.in_valid.value = 0
    dut.a_xy_in.value = 0
    dut.Px.value = 0
    dut.Py.value = 0
    dut.Cx.value = 0
    dut.Cy.value = 0
    await settle(dut)
    assert int(dut.valid.value) == 0, "valid asserted while in_valid=0"
    dut._log.info("PASS: in_valid=0 suppresses all valid bits")