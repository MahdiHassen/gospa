"""
test_mult_pipe.py -- cocotb test for mult_pipe.sv (pipelined signed multiplier)
GoSPA Project -- Team 19, ECE 720 (Spring 2026)

    make MODULE=test_mult_pipe                 # 16x16
    make MODULE=test_mult_pipe A_W=8 B_W=8

Drives signed operands one/cycle (in_valid), collects results on out_valid, and
checks the pipelined output against a*b. Order is preserved (1 result / input).
"""

import os
import random

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ReadOnly

A_W = int(os.environ.get("A_W", "16"))
B_W = int(os.environ.get("B_W", "16"))
P_W = A_W + B_W


def _signed(v, bits):
    return v - (1 << bits) if (v >> (bits - 1)) & 1 else v


def _mask(v, bits):
    return v & ((1 << bits) - 1)


def _clog2(n):
    return 0 if n <= 1 else (n - 1).bit_length()


@cocotb.test()
async def test_mult(dut):
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())

    dut.rst_n.value    = 0
    dut.in_valid.value = 0
    dut.a.value        = 0
    dut.b.value        = 0
    dut.aux_in.value   = 0
    for _ in range(4):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)

    # Test vectors: corners x corners, then random.
    amax, amin = (1 << (A_W - 1)) - 1, -(1 << (A_W - 1))
    bmax, bmin = (1 << (B_W - 1)) - 1, -(1 << (B_W - 1))
    corners_a = [0, 1, -1, amax, amin]
    corners_b = [0, 1, -1, bmax, bmin]
    pairs = [(a, b) for a in corners_a for b in corners_b]
    rng = random.Random(0xABCD)
    for _ in range(600):
        pairs.append((rng.randint(amin, amax), rng.randint(bmin, bmax)))

    results = []

    async def monitor():
        while True:
            await ReadOnly()
            if dut.out_valid.value == 1:
                results.append(_signed(int(dut.p.value), P_W))
            await RisingEdge(dut.clk)

    cocotb.start_soon(monitor())

    expected = []
    for (a, b) in pairs:
        dut.in_valid.value = 1
        dut.a.value = _mask(a, A_W)
        dut.b.value = _mask(b, B_W)
        expected.append(a * b)
        await RisingEdge(dut.clk)
    dut.in_valid.value = 0

    LAT = _clog2(1 << _clog2(B_W)) + 2
    for _ in range(LAT + 8):
        await RisingEdge(dut.clk)

    assert len(results) == len(expected), \
        f"got {len(results)} results, expected {len(expected)}"
    for i, (got, exp) in enumerate(zip(results, expected)):
        a, b = pairs[i]
        assert got == exp, f"[{i}] {a} * {b}: got {got}, expected {exp}"

    dut._log.info(f"[mult_pipe] PASS -- {len(expected)} products "
                  f"(A_W={A_W} B_W={B_W}, latency LAT={LAT})")
