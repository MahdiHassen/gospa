import os
import sys
import random

import cocotb
from cocotb.clock import Clock

_TEST_DIR = os.path.dirname(__file__)
sys.path.insert(0, os.path.abspath(os.path.join(_TEST_DIR, "..", "..", "sw")))
sys.path.insert(0, os.path.abspath(os.path.join(_TEST_DIR, "..", "ref")))
sys.path.insert(0, _TEST_DIR)

import functional as fm                                  # noqa: E402
fm._VERBOSE = False

import gospa_tb as tb                                    # noqa: E402
from gospa_tb import (H, F, S, N_PE, N_MULTS, E, CLK_NS,  # noqa: E402
                      load_mobilenet_kernels, make_activation)


@cocotb.test()
async def test_end_to_end(dut):
    """One input channel: N_PE kernels, check every output channel."""
    cocotb.start_soon(Clock(dut.clk, CLK_NS, unit="ns").start())
    rng = random.Random(0xC0FFEE)

    kernels = load_mobilenet_kernels(0)
    matrix  = make_activation(rng, density=0.5)
    n_nz    = sum(1 for row in matrix for v in row if v)
    dut._log.info(
        f"gospa cfg: H={H} F={F} S={S} -> E={E}  N_PE={N_PE} channels x "
        f"N_MULTS={N_MULTS} lanes  act {n_nz} nz")

    golden = [fm.conv2d_reference(matrix, kernels[pe], S) for pe in range(N_PE)]

    await tb.reset(dut)
    await tb.load_pe_weights(dut, kernels)
    await tb.arm_pe_array(dut)
    await tb.run_tiled_channel(dut, matrix)     # single-bank 2-D input tiling
    got = await tb.drain_all(dut)

    tb.compare(dut, got, golden, "end_to_end")
    dut._log.info(f"PASS -- all {N_PE} output channels match conv2d_reference")


@cocotb.test()
async def test_back_to_back_partial_sums(dut):
    """Two input channels accumulated into the same PE banks, then drained.
    Exercises activation refill, PE weight reload + re-arm mid-stream, and
    accumulator persistence across it."""
    cocotb.start_soon(Clock(dut.clk, CLK_NS, unit="ns").start())
    rng = random.Random(0xACC0)

    k0 = load_mobilenet_kernels(0)
    k1 = load_mobilenet_kernels(1)
    a0 = make_activation(rng, density=0.5)
    a1 = make_activation(rng, density=0.5)

    def _add(a, b):
        return [[a[r][c] + b[r][c] for c in range(E)] for r in range(E)]

    golden = [_add(fm.conv2d_reference(a0, k0[pe], S),
                   fm.conv2d_reference(a1, k1[pe], S)) for pe in range(N_PE)]

    await tb.reset(dut)

    await tb.load_pe_weights(dut, k0)
    await tb.arm_pe_array(dut)
    await tb.run_tiled_channel(dut, a0)

    await tb.load_pe_weights(dut, k1)          # accumulators persist across reload
    await tb.arm_pe_array(dut)
    await tb.run_tiled_channel(dut, a1)

    got = await tb.drain_all(dut)
    tb.compare(dut, got, golden, "back_to_back")
    dut._log.info(f"PASS -- all {N_PE} channels match sum of 2 input channels")


@cocotb.test()
async def test_perf(dut):
    """One pass: cycles/phase + multiplier utilisation from the beat handshake,
    verified against conv2d_reference. Informational numbers (no hard assert)."""
    cocotb.start_soon(Clock(dut.clk, CLK_NS, unit="ns").start())
    rng = random.Random(0x9E3779B9)

    kernels = load_mobilenet_kernels(0)
    matrix  = make_activation(rng, density=0.5)
    golden  = [fm.conv2d_reference(matrix, kernels[pe], S) for pe in range(N_PE)]

    await tb.reset(dut)
    await tb.load_pe_weights(dut, kernels)
    await tb.arm_pe_array(dut)

    perf = {}
    await tb.run_tiled_channel(dut, matrix, perf=perf)
    got = await tb.drain_all(dut)
    tb.compare(dut, got, golden, "perf")

    macs        = perf.get("macs")
    scan_cycles = perf.get("scan_cycles", 0)
    s2c         = perf.get("active_cycles", 0)
    total       = scan_cycles + s2c
    f_clk = 1e9 / CLK_NS

    dut._log.info("---- per-pass performance (one input channel) ----")
    dut._log.info(f"  scan (stage1) cycles : {scan_cycles}")
    dut._log.info(f"  stage2+PE     cycles : {s2c}")
    dut._log.info(f"  pass total    cycles : {total}")
    if macs is not None and s2c:
        peak = s2c * N_PE * N_MULTS
        dut._log.info(f"  useful MACs          : {macs}")
        dut._log.info(f"  multiplier util      : {macs/peak:.1%}  (over stage2+PE window)")
    dut._log.info(f"  fps (this pass only) : {f_clk/total:,.0f}")
    dut._log.info("PASS -- perf pass output verified against conv2d_reference")
