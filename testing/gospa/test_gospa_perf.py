"""
test_gospa_perf.py -- cycle / MACs-per-cycle measurement for the mobilenet
config. Reuses the test_gospa harness; counts real clock edges per phase and
the true MAC count (one accumulate per nonzero-activation x nonzero-weight
overlap, which is exactly what the PE lanes do).

Run (mobilenet shape):
    make MODULE=test_gospa_perf SIM=verilator \
         H=10 F=3 S=2 N_PE=8 N_MULTS=4 N_ROWS=10 N_NZ_MAX=1024 FIFO_D=2048
"""
import os
import sys
import random

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge

sys.path.insert(0, os.path.dirname(__file__))
import test_gospa as tg                                   # noqa: E402


def count_macs(matrix, kernels, S, F, E):
    """Number of multiply-accumulates the PE array performs: for every output
    position and kernel tap, one MAC iff both the weight and the overlapped
    activation are nonzero (GoSPA skips zero acts and WSP-gates zero weights)."""
    total = 0
    for ker in kernels:
        for oi in range(E):
            for oj in range(E):
                bi, bj = oi * S, oj * S
                for ki in range(F):
                    for kj in range(F):
                        if ker[ki][kj] != 0 and matrix[bi + ki][bj + kj] != 0:
                            total += 1
    return total


@cocotb.test()
async def perf_mobilenet(dut):
    cocotb.start_soon(Clock(dut.clk, tg.CLK_NS, units="ns").start())
    rng = random.Random(0xC0FFEE)                 # same seed as the e2e test
    matrix = tg._make_padded_activation(rng, sparsity=0.5)

    n_nz = sum(1 for row in matrix for v in row if v != 0)
    macs = count_macs(matrix, tg.KERNELS, tg.S, tg.F, tg.E)

    cnt = {"n": 0}
    async def ticker():
        while True:
            await RisingEdge(dut.clk)
            cnt["n"] += 1
    cocotb.start_soon(ticker())

    await tg.reset(dut)
    t0 = cnt["n"]
    await tg.load_apu_wsps(dut)
    await tg.load_pe_weights(dut)
    await tg.load_pe_wsps(dut)
    await tg.arm_pe_array(dut)
    await tg.fill_activation_csr(dut, matrix)
    t_setup = cnt["n"]
    await tg.run_scan(dut)
    t_scan = cnt["n"]
    await tg.run_stage2(dut)
    t_s2 = cnt["n"]
    await tg.drain_all(dut)
    t_drain = cnt["n"]

    setup_c   = t_setup - t0
    scan_c    = t_scan - t_setup
    s2_c      = t_s2 - t_scan
    drain_c   = t_drain - t_s2
    compute_c = scan_c + s2_c
    total_c   = t_drain - t0
    peak      = tg.N_PE * tg.N_MULTS

    L = dut._log.info
    L("================ PERF: mobilenet H=%d F=%d S=%d, %dx%d=%d channels ================"
      % (tg.H, tg.F, tg.S, tg.N_PE, tg.N_MULTS, tg.N_CHAN))
    L("activation nonzeros           = %d" % n_nz)
    L("MAC ops (useful work)         = %d" % macs)
    L("cycles  setup(load)           = %d" % setup_c)
    L("cycles  scan   (stage1->A)    = %d" % scan_c)
    L("cycles  stage2 (A->B->MAC)    = %d" % s2_c)
    L("cycles  drain  (readout)      = %d" % drain_c)
    L("cycles  compute(scan+stage2)  = %d" % compute_c)
    L("cycles  TOTAL  (reset->drain) = %d" % total_c)
    L("MACs/cycle  over stage2       = %.2f   (peak = %d lanes)" % (macs / s2_c, peak))
    L("MACs/cycle  over compute      = %.2f" % (macs / compute_c))
    L("MACs/cycle  over TOTAL        = %.2f" % (macs / total_c))
    L("PE-array utilization (stage2) = %.1f%% of %d-MAC/cyc peak"
      % (100.0 * macs / s2_c / peak, peak))
