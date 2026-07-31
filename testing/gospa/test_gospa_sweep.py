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
from gospa_tb import (H, F, S, E, N_PE, N_MULTS, CLK_NS,  # noqa: E402
                      TILE_H, TILE_W, make_activation, rand_kernels)

FCLK_MHZ = os.environ.get("FCLK_MHZ") or None            # override quoted clock
TRIALS   = int(os.environ.get("TRIALS", "4"))            # random draws averaged/point
PASSES   = int(os.environ.get("PASSES", "4"))            # accumulated passes/point before drain
S1_BATCH = int(os.environ.get("STAGE1_BATCH", "1"))      # scanner lanes/cycle (report only)


# Equal-density diagonal: d_a = d_w from 0.1 to 1.0 in 0.1 steps.
GRID = [(round(0.1 * i, 1), round(0.1 * i, 1)) for i in range(1, 11)]


def _add_maps(a, b):
    return [[x + y for x, y in zip(ra, rb)] for ra, rb in zip(a, b)]


async def _run_point(dut, da, dw, rng):
    """PASSES accumulated input-channel passes at (da, dw): each pass reloads
    fresh random weights/activation, scans + routes/computes into the SAME
    resident per-CID banks, then one drain reads the summed result. Returns
    (scan, stream, drain, macs) totalled over all PASSES, or raises on a
    functional mismatch."""
    await tb.reset(dut)

    golden = [[[0] * E for _ in range(E)] for _ in range(N_PE)]
    scan_total, active_total, macs_total = 0, 0, 0
    for _ in range(PASSES):
        kernels = rand_kernels(rng, N_PE, dw)
        matrix  = make_activation(rng, da, pad=1)   # AlexNet conv5: pad=1
        for pe in range(N_PE):
            golden[pe] = _add_maps(golden[pe], fm.conv2d_reference(matrix, kernels[pe], S))

        await tb.load_pe_weights(dut, kernels)
        await tb.arm_pe_array(dut)

        perf = {}                                   # single-bank 2-D input tiling
        await tb.run_tiled_channel(dut, matrix, TILE_H, TILE_W, perf=perf)
        scan_total   += perf.get("scan_cycles", 0)
        active_total += perf.get("active_cycles", 0)
        macs_total   += perf.get("macs", 0)

    drain_perf = {}
    got = await tb.drain_all(dut, perf=drain_perf)
    tb.compare(dut, got, golden, f"sweep[a={da},w={dw}]")
    return (scan_total, active_total, drain_perf.get("drain_cycles", 0), macs_total)


@cocotb.test()
async def test_sparsity_sweep(dut):
    cocotb.start_soon(Clock(dut.clk, CLK_NS, unit="ns").start())
    rng = random.Random(0x5A17)

    if FCLK_MHZ is not None:
        f_clk_hz, f_src = float(FCLK_MHZ) * 1e6, "specified"
    else:
        f_clk_hz, f_src = 1e9 / CLK_NS, "simulated"

    dut._log.info(f"goSPA sparsity sweep  H={H} F={F} S={S} E={E} "
                  f"N_PE={N_PE} N_MULTS={N_MULTS} PASSES={PASSES} "
                  f"STAGE1_BATCH={S1_BATCH}  tile={TILE_H}x{TILE_W}")

    rows = []
    for da, dw in GRID:
        acc = [0, 0, 0, 0]                                # scan, strm, drain, macs
        for _ in range(TRIALS):
            for i, v in enumerate(await _run_point(dut, da, dw, rng)):
                acc[i] += v
        scan, stream, drain, macs = (x / TRIALS for x in acc)
        rows.append((da, dw, scan, stream, drain, macs))
        dut._log.info(f"  d={da:.1f}  scan={scan:.1f} strm={stream:.1f} "
                      f"drain={drain:.1f} macs={macs:.1f}  (avg of {TRIALS})")

    dense_total = next((r[2] + r[3] + r[4] for r in rows if r[0] == 1.0), 0)
    peak_gmacs  = N_PE * N_MULTS * f_clk_hz / 1e9         # 1 MAC/lane/cyc at peak

    hdr = (f"{'Den':>5} {'scanCyc':>8} {'strmCyc':>8} "
           f"{'drnCyc':>7} {'totCyc':>7} {'useMACs':>8} {'multUtil%':>10} "
           f"{'GMAC/s':>8} {'speedup':>8} {'fps':>12}")
    lines = [
        "goSPA SPARSITY SWEEP  (V2 act dataflow, bare non-pipelined PE, RTL-measured)",
        "H=%d F=%d S=%d E=%d  N_PE=%d N_MULTS=%d  PASSES=%d  STAGE1_BATCH=%d  f_clk=%.0f MHz (%s)"
        % (H, F, S, E, N_PE, N_MULTS, PASSES, S1_BATCH, f_clk_hz / 1e6, f_src),
        "input tiled %dx%d (single bank); FIFO-A holds <=%d entries, independent of H"
        % (TILE_H, TILE_W, TILE_H * TILE_W),
        "peak = %.3f GMAC/s (%d lanes)   dense frame = %d cyc"
        % (peak_gmacs, N_PE * N_MULTS, dense_total),
        "d_a = d_w (equal-density diagonal); density = non-zero fraction",
        "one full inference of this HxH map: Cin=%d accumulated passes, Cout=N_PE output channels"
        % PASSES,
        "",
        hdr,
        "-" * len(hdr),
    ]

    for da, dw, scan, stream, drain, macs in rows:
        total = scan + stream + drain
        util  = macs / (stream * N_PE * N_MULTS) if stream else 0.0
        gmacs = macs * f_clk_hz / 1e9 / total if total else 0.0
        spd   = dense_total / total if total else 0.0
        fps   = f_clk_hz / total if total else 0.0
        lines.append(
            f"{da:>5.1f} {scan:>8.1f} {stream:>8.1f} {drain:>7.1f} "
            f"{total:>7.1f} {macs:>8.1f} {util*100:>10.1f} {gmacs:>8.3f} "
            f"{spd:>7.2f}x {fps:>12,.0f}")

    lines += [
        "-" * len(hdr),
        "legend:",
        "  scanCyc   = Stage-1 scan (1 activation/cycle; ~n_nz). Independent of",
        "              weight sparsity -- the front-end floor.",
        "  strmCyc   = Stage-2 routing + PE compute: s2_start -> last accepted beat.",
        "  drnCyc    = accumulator drain (drain_start -> drain_done).",
        "  totCyc    = scanCyc + strmCyc + drnCyc: full inference of this tile.",
        "  useMACs   = real accumulates performed (zeros never reach a multiplier).",
        "  multUtil% = useMACs / (strmCyc x N_PE x N_MULTS): PE-array utilisation",
        "              over the compute window (not per-beat lane fill).",
        "  GMAC/s    = useMACs x f_clk / totCyc: effective delivered throughput.",
        "  speedup   = dense totCyc / this totCyc: end-to-end vs the dense frame.",
        "  fps       = f_clk / totCyc: inferences/second for this tile at f_clk.",
        "",
    ]

    report_path = os.path.join(_TEST_DIR, "gospa_perf.txt")
    with open(report_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    dut._log.info(f"sweep report written to {report_path}")
