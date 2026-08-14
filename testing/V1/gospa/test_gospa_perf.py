"""
test_gospa_perf.py -- V1 architecture RTL performance measurement.

Runs the same synthetic density-swept workload as the V2 sweep
(testing/gospa/test_gospa_dseg.py: 3 input channels, N_PE*N_MULTS output
kernels, Bernoulli densities, identical generator distributions and seed)
through the V1 gospa top, golden-checks every output channel against the
functional model, and reports end-to-end cycles / useful MACs / multiplier
utilization per density point.

Build: make MODULE=test_gospa_perf H=32 F=3 S=1 N_PE=8 N_MULTS=4
Env:   DENS="0.3,0.6,1.0"  (default)  PERF_CSV=path  PERF_TAG=V1
"""
import os
import random

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge

from test_gospa import (                                    # noqa: E402
    H, F, S, N_PE, N_MULTS, E, N_CHAN, CLK_NS,
    reset, drain_all, _compute_per_channel_state, _load_pe_weights_for,
    _load_pe_wsps_for, _run_one_input_channel,
)
import functional as fm                                     # noqa: E402
fm._VERBOSE = False

_TEST_DIR = os.path.dirname(__file__)
GRID = [float(s) for s in os.environ.get("DENS", "0.3,0.6,1.0").split(",")]
CSV = os.environ.get("PERF_CSV", os.path.join(_TEST_DIR, "v1_perf_rows.csv"))
TAG = os.environ.get("PERF_TAG", "V1")
N_IN = 3                                   # input channels, matches V2 dseg


def rand_kernel(rng, density):
    k = [[(rng.randint(1, 50) * (1 if rng.random() < 0.5 else -1)
           if rng.random() < density else 0)
          for _ in range(F)] for _ in range(F)]
    if all(v == 0 for row in k for v in row):
        k[0][0] = 3
    return k


def make_activation(rng, density, pad=0):
    m = [[0] * H for _ in range(H)]
    core = H - 2 * pad
    for r in range(core):
        for c in range(core):
            if rng.random() < density:
                v = rng.randint(1, 80)
                m[r + pad][c + pad] = v if rng.random() < 0.5 else -v
    return m


def useful_macs(acts, kers):
    """Exact useful MAC count: nonzero activation x nonzero weight pairs
    over all valid output positions, summed across channels and kernels."""
    total = 0
    for ch in range(N_IN):
        a = acts[ch]
        # per-offset count of nonzero activations covering the output grid
        cover = [[0] * F for _ in range(F)]
        for fr in range(F):
            for fc in range(F):
                cnt = 0
                for orow in range(E):
                    for ocol in range(E):
                        if a[orow * S + fr][ocol * S + fc] != 0:
                            cnt += 1
                cover[fr][fc] = cnt
        for k in kers[ch]:
            for fr in range(F):
                for fc in range(F):
                    if k[fr][fc] != 0:
                        total += cover[fr][fc]
    return total


MOBILENET = os.environ.get("MOBILENET", "0") == "1"
FCLK_MHZ = float(os.environ.get("FCLK_MHZ", "100"))


async def _run_workload(dut, acts, kers, golden):
    """Load, stream all input channels, drain, golden-check. Returns cycles."""
    await reset(dut)
    t0 = cocotb.utils.get_sim_time("ns")
    chunks = None
    for ch in range(len(acts)):
        wsp, sw, chunks, union = _compute_per_channel_state(kers[ch])
        await _load_pe_weights_for(dut, sw, chunks)
        await _load_pe_wsps_for(dut, wsp, chunks)
        await _run_one_input_channel(dut, acts[ch], wsp, sw,
                                     chunks, union, f"ch{ch}")
    got = await drain_all(dut)
    t1 = cocotb.utils.get_sim_time("ns")
    for pe in range(N_PE):
        chunk = chunks[pe] if pe < len(chunks) else []
        for lane in range(min(N_MULTS, len(chunk))):
            k = chunk[lane]
            out = [[got[pe][lane].get(r * E + c, 0) for c in range(E)]
                   for r in range(E)]
            assert out == golden[k], (
                f"channel {k} (PE{pe} lane{lane}) mismatch vs golden")
    return int(round((t1 - t0) / CLK_NS))


@cocotb.test(skip=not MOBILENET)
async def test_perf_mobilenet_conv1(dut):
    """MobileNetV2 first conv (real 80x80 apple input, real quantized
    weights) on the V1 architecture, one output-channel group per pass.
    Build: H=80 F=3 S=2 N_ROWS=80 N_NZ_MAX=8192 FIFO_D=8192."""
    import apple_input                                      # noqa: E402
    cocotb.start_soon(Clock(dut.clk, CLK_NS, unit="ns").start())
    lanes = N_PE * N_MULTS

    chans = apple_input.load_channels()                     # 3 x (80x80)
    acts = [[[int(v) for v in row] for row in ch] for ch in chans]
    kraw = apple_input.first_conv_kernels(N_CHAN)           # [in][out] FxF
    kers = [[[[int(v) for v in row] for row in kraw[c][k]]
             for k in range(N_CHAN)] for c in range(len(kraw))]
    golden = [[[int(v) for v in row] for row in g]
              for g in apple_input.golden_output(N_CHAN)]

    cycles = await _run_workload(dut, acts, kers, golden)
    um = useful_macs(acts, kers)
    util = 100.0 * um / (cycles * lanes)
    fps = FCLK_MHZ * 1e6 / cycles
    dut._log.info(f"[{TAG} mobilenet conv1] cycles={cycles} useMACs={um} "
                  f"util={util:.1f}% fps@{FCLK_MHZ:.0f}MHz={fps:.1f} "
                  f"(all {N_CHAN} channels golden-checked)")
    with open(CSV, "a") as fh:
        fh.write(f"{TAG},mobilenet_conv1,{cycles},{um},{util:.2f},{fps:.1f}\n")


@cocotb.test(skip=MOBILENET)
async def test_perf_density_sweep(dut):
    cocotb.start_soon(Clock(dut.clk, CLK_NS, unit="ns").start())
    rng = random.Random(0xD5EE9)           # same seed as the V2 dseg sweep
    lanes = N_PE * N_MULTS

    hdr = (f"{'dens':>5} {'cycles':>8} {'useMACs':>9} {'util%':>6} "
           f"{'MAC/cyc':>8}")
    dut._log.info(f"[{TAG} perf] H={H} F={F} S={S} {N_PE}x{N_MULTS} "
                  f"({lanes} mults), {N_IN} input ch, {N_CHAN} out ch")
    dut._log.info(hdr)

    rows = []
    for d in GRID:
        kers = [[rand_kernel(rng, d) for _ in range(N_CHAN)]
                for _ in range(N_IN)]
        acts = [make_activation(rng, d, pad=0) for _ in range(N_IN)]

        golden = []
        for k in range(N_CHAN):
            acc = [[0] * E for _ in range(E)]
            for ch in range(N_IN):
                ref = fm.conv2d_reference(acts[ch], kers[ch][k], S)
                for r in range(E):
                    for c in range(E):
                        acc[r][c] += ref[r][c]
            golden.append(acc)

        await reset(dut)
        t0 = cocotb.utils.get_sim_time("ns")

        chunks = None
        for ch in range(N_IN):
            wsp, sw, chunks, union = _compute_per_channel_state(kers[ch])
            await _load_pe_weights_for(dut, sw, chunks)
            await _load_pe_wsps_for(dut, wsp, chunks)
            await _run_one_input_channel(dut, acts[ch], wsp, sw,
                                         chunks, union, f"d{d}_ch{ch}")

        got = await drain_all(dut)
        t1 = cocotb.utils.get_sim_time("ns")
        cycles = int(round((t1 - t0) / CLK_NS))

        for pe in range(N_PE):
            chunk = chunks[pe] if pe < len(chunks) else []
            for lane in range(min(N_MULTS, len(chunk))):
                k = chunk[lane]
                out = [[got[pe][lane].get(r * E + c, 0) for c in range(E)]
                       for r in range(E)]
                assert out == golden[k], (
                    f"[d={d}] channel {k} (PE{pe} lane{lane}) mismatch "
                    f"vs functional golden")

        um = useful_macs(acts, kers)
        util = 100.0 * um / (cycles * lanes)
        rows.append((d, cycles, um, util))
        dut._log.info(f"{d:>5.1f} {cycles:>8} {um:>9} {util:>6.1f} "
                      f"{um / cycles:>8.2f}")

    with open(CSV, "a") as fh:
        for (d, cyc, um, util) in rows:
            fh.write(f"{TAG},{d},{cyc},{um},{util:.2f}\n")
    dut._log.info(f"[{TAG} perf] {len(rows)} rows appended to {CSV} "
                  f"(all golden-checked)")
