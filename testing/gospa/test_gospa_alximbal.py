"""
test_gospa_alximbal.py -- B (S2_BEATS) vs FIFO-B load imbalance on AlexNet
conv1 (real weights, real apple activations, 64x64 center crop of the padded
input, first 8 output channels, all 3 input channels accumulated).

Load imbalance := max_k(peak FIFO-B occupancy of PE k)
               /  mean_k(peak FIFO-B occupancy of PE k)
sampled every cycle from the internal fifo counters
(u_apu.u_stage2.g_fifob[k].u_fifob.cnt). Also records the time-averaged
occupancy imbalance. Golden-checked like every other run.

One build per S2_BEATS value; appends to IMBAL_CSV:
    s2_beats, imbal_peak, imbal_avg, peak0..peak7
"""
import os
import sys

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import Event, ReadOnly, RisingEdge

_TEST_DIR = os.path.dirname(__file__)
sys.path.insert(0, os.path.abspath(os.path.join(_TEST_DIR, "..", "..", "sw")))
sys.path.insert(0, os.path.abspath(os.path.join(_TEST_DIR, "..", "ref")))
sys.path.insert(0, _TEST_DIR)

import numpy as np                                        # noqa: E402
import functional as fm                                   # noqa: E402
fm._VERBOSE = False

import gospa_tb as tb                                     # noqa: E402
import alexnet_layers as ax                               # noqa: E402
from test_gospa_apple import _load_group_weights          # noqa: E402

CSV = os.environ.get(
    "IMBAL_CSV", os.path.join(_TEST_DIR, "gospa_alximbal_rows.csv"))
CROP = 32
N_CH = 3


@cocotb.test()
async def test_conv1_imbalance(dut):
    L = ax.get_layer(1)
    acts, weights = L["acts"], L["weights"]
    f, stride, pad = weights.shape[2], L["stride"], L["pad"]
    assert tb.H == CROP and tb.F == f and tb.S == stride

    hp = acts.shape[1] + 2 * pad
    acts_p = np.zeros((N_CH, hp, hp), dtype=np.int64)
    acts_p[:, pad:pad + acts.shape[1], pad:pad + acts.shape[1]] = acts
    o = (hp - CROP) // 2
    o -= o % stride                                  # stride-aligned crop
    crop = acts_p[:, o:o + CROP, o:o + CROP]
    gch = list(range(tb.N_PE))
    golden = ax.golden_conv(crop, weights[:tb.N_PE, :N_CH], stride)

    # internal FIFO-B occupancy counters
    cnts = []
    for k in range(tb.N_PE):
        try:
            h = dut.u_apu.u_stage2.g_fifob[k].u_fifob.cnt
            int(h.value)
            cnts.append(h)
        except Exception as exc:
            raise AssertionError(
                f"cannot reach internal FIFO-B counter {k}: {exc}")

    cocotb.start_soon(Clock(dut.clk, tb.CLK_NS, unit="ns").start())
    peaks = [0] * tb.N_PE
    sums = [0] * tb.N_PE
    nsamp = 0
    stop = Event()

    async def sampler():
        nonlocal nsamp
        while not stop.is_set():
            await RisingEdge(dut.clk)
            await ReadOnly()
            nsamp += 1
            for k in range(tb.N_PE):
                v = int(cnts[k].value)
                sums[k] += v
                if v > peaks[k]:
                    peaks[k] = v
    cocotb.start_soon(sampler())

    def sp(c):
        return [fm.kernel_to_sparse(weights[k][c].tolist())[1] or [(0, 0)]
                for k in gch]

    await tb.reset(dut)
    await tb.fill_activation_csr(dut, crop[0].tolist())
    await tb.run_scan(dut)
    await _load_group_weights(dut, sp(0))
    await tb.swap_weights(dut)
    for c in range(N_CH):
        await tb.kick_stage2(dut)
        tload = None
        if c + 1 < N_CH:
            tload = cocotb.start_soon(_load_group_weights(dut, sp(c + 1)))
            await tb.fill_activation_csr(dut, crop[c + 1].tolist())
            await tb.run_scan(dut)
        if tload is not None:
            await tload
        await tb.wait_pes_idle(dut)
        if c + 1 < N_CH:
            await tb.swap_weights(dut)
    got = await tb.drain_all(dut)
    stop.set()

    bad = 0
    for p, k in enumerate(gch):
        out = np.array([[got[p].get(r * tb.E + cc, 0) for cc in range(tb.E)]
                        for r in range(tb.E)], dtype=np.int64)
        bad += int((out != golden[k]).sum())

    # Record the measurement FIRST (with a verified flag), then judge.
    avgs = [s / max(1, nsamp) for s in sums]
    imb_peak = max(peaks) / (sum(peaks) / len(peaks))
    imb_avg = max(avgs) / (sum(avgs) / len(avgs))
    with open(CSV, "a") as fh:
        fh.write(f"{tb.S2_BEATS},{imb_peak:.4f},{imb_avg:.4f},"
                 f"{1 if bad == 0 else 0},"
                 + ",".join(str(p) for p in peaks) + "\n")
    dut._log.info(
        f"S2_BEATS={tb.S2_BEATS}: peaks={peaks} "
        f"imbal_peak={imb_peak:.3f} imbal_avg={imb_avg:.3f} "
        f"{'PASS' if bad == 0 else f'GOLDEN-MISMATCH ({bad} values)'}")
    assert bad == 0, f"conv1-crop: {bad} output values mismatch"
