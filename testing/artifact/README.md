# Artifact — reproduce the report's RTL simulation results

One-command regeneration of the RTL-measured numbers in the report. Every
run golden-checks its outputs inside the simulation (PyTorch/functional
model reference); a mismatch fails the run, so a completed target is a
verified result.

## Requirements

Verilator ≥ 5.0, the repo venv active (cocotb 2.x, numpy, torch,
matplotlib).

## Targets

```
./run.sh smoke       # V1 + V2 functional suites                   (~2 min)
./run.sh v1v2        # RTL-measured V1 vs V2 density comparison    (~5 min)
./run.sh mobilenet   # V2 MobileNetV2 conv1 golden + perf          (~2 min)
./run.sh mobilenet-e2e  # V2 full 52-layer MobileNetV2 at 32x4     (slow)
./run.sh alexnet     # V2 AlexNet conv3/4/5 tiled layers           (~30-45 min)
./run.sh all         # smoke + v1v2 + mobilenet
```

Results append to `results/*.csv`; `v1v2` prints a merged comparison table
(see `summarize.py`).

## What v1v2 measures

Both architectures run the *identical* synthetic workload (same RNG seed
and draw order: 3 input channels, 32 output kernels, H=32 F=3 S=1,
Bernoulli densities 0.3/0.6/1.0) at 8 PEs x 4 lanes:

- **V1** (`rtl/V1`, multiple kernels/PE): via the new
  `testing/V1/gospa/test_gospa_perf.py`. Built with `FIFO_D=1024` because
  V1's serial scan-then-route flow deadlocks when a per-PID FIFO-A fills
  mid-scan (its original tests only used H ≤ 16).
- **V2** (`rtl/V2`, one kernel/PE, wide front end: `FILL_W=16,
  STAGE1_BATCH=16, S2_BEATS=16, DRAIN_W=64`): the configuration reported
  in the paper.

Reference numbers (Verilator 5.048, cycles / multiplier utilization):

| d   | V1             | V2            |
|-----|----------------|---------------|
| 0.3 | 10,876 / 19.5% | 4,291 / 49.5% |
| 0.6 | 20,612 / 44.2% | 13,010 / 70.0% |
| 1.0 | 32,755 / 74.2% | 25,164 / 96.6% |

Headline: V2 is 1.74x V1 (geomean), growing to 2.53x at d=0.3; 96.6%
utilization at full density.

## What mobilenet measures

MobileNetV2's first convolution (real 80x80 apple input, real quantized
weights, 32 output channels, golden-checked) on both architectures at
8x4, reporting layer fps at 100 MHz:

| arch | cycles | util  | fps @100 MHz |
|------|--------|-------|--------------|
| V1   | 82,188 | 39.1% | 1,216.7      |
| V2   | 50,507 | 63.6% | 1,979.9      |

## What mobilenet-e2e measures

The full 52-layer quantized MobileNetV2 (real apple-80 tensors, every
layer golden-checked) through the RTL at 32x4 with the wide config.
Reference (post drain-race fix, see below):

```
NETWORK: useful MACs=32377599  cycles=669094  util=37.8%
         latency=6.69 ms @100MHz  (149.5 fps)
```

Historical note: the first re-run of this artifact caught a real RTL bug —
the pipelined-multiplier commit (`b45afd3`) staggered per-PE drain starts,
and with a wide `DRAIN_W` the OR'd `drain_busy` could gap and pulse
`drain_done` before delayed PEs drained, silently losing their output.
Fixed in `rtl/V2/pe/pe.sv` (`drain_busy = draining || drain_req_q`). The
fix costs ~1.1k cycles network-wide (149.7 -> 149.5 fps).

## Other paper results

- Density sweep / knob sweeps for the figures: `testing/gospa/gen_graph_data.py`
- AlexNet model-side V1/V2 comparison (labeled *modeled* in the paper):
  `sw/alexnet_arch_compare.py`
- Paper figures: `omiited-report/ECE720__GOSPA/gen_paper_figures.py`
- FPGA synthesis: `rtl/V2/synth/` (Vivado 2024.2, Kria KV260)
