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
./run.sh smoke          # functional suite                       (~2 min)
./run.sh mobilenet      # MobileNetV2 conv1 golden + perf        (~2 min)
./run.sh mobilenet-e2e  # full 52-layer MobileNetV2 at 32x4      (slow)
./run.sh alexnet        # AlexNet conv3/4/5 tiled layers         (~30-45 min)
./run.sh all            # smoke + mobilenet
```

Results append to `results/*.csv`.

## What mobilenet measures

MobileNetV2's first convolution (real 80x80 apple input, real quantized
weights, 32 output channels, golden-checked) at 8x4, reporting layer fps
at 100 MHz. Reference: 50,507 cycles / 63.6 % utilization / 1,979.9 fps.

## What mobilenet-e2e measures

The full 52-layer quantized MobileNetV2 (real apple-80 tensors, every
layer golden-checked) through the RTL at 32x4 with the wide front-end
config. Reference (post drain-race fix, see below):

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

## Other report results

- Density sweep / knob sweeps for the figures: `testing/gospa/gen_graph_data.py`
- FPGA synthesis: `rtl/V2/synth/` (Vivado 2024.2, Kria KV260)
