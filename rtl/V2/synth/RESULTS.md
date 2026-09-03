# GoSPA — FPGA Synthesis Results

Vivado 2024.2, out-of-context post-route implementation of `gospa.sv` on the
Kria KV260 (`xck26-sfvc784-2LV-c`).

Configuration: N_PE = 8, N_MULTS = 4, H = 8, F = 3, S = 1, DATA_W = 16,
ACC_W = 32, FIFO depth = 64 (32 multipliers). Target clock 1 GHz (1.000 ns),
deliberately over-constrained: the tool then reports the true achievable
frequency instead of stopping once a loose constraint is met.

The design instantiates a custom pipelined multiplier
(`common/arith/mult_pipe.sv`) rather than inferring DSP blocks.

Reproduce with:

```bash
cd rtl/V2/synth/vivado && vivado -mode batch -source run_synth.tcl
```

## Results

| Metric | Value |
|---|---|
| Target clock | 1 GHz (1.000 ns) |
| Worst negative slack (WNS) | −4.855 ns |
| Achievable period | 5.855 ns |
| **Maximum frequency** | **≈ 171 MHz** |
| LUT | 42 698 (36.5 %) |
| — LUT as logic | 42 194 |
| — LUT as memory | 504 |
| FF | 72 610 (31.0 %) |
| DSP | 0 (0 %) |
| BRAM | 0 (0 %) |

Available on the KV260: 117 120 LUT, 234 240 FF, 1 248 DSP, 144 BRAM.

## Critical path

From `u_apu/u_scanner/row_ptr_q_reg` to `u_apu/u_stage1/g_fifoa[1].u_fifoa`
(bank write). The multiplier is not on the critical path — pipelining
`mult_pipe` moved the multiply off it; the design is limited by the Stage-1
scanner → FIFO-A write path.

## Resource sweep

`sweep/sweep_results.csv` (synthesis-only) sweeps N_PE, N_MULTS and FIFO
depth; `sweep/plot_sweep.py` plots the LUT scaling. LUT count grows roughly
linearly with N_PE and FIFO depth, but **super-linearly with N_MULTS**
(19 328 → 108 193 LUT for 1 → 8 lanes), making the multiplier lanes the
dominant area knob.
