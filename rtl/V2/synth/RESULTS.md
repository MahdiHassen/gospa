# GoSPA — FPGA Synthesis Results

Vivado 2024.2, out-of-context post-route implementation of `gospa.sv` on the
Kria KV260 (`xck26-sfvc784-2LV-c`).

Both architectures are synthesized with the **same configuration** (N_PE = 8,
N_MULTS = 4, H = 8, F = 3, S = 1, DATA_W = 16, ACC_W = 32, FIFO depth = 64 — 32
multipliers either way) and the **same 1 GHz (1.000 ns) target**. The target is
deliberately over-constrained: the tool then reports the true achievable frequency
instead of stopping once a loose constraint is met, so the two columns are directly
comparable.

Both versions instantiate the same custom pipelined multiplier
(`common/arith/mult_pipe.sv`), so the comparison isolates the *dataflow* rather than
the arithmetic implementation.

Reproduce with:

```bash
cd rtl/V1/synth/vivado && vivado -mode batch -source run_synth.tcl
cd rtl/V2/synth/vivado && vivado -mode batch -source run_synth.tcl
```

## Results

| Metric | V1 (multiple kernels/PE) | V2 (one kernel/PE) |
|---|---|---|
| Target clock | 1 GHz (1.000 ns) | 1 GHz (1.000 ns) |
| Worst negative slack (WNS) | −3.717 ns | −4.855 ns |
| Achievable period | 4.717 ns | 5.855 ns |
| **Maximum frequency** | **≈ 212 MHz** | **≈ 171 MHz** |
| LUT | 36 349 (31.0 %) | 42 698 (36.5 %) |
| — LUT as logic | 35 259 | 42 194 |
| — LUT as memory | 1 090 | 504 |
| FF | 59 952 (25.6 %) | 72 610 (31.0 %) |
| DSP | 0 (0 %) | 0 (0 %) |
| BRAM | 0 (0 %) | 0 (0 %) |

Available on the KV260: 117 120 LUT, 234 240 FF, 1 248 DSP, 144 BRAM.

## Critical paths

| Version | From | To |
|---|---|---|
| V1 | `u_apu/u_stage2/g_fifob[5].u_fifob/ob_head_reg` | `u_pe_array/g_pe[5].u_pe/u_wsram` (port-B read) |
| V2 | `u_apu/u_scanner/row_ptr_q_reg` | `u_apu/u_stage1/g_fifoa[1].u_fifoa` (bank write) |

In neither version is the multiplier on the critical path — pipelining `mult_pipe`
moved the multiply off it. V1 is limited by the FIFO-B → weight-SRAM fetch path;
V2 by the Stage-1 scanner → FIFO-A write path.

## Reading the comparison

V2 costs ~17 % more LUTs, ~21 % more FFs and ~19 % lower Fmax than V1. That is the
price of the wider front end and the per-lane CID accumulator banks. It buys a large
increase in multiplier utilization and throughput (see
`testing/artifact/results/`), which is where the V2 dataflow wins.

## Resource sweep

`sweep/sweep_results.csv` (V2, synthesis-only) sweeps N_PE, N_MULTS and FIFO depth;
`sweep/lut_sweep.pdf` plots the LUT scaling. LUT count grows roughly linearly with
N_PE and FIFO depth, but **super-linearly with N_MULTS** (19 328 → 108 193 LUT for
1 → 8 lanes), making the multiplier lanes the dominant area knob.
