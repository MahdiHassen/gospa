# GoSPA — A Sparse CNN Accelerator in SystemVerilog

A from-scratch SystemVerilog implementation of **GoSPA** (Deng *et al.*,
*GoSPA: An Energy-efficient High-performance Globally Optimized SParse
Convolutional Neural Network Accelerator*, ISCA 2021), with a Python golden
functional model, a cycle-accounting performance model, cocotb/Verilator
verification down to every leaf module, and an FPGA synthesis flow — running
real quantized MobileNetV2 and AlexNet layers end to end.

![GoSPA architecture](docs/GoSPA.jpg)

## Architecture overview

CNNs after pruning and ReLU are 70–90 % zeros on both the weight and
activation side, but a dense accelerator still spends a cycle on every
multiply. GoSPA skips the zero work through **on-the-fly intersection**
instead of a dedicated intersection unit:

- Every nonzero activation is tagged with a **Position ID (PID)** — which of
  the F×F kernel positions it meets — and a **Convolution ID (CID)** — which
  output pixel it contributes to.
- Each kernel's sparsity is captured in a **Weight Sparsity Pattern (WSP)**,
  an F²-bit mask of its nonzero positions. Since weights are known before
  inference, the WSP is static: matching an activation to a nonzero weight
  collapses to a single bitmask lookup.

The design (`rtl/V2/gospa.sv`) is an **Activation Processing Unit (APU)**
feeding an array of **Processing Elements (PEs)**:

- **APU Stage 1 — ID generation.** The input channel is held on-chip in CSR
  form, so only nonzeros are stored. A scanner walks the CSR structure,
  `csr_decode` recovers each entry's (x, y) coordinate, and an array of IDGen
  units enumerates every kernel position that overlaps it, emitting all valid
  (PID, CID) pairs into a bank of F² **FIFO-A** queues indexed by PID (the
  PID itself is never stored).
- **APU Stage 2 — WSP-gated routing.** The router drains the FIFO-A banks in
  PID order, packs up to `N_MULTS` same-PID activations into a *beat*, and
  broadcasts it toward all `N_PE` **FIFO-B** queues — gated per PE by that
  PE's WSP bit for the current PID. Activations paired exclusively with zero
  weights are dropped in flight: the intersection costs one AND gate per PE,
  and no zero operand ever reaches a PE.
- **PE array.** Each PE holds *one* filter kernel ("1 weight × M
  activations"): its nonzero weights sit PID-sorted in a local SRAM
  (`pe_mem`, double-buffered, WSP derived in hardware from the loaded
  weights), a Curr/Next window (`pe_fetch`) matches the incoming beat PID and
  broadcasts the selected weight to all `N_MULTS` multiplier lanes, and each
  lane accumulates into its own CID-indexed accumulator bank — one output
  channel per PE per pass, `N_PE × N_MULTS` useful MACs per cycle at full
  occupancy.

Because ID matching is deterministic arithmetic rather than a runtime search,
the whole path is pipelined with no intersection stalls on the critical path.
Every architectural dimension (`H, F, S, N_PE, N_MULTS, FIFO_D, …`) and the
front-end width knobs (`FILL_W, STAGE1_BATCH, S2_BEATS, DRAIN_W`) are
synthesis-time parameters.

The full design discussion is in [reports/final_report.pdf](reports/final_report.pdf);
the block-level diagram source is [docs/GoSPA.drawio](docs/GoSPA.drawio).

## Results

**FPGA** (Vivado 2024.2, Kria KV260, `N_PE=8 × N_MULTS=4`, post-route —
details in `rtl/V2/synth/RESULTS.md`): 42,698 LUT (36.5 %), 72,610 FF, 0 DSP,
0 BRAM, Fmax ≈ 171 MHz. Zero DSPs because the design uses a custom pipelined
multiplier (`common/arith/mult_pipe.sv`).

**RTL simulation** (golden-checked against the functional model, 100 MHz):

| Workload | Config | Cycles | Multiplier util. | Throughput |
|---|---|---|---|---|
| MobileNetV2, 52 layers end-to-end | 32 PE × 4 | 669,094 | 37.8 % | **149.5 fps** |
| MobileNetV2, 52 layers end-to-end | 8 PE × 4 | 1,747,377 | 57.9 % | 57.2 fps |

## Repository layout

```
rtl/V2/           SystemVerilog RTL
  apu/              stage1/ (CSR scan, decode, ID gen) + stage2/ (routing) + FIFOs
  pe/               pe, pe_array, pe_mem, pe_fetch, pe_lane
  common/           fifo, sram, arith/ (mult_pipe, mac_pipe, rca_add)
  gospa.sv          top level (APU + PE array)
  synth/            Vivado flow, resource sweep, RESULTS.md
testing/          cocotb + Verilator testbenches, mirrors rtl/
  ref/              MobileNetV2 / AlexNet reference tensors
  artifact/         one-command reproduction runner (run.sh)
sw/               Python models
  functional.py     golden functional model (bit-exact RTL reference)
  perf_model.py     cycle-accounting performance model + sim.py driver
docs/             architecture diagram (drawio + jpg)
reports/          final report, poster, analysis notes
```

## Reproducing the results

Environment: Vivado 2024.2, Verilator ≥ 5.0, Python 3.12 with cocotb, numpy,
torch, matplotlib (activate the virtualenv first — the cocotb Makefiles set
`PYTHONPATH` to `sw/` and `testing/ref/` automatically).

```bash
# One-command RTL-measured results (each run golden-checks in-simulation)
cd testing/artifact
bash run.sh smoke          # functional suite (~2 min)
bash run.sh mobilenet      # MobileNetV2 conv1, real 80×80 input (~2 min)
bash run.sh mobilenet-e2e  # full 52-layer MobileNetV2 at 32×4 (slow)
bash run.sh alexnet        # AlexNet conv3/4/5 tiled layers (~30–45 min)

# FPGA synthesis + place & route
cd rtl/V2/synth/vivado && vivado -mode batch -source run_synth.tcl

# Software performance model
cd sw && python3 sim.py            # network-level model
cd sw && python3 sim.py --sweep    # density sweep
```

Per-module testbenches are listed in [testing/README.md](testing/README.md).

## Credits

Built as a five-person course project (ECE 493/720 — Machine Learning
Hardware Systems, University of Waterloo) with Sara Ahmad, Fred Huang,
Adil Kazimov, and Emon Sarkar. The GoSPA architecture is from Deng *et al.*,
ISCA 2021; this repository is an independent educational reimplementation.
