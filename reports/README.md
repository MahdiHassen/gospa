# GoSPA — Sparse CNN Accelerator (Team 19)

A SystemVerilog reproduction of **GoSPA: An Energy-Efficient High-Performance
Globally Optimized Sparse Convolutional Neural Network Accelerator** (Deng *et al.*,
ISCA 2021). GoSPA exploits two-sided *unstructured* sparsity via on-the-fly
intersection: every nonzero activation is tagged with a **Position ID (PID)** and a
**Convolution ID (CID)**, so matching a nonzero activation to a nonzero weight
collapses to a single bitmask lookup (the **Weight Sparsity Pattern**, WSP) instead
of a dedicated intersection unit.

This repo holds the RTL, the verification infrastructure, the FPGA synthesis flow,
and the software functional/performance models, with scripts to reproduce the
reported results. All paths below are relative to the repository root.

---

## Repository layout

```
rtl/                SystemVerilog RTL — two architectures, kept side by side
  V2/                 current design: ONE kernel per PE, weight-stationary
    apu/
      stage1/           act_sram_scanner, csr_decode, position_encode, idgen,
                        zero_act, apu_stage1   (CSR scan + ID generation → FIFO-A)
      stage2/           routing, apu_stage2    (WSP-gated multicast → FIFO-B)
      apu.sv            Stage-1 + Stage-2 + FIFOs
    pe/                 pe, pe_array, pe_mem, pe_fetch, pe_lane
    common/             fifo, sram, arith/{mult_pipe, mac_pipe, rca_add}
    gospa.sv            top level (APU + PE array)
    synth/              Vivado flow + resource sweep + RESULTS.md
  V1/                 earlier design: MULTIPLE kernels per PE, union-WSP routing
    apu/                + wsp_file.sv, act_sram_scanner.sv at apu/ level
    pe/                 pe, pe_acc, pe_array
    common/             fifo, sram, arith/  (same custom multiplier as V2)
    synth/              Vivado flow (matched to V2's for a fair comparison)

testing/            RTL verification — cocotb + Verilator, mirrors rtl/
  common/ arith/ apu/ pe/ gospa/     per-module and full-accelerator testbenches
  V1/                                V1 testbenches (own vendored functional model)
  ref/                               MobileNetV2 / AlexNet reference tensors
  artifact/                          reproduction runner (run.sh) + results/

sw/                 Software models
  functional.py       golden functional model — bit-exact reference for the RTL
  perf_model.py, perf_pe.py, sim.py, layer.py, config.py    cycle-accounting model
  workloads/          AlexNet, MobileNetV2 layer descriptors

reports/            plan.md, progress_*.md, analysis notes, this README
```

### V3

Current development lives on the **`V3_Final`** branch. It drops the side-by-side
`rtl/V1` / `rtl/V2` split above in favor of a single flat RTL tree — `rtl/apu`,
`rtl/pe`, `rtl/common`, `rtl/gospa.sv`, `rtl/controller.sv` — with a matching `testing/gospa` testbench tree.

The artifact runner is `testing/artifact/run.sh`, documented in
`testing/artifact/README.md`:

```bash
cd testing/artifact
bash run.sh alexnet      # 5-layer AlexNet,   N_PE=8  x N_MULTS=4 = 32 mult
bash run.sh mobilenet    # 52-layer MobileNetV2, N_PE=16 x N_MULTS=4 = 64 mult
```

Each run golden-checks against the functional model and copies its report to
`testing/artifact/results/`, timestamped, for comparison against the checked-in
reference files in `testing/gospa/`.

---


### V1 vs V2

Both instantiate **32 multipliers** (`N_PE × N_MULTS`), but arrange them differently:

| | V1 | V2 |
|---|---|---|
| Kernels per PE | `N_MULTS` (4) | 1 |
| Output channels per pass | `N_PE × N_MULTS` = 32 | `N_PE` = 8 |
| Routing gate | **union** of the PE's lane WSPs | that PE's single WSP (exact) |
| Beat | 1 activation, fanned to all lanes | `N_MULTS` activations sharing one PID |
| Front-end width knobs | none (1 activation/cycle) | `STAGE1_BATCH`, `FILL_W`, `S2_BEATS`, `DRAIN_W` |
| Multiplier | `mult_pipe` (same as V2) | `mult_pipe` |

Both use the **same custom pipelined multiplier**, so a V1-vs-V2 comparison isolates
the *dataflow* rather than the arithmetic implementation. V2 is the current design and
the source of the reported results.

---

## Development environment (Applies to V3 as well)

| Tool | Version used | Purpose |
|---|---|---|
| Vivado | 2024.2 | FPGA synthesis + place & route (area / Fmax) |
| Verilator | 5.042 | primary RTL simulator |
| Icarus Verilog | 12.0 | alternative simulator (default in a few dirs) |
| Python | 3.12 | models, cocotb tests, plotting |
| cocotb | 1.9.2 | RTL <=> Python co-simulation |
| numpy, torch, matplotlib | — | reference tensors, quantization, plots |

**Activate the Python environment before running anything** — cocotb, numpy, torch and
matplotlib are all imported from it, and a bare system `python3` will not have them.
The cocotb Makefiles set `PYTHONPATH` to `sw/` and `testing/ref/` automatically.

> **`noexec` filesystems:** Verilator compiles the DUT to a native binary under
> `sim_build/`. If the checkout lives on a `noexec` mount, redirect the build:
> `SIM_BUILD=$HOME/.cache/gospa_build make …`
>
> Shell scripts below are invoked as `bash <script>` so they work regardless of the
> executable bit on the checkout's filesystem.

---

## Reproducing the results

### 1. FPGA synthesis — resources and frequency

Out-of-context synthesis + place & route on the Kria KV260 (`xck26-sfvc784-2LV-c`).
Both flows use the **same** board, clock target (1 ns / 1 GHz, deliberately
over-constrained so the tool reports true Fmax) and array configuration
(`N_PE=8, N_MULTS=4, H=8, F=3, S=1, FIFO_D=64, DATA_W=16, ACC_W=32`), so the two
results are directly comparable. Edit the knobs at the top of each TCL to change it.

```bash
# V2 (current architecture)
cd rtl/V2/synth/vivado && vivado -mode batch -source run_synth.tcl

# V1 (earlier architecture)
cd rtl/V1/synth/vivado && vivado -mode batch -source run_synth.tcl
```

Each writes, in its own directory:
`utilization_route.rpt` (LUT / FF / DSP / BRAM), `timing_route.rpt` and
`critical_paths.rpt` (Fmax = 1 / (target period − WNS), critical path).

**Resource sweep** over `N_PE`, `N_MULTS` and `FIFO_D`, plus the LUT-scaling plots:

```bash
cd rtl/V2/synth/sweep
bash run_sweep.sh                 # synthesis only
RUN_IMPL=1 bash run_sweep.sh      # full place & route
python3 plot_sweep.py             # → lut_sweep.pdf, lut_vs_{N_PE,N_MULTS,FIFO_D}.pdf
```

> `run_sweep.sh` **truncates** the committed `sweep_results.csv` before its first
> Vivado run — back it up first if you want to keep the reference data.

Signed-off numbers live in `rtl/V2/synth/RESULTS.md`. Vivado reads the SystemVerilog
directly; no conversion step is needed.

### 2. RTL verification

Every cocotb testbench golden-checks the RTL against the Python functional model
(`sw/functional.py`), which is itself checked against a dense convolution and against
PyTorch — so a passing simulation is a verified result.

**One-command reproduction.** `testing/artifact/run.sh` regenerates the RTL-measured
results; each target golden-checks inside the simulation (a mismatch aborts the run)
and appends to `testing/artifact/results/`:

```bash
cd testing/artifact
bash run.sh smoke          # V1 + V2 functional suites
bash run.sh v1v2           # RTL-measured V1 vs V2 density comparison
bash run.sh mobilenet      # MobileNetV2 conv1, golden + performance, both versions
bash run.sh mobilenet-e2e  # full 52-layer MobileNetV2 (V2, 32 PEs × 4 lanes)
bash run.sh alexnet        # AlexNet conv3/4/5 tiled layers (V2)
bash run.sh all            # smoke + v1v2 + mobilenet
```

**Per-module tests (V2).** Run from each block's directory. `SIM=verilator|icarus`
selects the simulator; `MODULE=` selects the testbench where a directory has several:

```bash
cd testing/common      && make                          # FIFO (SystemVerilog TB, 10 tests / 92 checks)
cd testing/common/sram && make                          # dual-port SRAM
cd testing/arith       && make MODULE=test_mult_pipe    # pipelined signed multiplier
cd testing/arith       && make MODULE=test_mac_pipe     # fused multiply-add
cd testing/apu/idgen   && make                          # CID / PID generation
cd testing/apu/stage1  && make MODULE=test_csr_decode   # CSR decode
cd testing/apu/stage1  && make MODULE=test_zero_act     # zero-activation filter
cd testing/apu/stage1  && make MODULE=test_apu_stage1   # Stage-1 chain
cd testing/apu/full    && make                          # full APU
cd testing/apu/full    && make mobilenet                # APU on real MobileNet conv1
cd testing/pe          && make pe                       # single PE
cd testing/pe          && make array                    # PE array
cd testing/gospa       && make test_gospa               # full accelerator
cd testing/gospa       && make mobilenet                # MobileNetV2 first conv
cd testing/gospa       && make apple                    # real 80×80 MobileNetV2 conv1
```

**Per-module tests (V1).** Makefiles point at `rtl/V1` and use a vendored functional
model, so V1 tests never pick up the V2 model:

```bash
cd testing/V1/gospa       && make MODULE=test_gospa       # full accelerator
cd testing/V1/gospa       && make mobilenet               # MobileNetV2 first conv
cd testing/V1/pe          && make MODULE=test_pe          # single PE
cd testing/V1/apu/full    && make                         # full APU
cd testing/V1/apu/idgen   && make                         # CID / PID generation
cd testing/V1/apu/routing && make                         # Stage-2 routing
cd testing/V1/apu/stage1  && make MODULE=test_apu_stage1  # Stage-1 chain
```

Layer and array geometry are overridable, e.g.
`make test_gospa H=10 F=3 S=2 N_PE=8 N_MULTS=4`. Note `testing/arith` and
`testing/V1/pe` default to **Icarus**; the other directories default to Verilator.

Sweep targets also exist: `make sweep_pe` / `sweep_array` (PE), `make sweep_gospa` /
`conv5` (full accelerator), `make sweep_s1` (Stage-1), `make sweep_apu`, `make sweep`
(idgen, SRAM).

### 3. Software models

`sw/functional.py` is the golden functional model — it reimplements the dataflow
(CSR decode, PID/CID generation, FIFO-A binning, WSP-gated routing, CID-indexed
accumulation) and models both the V1 and V2 PE mappings. `sw/perf_model.py`,
`perf_pe.py` and `sim.py` form the cycle-accounting performance model used for
design-space exploration; it attributes each layer to a Stage-1 / Stage-2 / PE /
memory bottleneck, which is what motivated widening the Stage-2 router.

```bash
cd sw && python3 sim.py            # network-level performance model
cd sw && python3 sim.py --sweep    # activation/weight density sweep
```

---

## Reference results

**FPGA, V2** (`rtl/V2/synth/RESULTS.md`; KV260, `N_PE=8, N_MULTS=4`, post-route):

| Resource | Used | Available | Utilization |
|---|---|---|---|
| LUT | 42,698 | 117,120 | 36.5 % |
| FF | 72,610 | 234,240 | 31.0 % |
| DSP | 0 | 1,248 | 0 % |
| BRAM | 0 | 144 | 0 % |

Fmax ≈ **171 MHz** (WNS −4.855 ns against a 1 ns target). Zero DSP is a direct
consequence of the custom `mult_pipe`; at these depths the FIFOs and accumulator
banks infer flip-flops rather than block RAM.

**RTL simulation, V2** (`testing/artifact/results/`, golden-checked, 100 MHz):

| Workload | Config | Cycles | Utilization | Throughput |
|---|---|---|---|---|
| MobileNetV2, 52 layers, end-to-end | 32 PE × 4 | 669,094 | 37.8 % | 6.69 ms → **149.5 fps** |
| MobileNetV2, 52 layers, end-to-end | 8 PE × 4 | 1,747,377 | 57.9 % | 17.47 ms → 57.2 fps |

Per-layer-class utilization at 32×4: conv 72.1 %, pointwise 64.9 %, depthwise 3.0 %
(depthwise is 44 % of runtime and is the current bottleneck).

**Resource sweep** (`rtl/V2/synth/sweep/sweep_results.csv`, synthesis-only): LUT
scales roughly linearly with `N_PE` (22,822 → 67,878 for 2 → 16) and with FIFO depth
(32,508 → 55,820 for 16 → 128), but **super-linearly with `N_MULTS`** (19,328 →
108,193 for 1 → 8) — the multiplier lanes are the dominant area knob.

> **V1-vs-V2 comparison — regenerate before citing.** `rtl/V1/pe/pe.sv` was recently
> changed to instantiate the same `mult_pipe` as V2 (it previously used a `*`
> operator that inferred DSP blocks). This makes the comparison fair, but every
> previously recorded **V1** cycle count is stale. Regenerate with
> `bash run.sh v1v2` and `bash run.sh mobilenet`, and produce the matching V1 FPGA
> point with `rtl/V1/synth/vivado/run_synth.tcl`, before quoting any V1 number.
> V2 numbers are unaffected.
