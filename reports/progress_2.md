# ECE 493 / 720 Course Project — Progress Report 2

**Team 19 — GoSPA: An Energy-efficient High-performance Globally Optimized Sparse CNN Accelerator** (ISCA 2021)


## 1. Per-member progress

### 1.1 Emon Sarkar

**Status: ahead of schedule.** Completed and verified the APU Stage-1 datapath, built the Processing
Element and PE array, designed a reusable deeply-pipelined arithmetic unit and integrated it into the
PE, and brought up the FPGA synthesis/implementation flow with post-route results. This covers the
July PE milestone early and starts the synthesis milestone ahead of plan.

**Completed:**

- **APU Stage-1 integration (`rtl/apu/stage1/apu_stage1.sv`) — verified.** Finished the Stage-1 top
  (`csr_decode -> zero_act -> position_encode -> idgen array -> FIFO-A bank`) with an all-or-nothing
  fan-out join, and verified it end-to-end against the functional model in
  `testing/apu/stage1/test_apu_stage1.py` (7 layer configs x 5 cases pass).
- **Processing Element (`rtl/pe/pe.sv`, `rtl/pe/pe_array.sv`) — verified vs dense convolution.** Built
  the PE (PID-ordered weight store with a Curr/Next reuse window, multiply, CID-indexed accumulator,
  consuming the FIFO-B stream) and the N_PE PE array wired onto the APU's FIFO-B ports. Verified
  against true dense convolution (`conv2d_reference`) across kernel sizes, strides, and sparsity
  levels, single-PE and multi-channel. This also surfaced a correctness gap in the functional model's
  `pe_process` (it multiplies by the wrong weight when a non-zero weight receives no activation --
  ~14% of random sparse cases disagree with dense convolution); the RTL PE handles that case
  correctly. Flagged to the software team (see Section 3).
- **Reusable deeply-pipelined arithmetic unit (`rtl/common/arith/`) — verified.** Designed a custom
  signed multiplier (`mult_pipe`) and fused multiply-add (`mac_pipe`) built structurally from
  explicit partial-product generation and a registered ripple-carry adder tree (`rca_add`) -- no
  `*`/`+` operators and no SystemVerilog functions -- parameterized by operand width and portable
  across FPGA and ASIC. Fully pipelined (one result/cycle) with an aux passthrough that keeps the
  output-pixel index aligned to the product at any depth. Verified with cocotb against a reference
  multiply/MAC over multiple widths and corner cases (`testing/arith/`).
- **Arithmetic-unit integration into the PE datapath.** Replaced the PE lane's multiply stage with
  the pipelined `mult_pipe` (CID carried on the aux passthrough; the accumulator drain gated on the
  pipeline's in-flight status), and added a 1-cycle `mac_fire` probe for the MAC performance counter.
  The single PE and the PE array pass all correctness cases against the dense-convolution reference.
- **FPGA synthesis/implementation flow + results (`rtl/synth/`).** Set up an out-of-context Vivado
  synthesis and place-and-route script (`rtl/synth/vivado/run_synth.tcl`) for the GoSPA top and ran it
  to post-route signoff on the Kria KV260. Results recorded in `rtl/synth/RESULTS.md`: maximum
  frequency ~226 MHz (post-route), critical path in the FIFO-B read / CID-indexed accumulate control
  (the multiplier is off the critical path), and resource utilisation (LUT 31%, FF 25.5%).

**Next:** ASIC implementation (OpenLane / sky130) for area and power, and continued verification and
metric gathering toward the final report.

---

### 1.2 Fred Huang

---

### 1.3 Mahdi Hassen

---

### 1.4 Sara Ahmad

---

### 1.5 Adil Kazimov


## 2. Weekly milestones — met / delayed / skipped

Milestones are taken from the per-member table in `reports/plan.md`. Period covered: June 19 -> July 17.

### Sub-period ending July 5

| Member | Planned task | Status | Notes |
|---|---|---|---|
| Fred |  |  |  |
| Mahdi |  |  |  |
| Sara |  |  |  |
| Emon | `apu.sv` with router & FIFOs | **Met** | APU top completed within the team; effort redirected to the PE |
| Adil |  |  |  |

### Sub-period ending July 12

| Member | Planned task | Status | Notes |
|---|---|---|---|
| Fred |  |  |  |
| Mahdi |  |  |  |
| Sara |  |  |  |
| Emon | APU full TB | **Met** | Full-APU test completed within the team |
| Adil |  |  |  |

### Sub-period ending July 19 (in progress at time of report)

| Member | Planned task | Status @ Jul 17 |
|---|---|---|
| Fred |  |  |
| Mahdi |  |  |
| Sara |  |  |
| Emon | Help finish PE remaining modules | **Met / exceeded** — built the full PE + PE array, a reusable pipelined arithmetic unit integrated into the PE, and the FPGA implementation flow (post-route Fmax ~226 MHz) |
| Adil |  |  |


## 3. Refinements to the initial plan

1. **Synthesis pulled forward, FPGA-first.** Rather than starting with the ASIC (OpenLane) flow at the
   Aug 9 milestone, an FPGA implementation (Vivado, Kria KV260) was brought up first -- it stands up
   quickly and gives post-route timing, critical path, and utilisation. This established a baseline
   (~226 MHz) and identified the CID-indexed accumulator read-modify-write as the timing limiter (the
   multiplier is not on the critical path). The ASIC/OpenLane flow follows for area and power.

2. **Reusable custom arithmetic IP.** A deeply-pipelined, structural, parameterized multiply/MAC unit
   was built as a self-contained block (no operators, no functions; FPGA- and ASIC-portable) and
   integrated into the PE. It decouples the arithmetic from the datapath and enables MAC-level
   optimisation exploration independent of any later architecture change.

3. **Functional-model PE correctness.** The RTL PE is verified against dense convolution rather than
   the functional model, because the model's `pe_process` mishandles the weight-skip case for sparse
   activations. Recommend aligning the software model to the corrected (skip-handling) dataflow so the
   performance model's sparse-case counts stay accurate.

4. **No change to downstream milestones.** Top-level integration, full-CNN bring-up, and final
   verification/metrics (July 26 -> Aug 9) remain as planned.
