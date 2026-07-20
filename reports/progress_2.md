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
- **Profiling performance model** done based on a sweep of activation/weight density, determined discrepancy of bottleneck between paper & implemented architecture.
  - Collaborated with other team members to bring up full workloads on performance model in a PE -> Pass -> Layer -> Sim hierarchy
  - Added configurable synthetic workload where weight and activation density can be controlled by `--dw` and `da`.
  - Verified the observation that multiplier utilization scales linearly with almost a factor of 1 with weight density.
- **Performance model refinement** according to newly proposed architecture (See section 3).
  - Added parallel mode to APU software model to accurately capture the hardware behavior (update on previous assumption)
  - Added `act` option for `--arch` in performance model to demostrate updated dataflow: one kernel/WSP per PE, `M` activations drained from FIFO-A's to FIFO-B's per cycle, observed increased per-lane utilization calculated as `useful_macs/pe_cycles`.

Next:
 - Modify lane utilization accounting logic to account for total compute cycles **not** assuming PEs are the bottleneck.
 - Run experiments on design space exploration on the number of lanes per PE, plot number of lanes vs. util & wall-clock time, compare with original architecture.

---

### 1.3 Mahdi Hassen

Ahead of schedule

Progress:
- Stitching together RTL for the goSPA module, completing the initial top-level:
  - updated PE sv to match architecture
  - Added SRAM to PE and APU
  - modify APU TB to measure perf metrics
  - Changed all modules to have synchronous reset
  - Added pipeline stage in PE
- Verified compared to functional model on MobileNetV2 Layer 1
  - Extracted Utilization numbers for MobileNetV2
  - Added various verification tests including end to end classification of a img (class: granny smith apple)
- Wrote mini-complier which maps larger input computation to smaller GoSPA module

Chnages to plan: 
- No longer designing modules for intermediate layers, will be done inside the TB or in SW 
- Added mini-compiler to map more general computation into GoSPA modules

Next Steps:
- Evaluate and assist with implementing the new architecture interpretation 
- Map various CNN architectures and extract perf metrics
- Work on optimizing architecture
- Write new mini-compiler to map CNN architectures to physical GoSPA modules




---

### 1.4 Sara Ahmad

**Completed:**

- **MobileNetV2 weights + simulation testing (`sw/workloads/mobilenetv2.py`).** Added the full
  MobileNetV2 layer list (Table 2: initial conv, all 7 inverted-residual bottleneck groups
  expand/depthwise/project, final 1x1 conv, classifier) so the perf model runs the network
  end-to-end via `sim.py --net mobilenetv2`.
- **`new_arch` branch verification.** Verified the new architectural design change (one kernel/PE
  dataflow); checking the new PE cycle counts and utilization computations which supported
  the team in bringing the `new_arch` modifications into master.
- **Functional model of the new architecture (`sw/functional.py`).** Added `pe_process_actparallel`,
  a one-kernel-per-PE, wide-activation-feed PE model, and wired it into the existing
  `goSPA_route`/`goSPA_run` pipeline alongside the other PE interpretations. 
- **New architecture testing (`sw/test_functional_actparallel.py`).** Wrote functional tests for the
  activation-parallel, stationary-weight PE (`pe_process_actparallel`): stationary-weight grouping
  across full and tail activation batches, confirmed the accumulated outputs are identical regardless
  of `feed_width` (varying it only changes the scheduling of activations into the PE, not the
  computed result), the v1 pipeline (`goSPA_run`) against the dense-convolution reference, the
  empty-stream edge case, and input validation (`feed_width`, missing stationary weight, PID
  ordering, duplicate sparse weights).

**Next:** Since the new architecture identified the APU as the new bottleneck, build a small-scale
model to verify this without relying on the current performance model's assumptions, and design a
new APU stage architecture to reduce that bottleneck.

---

### 1.5 Adil Kazimov
Ahead of schedule 

Note:
PE module had 3 versions. V1.0, V2.0 and V2.1. V1.0 was completed by Emon and later upgraded by Mahdi to V2.0. PE V2.1 was written by me. ("Reworked PE logic" commit)

Progress:
- Set up Cadence ASIC Flow using ASAP7 PDK (Files can be found in adil_testing branch)
  - Wrote custom Cadence Genus TCL script
  - Ran synthesis on pe_array and gospa top level. Set up Xcelium for post-synthesis simulation and verified pe_array. 
  - Wrote TCL scripts for Innovus PAR. Patched numerous issues that were preventing PAR from completing.
  - Manually defined the floorplan, power rings and other things to run PAR.
  - Ran 16 hour pe_array PAR that resulted in a design with over 10,000 DRC violations.
  - Tried to patch ASAP7 further to resolve DRC violations but was halted by little to no support for this PDK in Innovus.
  - Coordinated the switch to OpenLane with Emon, leaving Cadence Flow as a backup plan, given that Genus synthesis works correctly.     

- PE Module (PE V1.0)
  - Completed pe_acc.sv that was used by Emon to finish PE V1.0

- PE Module Rework (PE V2.0 --> V2.1)
  - Performed significant PE module refactoring and optimization. 
  - Replaced 1 single SRAM with SRAM banks and implemented correct dataflow.
  - Removed SRAM side arbiter and simplified the overall logic of PE.
  - Patched Makefile to make it easier to run codes
  - Introduces performance measurement tests into the testbench.
  - Obtained hardware accurate performance metrics, such as multiplier utilization, for pe.sv. Performance log is generated by running `make test_pe`
  - Found a significant discrepancy between the performance metrics of our proposed architecture and those reported in the paper.
  - Researched a new architecture to replace our current union-WSP architecture.

- New Architecture (The codes can be found under new_arch branch)
  - Proposed a new "1 weight x 4 activations" architecture that requires APU and PE restructure. 
  - Modified performance and functional models to implement this new architecture and obtained new performance metrics that were similar to the ones in the paper.
  - Brainstormed optimization and parallelization ideas to further improve performance of our GoSPA architecture. 
  - Passed my modifications to the software team for review. 
  
Next Step:
  - Start PE and APU RTL rewrite for the new architecture, spliting the work between group members.
  - Complete migration to OpenLane ASIC Flow.

## 2. Milestones — met / delayed / skipped / added

Milestones are taken from the per-member table in `reports/plan.md`. Period covered: June 19 -> July 19. New milestones are tagged with `Added`

| Member | Planned task | Status | Notes |
|---|---|---|---|
| Mahdi | Intermediate CNN layers (maxpooling, fully connected) in HW | **Skipped** | No longer designed as dedicated RTL modules — handled in the TB / SW instead (see Section 3); effort redirected to top-level RTL integration |
| Mahdi | Stitch RTL into the initial GoSPA top-level | **Met / exceeded** | Completed the initial top-level: updated `pe.sv` to match the architecture, added SRAM to the PE and APU, converted all modules to synchronous reset, added a pipeline stage in the PE, and modified the APU TB to measure perf metrics |
| Mahdi | Verify against the functional model | **Met** | Verified against the functional model on MobileNetV2 Layer 1, extracted MobileNetV2 utilization numbers, and added verification tests including end-to-end classification of an image (class: granny smith apple) |
| Mahdi | Mini-compiler | **Added** | Wrote a mini-compiler that maps a larger input computation onto the smaller GoSPA module |
| Fred | Finish Perf Model | **Met** | Perf model for first-generation architecture done, second-generation largely implemented with a few details pending discussion |
| Fred | PE testing | **Met** | Ran sweep on activation and weight densities, confirmed discrepancy in lane utilization together with the TB results from other group members |
| Fred | Design Space Exploration | **Added** | With the new dataflow, run experiments with various hardware configurations to identify the best design specs |
| Sara | Compare perf APU with HW APU | **Redirected** | Effort redirected to verifying the `new_arch` interpretation (PE cycle counts, utilization) that supported merging it into master, adding its functional model and tests, and adding MobileNetV2 weights + simulation testing |
| Sara | Verification and testing of the `new_arch` model | **Added** | Verified the new architecture's PE cycle counts and utilization computations against the functional model, added the functional model of the new architecture, and tested it -- work that supported the team in bringing the `new_arch` modifications into master |
| Emon | `apu.sv` with router & FIFOs | **Met** | APU top completed within the team; effort redirected to the PE |
| Emon | APU full TB | **Met** | Full-APU test completed within the team |
| Emon | Help finish PE remaining modules | **Met / exceeded**: built the full PE + PE array, a reusable pipelined arithmetic unit integrated into the PE, and the FPGA implementation flow (post-route Fmax ~226 MHz) |  |
| Adil | Code pe_acc.sv | **Met** |  |
| Adil | Code PE | **Skipped** | PE V1.0 RTL + Testbench done by Emon, V2.0 done by Mahdi
| Adil | Rework PE | **Added** | PE V2.1 release. Reworked and optimized PE for union-WSP architecture
| Adil | Quantify Perf of PE | **Added** | Measured performance in the testbench
| Adil | Devise new architecture | **Added (In Progress)** | Working on scaling APU to 4 activations
| Adil | Setup ASIC flow | **Added (In Progress)** | Working on setting up OpenLane with Emon


## 3. Refinements to the initial plan

The biggest change in the plan is changing out interpretation of the GoSPA architecture. The paper itself is ambiguous on certain aspects of the architecture, specifically how multiple kernels map within a PE. Our interpretation (dubbed V2 interpretation in functional.py) relies on each kernel mapping to one multiplier, and thus multiple kernels per PE. Upon implementing our interpretation in both the performance model and the RTL, we determined that our interpretation of the architecture does not yield the same results from the paper in terms of multiplier utilization.

When evaluating an alternative interpretation in which a PE holds a single kernel, we achieve an increased multiplier utilization value. It's worth noting that this implementation seems to contradict the paper's claim that one activation is fed into a PE per cycle. To keep the PE fed, we must push in multiple activations every cycle. This contradiction was the main reason for the previous interpretation. One of the key challenges with this new dataflow is that with the PE array holding `M`x less kernels at a time, the activations will need to be streamed `M`x more times. It's hard to qualitatively decide whether this will result in an increased performance because the comput density within each PE has also increased (i.e., PE guraranteed 4 ops/cycle when active). To address this concern, more design space exploration will be needed as this is an optimization problem with `N_PE` and `M_LANES` as variables.

Since the team is very ahead of schedule, we see value in spending time to implement this new interpretation in RTL and comparing it with our previous one.

The downstream plan remains the same: new top level done by July 26, running full workloads by August 2.

Other changes include:

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

4. **Mini Compiler.** Add a small component to flexibly map CNN architectures onto GoSPA modules
