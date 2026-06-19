# ECE 493 / 720 Course Project — Progress Report 1

**Team 19 — GoSPA: An Energy-efficient High-performance Globally Optimized Sparse CNN Accelerator** (ISCA 2021)


## 1. Per-member progress

### 1.1 Emon Sarkar

**Status: ahead of schedule.** Completed the full APU Stage-1 input datapath in RTL, from CSR decode
through to the FIFO-A bank, and verified the assembled stage against the team functional model (our
first RTL vs SW co-simulation). The June 21 integration milestone is done early.

**Completed:**

- **`rtl/common/fifo.sv`** + **`testing/common/fifo_tb.sv`** -- parameterized synchronous FIFO (used
  for both FIFO-A and FIFO-B) with a self-checking testbench. **10 tests / 92 checks, all passing**
  (reset, full/empty, overflow/underflow protection, FIFO ordering, simultaneous push+pop, the
  paper's F=2,H=3,S=1 scenario, randomized stress). Runs under Verilator and Icarus. (June 1, ahead
  of its June 14 slot.)
- **`rtl/apu/stage1/csr_decode.sv`** -- FSM decoder turning a CSR sparse activation matrix into a
  `(value, x, y)` stream, one tuple/cycle, with valid/ready backpressure and empty-row handling. HW
  form of the model's `csr_to_positional`.
- **`rtl/apu/stage1/zero_act.sv`** -- combinational zero-activation filter (HW form of the model's
  `if val == 0: continue`), so zeros never reach the IDGen array.
- **`rtl/apu/stage1/apu_stage1.sv`** *(new)* -- Stage-1 top that ties the chain together,
  `csr_decode -> zero_act -> position_encode -> idgen array (G x G) -> FIFO-A bank (F^2 slots)`. It
  adds an **all-or-nothing fan-out join** so the front end stalls until every targeted FIFO-A can
  accept (IDGen is a combinational G^2-wide fan-out with no ready handshake). It exposes the F^2
  FIFO-A read ports for Stage 2.
- **cocotb unit tests** -- `test_csr_decode.py` (8 tests), `test_zero_act.py`, and the new
  **`test_apu_stage1.py`** (full-stage co-simulation, below).
- **`.gitignore`** plus the project RTL/testbench conventions (layout mirroring `rtl/`,
  Verilator/Icarus + cocotb flow) documented in `testing/README.md`.

**Verification (RTL vs SW co-simulation):** `test_apu_stage1.py` drives a CSR activation matrix into
`apu_stage1`, drains all F^2 FIFO-A ports, and checks each slot against the golden built from Mahdi's
`functional.py` front end (`csr_to_positional -> zero_act_filter -> axy_to_pcid -> pcid_to_cid_pid ->
route_to_fifo_a`). **7 layer configs x 5 tests** (empty, single-nonzero, dense, randomized, paper
toy) all pass under Icarus, including the paper's F=2,H=3,S=1 example. This confirms Stage-1 emits
the correct PID-binned `(Axy, CID)` FIFO-A contents. Note Stage-1 does not apply WSP; that is done in
Stage 2, which matches both the RTL and `functional.py`.

**Next (July):** Stage-2 routing module `apu.sv` (per-PE FIFO-B routing, Jul 5), then APU full TB
(Jul 12), assist PE modules (Jul 19), top-level (Jul 26), and synthesis area/power (Aug 9).

---

### 1.2 Fred Huang

**Completed:**

- **`sw/config.py`:** dataclass for integrated hardware configuration in performance model.
- **`sw/PERF_MODEL_PLAN.md`:** detailed breakdown components and tasks for performance model. Guidance for collaboration with other team members on software simulation.
- **`sw/perf_pe.py`:** performance accounting for a single PE. Designed to support 3 weight reloading models, ranging from `ideal` (no reloading penalty/reload completely hidden by computation) to `double_buffer` (overall reload penalty depends on previous PID's sparsity, visible only when compute takes shorter than reload). Also tracks multiplier utilization. Statistics reported as a dataclass `PEStats`.
- Modified **`sw/functional.py`** to expose an API for APU processing (`goSPA_route`), reusing logic for data routing in performance model. 
- **`sw/perf_model.py`:** performance accounting for a single pass on full GoSPA architecture, with `N_PE` PEs and `M` multipliers per PE. Records cycles for `stage1`, `stage2`, `pe` and `mem`, determines the system bottlenect as well as lane utilization, and reports with a dataclass `PassStats`.

**Next:** work with other team members to bring up a full end-to-end workload on the software model. 

---

### 1.3 Mahdi Hassen

**Status: met early** (planned completion June 21).

**Completed:**

- **Architecture block diagram** (`testing/GoSPA.drawio` + `GoSPA.jpg`) for the HW design.
- **`sw/functional.py`** (≈970 lines) — the golden functional model, built bottom-up: CSR decode ->
  positional stream -> `(Px,Py,Cx,Cy)` -> `(CID,PID)` enumeration -> zero filter -> FIFO-A routing ->
  WSP-gated FIFO-B broadcast -> PE MAC. Extended to the **multi-PE** accelerator (`goSPA_run`) and
  **multi-input-channel** convolution (`goSPA_multichannel`) with accumulators resident across Cin.
- **`testing/ref/alexnet.py`, `testing/ref/mobilenet.py`** — PyTorch reference models.
- **End-to-end validation:** functional model verified against PyTorch on MobileNetV2's first conv
  (RGB, 32 output channels, including the quantize/bias/ReLU/requant tail).

**Next (per plan):** intermediate layers needed for full-CNN testing (max-pool, fully-connected)
on the HW side, starting in the July sub-periods.

---

### 1.4 Sara Ahmad

_To be completed by Sara._

---

### 1.5 Adil Kazimov
**Status: Ahead of Schedule**


**Completed:**

- **`rtl/apu/stage1/idgen.sv`** -- CID/PID generator. Contains `idgen` + `idgen_unit` modules. The `idgen` bundle instantiates one `idgen_unit` per `(m,n)` in the GxG grid, each computing the bound checks and `CID=a·E+b`, `PID=c·F+d`.
- **`testing/apu/idgen/tb_idgen.py`** -- cocotb testbench that sweeps every activation coordinate `(x,y)`, drives the bundle, and checks the emitted `(CID,PID)` set against a self-contained convolution reference computed in the testbench. It runs across swept layer configs and both `PIPE` timing modes via the Makefile; 2 tests passing under Verilator and Icarus.
- **`rtl/apu/stage1/position_encode.sv`** -- combinational coordinate decomposition `Px=x%S`, `Py=y%S`, `Cx=x/S`, `Cy=y/S` that feeds `idgen`.
- **`rtl/apu/stage2/routing.sv`** -- APU Stage 2 routing module. Drains the `N_PID` FIFO-A lanes in order, one head per cycle, and all-or-nothing multicasts each `{ACT,CID}` into every FIFO-B whose WSP (MSB-first by PID) selects that lane, with start/busy/done framing and backpressure handling.
- **`testing/apu/routing/tb_routing.py`** -- cocotb testbench that cosimulates the RTL against the `sw/functional.py` golden model: per-PE FIFO-B contents are checked against `broadcast_to_fifo_b`. Extra tests are run to check timing properties that the model can't express (backpressure, no-idle-bubble, framing, empty lanes, MSB-first WSP orientation). 5 tests passing under Verilator and Icarus.
- **`rtl/common/sram.sv`** -- parameterized synchronous SRAM used across the design.
- **`testing/common/sram/tb_sram.py`** -- self-checking testbench against a Python reference memory. 5 tests passing (write/read-back, write-first, valid qualifiers, dual-port reads, reset) under Verilator and Icarus.

**In progress:**

- **`rtl/apu/pe/pe_acc.sv`** -- draft of the PE's accumulator block that compiles clean under Verilator and Icarus, with functional testbench to follow in the coming weeks.

**Note**: `dram.sv` module has been discarded. See Section 3 for details.

**Next**: Complete PE module following the team's schedule.


## 2. Weekly milestones — met / delayed / skipped

Milestones are taken verbatim from the per-member table in `reports/plan.md`. Period covered:
project start -> June 18.

### Sub-period ending June 7

| Member | Planned task | Status | Notes |
|---|---|---|---|
| Fred | Research perf-model requirements, determine abstraction level | **Met** | Concepts recorded in `PERF_MODEL_PLAN.md` |
| Mahdi | Finish HW-design block diagram | **Met** | `GoSPA.drawio` / `.jpg` committed Jun 1 |
| Sara | Break implementation into submodules for perf-model tests | | |
| Emon | HW constraints, comms protocol w/ Adil, lab-PC setup, `fifo.sv` | **Met (early)** | FIFO + TB done Jun 1 |
| Adil | Design memory `.sv` (sram, dram) |**Met** | DRAM module discarded |

### Sub-period ending June 14

| Member | Planned task | Status | Notes |
|---|---|---|---|
| Fred | Perf-model skeleton + abstraction level | **Met** | Reusing logic for data routing from `functional.py`, account for latency & utilization |
| Mahdi | Work on functional model | **Met / exceeded** | Multi-PE + multi-channel + PyTorch validation done |
| Sara | Model HW components at abstraction level | | |
| Emon | `csr_decode.sv`, `zero_act.sv` | **Met** | Both done Jun 8 with cocotb tests |
| Adil | `position_encode.sv`, `idgen.sv` | **Met**| |

### Sub-period ending June 21 (in progress at time of report)

| Member | Planned task | Status @ Jun 18 |
|---|---|---|
| Fred | Work on perf model | Working with other team members toward an end-to-end workload |
| Mahdi | Finish functional model | **Effectively done** — validated end-to-end |
| Sara | Work on perf model | |
| Emon | Merge into `apu_stage1.sv` with IDGen + FIFOs | **Met (early)** — `apu_stage1.sv` integrated and verified vs the SW model (7 configs × 5 tests pass) |
| Adil | `router.sv` | **Met**|



## 3. Refinements to the initial plan

1. **RTL <=> SW co-simulation pulled forward — and already started.** Rather than waiting for the
   July co-sim milestone, the functional model is now the golden reference for the RTL: `apu_stage1`
   is checked end-to-end against `functional.py`. This becomes the standard acceptance check for
   each new RTL block as it lands.

2. **HW sequencing worked; integration is complete early.** `apu_stage1.sv` integration was finished and verified ahead of the June 21 target. The next HW
   priority are the Stage-2 and top-level apu modules. The DRAM model was discarded, as the
   memory hierarchy is simplified to fixed latency and modeled in the testbench.

4. **No change to downstream milestones.** APU Stage 2, PE, top-level, full-CNN, and metrics
   milestones (July 12 -> Aug 9) remain as planned.


## 4. Repository pointers (evidence)

- SW functional model: `sw/functional.py`, refs in `testing/ref/`
- SW performance model: `sw/perf_model.py`, `sw/perf_pe.py`, `sw/config.py`, `sw/PERF_MODEL_PLAN.md`
- APU Stage-1 RTL: `rtl/apu/stage1/{csr_decode,zero_act,position_encode,apu_stage1}.sv`
- Shared RTL: `rtl/common/fifo.sv`, `rtl/common/sram.sv`
- Tests: `testing/common/` (FIFO, 92 checks), `testing/apu/stage1/` (cocotb: `test_csr_decode`,
  `test_zero_act`, `test_apu_stage1` — full-stage co-sim vs `functional.py`, 7 configs × 5 tests pass)
