# ECE 493 / 720 Course Project — Progress Report 1

**Team 19 — GoSPA: An Energy-efficient High-performance Globally Optimized Sparse CNN Accelerator** (ISCA 2021)

---

## 1. Per-member progress

### 1.1 Emon Sarkar

**Status: on/ahead of schedule.** Delivered the APU Stage-1 input front-end in RTL with full
unit-test coverage, plus the shared FIFO used throughout the design.

**Completed:**

- **`rtl/common/fifo.sv`** — parameterized synchronous FIFO (data width + depth) used for both
  FIFO-A (per-PID buckets) and FIFO-B (per-PE) in the GoSPA datapath. *(Originally planned for the
  June 14 sub-period; delivered June 1, ahead of schedule.)*
- **`testing/common/fifo_tb.sv`** + Makefile + `run_sim.sh` — self-checking SystemVerilog testbench
  runnable under both Verilator and Icarus. **10 tests / 92 checks, all passing**, including
  reset/full/empty/overflow/underflow protection, FIFO-order integrity, simultaneous push+pop, the
  paper's F=2,H=3,S=1 FIFO-A toy scenario, and a 50/50 randomized stress test.
- **`rtl/apu/stage1/csr_decode.sv`** — FSM-based decoder converting a CSR-format sparse activation
  matrix into a `(value, x, y)` stream, one tuple/cycle, with valid/ready backpressure and correct
  empty-row handling. This is the software model's `csr_to_positional` realized in hardware.
- **`rtl/apu/stage1/zero_act.sv`** — combinational zero-activation filter that suppresses `valid`
  for zero-valued activations so they never reach the IDGen array (the HW equivalent of the model's
  `if val == 0: continue`), saving G² wasted comparisons per zero.
- **`testing/apu/stage1/test_csr_decode.py`** (cocotb, 8 tests) and **`test_zero_act.py`** —
  golden-model-checked unit tests covering single/multi-entry rows, skipped empty rows, all-empty,
  fully-populated, backpressure stalls, randomized 8×8 matrices, and mid-stream reset. All passing
  (see `testing/apu/stage1/results.xml`).
- **`.gitignore`** for simulation build artifacts to keep the repo clean.
- Established the project's RTL/testbench conventions (directory layout mirroring `rtl/`,
  Verilator + cocotb flow) documented in `testing/README.md`.

**In progress / next (toward the June 21 sub-period):**

- Integrate the Stage-1 datapath into `apu/stage1/apu_stage1.sv`, wiring `csr_decode → zero_act →
  IDGen array → FIFO-A`. This depends on the IDGen / position_encode modules (see Adil's section),
  so the integration shell and the FIFO-A array instantiation are being prepared in parallel so
  that merge is fast once IDGen lands.

**Blockers:** Stage-1 integration is gated on the IDGen module. Mitigation: building the
integration top and FIFO-A wiring against the IDGen *interface* now so only the IDGen instance
needs to drop in.

---

### 1.2 Fred Huang

_To be completed by Fred._

---

### 1.3 Mahdi Hassen

**Status: met early** (planned completion June 21).

**Completed:**

- **Architecture block diagram** (`testing/GoSPA.drawio` + `GoSPA.jpg`) for the HW design.
- **`sw/functional.py`** (≈970 lines) — the golden functional model, built bottom-up: CSR decode →
  positional stream → `(Px,Py,Cx,Cy)` → `(CID,PID)` enumeration → zero filter → FIFO-A routing →
  WSP-gated FIFO-B broadcast → PE MAC. Extended to the **multi-PE** accelerator (`goSPA_run`) and
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

_To be completed by Adil._

---

## 2. Weekly milestones — met / delayed / skipped

Milestones are taken verbatim from the per-member table in `reports/plan.md`. Period covered:
project start → June 17.

### Sub-period ending June 7

| Member | Planned task | Status | Notes |
|---|---|---|---|
| Fred | Research perf-model requirements, determine abstraction level | | |
| Mahdi | Finish HW-design block diagram | **Met** | `GoSPA.drawio` / `.jpg` committed Jun 1 |
| Sara | Break implementation into submodules for perf-model tests | | |
| Emon | HW constraints, comms protocol w/ Adil, lab-PC setup, `fifo.sv` | **Met (early)** | FIFO + TB done Jun 1 |
| Adil | Design memory `.sv` (sram, dram) | | |

### Sub-period ending June 14

| Member | Planned task | Status | Notes |
|---|---|---|---|
| Fred | Perf-model skeleton + abstraction level | | |
| Mahdi | Work on functional model | **Met / exceeded** | Multi-PE + multi-channel + PyTorch validation done |
| Sara | Model HW components at abstraction level | | |
| Emon | `csr_decode.sv`, `zero_act.sv` | **Met** | Both done Jun 8 with cocotb tests |
| Adil | `position_encode.sv`, `idgen.sv` | | |

### Sub-period ending June 21 (in progress at time of report)

| Member | Planned task | Status @ Jun 17 |
|---|---|---|
| Fred | Work on perf model | |
| Mahdi | Finish functional model | **Effectively done** — validated end-to-end |
| Sara | Work on perf model | |
| Emon | Merge into `apu_stage1.sv` with Adil's work + FIFOs | **In progress** — blocked on IDGen; integration shell being prepared |
| Adil | `router.sv` | |

---

## 3. Refinements to the initial plan

1. **Software is ahead — pull verification forward.** With the functional model complete and
   PyTorch-validated, we will start **RTL ↔ SW co-simulation earlier**: the functional model can
   serve as the golden reference for the Stage-1 RTL (`csr_decode`/`zero_act` already mirror
   specific model functions), instead of waiting for the July co-sim milestone.

2. **Sequence the HW work to unblock integration.** Prioritize **`idgen.sv` before `router.sv`**,
   since IDGen is the dependency for the `apu_stage1.sv` integration. The DRAM model is lower
   priority and can follow, as the memory hierarchy is being simplified to fixed latency (per the
   plan's stated modification) and can be modeled in the testbench.

3. **Build the Stage-1 integration shell against interfaces, not implementations.** To decouple
   from the IDGen schedule, the Stage-1 top and FIFO-A array are being written against the agreed
   IDGen port interface so the module drops in once ready.

4. **Ensure commit visibility.** All members commit their own work to the repo, as individual
   contribution is part of the project evaluation.

5. **No change to downstream milestones.** APU Stage 2, PE, top-level, full-CNN, and metrics
   milestones (July 12 → Aug 9) remain as planned.

---

## 4. Repository pointers (evidence)

- SW functional model: `sw/functional.py`, refs in `testing/ref/`
- SW performance model: `sw/perf_model.py`, `sw/perf_pe.py`, `sw/config.py`, `sw/PERF_MODEL_PLAN.md`
- APU Stage-1 RTL: `rtl/apu/stage1/csr_decode.sv`, `rtl/apu/stage1/zero_act.sv`
- Shared RTL: `rtl/common/fifo.sv`, `rtl/common/sram.sv`
- Tests: `testing/common/` (FIFO, 92 checks), `testing/apu/stage1/` (cocotb, passing — `results.xml`)
