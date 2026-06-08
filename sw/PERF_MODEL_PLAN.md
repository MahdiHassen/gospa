# GoSPA Performance Simulator — Design Plan

**Author:** Fred Huang (Architect) · Team 19 · ECE 720 (Spring 2026)
**Paper:** GoSPA: An Energy-efficient High-performance Globally Optimized SParse
Convolutional Neural Network Accelerator (ISCA 2021),
Deng, Sui, Liao, Qian, Yuan. Reported speedups (vs. dense baseline):
AlexNet 1.38×, VGG 1.28×, GoogLeNet 1.23×, MobileNet 1.17×, ResNet 1.21×,
ResNeXt 1.28×; energy-efficiency 2.06–5.38×.

This document specifies the **performance** simulator. It is *not* cycle-accurate;
it estimates per-layer and end-to-end cycle counts to a useful fidelity. The
existing `sw/functional.py` is the **functional** golden model and is left as-is.

---

## 1. Scope and relationship to the functional model

`functional.py` correctly implements the GoSPA *algorithm* for a single input and
single output channel: CSR decode → `(Px,Py,Cx,Cy)` → `(CID,PID)` enumeration →
FIFO-A routing → WSP-gated broadcast → PE accumulate.

The perf model **reuses the correct front-end stages** (CSR decode, ID generation,
FIFO-A routing — Figs. 5/6) but **does not reuse `pe_process`**, because that
function models the PE multipliers along the wrong axis (see §2). The perf model
implements its own corrected PE assignment (union-of-WSP gating) and PE timing.

---

## 2. Corrected PE microarchitecture (critical)

The `M` multipliers inside each PE are **spatial across output channels (weights)**,
not across activations. This corrects an assumption baked into
`functional.py:pe_process` (which batches activations sharing a PID across lanes).

- Each PE handles **`M` output channels**.
- Each cycle a PE consumes **one** activation `(Axy, CID, PID)` and **broadcasts it
  to all `M` multipliers**. Multiplier `c` computes `Axy × W_c[PID]` and accumulates
  into that channel's own CID-indexed accumulator. → reuse axis is *activation
  reused across output channels* (weight-stationary).
- Each PE stores **`M` WSPs** (one per output channel). FIFO-B admission is gated by
  the **union** of those `M` WSPs: an activation enters the PE iff *any* of its `M`
  channels has a non-zero weight at that PID.
- **Partial-utilization effect:** because admission is by union, on a given cycle
  only the channels with a non-zero at the consumed PID do useful work; the other
  multipliers idle. The model tracks this as
  `lane_utilization = useful_MACs / (PE_cycles × M)`.
- **Weight reuse / reordering** (`Curr`/`Next`) operates on a **column of `M`
  weights per PID**: same-PID activations are batched so the column stays resident
  and is reused; a reload penalty `W_UPDATE_PENALTY` is charged on PID change.
- **Throughput:** **1 activation / cycle / PE.**
- **Output-channel tiling:** `Cout` is mapped across `N_PE × M` channels per pass;
  `ceil(Cout / (N_PE·M))` passes per layer.

---

## 3. Modeling approach — structural per-stage cycle accounting

> Decision: **structural per-stage** (chosen over closed-form analytical and full
> cycle-stepped).

No global clock and no per-cycle loop. We run the (corrected) functional pipeline
**once** to produce the *real* routed data structures, then for each stage we count
how many cycles consuming that data would take at the hardware's throughput. Because
the counts are taken over the actual sparse, routed data (not averages), this
captures the two effects that make GoSPA's real speedup fall short of ideal:

1. **Per-PE load imbalance** — different output-channel groups have different
   weight-sparsity, so different FIFO-B depths; the **slowest PE sets layer time**.
2. **Per-lane (union-gating) under-utilization** — idle multipliers when only some
   of a PE's `M` channels are non-zero at the consumed PID.

What it approximates away: FIFO **backpressure**/stalls (assumes FIFOs deep enough
that each stage runs at its natural rate). This is the natural thing to refine later
against the RTL testbenches (July milestones) if the discrepancy proves material.

---

## 4. Per-stage timing equations

Run the functional pass to obtain, per layer/pass: the decoded `(CID,PID)` pair
stream, the F²×FIFO-A contents, and `fifo_b[k]` for each PE `k`.

- **APU Stage 1 (ID-gen):** ~1 non-zero activation decoded/cycle, each emitting up
  to `G² = (F/S)²` `(CID,PID)` pairs.
  `stage1 ≈ Σ_nz (1 + pairs_emitted)` (or `max(#nz, #pairs)` if the enumerator is
  unrolled — config knob). Input rate is bounded by activation read bandwidth.
- **APU Stage 2 (router/broadcast):** drains FIFO-A in PID order, ~1 entry
  broadcast/cycle (fan-out to matching PEs is parallel).
  `stage2 ≈ Σ_pid len(FIFO_A[pid])`.
- **PE (usual bottleneck):** for each PE walk `fifo_b[k]`; 1 activation/cycle, plus a
  reload on PID change:
  `pe_cycles(k) ≈ len(fifo_b[k]) + (#PID_changes_in_k × W_UPDATE_PENALTY)`.
  `pe_stage = max_k pe_cycles(k)`  ← load imbalance.
- **Memory (fixed-latency + bandwidth, matches RTL):**
  `mem_cycles = ceil(bytes_moved / B) + L` for activation loads, weight loads, and
  output stores. 1×1 and depthwise layers are the bandwidth-bound stress cases.

**Layer latency:**
```
layer_cycles = fill
             + Σ_passes [ max(stage1, stage2, pe_stage, mem_cycles) ]
             + drain
```
`fill`/`drain` model pipeline startup and the FIFO-A barrier ("no new input tile
until FIFO-A drains" — see the note in `functional.py:route_to_fifo_a`). Both are
config constants to be calibrated against RTL.

---

## 5. Scaling to real layers

`functional.py` is single in/out channel; the perf model adds:
- **`Layer` descriptor:** `H, W, F, S, pad, Cin, Cout, type ∈ {conv, 1x1, dw, fc}`.
- **Input-channel accumulation** over `Cin`; **output-channel tiling** over
  `N_PE × M` (§2).
- A **stats-only fast path** alongside the full functional path: for large layers we
  derive per-PE FIFO-B lengths and PID-change counts from sparsity without
  materializing every MAC (keeps full-network sweeps fast); the full path is used
  for small correctness/spot-check cases.

---

## 6. Sparsity sources

> Decision: **synthetic first, real later.**

- **Synthetic (phase 1):** inject per-layer activation density `d_a` and weight
  density `d_w` (Bernoulli / structured). Fast bring-up + sensitivity studies.
- **Real (phase 2):** run dense AlexNet/VGG16/GoogLeNet/MobileNet/ResNet in
  numpy/PyTorch, capture post-ReLU activation maps → real activation sparsity; load
  pruned weights (or the paper's ratios) → real WSP. Required to replicate the
  paper's speedups.

---

## 7. Proposed file layout (`sw/`)

```
functional.py     # exists — golden functional model (left as-is)
config.py         # HwConfig dataclass: N_PE, M (mults/PE), FREQ_HZ,
                  #   FIFO_A_DEPTH/FIFO_B_DEPTH, W_UPDATE_PENALTY, FILL/DRAIN,
                  #   ACT_W/PID_W/CID_W, mem latency L / bandwidth B
layer.py          # Layer descriptor + multi-channel / tiling iteration
perf_model.py     # corrected PE assignment (union-WSP) + per-stage cycle counting
sparsity.py       # synthetic providers now; PyTorch-captured real later
workloads/        # alexnet.py, vgg16.py, ... as Layer lists
sim.py            # driver: network -> per-layer & total cycles, latency, FPS,
                  #   utilization, load-imbalance factor, speedup vs dense
tests/            # unit tests: per-stage cycle counts + invariants
```

---

## 8. Outputs / metrics

Per layer and per network:
- total cycles → latency (ms) and FPS via `FREQ_HZ`;
- multiplier (lane) utilization and per-PE **load-imbalance factor**
  (`max_k pe_cycles / mean_k pe_cycles`);
- achieved vs. ideal MAC throughput;
- **speedup vs. a dense baseline** (same array, no sparsity skipping) — the headline
  number to compare against the paper's Table.

---

## 9. Validation strategy

1. **Sanity limits:** density=1 (dense) approaches the ideal roofline; a single
   non-zero approaches minimum latency.
2. **Front-end cross-check:** perf model's Stage-1/Stage-2 routed structures match
   `functional.py` on its existing small cases.
3. **RTL co-sim (July):** compare per-layer cycles against the SystemVerilog APU/PE
   testbenches; calibrate `W_UPDATE_PENALTY`, `FILL`, `DRAIN`.
4. **Paper replication:** end-to-end speedup vs. dense within a reasonable margin;
   reason about gaps (load imbalance, union under-utilization, bandwidth, unmodeled
   backpressure).

---

## 10. Open items — pull from the paper's evaluation section

Set these `HwConfig` defaults from the paper's config table (PDF not fetchable in
the current environment; flagged rather than guessed): **`N_PE`, `M` (multipliers /
PE), clock frequency, on-chip buffer sizes, datatype width.** Confirmed from the
abstract / secondary sources: 28nm CMOS, weight-stationary dataflow, implicit
on-the-fly intersection. Keep all of these as parameters so filling them in is a
one-line change.

---

## 11. Build order (next milestones)

1. `config.py` — `HwConfig` with all knobs (placeholder defaults + TODOs).
2. `perf_model.py` — corrected PE assignment + §4 stage counters; validate on the
   three cases already in `functional.py`.
3. `layer.py` + tiling — lift to multi-channel / multi-PE.
4. `sim.py` + AlexNet end-to-end with synthetic sparsity; then wire in real sparsity.
