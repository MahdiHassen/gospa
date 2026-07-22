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

`functional.py` implements the full GoSPA *algorithm* and is **no longer limited to a
single PE / single channel.** Bottom-up it provides:
- the per-stage front end — CSR decode (`csr_to_positional`) → `(Px,Py,Cx,Cy)`
  (`axy_to_pcid`) → `(CID,PID)` enumeration (`pcid_to_cid_pid`) → zero filter →
  FIFO-A routing (`route_to_fifo_a`) → WSP-gated broadcast (`broadcast_to_fifo_b`);
- the **multi-PE accelerator** `goSPA_run(..., num_pes, num_mults, interpretation)`:
  it packs output channels onto an `N_PE × M` array and routes the *real* FIFO-B
  stream to each PE (v2: up to `num_pes·num_mults` kernels, one per lane, each PE
  gated by the **union** of its lanes' WSPs);
- **multi-input-channel** convolution `goSPA_multichannel(...)`, which threads
  `initial_outputs` so each PE's CID-indexed accumulator **persists across input
  channels** (partial sums stay resident and are summed over `Cin`).

Verified end-to-end against PyTorch on MobileNetV2's first conv (RGB, 32 output
channels, including the quantize/bias/ReLU/requant tail).

The perf model **reuses the correct front-end stages** (CSR decode, ID generation,
FIFO-A routing — Figs. 5/6) and the **corrected v2 PE dataflow** that already exists
in `functional.py` (`pe_process_v2` / `goSPA_run(interpretation="v2")`: M kernels per
PE, union-of-WSP FIFO-B gating). It **does not reuse `pe_process`** (v1), because that
function models the PE multipliers along the wrong axis (see §2). Rather than
reimplement the v2 dataflow, the PE perf model is an **instrumentation layer** over
it: it walks the already-routed `fifo_b` and counts cycles/utilization — it does not
recompute MACs. Implemented in `perf_pe.py`.

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
  and is reused. A `Curr`/`Next` **double buffer** prefetches the next column, so a
  PID change is *hidden* when the current PID's run is long enough to cover the
  fetch. The reload penalty **scales with `M`** (the column size):
  `P = W_FETCH_LATENCY + ceil(M / W_FETCH_BW)` (defaults give `P = M`). §4 shows how
  the stall is charged. *(In `pe_process_v2` each lane advances its **own** sparse
  weight stream and reloads only when it is active at a new PID; the perf model
  abstracts these independent per-lane updates as one M-wide column reload at each
  PID-group boundary — matching the hardware's `Curr`/`Next` column double-buffer.)*
- **Throughput:** **1 activation / cycle / PE.**
- **Output-channel tiling:** one pass packs up to `N_PE × M` output channels.
  `goSPA_run(interpretation="v2")` **already does this packing** (PE `k` gets kernels
  `[k·M : (k+1)·M]`, union-WSP per PE; the last PE may be partially filled, so its
  effective lane count `M_eff < M`). It *raises* if `Cout` exceeds `N_PE·M`, so the
  **perf model adds only the pass loop** `ceil(Cout / (N_PE·M))` on top.

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

- **APU Stage 1 (ID-gen):** the RTL (`rtl/apu/stage1/idgen.sv`, `apu_stage1.sv`) is a
  purely **combinational `G²` fan-out** (`PIPE=0`): one activation's up-to-`G²` pairs
  carry **distinct PIDs** (the `(m,n)→PID` map is injective) and land in `G²`
  *different* FIFO-A lanes in the **same cycle**, so throughput is **1 activation/cycle
  → `stage1 = n_nz`** (`STAGE1_ENUM = "parallel"`, the RTL-faithful **default**).
  `G = ceil(F/S)` (`ceil`, not `floor`); the `E`/`F`-range gates in `pcid_to_cid_pid`
  drop out-of-bounds `(m,n)`. Kept as analysis knobs: `"unrolled"` = `max(n_nz,
  n_pairs)` (1-pair/cycle emit) and `"serial"` = `n_nz + n_pairs` (serial emit, the old
  default). Stage-1 stalls on FIFO-A-full (the all-or-nothing join in `apu_stage1.sv`)
  are captured by the per-pass `max()` over stages, not charged here.
- **APU Stage 2 (router/broadcast):** drains FIFO-A in PID order, popping `B` heads
  per cycle (fan-out to matching PEs is parallel): **`stage2 = ceil(n_pairs / B)`**.
  `B` is `STAGE2_BATCH`; `"native"` (the **default**) is the as-built width — `B = M`
  in `"act"` (the FIFO-A→FIFO-B transfer is widened to `M` activations/cycle) and
  `B = 1` in `"channel"`. **`n_pairs` is gated on zero *activations* only** — the
  router pops a head whether or not any PE's WSP wants it (`functional.py`
  `n_pairs=len(zero_act_filter(pairs))`; `routing.sv` `go = head_valid && all_ready`
  with `go` independent of `sel`). That is the origin of the `d_w` cap in §12.
  Knobs: a fixed integer `B`, or `"auto"` → `B*` (below).
- **PE (usual bottleneck):** for each PE walk `fifo_b[k]`; 1 activation/cycle, plus
  reload stalls on PID change. With the `Curr`/`Next` double buffer (the default,
  faithful model) a reload is hidden behind the previous PID's run:
  `pe_cycles(k) = len(fifo_b[k]) + Σ_{i≥1} max(0, P − run_len(group_{i-1}))`,
  where the groups are the contiguous same-PID runs (FIFO-B is PID-ordered out of
  Stage 2) and `P` is the M-scaled penalty from §2. A `simple` model
  (`+ #PID_changes × P`, no hiding) and an `ideal` model (no stall) are kept as
  analysis knobs. `pe_stage = max_k pe_cycles(k)`  ← load imbalance.
  *(Implemented in `perf_pe.py:pe_perf_from_stream`.)*
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

`functional.py` already covers multi-PE packing (`goSPA_run`) and input-channel
accumulation with resident accumulators (`goSPA_multichannel`). On top of that the
perf model adds:
- **`Layer` descriptor:** `H, W, F, S, pad, Cin, Cout, type ∈ {conv, 1x1, dw, fc}`.
- **Output-channel pass loop:** `ceil(Cout / (N_PE·M))` passes, since one `goSPA_run`
  packs at most `N_PE·M` channels (§2).
- **Input-channel accumulation is reused** from `goSPA_multichannel`: the CID-indexed
  accumulators stay resident across `Cin` (threaded via `initial_outputs`), so there
  is **no extra readout/reload between input channels** — each input channel just adds
  another front-end pass over the same PEs.
- **Stream-only accounting:** the perf model runs the real functional routing and
  walks the materialized `fifo_b` per PE (the stream entry point in `perf_pe.py`).
  *(The earlier "stats-only occupancy fast path" is dropped — the model reports the
  architecture's real routed behavior, it does not model an optimized variant.)*

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
functional.py     # golden functional model + goSPA_route() routing accessor
                  #   (additive: shared front end for goSPA_run and the perf model)
config.py         # DONE: HwConfig dataclass: N_PE, M (mults/PE), FREQ_HZ,
                  #   FIFO_A_DEPTH/FIFO_B_DEPTH, W_UPDATE_PENALTY, FILL/DRAIN,
                  #   ACT_W/PID_W/CID_W, mem latency L / bandwidth B,
                  #   STAGE1_ENUM, STAGE2_BATCH (router width B -- see sec.13)
layer.py          # Layer descriptor + output-channel pass loop;
                  #   reuses goSPA_multichannel for Cin accumulation
perf_pe.py        # DONE: single-PE timing — instruments the v2 PE; double-buffer
                  #   reload model, M-scaled penalty, lane-utilization stats
perf_model.py     # DONE: per-pass (APU1/APU2/mem) counting + multi-PE aggregation
                  #   (max_k pe_cycles); calls perf_pe per PE via goSPA_route
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
- **array utilization (wall-clock)** `array_util = Σ useful_MACs / (Σ pass_cycles ·
  N_PE · M)` and per-PE **load-imbalance factor** (`max_k pe_cycles / mean_k
  pe_cycles`). The denominator is `pass_cycles` (the *whole* physical array over the
  pass wall-clock), **not** each PE's `pe_cycles`, so it charges every multiplier-slot
  idled when the PE array is not the bottleneck — front-end/mem stall (`pass_cycles >
  pe_stage`), cross-PE imbalance, and PEs left idle on a partial tile. It never exceeds
  the PE-intrinsic ratio `useful / (Σ pe_cycles·M)`, with equality only when the pass is
  PE-bound, balanced and full. (The PE-intrinsic per-lane ratio — the union-penalty
  diagnostic of §"channel" — still lives in `perf_pe.py` as `overall/compute_lane_util`;
  it answers a *different* question and is not the reported aggregate.) This aggregate
  equals `achieved_macs_cycle / ideal` by construction;
- achieved vs. ideal MAC throughput;
- **Stage-2 router width** `B` used, and **`B*`** per layer and per network — the
  narrowest router that would stop Stage 2 binding (§13). `B*` is reported even when
  `B` is pinned, so the headroom above the `d_w` cap is always visible;
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

Single-pass `N_PE×M` packing are **reused** from `functional.py`, not rebuilt here.

1. **`config.py`** — **DONE**: `HwConfig` parameter bag: all knobs (`N_PE`, `M`,
   `FREQ_HZ`, `FIFO_A/B_DEPTH`, `W_UPDATE_PENALTY` / `W_FETCH_LATENCY` / `W_FETCH_BW`,
   `RELOAD_MODEL`, `STAGE1_ENUM`, `FILL`/`DRAIN`, mem `L`/`B`/ports/byte-widths,
   `ACT_W`/`PID_W`/`CID_W`). Placeholder defaults + TODOs (§10).

2. **`perf_pe.py`** — **DONE**: single-PE timing (stream entry point). Instruments the
   v2 PE; double-buffer reload, M-scaled penalty, lane-utilization stats. Validated by
   its own self-test (Fig.13 replay, roofline, double-buffer stalls, union
   under-utilization) and cross-checked on a real routed `fifo_b` from `functional.py`.

3. **`perf_model.py`** — **DONE**: the per-pass engine. Drives the real front end via
   the new pure `functional.goSPA_route` (shared with `goSPA_run`) to obtain each PE's
   `fifo_b` + lane WSPs, counts APU-Stage-1 / Stage-2 / memory (§4), calls `perf_pe`
   per PE, and returns `pass_cycles = max(stage1, stage2, max_k pe_cycles, mem)` with a
   breakdown (bottleneck, load-imbalance factor, wall-clock `array_util`). `FILL`/
   `DRAIN` and the Cin / output-tile loops are deferred to `layer.py`. Validated by its
   own self-test (front-end cross-check, roofline, load-imbalance, `STAGE1_ENUM` knob).

4. **`layer.py`** — `Layer` descriptor (§5) plus the two outer loops `perf_model`
   doesn't own: the **output-channel pass loop** `ceil(Cout/(N_PE·M))`, and
   **input-channel accumulation** delegated to `goSPA_multichannel` (accumulators stay
   resident — no inter-channel reload to charge). Expands a `Layer` into the sequence
   of passes `perf_model` scores.

5. **`sparsity.py`** — sparsity providers: synthetic `d_a`/`d_w` (Bernoulli /
   structured) now for bring-up; PyTorch-captured real maps later (§6).

6. **`workloads/`** — networks as `Layer` lists (`alexnet.py`, `vgg16.py`, …). Data
   only, no logic.

7. **`sim.py`** — top driver: run a workload through `layer` + `perf_model`, sum to
   total cycles → latency/FPS, report wall-clock `array_util`, per-PE load-imbalance factor,
   and **speedup vs. dense** (§8). AlexNet end-to-end with synthetic sparsity first,
   then wire in real sparsity.

8. **`tests/`** — unit tests: per-stage cycle counts + the §9 invariants (density=1
   roofline, single-nonzero floor, front-end structures match `functional.py`).
   Generalizes `perf_pe.py`'s inline self-test.

**Critical path to a first end-to-end number: 3 → 4 → 7** (with 1 as prerequisite).
Steps 5, 6, and 8 can land in parallel.

---

## 12. Dataflow architectures (`--arch`)

`§2` describes the **`channel`** dataflow (the default). A second dataflow, **`act`**,
was added to attack `channel`'s core inefficiency. Selected by `HwConfig.ARCH` /
`sim.py --arch {channel,act}`. The whole front end (`goSPA_route`, Stage 1) is shared;
only the PE mapping, the FIFO-A→B transfer, and the output-channel tiling differ. No
`functional.py` change was needed — `act` reuses `goSPA_route(interpretation="v1")`.

**Motivation.** In `channel`, FIFO-B admission is by the **union** of a PE's `M`
per-lane WSPs, but lane `c` only does useful work when *its own* `WSP_c[PID]=1`. So
multiplier utilization is capped at ~weight density (`sw/unioin_wsp_util.csv`: util
tracks `d_w` almost exactly — 0.49 @ `d_w=0.5`, 0.27 @ `d_w=0.1`).

| | **`channel`** (default) | **`act`** |
|---|---|---|
| per PE | `M` kernels (`M` output channels) | **1 kernel**, shared by all `M` mults |
| within PE | 1 activation → `M` lanes | **`M` activations → `M` mults** |
| WSP | `M` per-lane, **union**-gated | **single** WSP, no union |
| output chans/pass | `N_PE·M` | **`N_PE`** (⇒ `M×` more tiles/passes) |
| Stage 2 | `ceil(n_pairs/B)`, `B=1` native | `ceil(n_pairs/B)`, **`B=M`** native |
| PE `k` cycles | `\|FIFO_B_k\|` + reload(`P=M`) | **`Σ_g ceil(run_len_g / M)`**, no reload |
| useful MACs / PE | `Σ popcount(lane WSPs)` | `\|FIFO_B_k\|` |
| util ceiling | ≈ `d_w` | ≈ `1 − tail`, set by PID-run length ⟂ `d_w` |

**`act` PE model (decision B2, `perf_pe.py:pe_perf_actparallel`).** FIFO-B is
PID-ordered, so it is bundled into contiguous PID runs; up to `M` activations of a run
are consumed per cycle against that run's single resident weight → `ceil(run_len/M)`
cycles per run. The whole `F²` kernel is resident, so there is **no per-PID reload**
(one-time kernel load folds into `FILL`). The only idle lane-slots are each run's
partial last cycle (the "tail"), so utilization is governed by average run length
`≈ n_pairs/F²`, which is **independent of weight density**.

**Key finding (synthetic phase-1).** `act` raises the *PE-intrinsic* lane utilization
exactly as predicted, but it is **not** a free win on wall-clock:
- The **PE-intrinsic** lane util (`perf_pe`'s `overall_lane_util`, union penalty gone)
  jumps on the small 2-layer probe **0.49 → 0.96** (conv layers ~0.98). But this is a
  per-PE ratio over `pe_cycles` — it assumes the PE is the bottleneck.
- The **reported wall-clock `array_util`** does *not* follow it: because it normalizes by
  `pass_cycles · N_PE · M`, the `M×` re-stream (below) and idle PEs pull it back down, so
  `act` is roughly neutral vs `channel` on `3×3` and *worse* on `1×1`/FC. This is the
  whole reason the aggregate metric uses wall-clock cycles rather than `pe_cycles`.
- `act` re-streams the activation `M×` more (one front-end run per `N_PE`-channel
  tile), and with Stage 1 left un-widened at `n_nz`, the **front end becomes the
  bottleneck**. Total Stage-2 work is mode-invariant (the `M`-wide transfer cancels the
  `M×` passes), but total Stage-1 work is `M×` higher.
- Net: `act` is roughly **wall-clock-neutral on `3×3` convs** (long PID runs, PE-bound)
  but **loses badly on `1×1`/FC** (`F²=1` ⇒ `n_nz = n_pairs`, Stage-1-bound, full `M×`
  penalty). In the 2-layer probe: `3×3` 3447→3495 cyc; `1×1` 1741→5172 cyc.
- **Cross-arch comparison must use absolute cycles**, not *speedup-vs-dense*: each arch
  is compared to its *own* dense baseline, which flatters `act` (1.71× vs 1.53×) even
  though it is slower in absolute cycles. Array efficiency is the honest headline
  (`channel` 45% → `act` 22% on the probe).

**Takeaway.** The union penalty and the re-stream penalty are *two different*
inefficiencies; `act` trades the first for the second. It is attractive only where PID
runs are long **and** the front end is not the binding stage — so a genuine `act` win
likely also needs a widened Stage 1 (the `STAGE1_ENUM` knob can explore this). Real
sparsity maps (§6) may shift the crossover; the model now reports the bottleneck
per layer so the trade is measured, not assumed.

---

## 13. The `d_w` utilization cap and the Stage-2 router width (`STAGE2_BATCH`)

**Observation.** Under `"native"`, `array_util ≈ d_w` for *both* arches, and it is
**flat in `d_a`** — a `d_a` sweep at `d_w=0.8` puts every AlexNet conv layer at
`util ≈ 0.8` regardless of which stage the report names as the bottleneck.

**Cause — a rigorous, bottleneck-independent cap.** In `act`, useful work is exactly
`N_PE · d_w · n_pairs` (each PE holds one kernel and admits the `d_w` fraction of the
shared pair stream), while `pass_cycles ≥ stage2 = ceil(n_pairs/B)`. Therefore

```
array_util  ≤  d_w · B / M                     (act, B/M = 1 natively)
```

**always** — this needs Stage 2 only to be a *lower bound* on wall-clock, not to be the
binding stage. Being PE-bound makes `pass_cycles` larger and pushes util further
*below* the cap; it can never lift it above. The union penalty of §12 is a *symptom*
of this, not the root cause: `act` did not remove the union, it **relocated** it from
intra-PE lanes to the shared inter-PE router.

**Why util sits so close to the cap.** With `pe_cycles_k ≈ (nnz_k/F²)·(n_pairs/M)/tail`:

```
array_util = d_w · min(1, tail · F² / nnz_max)      ⇒   d_w·tail ≤ util ≤ d_w
```

and since `nnz_max ≤ F²`, **PE-bound ⟺ `nnz_max > tail·F²` ⟺ the fullest kernel is
essentially dense** — which is exactly the condition making the load-imbalance ratio
`mean_k(nnz)/max_k(nnz) ≈ d_w·F²/F² = d_w`. The two regimes are not a coincidence:
being PE-bound *implies* the imbalance ratio is ≈`d_w`, and they hand off continuously.
Verified (`act` 8×4, `d_a=1.0`, `d_w=0.8`): F=3 → PE-bound, `mean/max=0.836≈d_w`;
F=9 → Stage-2-bound, `mean/max=0.927` (well above `d_w`) yet `util=0.795≈d_w`.
Shrinking to `N_PE=M=1` removes the union *and* the tail but lands **exactly** on the
cap (`util=d_w`, 100% Stage-2-bound) at 1/32 the throughput — the cap is structural.

**The fix — `B*`.** Widening the router scales the cap by `B/M`. The knee is

```
B* = max(1, ceil(n_pairs / max(stage1, pe_stage, mem)))
```

the narrowest router that stops being the bottleneck. `stage1`/`pe_stage`/`mem` are all
**independent of `B`**, so `B*` is computable in one shot with no circular dependency,
and `perf_model.rescore_at_batch` / `layer.rescore_layer_at_batch` can re-derive any
`B` from stored counters **without re-routing** — which is what makes network-scope
`"auto"` free. `B*` generalizes the analytic `B* ≈ M/d_w` law (verified: knee at 8/12/18
for `d_w` = 0.5/0.3/0.2 with `M=4`) and correctly stops short on small kernels where
Stage 1 (`n_nz`, 1 activation/cycle) takes over first.

**Scope.** `"auto"` is resolved at **network scope** — one fixed width for every layer,
`B = max` of per-pass `B*` over the whole network, because that is what fixed-width
hardware is. This over-provisions the small-kernel layers; the per-layer `B*` column in
`sim.py`'s report makes that visible. The dense baseline is re-scored at the *same*
width (same hardware), so speedup stays a data comparison.

**Payoff** (`act` 8×4, F=9, `d_a=1.0`) — concentrated exactly where GoSPA claims its
value, and worth nothing in the dense corner:

| `d_w` | `util` @ `B=M` | `util` @ `B*` | speedup |
|---|---|---|---|
| 0.2 | 0.203 | 0.713 | 3.5× |
| 0.3 | 0.298 | 0.743 | 2.5× |
| 0.5 | 0.498 | 0.801 | 1.6× |
| 0.8 | 0.795 | 0.873 | 1.10× |
| 1.0 | 0.942 | 0.942 | 1.00× |

After the fix the binding stage is the **PE** (ceiling = `tail × mean/max` imbalance,
≈0.87) on large kernels, or **Stage 1** on `3×3` — the next thing to attack. On a
2-layer `d_a=0.4/d_w=0.3` probe, `auto` (B=9) moved util 0.277 → 0.975 and
speedup-vs-dense 2.47× → 8.40×; the dense baseline is PE-limited and flat in `B`
(146,016 cycles for every `B ≥ 4`), so widening the router is precisely what converts
weight sparsity into speedup.

**Caveats.** (1) `"auto"` is an *analysis* knob — it answers "how wide would it have to
be", not "how wide is it"; the default stays `"native"` so the model keeps matching the
RTL. (2) Cost is not modeled: `routing.sv` pops one head/cycle, so `B` heads/cycle means
a `B`-way arbiter, `B×` broadcast width and `B×` FIFO-B write ports — at `d_w=0.2`,
`B*≈20` is not a small router. (3) Back-pressure is not modeled (§3), so at `B > M` the
arrival rate is bursty above the PE drain rate and `FIFO_B_DEPTH` becomes load-bearing
in a way this model cannot see. `B*` is a mean-rate argument.
