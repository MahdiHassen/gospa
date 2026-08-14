# Where the apple-layer cycles actually go (V2 RTL, measured)

**Question:** avg MACs/cycle is 25 of 32 — shouldn't stage 2 empty FIFO-As in
huge batches so fast that this loss disappears? Where is the real bottleneck?

**Answer up front:** stage 2 is measured running at **99.9% occupancy, popping
4 pairs/cycle every cycle** — it leaves nothing on the table at its width, and
widening it (the perf model's `--stage2-batch`) buys **exactly zero** on this
layer. The streaming ceiling of 78% is architectural (weight density × the
shared-PID multicast), and the busiest PE ties the router cycle-for-cycle, so
no router speed can beat it. End-to-end, the actual bottleneck is the
**serialized 1-entry/cycle activation fill** (53% of all cycles at N_PE=8).
Everything below is RTL-measured or cross-checked bit-exact against the
functional model.

All numbers: MobileNetV2 first conv (F=3, S=2, dw=0.785, da≈0.99) on the real
80×80 apple, V2 RTL, N_PE=8 × M=4 (32 mults), 4 group passes × 3 input
channels. Reports: `testing/gospa/gospa_apple_perf{,_npe8,_npe32}.txt`.

## 1. The end-to-end cycle budget (the metric that matters)

| phase | cycles | share | note |
|---|---|---|---|
| activation CSR fill | 77,488 | **53%** | 1 entry/cycle, × 4 group refills |
| stage2 windows | 41,208 | 28% | router+PE streaming |
| scan (stage 1) | 19,248 | 13% | 4-wide scanner (SB=4) |
| drain | 6,096 | 4% | 4 × 1,524 |
| load/arm/reset | 874 | 1% | |
| **total** | **144,914** | | end-to-end util 22.2% |

The single biggest lever is not in the compute path at all: the fill port
writes one CSR entry per cycle, serialized before every scan. 6,340–6,398
entries per channel ≈ 6.4k cycles — **1.9× longer than the stage2 window it
feeds** (≈3,420 cycles/channel).

## 2. Stage 2 is already running flat out (measured)

- Router beats emitted = **260,458 = the analytic ideal** Σ_pid ceil(pairs/4):
  every single pop was full-width except the unavoidable per-lane tails
  (packing efficiency 100.0%). The FIFO show-ahead sustains 4-wide reads.
- The 10 full-WSP output channels (9/9 weights in all 3 input channels)
  consume **every** router beat. Their measured starve = 63 cycles over 3
  stage2 windows = exactly the s2_start ramp (~3) + end-of-window idle
  detection (~17) per window. During streaming the router emitted a beat
  **every cycle** — zero bubbles, zero FIFO-B backpressure.
- Weight-switch stalls: 3 per PE per pass = the one cold SKIP per re-arm.
  Curr/Next prefetch removes the rest.
- FIFO-A lanes hold ~1,506 pairs each (80×80 untiled channel), so a lane
  drains in ~377 cycles at the RTL's 4/cycle — the "empty a FIFO-A in 1–2
  cycles" picture is true only for the *tiled sweep* configs (11×11 tile ≈
  13 pairs/lane) and for the perf model's B=8 analysis knob. The RTL has no
  stage2 batch: FIFO-A read width is hardwired `NUM_MULTS` = 4
  (`apu_stage1.sv` PW, `routing.sv` a_pop).

## 3. Why 32 MACs/cycle is not reachable on this layer — two ceilings, same number

**(a) Shared-PID multicast ceiling.** Each cycle the router broadcasts one
PID; only PEs whose WSP holds that PID work. Expected useful lanes/cycle =
32 × avg(dw_k) = **25.1**. Measured: 25.06 useful MACs/cycle streaming =
**99.9% of this ceiling** (78.3% util).

**(b) Busiest-PE (imbalance) ceiling.** Suppose an infinitely wide router
with perfect per-PE beat compaction (every PE consumes only its own PIDs,
packed 4-wide, from a deep FIFO-B). The pass still cannot end before the
busiest PE finishes. Every group contains ≥1 of the 10 full-WSP channels,
whose work = all 10,236 beats of the pass:

| grouping | Σ group floors (beats) | vs today |
|---|---|---|
| today's streaming window (measured) | 40,992 | — |
| ideal infinite-B + per-PE compaction, sequential groups | 40,944 | **−0.1%** |
| same + channels grouped by density | 38,296 | −6.5% |
| perfect work balance across PEs (split kernels) | 32,130 | −21.6% (= 32-mult floor) |

So batching + compaction is worth 0.1% here. The team's own perf model agrees
when fed the real apple data (ARCH=act, native): **B\* = 3** — the as-built
4-wide router is already *wider* than the knee — and re-scoring every pass at
B\* changes total pass cycles by **1.0×** (76,504 → 76,504). The
`new_arch_reasoning.md` §4 result (B\*=8–10, util +30%) came from the
synthetic da=dw=0.5 AlexNet sweep; it does not transfer to conv1 because
quantized MobileNetV2 weights are nearly dense (this CSV row: dw=0.785, and
0.78–0.99 across *all* layers in `mobilenet_sparsity_apple80.csv` — the B
lever has no headroom anywhere in this network).

The only lever that raises the *streaming* ceiling is **work rebalancing**
(density-aware channel grouping ≈ +6%, PID/channel splitting of dense kernels
up to +28%), not router bandwidth.

## 4. What to fix, in order of measured impact

1. **Fill bandwidth** (53% of cycles): widen the CSR fill port to 4
   entries/cycle (the entry SRAM is already banked 4-way for the scanner) →
   77.5k → ~19.4k cycles.
2. **Overlap fill/scan with compute**: double-buffer the activation SRAM (or
   fill channel c+1 while c streams). Together with (1), end-to-end converges
   toward the 78% streaming ceiling instead of 22%.
3. **Scale N_PE instead of group passes**: multicast makes output channels
   free on the front end — N_PE=32 measured identical 77.9% streaming util
   and 2.23× V1 wall clock, and removes the 4× refill tax entirely.
4. **Load balance** (the only way past ~78%): density-sorted grouping (+6.5%)
   or splitting dense kernels across PEs (+27.6%, reaches the 32-mult floor).
5. (Looming, out of scope here: depthwise layers map 1 input channel → 1
   output channel; with one resident input channel only one PE has work —
   needs a co-scheduling story before MobileNet bring-up proceeds past conv1.)

## Addendum: fix #1 applied — 16-wide CSR fill (FILL_W)

The fill port is now `FILL_W` entries/cycle (RTL default 1, test config 16):
the entry SRAM banks by max(FILL_W, STAGE1_BATCH) with sequential entries
striped across banks, so a 16-wide aligned beat hits 16 distinct banks;
row-pointer fill got the same lanes. Scan reads its 4-wide batch through a
subgroup mux. ~60 lines across `act_sram_scanner.sv` / `apu.sv` / `gospa.sv`,
plus the `gospa_tb.py` driver. All correctness tests + apple golden pass at
FILL_W=16 and FILL_W=1.

Fill fell from 53% of cycles to 7%.

## Addendum 2: fix #2 applied — batched router + pipelined channels (S2_BEATS)

`S2_BEATS` (default 1): the router pops S2_BEATS×M entries/cycle from the
current FIFO-A lane and multicasts up to S2_BEATS packed beats into the
selected FIFO-Bs (FIFO-B `PORT_WIDTH=S2_BEATS` writes, 1-beat reads). At
S2_BEATS=4 the router outruns the PEs 4× and FIFO-B becomes the decoupling
backlog, so the front end refills FIFO-A with the NEXT channel while the PEs
work — measured: router drains a channel in ~857 cycles, the next channel's
fill+scan (~2,015 cy) hides completely under the ~3,420-cycle PE backlog with
~540 cycles to spare. The dense PE's starvation inside a pipe window is 27
cycles (startup ramp only). The weight swap (~70 cy/channel) stays serialized
— beats carry no channel tag, so channels must not mix in FIFO-B. FIFO_D
4096 so the peak backlog (~2,560 beats) fits.

Full progression, all golden-checked vs PyTorch:

| config | total cyc | end-to-end util | wall clock vs V1 |
|---|---|---|---|
| V1 baseline (32 mults) | 82,114 | 39.1% | 1.00× |
| V2 as-committed (serial, FILL_W=1) | 144,914 | 22.2% | 0.57× |
| V2 + FILL_W=16 | 72,298 | 44.4% | 1.14× |
| **V2 + FILL_W=16 + S2_BEATS=4 pipelined** | **56,030** | **57.3%** | **1.47×** |
| V2 N_PE=32 + FILL_W=16 | 18,589 | 43.2% | 4.42× |
| **V2 N_PE=32 + FILL_W=16 + S2_BEATS=4** | **14,522** | **55.3%** | **5.65×** |

Utilization over the pipe windows is 78.2% = the weight-density ceiling (§3);
the pipeline achieves it wall-to-wall inside a pass. Remaining exposed
overheads per group: the first channel's fill+scan preamble (~2,000 cy, no
prior backlog to hide under) and the drain (~1,524 cy). Hiding those needs
drain-with-clear (drain one group while the next group's preamble runs) —
worth it mainly at N_PE=8; past that, only work rebalancing lifts the 78%
ceiling (§3).

## Addendum 3: the ceiling is imbalance, not density — measured

On the batched machine the pass ends when the *last* PE finishes its own
beat list (FIFO-B backlog decouples the PEs from the shared stream), so the
utilization ceiling is avg(dw_k)/max(dw_k) — the density *spread* — not
avg(dw). Apple conv1 reads 78% only because 10/32 kernels are fully dense
(max = 1) while others are near-empty, making the two formulas coincide.

Experiment (`make apple EQUAL_NW=7`): real apple activations, synthetic
kernels ALL exactly 7/9 non-zero — same ~0.78 average density, zero spread:

| kernels (avg dw ≈ 0.78 both) | pipe util | end-to-end | total cyc |
|---|---|---|---|
| real conv1 weights (spread 0 → 9/9) | 78.2% | 57.3% | 56,030 |
| balanced synthetic (all 7/9) | **93.3%** | **64.9%** | **49,041** |

Balanced PEs all finish together (~2,645 cy/channel instead of 3,420), so
the pass shortens AND nobody idles early. The residual ~7%: with the PE work
that short, the serial router -> fill+scan chain (853 + 2,015 cy) now slightly
exceeds it — the bottleneck flips to the front end. Letting the scanner fill
FIFO-A *while* the router drains it (budgeted drain / virtual ping-pong)
would overlap those and push pipe util to ~99%.

Implication for the full network: quantized MobileNetV2's deeper layers have
uniform 0.88–0.99 weight densities (near-balanced naturally), so they should
run near the ceiling as-is; conv1's extreme spread is the outlier, and its
fix is assignment (density-aware grouping / channel pairing / kernel
splitting), not more bandwidth.

Measured (`make apple GROUP_SORT=1`, real weights, channels grouped by beat
count): pipes 41,064 → 40,158 cy, end-to-end 57.3% → 58.2% (55,124 total).
The gain is smaller than the 6.5% beat-floor prediction because the sparse
groups' windows are now floored by the serial router→fill+scan chain
(~2,868 cy/channel) rather than PE work — the same front-end serialization
the balanced-kernel run exposed. Budgeted-drain overlap unlocks both.

## Addendum 4: first four MobileNetV2 layers, real tensors, measured

Layers 1–3 run on their REAL quantized inputs/weights (extracted from the
apple-80 forward pass, `testing/ref/mobilenet_layers.py`), each golden-checked
vs conv2d_reference. N_PE=8, FILL_W=16, S2_BEATS=4:

| # | layer | shape | useful MACs | total cyc | pipe util | e2e util |
|---|---|---|---|---|---|---|
| 0 | conv3×3 s2 | 3→32, 80×80 | 1,027,158 | 56,030 | 78.2% | 57.3% |
| 1 | dw3×3 (32 grp) | 32→32, 40×40 | 238,830 | 119,979 | 12.1% | **6.2%** |
| 2 | pw1×1 | 32→16, 40×40 | 494,864 | 32,231 | 56.1% | 48.0% |
| 3 | pw1×1 | 16→96, 40×40 | 2,243,925 | 138,995 | 64.0% | 50.4% |
|   | **4-layer total** | | 4,004,777 | 347,235 | | **36.0%** |

Findings:
- **Depthwise is the architectural gap, quantified.** One resident input
  channel multicast to all PEs means only 1 of 8 PEs can ever work (12.5%
  ceiling; measured pipe util 12.1% — the active PE itself runs at ~97%).
  Per-channel drains (32 × 1,447 cy = 39% of the layer) finish the damage.
  Fix directions: CID-range routing (PEs own spatial tiles of the same
  channel) or per-PE resident channels (banked activation SRAM).
- **Pointwise works (F=1 RTL corner verified) at ~50% e2e.** The pipe window
  per input channel is front-end-bound: route ≈ nnz/4 ≈ 380 cy < scan+fill ≈
  480 cy — F=1 has fan-out exactly 1, so stage 1 no longer amplifies and the
  scanner (4/cycle) is the long pole. Levers: STAGE1_BATCH=8/16, scan/route
  overlap, and drain hiding (layer 3 pays 12 group drains = 14% + 192 weight
  swaps at a measured ~22 cy each).
- Weight-swap cost measured across all layers: ~21–22 cy per channel
  boundary (~3–4% of cycles); drain is the bigger fixed cost (10–39%).

## Addendum 5: full MobileNetV2 — pointwise, not depthwise, is the network bottleneck

All 52 conv layers (real apple-80 tensors, golden-checked; linear classifier
not mapped) via `run_mobilenet_all.py` → `gospa_mobilenet_network.txt`.
N_PE=8, FILL_W=16, S2_BEATS=4:

**Network: 32.4M useful MACs, 13.87M cycles, util 7.3%, 138.7 ms @100 MHz (7.2 fps).**

| type | useful MACs | cycles | e2e util | share of runtime |
|---|---|---|---|---|
| conv1 | 1.03M | 56k | 57.3% | 0.4% |
| depthwise (17 layers) | 1.11M | 990k | 3.5% | 7% |
| **pointwise (34 layers)** | **30.2M** | **12.82M** | **7.4%** | **92%** |

The pw depth cliff (e2e util by spatial size): H=40: 48–50%, H=20: 42–46%,
H=10: 19–29%, H=5: 6–12%, H=3: 2–5%. Cause: each (output-group × input-
channel) round pays fixed costs — weight load+arm ~22 cy plus kick/idle
framing ~10 cy — while the work quantum shrinks to ceil(nnz/4) ≈ 7 beats at
H=5 and ~3 at H=3. The `load%` column reaches 42–50% of ALL cycles for deep
pw layers; the router/PE machinery itself is fine (conv1 and the H=40 pw
layers prove it).

Fix plan (priority order):
1. **Weight double-buffer + fused rounds** — load the next round's weights
   during the current round, switch banks at the channel boundary instead of
   drain-and-reload. Kills the 22-cy swap and most framing; generic.
2. **Channel-as-PID mode for pw** — map input-channel index → PID so one
   pass processes N_PID input channels at once (boundaries amortize ~9×, or
   more with widened lanes). Reuses FIFO-A lanes, weight banks, and WSP
   routing unchanged; needs a stage-1 bypass (cid = x·E + y, pid = channel)
   and a channel field in the scanner.
3. **Depthwise** — after 1–2, dw grows to ~1/3 of remaining runtime; true
   fix is per-PE front ends (dw is 3.4% of network MACs, so bounded
   acceptance is also defensible).

Projection at 32 mults: deep pw at early-pw efficiency ≈ 2.9M cycles
(~22 fps, ~34% util); with dw ×8 as well ≈ 2.1M (~31 fps, ~49% util).

## Addendum 6: weight double-buffer + fused rounds — measured

`pe_mem.sv` now holds two banks: fills stream into the shadow bank (safe
during routing/MACs — port A never touches the live bank or the exported
WSP), and `wload_done` became a 1-cycle bank swap. The test flows fuse
rounds: kick router → in parallel {next weights → shadow bank, next channel
CSR → FIFO-A} → wait backlog → swap → kick. The serialized per-round cost
drops from ~34 cy (load+arm) to ~3 cy (swap).

Full-network re-run (all layers golden-checked): the pw `load%` column went
to 0.0 everywhere.

| | before | after DB | |
|---|---|---|---|
| network cycles | 13.87M | **8.83M** | 1.57× |
| network util | 7.3% | **11.5%** | |
| fps @100 MHz | 7.2 | **11.3** | |
| pw cycles | 12.82M | 7.79M | deep layers ~1.7× |
| dw cycles | 990k | 990k | (reset-per-pass flow; DB n/a until drain-with-clear) |

Remaining deep-pw structure: pipe% is now 97–99.8% but the per-round quantum
is still tiny (~3–6 cy of work vs ~12–15 cy of kick/idle/swap framing at
H=3–5) — which is exactly what the channel-as-PID batching removes (rounds
amortize ×N_PID).

## Addendum 7: channel-as-PID pointwise + depthwise mosaic — final network

Three RTL features landed (all modes parameter-gated, defaults = legacy;
base suite + apple regression green after each):

1. **PE-parallel weight fill**: `pe_wfill_*` became per-PE lanes — one PID
   per cycle loads all 8 PEs (a kernel row per cycle, ≤9 cycles/round).
2. **Channel-as-PID (`CH_PID`)**: pw layers batch 9 input channels as the 9
   "taps" of a virtual 3x3 kernel. Host lays the batch out as a 9-row CSR
   (row = channel, col = flattened pixel); stage 1 bypasses idgen (pid = row,
   cid = col); everything downstream (WSP routing, weight banks, drain) is
   unchanged. Used when Cin >= 19 (batch preamble amortizes over >= 3 rounds).
3. **Depthwise mosaic (`DW_COLW`)**: 8 channels tiled 3x3 into one composite
   map with zero gaps, padded so the output width is a power of two (cid =
   {row, col} bit-fields). One traditional-conv pass; the router demuxes
   beat lanes to PEs by 2-D CID band (per-lane band masks ANDed into
   lane_valid), and drains walk per-PE 2-D windows. Straddling beats split
   correctly by lane; gap outputs fall in no band and are dropped.

| network (N_PE=8, 32 mults) | cycles | util | fps @100MHz |
|---|---|---|---|
| baseline (this session start) | 13.87M | 7.3% | 7.2 |
| + wgt double-buffer / fused rounds | 8.83M | 11.5% | 11.3 |
| **+ CH_PID pw + dw mosaic** | **2.76M** | **36.6%** | **36.2** |

Per type: conv1 55k @58.1%; dw 990k → 294k (3.4x, util 11.8%); pw 12.82M →
2.42M (5.3x, util 39.1%). Worst-case singles: L44 (960->160 @3x3) 481k →
93.5k (5.1x); L1 (dw 40x40) 120k → 37.1k (3.2x); L43 (dw 3x3) 61.6k → 11.1k
(5.6x). All 52 layers golden-checked vs the real quantized tensors after
every change.

Remaining headroom: deep-pw rounds are still framing-bound (~12-15 cy of
kick/idle/swap around ~20 cy of work at H=3) — wider CH_PID batches (more
FIFO-A lanes) or round fusion would help; dw mosaic preambles (composite
fill+scan, 28-59% of dw cycles) are exposed — the fused-round prefetch
pattern applies but needs per-pass band/weight swap sequencing.

## Addendum 8: adaptive channel batches — deep-pw utils were flow artifacts

The 9-channel CH_PID batch was an artifact of reusing F=3 builds; in
channel-as-PID mode F has no geometric meaning, so building with F=8 gives
64 PID lanes = 64 channels/batch with ZERO RTL change (the driver picks the
largest batch in {9..64} that still leaves >= 3 rounds; build H =
max(W^2, F^2) so the scanner's row coordinate covers the channel index —
found via an aliasing bug at F=8/H=9). FIFO-B backlogs now span ~100+ beats
per round instead of ~13, and the fixed round framing amortizes ~7x.

| network (N_PE=8, 32 mults) | cycles | util | fps @100MHz |
|---|---|---|---|
| addendum 7 (batch=9) | 2.76M | 36.6% | 36.2 |
| **adaptive batches (<= 64)** | **2.40M** | **42.2%** | **41.7** |

Deep layers: L44 93.5k -> 55.6k (35.0%; 8.7x vs the original 481k), L50
47.6%, L51 43.5%, H=5 project layers 37 -> 48%. pw type: 2.42M -> 2.05M
(util 46.1%).

Remaining, decomposed honestly at W=3 (e.g. L44 at 35%): (a) partial-beat
ceiling ~62% — per-channel nnz ~5 of 9 pixels means beats average ~62% full;
packing lanes across PIDs (per-lane PID beats) is an architecture change;
(b) round supply: fill+scan of the next 64-channel batch (~125 cy) slightly
exceeds PE work (~100 cy) — STAGE1_BATCH=8/16 would flip it; (c) ~10-15 cy
of kick/idle/swap framing per round (partly testbench-loop granularity).

## Addendum 9: density x FIFO-B sweep — 4 MACs/PE/cycle confirmed at 1.0/1.0

`FIFOB_D` split from `FIFO_D` (FIFO-A must hold a full channel; FIFO-B is
the swept decoupling depth). Equal-density diagonal 0.1..1.0, pipelined
flow, H=32 F=3 S=1, Cin=3, Cout=8, 5 depths — `gospa_dsweep_B*.txt`.

At 1.0/1.0: pipe util by FIFOB_D = 89.7% (8), 90.3% (32), 93.0% (128),
**99.4% (512 and 2048)** = 3.98 MACs/PE/cycle. Depth is a threshold, not a
gradient: it must leave enough boundary backlog to hide the next channel's
fill+scan (~330 cy here); beyond that, nothing (512 == 2048 exactly).
End-to-end at 1.0 = 82.5%; the residue is the drain (903 cy, 12%) and the
first preamble (352 cy, 5%).

At low density the pipe util is NOT ~= density (26-47% at 0.1-0.3). E2E was
additionally dominated by the DENSE drain walk (903 cycles regardless of
sparsity = 70% of runtime at d=0.1). See Addendum 10 for the measured causal
split — the imbalance hypothesis originally written here turned out to be a
minor factor.

## Addendum 10: WSP-similarity scheduling + DRAIN_W parallel drains — measured

Two changes from the "physical mapping" review:
1. **DRAIN_W** (default 1): the per-lane accumulator banks are flop arrays,
   so the drain now reads DRAIN_W CIDs/PE/cycle (parallel sum trees; window
   drains step DRAIN_W along the column). At DRAIN_W=8 a 900-CID drain is
   116 cycles. (An SRAM-based accumulator would bank by cid-interleave to
   match.)
2. **Scheduling sweep**: pool of 32 output channels per density point, run
   both naive (index order) and compiler-style (grouped by weight count)
   schedules from the same pool — `gospa_dsweep_B512.txt`.

Results (FIFOB_D=512, DRAIN_W=8, da=dw diagonal): at 1.0/1.0 e2e util is now
**92.3%** (pipe 99.4%); the drain residue is gone. But WSP-similarity
scheduling buys only **1.01-1.05x** at any density — the low-density loss is
NOT kernel imbalance (this measurement corrects Addendum 9):

- At HIGH density the pass is PE-bound (max own-beats), so balance matters —
  that is why the apple EQUAL_NW experiment gained 78->93%.
- At LOW density the pass is ROUTER+front-end-bound: the router must walk
  ALL beats of the shared stream while each PE consumes only its dw
  fraction (the multicast dw-bound), and the fixed per-round costs
  (router walk ~54 cy + fill/scan ~45 cy + framing ~10 cy at d=0.1) dwarf
  ~23 cycles of per-PE work. Balance cannot help; wider S2_BEATS (shorter
  walk) and round fusion can.

Full network with DRAIN_W=8: **2.29M cycles, 44.2% util, 22.9 ms @100 MHz
(43.6 fps)** — conv1 64.3%, pw 48.0%, dw 12.7%. Session total: 13.87M ->
2.29M cycles = 6.05x, util 7.3% -> 44.2%.

## Addendum 11: max config — S2_BEATS=16, STAGE1_BATCH=16, DRAIN_W=64 + sparse drain

Per the low-density attribution: the FIFO-A->FIFO-B walk widened to 16
beats/cycle (64 pairs), the scanner to 16 activations/cycle, the drain to 64
CIDs/PE/cycle with non-zero-only lane emission (consumers default absent
CIDs to 0, so correctness is untouched; a real accelerator would emit this
as the next layer's CSR). All regression + sweep + network golden-checked.

Density sweep (`gospa_dseg_S16B16.txt`): d=0.1: 2,020 -> 1,055 cy (e2e 17.4
-> 33.4%; drain 23 -> 6.8%, walk 38 -> 24%); **d=1.0: e2e 96.6%** (3.86
MACs/PE/cycle wall-to-wall; walk 6.6%, fill+scan 4.5%, drain 0.3%, the
remaining 86% is pure PE backlog-chew at ~99% duty).

**Full network: 1.74M cycles, 58.0% util, 17.45 ms @100 MHz (57.3 fps)** —
conv1 72.1%, pw 67.2%, dw 11.8% (dw is now 17% of runtime and the clear next
target; the mosaic gained nothing from the wider walk because its demux
spreads a 16-beat group across bands, duplicating multicast pushes).

Session grand total, same 32 multipliers, every step golden-checked against
the real quantized model: **13.87M -> 1.74M cycles (7.95x), utilization
7.3% -> 58.0%, 7.2 -> 57.3 fps.**

## Addendum 12: metrics tool, architecture sweep, and the best-FPS config

**Tools**: `get_rtl_metrics.py` (density classes 0.1-1.0 + a 6-layer real-
tensor MobileNet subset; FPS/util/latency/GMAC/s at FCLK_MHZ) and
`arch_sweep.py` ((N_PE, N_MULTS) grid on a conv1 + 2-largest-pw proxy,
FIFO-B sensitivity on the winner). Reports: `gospa_rtl_metrics.txt`,
`gospa_arch_sweep.txt`.

**Arch sweep** (proxy cycles, all golden-checked): PE-scaling is PERFECT —
4x4/8x4/16x4/32x4 give 542k/272k/136k/68k cycles (exactly 2.00x per PE
doubling, util flat at 60.0%) because the multicast router feeds any N_PE
from one shared stream. M-scaling is poor: 8x8 is 1.6x slower than 16x4 at
equal multipliers (util 37% vs 60% — wider beats hit the partial-beat
ceiling at MobileNet's small spatial sizes). FIFO-B saturates at >= 1024
(256 costs only 1.7%).

**Winner: N_PE=32 x M=4 (128 mults), FIFO-B >= 1024.** Full network, real
tensors, golden-checked: **667,952 cycles = 6.68 ms @100 MHz = 149.7 fps,
util 37.9%** (conv1 72.1%, pw 65.0%, dw 3.0%). Session total: 7.2 -> 149.7
fps = 20.8x.

The next wall is now unambiguous: the depthwise mosaic is hardwired to a
3x3/8-PE tile grid, so dw did not scale — it is 44% of runtime at 3% util.
Generalizing the grid (6x6 tiles for 32 PEs) projects dw ~293k -> ~75k and
the network to ~450k cycles (~220 fps).

## Cross-checks backing these numbers (conv1 apple runs)

- RTL golden vs PyTorch: PASS, all 32 output channels, both N_PE=8 and 32.
- Monitored executed MACs = functional model exactly (1,040,813).
- Monitored beats = analytic ideal exactly (260,458).
- Useful MACs = 1,027,158 = V1 baseline exactly (same conv).
- Predicted stage2 floor 40,944 vs measured window 41,208 (12 windows × ~22
  cycles of ramp/idle-detect bookkeeping): 99.4%.

## Addendum 13: throughput in ops (GOPS / "TFLOP-style" numbers)

The datapath is all-integer (int16 data, int32 accumulate; synthesis shows
DSP = 0, BRAM = 0 — multipliers live in LUT fabric), so the honest unit is
**OPS, not FLOPS** (1 MAC = 2 ops). Clock from `rtl/synth/RESULTS.md`: V2
post-route Fmax ~= 171 MHz on the Kria KV260 (measured on the 8x4 build).

| metric (winner N_PE=32 x M=4, 128 mults) | @100 MHz | @171 MHz (Fmax) |
|---|---|---|
| peak                                   | 25.6 GOPS | 43.8 GOPS (0.044 TOPS) |
| dense 1.0/1.0 benchmark (98.1% util)   | 25.1 GOPS | 42.9 GOPS |
| MobileNet effective, useful MACs (32.38M / 667,952 cyc) | 9.7 GOPS | 16.6 GOPS |
| MobileNet dense-equivalent (41.59M dense MACs, sparsity credited) | 12.5 GOPS | 21.3 GOPS |
| MobileNet fps                          | 149.7 | 256.0 |

As-synthesized 8x4 config: 32 lanes -> 10.9 GOPS peak @171 MHz.

Caveats: (1) Fmax was measured post-route on the 8x4 build; the 32-PE build
is unrouted — the multicast fanout may cost some MHz. (2) LUT extrapolation
from the sweep (~3.2k LUT/PE incremental) puts 32x4 at ~119k LUTs vs the
KV260's 117k — marginally over; 24x4 (~93k LUTs, 96 lanes, 32.8 GOPS peak,
~192 fps by perfect PE scaling) is the largest config that clearly fits.
(3) Dense-equivalent is the standard sparse-accelerator credit metric:
dense-network MACs / measured time.
