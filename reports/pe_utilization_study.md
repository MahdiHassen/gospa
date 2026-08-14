# PE Multiplier-Utilization Study

**Metric:** `multUtil = useful MACs / (cycles x N_MULTS)`, plus input/output
data transfer per useful MAC. This report explains *why* the numbers are what
they are; every claim below is backed by RTL measurement and a cycle-exact
model.

## Artifacts

| What | Where |
|---|---|
| PE sweep harness (loss split, seeds, CSV) | `testing/pe/test_pe.py::test_perf` (knobs: `NSEEDS`, `PERF_TAG`, `H/F/S/NUM_MULTS`) |
| Measured sweeps: M=1/2/4/8, H=8/10/16, F=3/5, S=1/2, 5 seeds x 25 density cases | `testing/pe/perf_sweep/pe_perf_<cfg>.{txt,csv}` |
| Cycle-exact replay predictor (validates our understanding) | `testing/pe/perf_model_check.py` |
| Real-workload sparsity profile (quantized MobileNetV2 on the apple image) | `testing/ref/mobilenet_sparsity.py`, `mobilenet_sparsity_apple{80,224}.csv` |
| Real-layer end-to-end run (full gospa RTL, golden-checked) | `testing/gospa/test_gospa_apple.py`, `gospa_apple_perf.txt` |

## The accounting identity

Every streaming cycle is one of three things, so per case:
`multUtil% + idle% + stall% = 100` exactly.

- **idle** — union-WSP gating: the FIFO-B stream is admitted by the *union* of
  the M lanes' weight-sparsity patterns; a lane idles on an admitted beat iff
  its own kernel has no weight at that PID.
- **stall** — a whole-PE 1-cycle bubble whenever any lane's needed weight is
  neither held (Curr) nor prefetched (bank register): cold-start (first hit
  per lane per arm), activation-sparsity PID skips, and run-length-1
  adjacency. All simultaneously-skipping lanes share one bubble (per-lane
  SRAM banks).
- Fixed **load/arm/drain** overhead sits outside the streaming window and is
  reported separately (`latency` column, CSV `load_cycles`/`drain_cycles`).

## Understanding is proven, not just plausible

`perf_model_check.py` regenerates every sweep case offline and replays
`pe_fetch.sv` beat-by-beat. It reproduces **consumed, macs, and stalled
bit-exactly on all 875 measured rows** (7 configs x 25 cases x 5 seeds),
including 750 rows on configs it was never tuned on (M=1, F=5, stride-2).
Auxiliary identities verified on all rows: `consumed + stalled = offered`,
`stalled = stall_fetch`, `cycles = offered + load + drain`. Every lost cycle
is accounted for.

## Why utilization is what it is — the two laws

**1. Union-gating law (the utilization lever).** Per-lane hit rate on an
admitted beat is `dw_realized / (1 - (1-dw_realized)^M)`. Fit across all 175
pooled sweep points: R^2 = 0.996, MAE = 0.010. Consequences:

- Weight density sets utilization; activation density barely touches it
  (act10_wgt100 -> 97.8% util; act100_wgt10 -> 36.0% at M=4).
- Activation sparsity instead shortens the stream: straight speedup
  (10% act density -> 7x fewer streaming cycles at ~constant util when
  weights are dense).
- More lanes dilute util toward a floor of `dw` but *always* add absolute
  throughput (~M x dw MACs/cycle): at 50/50, M=1..8 gives util
  98.6/62.4/53.4/49.1% while MACs/cycle rises 0.99/1.25/2.14/3.93.
  Doubling gain saturates at low dw (x1.43 at 30/30 for M4->M8).
- Frontier: to keep util > 70%, need realized dw >= ~0.55 (M=2), ~0.7 (M=4),
  ~0.8 (M=8). Collapse below 40% at dw <= 0.3 (M=4) / 0.4 (M=8); never for M<=2.

**2. Stall law (a short-stream artifact).** Stalls are ~cold-start
(1.0/1.4/1.8/2.6 beats for M=1/2/4/8) plus PID-skip events at very sparse
activations; `stall% ~ 1.4/consumed` (r = 0.89). They exceed 10% of cycles
only when the admitted stream is under ~45 beats (all such cases have
da <= 0.3 at H=8; H=16 amortizes the same ~3 beats to 4%). At production
H (>= 40) stalls are noise.

## What real workloads look like (quantized MobileNetV2, apple image)

MAC-weighted over 53 layers: **activations 79% dense, weights 96.7% dense**
(identical at 80x80 and 224x224).

- Weight sparsity — the one thing that hurts utilization — is essentially
  absent unless the team prunes. The only sub-90% layer is the first conv
  (78.5% per-position density).
- Activation sparsity is positional: expansion 1x1 inputs ~93-99% dense (no
  ReLU upstream), ReLU6-fed depthwise/projection inputs 20-80% -> that is
  where cycle-count speedup lives.
- Caveat for scoping: 88% of MobileNetV2 MACs are 1x1 convs where F^2-position
  sparsity does not exist (NUM_PID = 1); the PID/WSP mechanism operates on the
  3x3 layers (~9% of MACs) unless layers are mapped differently.

## The actual layer on the actual chip (full gospa RTL)

MobileNetV2 first conv, real 80x80 apple input (99-100% dense), 3 input
channels accumulated, 8 PEs x 4 lanes = all 32 output channels. All outputs
match the PyTorch-derived golden; monitored FIFO-B beats match the functional
model exactly.

| Window | Cycles | multUtil |
|---|---|---|
| stage2 streaming only | 40,975 | **78.3%** |
| + scan (compute) | 60,374 | 53.2% |
| end-to-end incl. fills + drain | 82,114 | 39.1% |

- 78.3% streaming util is the union-gating prediction for dw = 0.785 kernels
  — the model transfers to real data unchanged. Idle is ~100% of the loss;
  stalls are 4-7 beats per PE per layer (cold starts); **starvation is 0.2%**:
  APU stage2 feeds all 8 PEs one beat/PE/cycle, so the PE, not the front end,
  is the streaming bottleneck.
- Per-PE util spreads 50.8%-98.0% purely by kernel-group density (all PEs
  consume the same-length stream; the spread is where MACs happen, not a
  cycle-count imbalance).
- The gap from 78% to 39% is **phase serialization**: per channel pass,
  act-fill (~6.4k) + scan (~6.4k) cycles run with all 32 multipliers idle,
  then stage2 (~13.6k) does all the work. Drain adds 1.5k once.
- Data transfer: host-side traffic is tiny (CSR fill 0.4 b/MAC, weights
  ~0, drain 1.5 b/MAC); internal FIFO-B broadcast is 9.9 b/MAC with each
  31-bit entry feeding 3.14 MACs on average.

## Mapping the layer onto a realistically-sized gospa (tiled reuse)

`sw/gospa_compile.py`'s schedule was executed on the real RTL
(`testing/gospa/test_gospa_tiled.py`): the same 80x80 layer, tiled onto a
fixed H x H gospa with the F-S halo, channels innermost, one drain per tile,
full reset between tiles. Stitched outputs match the full-layer golden
exactly at both candidate sizes.

| | monolithic H=80 | H=64 (2x2 tiles) | H=32 (3x3 tiles) |
|---|---|---|---|
| pe_acc storage (dominant) | 1.56 Mb | 0.98 Mb | **0.23 Mb** |
| schedule | 3 passes | 12 passes | 27 passes |
| total cycles | 82,114 | 91,340 (+11%) | 96,004 (+17%) |
| stage2 streaming util | 78.3% | 77.9% | 77.3% |
| end-to-end util | 39.1% | 36.3% | 34.6% |
| activation refetch (halo) | 1.00x | 1.05x | 1.10x |
| stage2 starvation | 0.2% | 0.6% | 1.4% |

- **Tiling is cheap.** A 7x smaller accumulator costs 17% wall clock and ~1
  point of streaming utilization. Union-gating behavior is size-invariant
  (~78% at every tile size, as the model predicts), and per-PE supply holds.
- The extra cycles are per-pass constants: 27 weight loads instead of 3
  (still only 0.11 b/MAC), more scan/stage2 tails, 9 small drains.
- **Boundary waste:** edge tiles compute windows whose outputs fall past the
  39x39 layer edge and get cropped at stitch — +3.4% MACs (1,062,354 computed
  vs 1,027,158 retained) at both tile sizes. Util on *retained* work is
  ~74.7% (H=32). A compiler-side clip of the last tile's scan window would
  recover it.
- The phase profile is unchanged by tiling (stage2 ~45%, scan ~23%,
  act-fill ~23%): scan/fill overlap remains the dominant system lever
  regardless of tile size.
- Full-MobileNetV2 note: `F` and `S` are synthesis parameters, so one build
  serves one layer-shape class. The 3x3-s2 build covers the first conv only;
  pointwise 1x1 (88% of network MACs) and depthwise layers need their own
  builds ("a bunch of gospas") or runtime-configurable F/S.

## Measurement caveats (methodology audit)

- `multUtil%` is streaming-phase, ideal-producer, per-PE — an **upper bound**
  on system utilization. The gospa run above closes that gap with measured
  starvation/phase numbers.
- `inBits/mac` in the PE sweep is input-side only; drain traffic
  (fixed 4.6 kb/pass at H=8) dominates per-MAC transfer at sparse corners
  (~214 b/MAC at 10/10). Drain amortizes only via multi-channel accumulation.
- Sparse-corner rows (dw <= 0.2) are noisy at NSEEDS=5 (union-size CV up to
  46%; the all-zero-kernel patch fires on 39% of dw=0.1 kernels). Trends are
  robust; individual sparse-row deltas < ~5 points are not.
- `sw/perf_pe.py` still models the pre-rework microarchitecture (P=M column
  reload + double buffer). Its stall numbers are wrong for the current RTL in
  both directions — retire or recalibrate it; `testing/pe/perf_model_check.py`
  is the faithful model now.
- Bernoulli sparsity vs. real structure: validated as identical behavior on
  the real layer above (78.3% vs 78% predicted), but pruned-weight structure
  (if pruning happens) should be re-measured.

## Implications (levers, in impact order — not yet acted on)

1. **Phase overlap** at the gospa level (scan/act-fill concurrent with
   stage2, double-buffered activation SRAM): 39% -> ~78% end-to-end on the
   real layer, a 2x wall-clock win, no PE change.
2. **Weight density is destiny** for util: with unpruned networks (dw ~ 0.97)
   the PE already runs near-roofline; if pruning is planned, co-design it
   with M (keep realized dw >= 0.7 at M=4, or group kernels by shared WSP to
   raise within-PE overlap).
3. **Lane count trades util% for throughput** predictably (`~M x dw`
   MACs/cycle); choose M from the target workload's dw, not from util alone.
4. Stalls and per-PE imbalance are not worth optimizing: measured at <0.1%
   and 0 cycles respectively on the real layer.
