# PE / PE Array — Interface and Operating Guide

> **⚠️ STALE — describes the V1 PE (one kernel per lane).** The `pe.sv` on
> main implements the V2 "one kernel per PE" dataflow: a single Curr weight,
> selected by the beat's PID, feeds all `N_MULTS` multipliers, and the M lanes
> consume M FIFO-B entries that share that PID but carry distinct CIDs
> (per-lane valid masks cover short beats). One WSP per PE is derived from the
> loaded weights by `pe_mem` and exported to the router — there are no
> `wsp_we`/`wsp_lane`/`wsp_data` ports, and weight fill is PID-ordered
> per PE (`wfill_we`/`wfill_pid`/`wfill_val`), not lane/slot addressed.
> Each PE produces ONE output channel per pass, drained `DRAIN_W` CIDs per
> cycle with per-lane banks summed per CID. See the header comments of
> `pe.sv`, `pe_mem.sv`, `pe_fetch.sv`, and `pe_lane.sv` for the current
> design; `testing/pe/test_pe.py` shows the current fill/arm sequence.

The Processing Element is the compute back end of the goSPA accelerator. Each
PE consumes its FIFO-B stream from the APU (`(Axy, PID, CID)` tuples), runs
N_MULTS multipliers in parallel — each holding its own kernel — and produces
one E×E output map per lane. The PE Array (`pe_array.sv`) instantiates N_PE PEs
and ties them to the APU's `fifob_rd_*` ports.

```
                          ┌────────────────── pe ────────────────────────────────┐
                          │                                                      │
                          │   ┌─────────────────────┐                            │
                          │   │  Weight SRAM        │  one per PE, holds         │
   wfill_we   ───────────►│   │  (sram.sv)          │  ALL N_MULTS kernels:      │
   wfill_lane ───────────►│   │  N_MULTS × N_PID    │  addr = lane*N_PID + slot  │
   wfill_slot ───────────►│   │  slots ×            │                            │
   wfill_pid  ───────────►│   │  (DATA_W + PID_W)   │                            │
   wfill_val  ───────────►│   │  bits/word          │                            │
                          │   └────────┬────────────┘                            │
                          │            │  one {val,pid}/cycle to one lane        │
                          │            │  (round-robin refill arbiter)           │
                          │            ▼                                         │
                          │   Per-lane Curr / Next flop pairs (the multiplier's  │
                          │   working window):                                   │
                          │     lane 0: curr_pid[0]/curr_val[0], next_pid[0]/... │
                          │     lane 1: curr_pid[1]/curr_val[1], next_pid[1]/... │
                          │     ...                                              │
   wsp_we    ────────────►│                                                      │
   wsp_lane  ────────────►│   Per-lane WSP register file (N_MULTS × N_PID bits)  │
   wsp_data  ────────────►│   wsp_q[k][p] = 1 iff lane k has a weight at PID p   │
                          │                                                      │
   b_valid/act/pid/cid  ─►│   Selector: for incoming b_pid, each lane picks     │
   b_ready             ◄──│   curr_val[k] or next_val[k] based on its own WSP   │
                          │                                                      │
                          │      mult_0    mult_1    ...    mult_{N_MULTS-1}     │
                          │        │         │                  │                │
                          │        ▼         ▼                  ▼                │
                          │      pe_acc_0  pe_acc_1  ...    pe_acc_{N-1}         │
                          │      (CID-indexed accumulator file, N_CID banks each)│
                          │                                                      │
   drain_start ──────────►│   drain in parallel; each lane streams N_CID beats   │
   drain_busy / done   ◄──│                                                      │
   out_valid[lane]     ◄──│                                                      │
   out_cid[lane]       ◄──│                                                      │
   out_acc[lane]       ◄──│                                                      │
   out_ready[lane]      ──┘                                                      │
                          └──────────────────────────────────────────────────────┘
```

## Parameters (synthesis-time)

| Parameter | Default | Meaning                                                              |
|-----------|---------|----------------------------------------------------------------------|
| `N_MULTS` | 4       | Multiplier lanes per PE = output channels held by this PE (V2).      |
| `N_PID`   | 9       | Kernel positions = F×F. Max weights any lane can hold.               |
| `N_CID`   | 36      | Output positions = E×E. Per-lane CID-indexed accumulator depth.      |
| `DATA_W`  | 16      | Bits per activation and per weight value (signed two's complement).  |
| `ACC_W`   | 32      | Bits per accumulator entry (sized for DATA_W² × small N_PID).        |

**Derived widths** (informational):
`PID_W = clog2(N_PID)`,
`CID_W = clog2(N_CID)`,
`LANE_W = clog2(N_MULTS)`,
`WPTR_W = clog2(N_PID + 1)`,
`WSRAM_DEPTH = N_MULTS × N_PID`,
`WSRAM_AW = clog2(WSRAM_DEPTH)`,
`WSRAM_DW = DATA_W + PID_W`,
`PROD_W = 2 × DATA_W`.

`pe_array.sv` adds one more parameter — `N_PE` — and uses the same `N_MULTS`,
`N_PID`, `N_CID`, `DATA_W`, `ACC_W`.

## Reset and clock

- Single positive-edge clock `clk`.
- **Asynchronous** active-low reset `rst_n` (matches `pe_acc.sv`). Hold low
  for ≥ 2 cycles.
- After reset: PE is in `S_LOAD`, every Curr/Next flop is 0,
  `have_curr/have_next = 0` for every lane, `wptr = 0`, `n_weights = 0`,
  WSP registers all zero, refill arbiter idle. With WSPs at 0 every lane is
  IDLE on every incoming activation, so the PE silently swallows any FIFO-B
  traffic until weights/WSPs are loaded and `wload_done` arms it.

## Weight loading (per-lane SRAM)

The shared on-chip weight SRAM holds all N_MULTS kernels. Each word packs
`{value, pid}`. The host writes lane `k`'s slot `s` via the addressed fill
port:

| Signal       | Width                   | Direction | Role                          |
|--------------|-------------------------|-----------|-------------------------------|
| `wfill_we`   | 1                       | in        | Write strobe.                 |
| `wfill_lane` | `LANE_W`                | in        | Lane (output channel) index.  |
| `wfill_slot` | `WPTR_W`                | in        | Slot inside the lane (0..N_PID-1; PID-monotone). |
| `wfill_pid`  | `PID_W`                 | in        | The PID this weight lives at. |
| `wfill_val`  | `DATA_W` (signed)       | in        | Weight value.                 |

Addresses get composed internally as `lane * N_PID + slot`; the SRAM is one
`sram.sv` instance per PE. The host should write slots in PID-monotone order
(slot 0 = lowest PID weight, ascending). Writes during the operational phase
are accepted at the SRAM level but never read back — re-arm to take effect.

## WSP loading (per-lane)

Each lane has its own N_PID-bit WSP register. WSP bit `p` set means "lane has
a weight at PID `p`"; on incoming `b_pid`, lanes with `wsp[b_pid] = 0` IDLE
(no MAC, no slide). Write one lane's full WSP per cycle:

| Signal      | Width          | Direction | Role                                  |
|-------------|----------------|-----------|---------------------------------------|
| `wsp_we`    | 1              | in        | Write strobe.                         |
| `wsp_lane`  | `LANE_W`       | in        | Lane index.                           |
| `wsp_data`  | `N_PID`        | in        | LSB-first by PID: bit 0 = WSP at PID 0, bit `N_PID-1` = WSP at PID `N_PID-1`. |

The WSP and the loaded sparse weights MUST be consistent: a lane's WSP set
of PIDs must equal the set of `wfill_pid` values written to that lane. The
PE does not check this — inconsistent state can desync the Curr/Next slide
logic.

Safe windows to write WSPs are the same as for weights: between resets / arm
cycles, or any time the PE is `S_LOAD`. Writing during `S_RUN` is unsafe
because the selector samples `wsp_q[k][b_pid]` combinationally each cycle.

## Arm (latch counts, pre-fill Curr/Next)

| Signal        | Width                              | Direction | Role                           |
|---------------|------------------------------------|-----------|--------------------------------|
| `wload_count` | `[N_MULTS][WPTR_W]` packed bus     | in        | Per-lane valid weight count.   |
| `wload_done`  | 1                                  | in        | 1-cycle pulse to arm the PE.   |

`wload_done` triggers the warm-up sequence: the FSM issues `2 × N_MULTS`
sequential SRAM reads, pre-loading slot 0 → Curr and slot 1 → Next for every
lane. Total warm-up = `2 × N_MULTS + 1` cycles. After warm-up the PE
transitions to `S_RUN` and accepts FIFO-B activations.

`pe_array.sv` exposes the per-PE count as a 3-D packed bus
`[N_PE][N_MULTS][WPTR_W]`; `wload_done` is broadcast and arms every PE in
the same cycle.

## FIFO-B input handshake

Standard valid/ready, **PID-monotone** by construction (Stage 2 routing
drains FIFO-A lanes in PID order).

| Signal     | Width             | Direction | Role                                     |
|------------|-------------------|-----------|------------------------------------------|
| `b_valid`  | 1                 | in        | Activation tuple is valid.               |
| `b_act`    | `DATA_W` (signed) | in        | Activation value `Axy`.                  |
| `b_pid`    | `PID_W`           | in        | Kernel position id.                      |
| `b_cid`    | `CID_W`           | in        | Output position id (= row·E + col).      |
| `b_ready`  | 1                 | out       | PE consumes this cycle.                  |

`b_ready` rules (combinational):
- Always 0 outside `S_RUN`.
- Held 0 while any lane's action is SLIDE (no MAC; just needs to advance).
- Held 0 while any lane has a refill in flight from its previous slide.
- Otherwise 1 — every KEEP/UPDATE lane MACs and consumes this cycle; IDLE/
  RETIRED lanes contribute nothing and don't block.

## Drain / per-lane output

Pulse `drain_start` once to stream every lane's accumulator out in parallel.

| Signal         | Width                                | Direction | Role                          |
|----------------|--------------------------------------|-----------|-------------------------------|
| `drain_start`  | 1                                    | in        | 1-cycle pulse.                |
| `drain_busy`   | 1                                    | out       | High while any lane drains.   |
| `drain_done`   | 1                                    | out       | 1-cycle pulse on the last beat. |
| `out_valid`    | `[N_MULTS]`                          | out       | Per-lane: this beat valid.    |
| `out_cid`      | `[N_MULTS][CID_W]` packed            | out       | Per-lane: current CID (= row·E + col). |
| `out_acc`      | `[N_MULTS][ACC_W]` packed (signed)   | out       | Per-lane: accumulator value.  |
| `out_ready`    | `[N_MULTS]`                          | in        | Per-lane: consume this beat.  |

Lanes share the start; each lane independently produces N_CID beats. The
host typically drives `out_ready = {N_MULTS{1'b1}}` so all lanes drain at
one beat/cycle. Total drain time = `N_CID + 1` cycles.

During drain (`drain_busy = 1`) the PE refuses MAC additions
(`add_en = consume && mac_en[k] && !drain_busy`), so it's safe to leave the
FIFO-B input idle without zeroing it.

## Internal FSM and refill arbiter

Three states:

| State     | What's happening                                                          |
|-----------|---------------------------------------------------------------------------|
| `S_LOAD`  | Accepting writes (weights, WSPs). `b_ready = 0`.                          |
| `S_WARM`  | `2 × N_MULTS` cycles of sequential SRAM reads (Curr then Next per lane). `b_ready = 0`. |
| `S_RUN`   | Per-cycle MAC + refill arbiter.                                           |

Per-lane action (combinational) when `b_valid` and `state == S_RUN`:

| Action    | Condition                                            | MAC | Slide |
|-----------|------------------------------------------------------|-----|-------|
| IDLE      | `!have_curr` or `wsp_q[k][b_pid] == 0`               | no  | no    |
| KEEP      | `b_pid == curr_pid[k]`                               | yes | no    |
| UPDATE    | `b_pid == next_pid[k] && have_next[k]`               | yes | yes   |
| SLIDE     | (otherwise)                                          | no  | yes   |

When multiple lanes want to slide on the same cycle, a **priority arbiter**
(lowest lane-index wins) picks one to refill from the SRAM. Other slide-
wanting lanes wait. A lane with `refill_in_flight = 1` (its rdata hasn't
been captured yet) is **not** eligible — it must be served before another
SRAM read for that lane is issued. The result: at most one SRAM read per
cycle, and each slide costs ≥ 2 cycles end-to-end (issue + capture).

## Storage size (per PE)

With defaults (`N_MULTS = 4, N_PID = 9, N_CID = 36, DATA_W = 16, ACC_W = 32`):

| Region                          | Size                                            |
|---------------------------------|-------------------------------------------------|
| Weight SRAM (`sram.sv`)         | `N_MULTS × N_PID × (DATA_W + PID_W)` = `4 × 9 × 20 = 720 b` (rounded to 2^WSRAM_AW deep) |
| Curr/Next flops                 | `N_MULTS × 2 × (PID_W + DATA_W)` = `4 × 2 × 20 = 160 b` |
| WSP register file               | `N_MULTS × N_PID` = `36 b`                      |
| `pe_acc` accumulator banks      | `N_MULTS × N_CID × ACC_W` = `4 × 36 × 32 = 4608 b` |
| **Total (per PE)**              | ≈ **5.5 Kbits**                                 |

The accumulator file dominates; everything else is essentially free at this
size. The synthesizer will likely map the weight SRAM and accumulator banks
to BRAMs (FPGA) or compiled SRAM macros (ASIC) and the Curr/Next + WSP to
flops.

## Operating sequence (per V2 pass)

```
                wfill_we / wfill_lane / wfill_slot / wfill_pid / wfill_val ─┐
                                                                            │ N writes total
                wsp_we / wsp_lane / wsp_data ───────────────────────────────┤ (one per lane
                                                                            │  cycle per write)
                wload_count (drive the per-lane bus) ───────────────────────┘
                                              │
                                              ▼
                                   wload_done (pulse 1 cyc)
                                              │
                          state: S_LOAD → S_WARM (2×N_MULTS+1 cyc)
                                              │
                                              ▼
                                          S_RUN
                              ┌──────────────────────────────────────┐
                              │  drive b_valid / act / pid / cid     │
                              │  wait for b_ready handshake          │
                              │  ... feed all FIFO-B entries ...     │
                              └──────────────────────────────────────┘
                                              │
                                              ▼
                                   drain_start (pulse 1 cyc)
                                              │
                                              ▼
                              ┌──────────────────────────────────────┐
                              │  collect out_valid / cid / acc for   │
                              │  each lane (N_CID beats per lane)    │
                              └──────────────────────────────────────┘
                                              │
                                              ▼
                                   drain_done (1-cyc pulse)
```

For a sequence of input channels (multi-input conv), the `pe_acc`
accumulator files persist across `drain_start` pulses unless reset
(`clear` is tied to 0 in `pe.sv`). The typical flow is:

1. Arm once (load weights, WSPs, pulse `wload_done`).
2. Stream channel 0's FIFO-B (do **not** drain).
3. **Re-pulse `wload_done`** to rewind the Curr/Next slide window back to
   slot 0 / 1 (accumulators are untouched). Optionally rewrite the weight
   SRAM and/or WSP file first if the kernel actually changes.
4. Stream channel 1's FIFO-B (do not drain).
5. ... repeat for each input channel ...
6. After the last channel, pulse `drain_start` and collect every
   `(PE, lane)` E × E output map.

The re-arm in step 3 is essential because every pass slides each lane's
`wptr` forward through the kernel weights; without rewinding, the next
pass starts with most lanes already retired. The re-arm path is handled by
the same FSM that does the initial arm — `wload_done` in `S_RUN`
transitions back to `S_WARM`, which re-fetches Curr/Next from the SRAM
(now showing the new or same weights), then returns to `S_RUN`. Total
re-arm cost = `2 × N_MULTS + 1` cycles.

This is exactly what `fm.goSPA_multichannel` models on the SW side.

## Backpressure summary

| Stall reason                              | Effect                                  |
|-------------------------------------------|-----------------------------------------|
| State ≠ `S_RUN` (loading or warming)      | `b_ready = 0`                           |
| Any lane wants SLIDE                      | `b_ready = 0` until the arbiter clears  |
| Any lane has refill_in_flight             | `b_ready = 0` until that data captured  |
| FIFO-B downstream of the PE (PE-internal) | n/a — the PE drains via `pe_acc`, no further FIFO |

There is no upstream backpressure on the FIFO-B writer beyond what
`apu_stage2.routing` already imposes; the PE simply de-asserts `b_ready`
and the APU's routing stalls accordingly.

## pe_array.sv — what's different

The array is `N_PE` PEs glued in parallel. The fill ports gain a PE select
(`wfill_pe`, `wsp_pe`); the per-PE counts become a 3-D packed bus
`wload_count[N_PE][N_MULTS][WPTR_W]`. The FIFO-B inputs, drain control, and
output streams are all `[N_PE]` (or `[N_PE][N_MULTS]`) replicas of the
single-PE interface.

| Signal         | Width                                | Notes                                 |
|----------------|--------------------------------------|---------------------------------------|
| `wfill_pe`     | `clog2(N_PE)`                        | Selects which PE's SRAM the fill targets. |
| `wsp_pe`       | `clog2(N_PE)`                        | Selects which PE's WSP file. |
| `wload_count`  | `[N_PE][N_MULTS][WPTR_W]`            | Per-(PE, lane) count.                 |
| `fifob_valid/data/ready` | `[N_PE]` (data is `[N_PE][FIFOB_W]`) | One stream per PE, matches `apu.fifob_rd_*`. |
| `out_valid/cid/acc/ready` | `[N_PE][N_MULTS]` (cid/acc packed) | Per-(PE, lane) output streams. |
| `drain_start/busy/done` | 1 | Broadcast / OR-reduce across PEs. |

`drain_busy` is OR'd across PEs; `drain_done` pulses on the trailing edge.
Total output channels per pass = `N_PE × N_MULTS`.

## Verification entry points

| Target                                                      | What it runs                              |
|-------------------------------------------------------------|-------------------------------------------|
| `make MODULE=test_pe`                                       | 4 V1-style single-channel tests + the new V2 multi-lane MobileNet test. Default `N_MULTS=1`. |
| `make MODULE=test_pe N_MULTS=4`                             | Same suite with the V2 PE actually running 4 lanes in parallel. |
| `make sweep_pe N_MULTS=4`                                   | Sweep of 7 layer configs at `N_MULTS=4`.  |
| `make MODULE=test_pe_array N_MULTS=4 N_PE=8`                | Array-level cosim (when its TB is updated to the V2 fill interface). |

The new V2 test (`test_v2_mobilenet_multilane` in
[testing/pe/test_pe.py](testing/pe/test_pe.py)) loads four real MobileNetV2
first-conv red-channel kernels into one PE, computes the union WSP via
`fm.wsp_union`, drives the resulting `fm.goSPA_route` FIFO-B into the DUT,
drains all lanes in parallel, and compares each lane's E×E output against
`fm.conv2d_reference(act, that lane's kernel)`.

## File map

- `pe.sv` — V2 PE: weight SRAM + per-lane Curr/Next + per-lane WSP + per-lane
  `pe_acc`, round-robin refill arbiter.
- `pe_array.sv` — `N_PE` PEs in parallel, addressed fill, broadcast arm/drain.
- `pe_acc.sv` — CID-indexed accumulator file (one per lane); flop-based.
- `../common/sram.sv` — shared SRAM primitive used for the weight store.
