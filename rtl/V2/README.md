# goSPA Accelerator — Top-Level Interface and Operating Guide

> **⚠️ STALE — describes the V1 (lane-per-kernel) build.** The RTL on main has
> since moved to the V2 "one kernel per PE" dataflow (commit `18e3633` and
> later). Key differences from what this document describes:
>
> - **One kernel per PE**: output channels per pass = `N_PE`, not
>   `N_PE × N_MULTS`. The `N_MULTS` lanes consume up to M FIFO-B activations
>   that share a PID but carry distinct CIDs ("1 weight × M activations").
> - **No WSP load ports**: `apu_wsp_*` / `pe_wsp_*` are gone. Each PE's
>   `pe_mem` derives its WSP from the loaded weights and exports it to the
>   Stage-2 router (`wsp`, `wsp_valid`).
> - **Weight fill is per-PE, PID-ordered**: `pe_wfill_we[N_PE]` /
>   `pe_wfill_pid` / `pe_wfill_val[N_PE]` — no lane/slot addressing. The
>   weight SRAM is double-buffered for round fusion.
> - **Wider host ports**: `FILL_W`-wide CSR fill, `S2_BEATS` router beats,
>   `DRAIN_W`-wide drain streams, plus `CH_PID` (channel-as-PID) and
>   `DW_COLW`/band ports (depthwise mosaic).
>
> The port lists and parameter meanings in `gospa.sv`, `pe/pe.sv`, and
> `apu/apu.sv` headers are the source of truth; `testing/gospa/gospa_tb.py`
> shows the current host sequence. The storage/backpressure discussion below
> is still broadly correct in spirit but the numbers refer to the V1 build.

`gospa.sv` is the SystemVerilog top of the goSPA accelerator. It instantiates
the APU (activation SRAM + Stage 1 ID-gen + Stage 2 routing) and the PE
Array (`N_PE` V2 PEs of `N_MULTS` lanes each), routes the APU's FIFO-B output
to the PE array's input, and exposes one combined host-facing interface.

See [apu/README.md](apu/README.md) for the APU internals and
[pe/README.md](pe/README.md) for the PE internals. This document covers the
top-level glue: how to load weights, WSPs, activations, run a pass, and read
out the result.

```
                  ┌────────────────────────── gospa.sv ──────────────────────────┐
                  │                                                              │
   fill_entry_*  ─┤  ┌────────────── apu ──────────────┐                         │
   fill_rptr_*  ──┤  │  act SRAM + Stage 1 + Stage 2   │ N_PE FIFO-B streams     │
   scan_*       ──┤  │  (CSR ─► (Axy,X,Y) ─► (CID,PID) │      │                  │
   apu_wsp_*    ──┤  │   ─► FIFO-A bank ─► WSP-gated   │      │                  │
   s2_*         ──┤  │   broadcast ─► FIFO-B bank)     │      │                  │
                  │  └─────────────────────────────────┘      │                  │
                  │                                           ▼                  │
   pe_wfill_*   ──┤  ┌──────────────────── pe_array ─────────────────┐           │
   pe_wsp_*     ──┤  │  N_PE × (weight SRAM + Curr/Next + WSP file + │           │
   pe_wload_*   ──┤  │  N_MULTS lanes + per-lane pe_acc)             │           │
                  │  └────────────────────────────────────────────────┘          │
   drain_*      ──┤                  │                                           │
   out_valid    ◄─┤                  ▼                                           │
   out_cid      ◄─┤    [N_PE][N_MULTS] per-channel E x E output streams         │
   out_acc      ◄─┤                                                              │
   out_ready    ──┤                                                              │
                  └──────────────────────────────────────────────────────────────┘
```

## Parameters

Picked at synthesis time; all are passed straight through to `apu` and/or
`pe_array`.

| Parameter   | Default | Meaning                                                                  |
|-------------|---------|--------------------------------------------------------------------------|
| `H`         | 32      | Activation map width (H × H, square). Pad in the host.                   |
| `F`         | 3       | Conv kernel size (F × F).                                                |
| `S`         | 1       | Conv stride.                                                             |
| `N_PE`      | 8       | Number of PEs in the array (and FIFO-B ports).                           |
| `N_MULTS`   | 4       | Multiplier lanes per PE (= output channels held inside one PE, V2).      |
| `N_ROWS`    | 32      | Activation SRAM depth in rows. Pad to `H` minimum.                       |
| `N_NZ_MAX`  | 1024    | Max non-zeros in the activation SRAM.                                    |
| `FIFO_D`    | 64      | Depth of every FIFO-A and FIFO-B (power of 2).                           |
| `DATA_W`    | 16      | Bits per activation and weight value (signed).                           |
| `ACC_W`     | 32      | Bits per accumulator entry.                                              |

**Derived widths** (informational):
`E = (H-F)/S + 1`,
`N_PID = F²`,
`N_CID = E²`,
`IDX_W = clog2(H)`,
`CID_W = clog2(N_CID)`,
`PID_W = clog2(N_PID)`,
`PESEL_W = clog2(N_PE)`,
`LANE_W = clog2(N_MULTS)`,
`WPTR_W = clog2(N_PID+1)`,
`WSRAM_AW = clog2(N_PID)`,
`ENT_AW = clog2(N_NZ_MAX)`,
`RPTR_AW = clog2(N_ROWS+1)`,
`N_CNT_W = clog2(N_ROWS+1)`,
`FIFOB_W = DATA_W + PID_W + CID_W`.

**Total output channels per pass = `N_PE × N_MULTS`** (V2 mapping).

## Reset and clock

- Single positive-edge clock `clk`.
- Active-low reset `rst_n`. The APU sub-blocks use synchronous reset; the
  PE sub-blocks use asynchronous reset (inherited from `pe_acc.sv`). Hold
  `rst_n` low for ≥ 2 cycles. Lint waives `SYNCASYNCNET` at the top.
- After reset: every storage is zero (activation SRAM unread, row_ptr flops,
  APU WSP file, per-PE weight SRAMs, per-PE WSP files, accumulator banks).
  Nothing is consumed from FIFO-B until weights + WSPs are loaded and the
  PE array is armed.

## Host-facing interface

The host's job is essentially:

1. **One-time / between layers** — load PE weights and per-(PE, lane) WSPs.
2. **Per input channel** — load APU's per-PE union WSP, fill activation SRAM,
   pulse `scan_start`, then `s2_start`.
3. **At the end of a multi-input conv** — pulse `drain_start` and collect
   the `N_PE × N_MULTS` output channels.

The signals fall into five groups.

### 1. Activation SRAM fill (CSR-encoded)

Host writes the activation as CSR before each input channel. The APU stores
the entry stream in `sram.sv` and the `row_ptr` table in a flop array.

| Signal              | Width             | Direction | Role                                  |
|---------------------|-------------------|-----------|---------------------------------------|
| `fill_entry_we`     | 1                 | in        | Entry write strobe.                   |
| `fill_entry_addr`   | `ENT_AW`          | in        | Entry index (CSR scan-order, 0..N-1). |
| `fill_entry_value`  | `DATA_W`          | in        | Non-zero activation value (signed).   |
| `fill_entry_col`    | `IDX_W`           | in        | Column index (= Y coordinate).        |
| `fill_rptr_we`      | 1                 | in        | row_ptr write strobe.                 |
| `fill_rptr_addr`    | `RPTR_AW`         | in        | row_ptr index, 0..N_ROWS.             |
| `fill_rptr_data`    | `PTR_W`           | in        | Pointer value (entry index of row r). |

One write per cycle on each port; the two ports are independent so the host
can interleave row_ptr and entry writes, but it's simplest to sequence them
(row_ptr first, then entries in row-major order).

### 2. APU per-PE union WSP

The APU's `wsp_file` holds one WSP per PE — the **UNION** of that PE's
N_MULTS per-lane WSPs (the host pre-computes this; see `fm.wsp_union(...)`
in `sw/functional.py`). The APU's routing uses these to gate FIFO-B
broadcasts.

| Signal           | Width      | Direction | Role                                   |
|------------------|------------|-----------|----------------------------------------|
| `apu_wsp_we`     | 1          | in        | Strobe.                                |
| `apu_wsp_waddr`  | `PESEL_W`  | in        | Which PE to update.                    |
| `apu_wsp_wdata`  | `N_PID`    | in        | New WSP, **MSB-first by PID** (bit `N_PID-1` = PID 0). |

Note the MSB-first ordering — `apu_stage2.routing` reads
`wsp[k][N_PID-1 - cur]`. The PE's per-lane WSP file uses LSB-first ordering
(see group 4 below); the two are intentionally different to mirror the
weight-SRAM metadata layout in the paper.

### 3. Scan + Stage 2 framing

After the activation SRAM is filled, the host kicks the APU front end:

| Signal              | Width      | Direction | Role                                          |
|---------------------|------------|-----------|-----------------------------------------------|
| `scan_start`        | 1          | in        | Pulse to begin walking the SRAM.              |
| `scan_n_rows`       | `N_CNT_W`  | in        | Rows to scan (1..N_ROWS).                     |
| `scan_base_row`     | `IDX_W`    | in        | Global row index of SRAM slot 0.              |
| `scan_busy`         | 1          | out       | High while scanning.                          |
| `scan_done`         | 1          | out       | 1-cycle pulse when the last tuple is accepted. |
| `s2_start`          | 1          | in        | Pulse to begin routing FIFO-A → FIFO-B.       |
| `s2_busy`           | 1          | out       | High while routing.                           |
| `s2_done`           | 1          | out       | 1-cycle pulse on the last lane drain.         |

The conventional sequence per input channel is `scan_start → wait scan_done
→ s2_start → wait s2_done`. The PE array consumes the FIFO-B stream while
Stage 2 is producing it (no separate trigger).

### 4. PE weights + per-(PE, lane) WSPs + arm

Host writes each PE's weight SRAM one `{lane, slot, pid, val}` at a time,
each PE's per-lane WSP file via `pe_wsp_*`, drives per-(PE, lane) valid-weight
counts on `pe_wload_count`, then pulses `pe_wload_done` to arm every PE
simultaneously. Each PE then issues `2 × N_MULTS` SRAM reads to pre-fill
its Curr/Next flops (~9 cycles for `N_MULTS=4`).

| Signal              | Width                                    | Direction | Role                                  |
|---------------------|------------------------------------------|-----------|---------------------------------------|
| `pe_wfill_we`       | 1                                        | in        | Weight write strobe.                  |
| `pe_wfill_pe`       | `PESEL_W`                                | in        | Target PE.                            |
| `pe_wfill_lane`     | `LANE_W`                                 | in        | Target lane inside that PE.           |
| `pe_wfill_slot`     | `WSRAM_AW`                               | in        | Slot inside the lane (PID-monotone).  |
| `pe_wfill_pid`      | `PID_W`                                  | in        | The PID this weight lives at.         |
| `pe_wfill_val`      | `DATA_W` signed                          | in        | Weight value.                         |
| `pe_wsp_we`         | 1                                        | in        | Per-lane WSP write strobe.            |
| `pe_wsp_pe`         | `PESEL_W`                                | in        | Target PE.                            |
| `pe_wsp_lane`       | `LANE_W`                                 | in        | Target lane.                          |
| `pe_wsp_data`       | `N_PID`                                  | in        | New WSP, **LSB-first by PID** (bit 0 = PID 0). |
| `pe_wload_count`    | `[N_PE][N_MULTS][WPTR_W]` packed         | in        | Per-(PE, lane) valid weight count.    |
| `pe_wload_done`     | 1                                        | in        | 1-cycle pulse — arms every PE.        |

### 5. Drain + per-channel output streams

| Signal         | Width                                    | Direction | Role                                         |
|----------------|------------------------------------------|-----------|----------------------------------------------|
| `drain_start`  | 1                                        | in        | 1-cycle pulse to begin draining accumulators.|
| `drain_busy`   | 1                                        | out       | High while any lane drains.                  |
| `drain_done`   | 1                                        | out       | 1-cycle pulse on the trailing edge.          |
| `out_valid`    | `[N_PE][N_MULTS]`                        | out       | Per-(PE, lane): this beat is valid.          |
| `out_cid`      | `[N_PE][N_MULTS][CID_W]` packed          | out       | Per-(PE, lane): CID (= row·E + col).         |
| `out_acc`      | `[N_PE][N_MULTS][ACC_W]` packed signed   | out       | Per-(PE, lane): accumulator value.           |
| `out_ready`    | `[N_PE][N_MULTS]`                        | in        | Per-(PE, lane): consume this beat.           |

Each lane streams `N_CID` beats; the host typically drives all `out_ready`
bits high so the whole array drains in `N_CID + 1` cycles. The lane's
`(channel_index, row, col)` mapping is implicit from the V2 chunking:

```
output channel index = pe * N_MULTS + lane
row    = CID / E
col    = CID % E
```

## How to load and run (per input channel)

```
1) (Once per layer)
   for pe in 0..N_PE-1:
     for lane in 0..N_MULTS-1:
       for slot, (pid, val) in enumerate(per_kernel_sw[pe][lane]):
         pe_wfill_we = 1; pe_wfill_pe = pe; pe_wfill_lane = lane;
         pe_wfill_slot = slot; pe_wfill_pid = pid; pe_wfill_val = val
       pe_wsp_we = 1; pe_wsp_pe = pe; pe_wsp_lane = lane;
       pe_wsp_data = LSB-first WSP for this lane's kernel
   pe_wload_count = pack {per-(pe, lane) #weights}
   pulse pe_wload_done   # warm-up = 2*N_MULTS+1 cycles

2) (Per input channel)
   for pe in 0..N_PE-1:
     apu_wsp_we = 1; apu_wsp_waddr = pe;
     apu_wsp_wdata = MSB-first union of that PE's lane WSPs

   write row_ptr table (N_ROWS+1 pointers) via fill_rptr_*
   write entries (N non-zeros) via fill_entry_*

   pulse scan_start (with scan_n_rows = H, scan_base_row = 0)
   wait scan_done

   pulse s2_start
   wait s2_done

   # PE accumulators have absorbed this channel's partial contribution.
   # For multi-input-channel layers, repeat step 2 for each channel.

3) (Once per layer, after the LAST input channel)
   pulse drain_start
   drive out_ready = all-ones
   collect out_valid / out_cid / out_acc until drain_done
```

For a **multi-input-channel conv** (e.g. RGB → 32 output channels), repeat
step 2 for each input channel. Critically, **re-pulse `pe_wload_done`** at
the start of every input channel (whether or not the weights actually
changed): the PE's per-lane Curr/Next slide window advances forward
through `wptr` over the course of a pass and doesn't auto-rewind, so the
next pass would start with most lanes already retired without the re-arm.
`pe_acc` accumulators are **not** disturbed by re-arm (`clear` is tied to 0
in `pe.sv`), so partial sums absorbed in pass N stay put for pass N+1.

The PE weight SRAM contents only need to be **rewritten** if the kernel
actually changes (e.g., switching from red-channel weights to green-channel
weights between input channels of an RGB conv). For "same kernel, different
activation" passes, just re-pulse `pe_wload_done` — the SRAM contents are
unchanged and the warm sequence will re-fetch slot 0 → Curr and slot 1 →
Next from the same data.

For a **multi-tile output conv** (more than `N_PE × N_MULTS` output
channels), drain the current tile, reload PE weights for the next tile, and
repeat.

## Operating sequence (timing summary)

```
  HOST ─►  load PE weights + PE WSPs + counts ──► pulse pe_wload_done
                                                      │  ~ 2*N_MULTS+1 cyc warmup
                                                      ▼
  HOST ─►  load APU per-PE union WSPs ─────────► (~ N_PE cycles)
  HOST ─►  load activation as CSR ────────────► (~ N_ROWS+1 + N_NZ cycles)
                                                      │
                                                      ▼
  HOST ─►  pulse scan_start ────────────────► wait scan_done
                                                      │  ~ 1 cyc / non-zero
                                                      ▼
  HOST ─►  pulse s2_start ───────────────────► wait s2_done
                                                      │  ~ 1 cyc / FIFO-A entry
                                                      │  PE array consumes FIFO-B
                                                      │  in parallel, MACs into
                                                      │  per-lane pe_acc banks
                                                      ▼
                                          [loop back to "load activation"
                                           for next input channel]
                                                      │
  HOST ─►  pulse drain_start ────────────────► collect ─► drain_done
                                                      │  N_CID + 1 cyc
                                                      ▼
                                          per-channel E × E output maps
```

## Backpressure (end-to-end)

A full FIFO-B in any PE stalls Stage 2 routing, which stalls the FIFO-A
drain, which (in concert with the front-end scanner's `out_ready`
handshake) would stall the APU scanner if scan and Stage 2 were overlapped.
The model TBs serialize them via `scan_done` / `s2_start`.

Per-lane stalls inside a PE (any lane needs a SLIDE refill from its weight
SRAM) de-assert that PE's FIFO-B `b_ready`. Routing's all-or-nothing
multicast holds the FIFO-A pop until every selected PE is ready, so a
slow-sliding lane in any PE briefly stalls the whole array.

## Storage size (per pass)

With the defaults (`H=32, F=3, S=1, N_PE=8, N_MULTS=4, N_ROWS=32,
N_NZ_MAX=1024, DATA_W=16, ACC_W=32`):

| Region                                                       | Size              |
|--------------------------------------------------------------|-------------------|
| Activation SRAM (1 / APU)                                    | ~21.5 Kbit        |
| APU WSP file (`N_PE × N_PID` flops)                          | 72 bit            |
| FIFO-A bank (`N_PID × FIFO_D × FIFOA_W`)                     | ~ 6 Kbit          |
| FIFO-B bank (`N_PE × FIFO_D × FIFOB_W`)                      | ~ 11 Kbit         |
| PE weight SRAMs (1 / PE × `N_MULTS × N_PID × (DATA_W+PID_W)`)| ~5.5 Kbit         |
| PE WSP files (1 / PE × `N_MULTS × N_PID`)                    | 288 bit           |
| PE Curr/Next flops (1 / PE × `N_MULTS × 2 × (PID_W+DATA_W)`) | ~1.3 Kbit         |
| PE `pe_acc` banks (1 / PE × `N_MULTS × N_CID × ACC_W`)       | ~36 Kbit          |
| **Total**                                                    | **≈ 80 Kbit ≈ 10 KB** |

The accumulator file dominates; the activation SRAM is the next-largest.
Both should map to BRAMs on FPGA or compiled SRAM macros on ASIC.

## Verification entry points

| Target                                                 | What it runs                                                       |
|--------------------------------------------------------|--------------------------------------------------------------------|
| `cd testing/gospa && make MODULE=test_gospa`           | Default config (small, 4 PEs × 4 lanes = 16 channels).             |
| `cd testing/gospa && make mobilenet`                   | Three end-to-end tests on MobileNetV2's first conv (32 channels): single-channel; 2-pass same-kernel accumulation; **4-pass partial-sum accumulation with mid-stream red→green weight swap**. |
| `cd testing/gospa && WAVES=1 make mobilenet`           | Same, also dumps `dump.vcd` for waveform viewing.                  |
| `cd testing/gospa && make MODULE=test_gospa H=... F=... ...` | Arbitrary parameter overrides.                              |

The test ([../testing/gospa/test_gospa.py](../testing/gospa/test_gospa.py))
does the full host sequence for one input channel:

1. Loads `N_PE × N_MULTS` red-channel MobileNet kernels (V2 chunked).
2. Loads per-PE union WSPs into the APU's `wsp_file`.
3. Loads per-(PE, lane) WSPs into each PE's WSP file.
4. Drives `pe_wload_count`, pulses `pe_wload_done` (warm-up).
5. Generates a synthetic sparse padded activation.
6. Streams it as CSR into the APU's activation SRAM.
7. Pulses `scan_start` → waits `scan_done`.
8. Pulses `s2_start` → waits `s2_done`.
9. Pulses `drain_start`, collects every `(PE, lane)` E × E map.
10. Compares each map against `fm.conv2d_reference(act, kernel)`.

A pass = every one of the `N_PE × N_MULTS` output channels matches the
functional model's dense-conv golden, exactly.

## File map

- `gospa.sv` — this module: instantiates `apu` + `pe_array`, wires
  N_PE FIFO-B streams between them, exposes one combined host interface.
- `apu/` — APU (activation SRAM scanner + Stage 1 + Stage 2 routing + FIFO-B).
  See [apu/README.md](apu/README.md).
- `pe/` — V2 PE array (N_PE PEs of N_MULTS lanes each, with per-lane weight
  SRAM access, WSP, and accumulator). See [pe/README.md](pe/README.md).
- `common/` — `fifo.sv`, `sram.sv` primitives.
