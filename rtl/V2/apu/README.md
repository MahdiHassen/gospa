# APU — Interface and Operating Guide

The Activation Processing Unit (APU) is the front end of the goSPA accelerator.
It receives sparse activations in CSR form, generates `(CID, PID)` matches per
non-zero, applies per-PE WSP gating, and presents `(Axy, PID, CID)` tuples to
the PE array over `N_PE` parallel FIFO-B read ports.

```
                     +-----------------------------------------------------------+
                     |                            apu.sv                         |
   fill_entry_* ---->|  +--------------------+     +-----------+   +----------+  |
   fill_rptr_*  ---->|  | act_sram_scanner   |---->| apu_stage1|-->| apu_stage2|->-> fifob_rd_* (beats)
   scan_*       ---->|  | (SRAM + CSR FSM)   |     | (FIFO-A)  |   | (routing |  |
                     |  +--------------------+     +-----------+   |   + FIFO-B)| |
   wsp (per-PE) ---->|-------------------------------------------------^        |
                     |    (driven by the PE array; each PE derives its own WSP)  |
   s2_start, ...   ->|                                                           |
                     +-----------------------------------------------------------+
```

The module tested by `testing/apu/full/` is `rtl/apu/apu.sv`. Everything below
lives on that top-level interface.

## Parameters (synthesis-time)

| Parameter   | Default | Meaning                                                        |
|-------------|---------|----------------------------------------------------------------|
| `H`         | 32      | Activation map size (H × H, square). Used for coord widths.    |
| `F`         | 3       | Convolution kernel size (F × F).                               |
| `S`         | 1       | Convolution stride.                                            |
| `N_PE`      | 8       | Number of PE / FIFO-B output ports.                            |
| `NUM_MULTS` | 4       | Activations per FIFO-B beat = FIFO-A read width (M).           |
| `N_ROWS`    | 32      | Activation rows the scanner can hold (== row_ptr flop depth).  |
| `N_NZ_MAX`  | 1024    | Max non-zeros the activation SRAM can hold.                    |
| `DATA_W`    | 16      | Bits per activation value.                                     |
| `FIFO_D`    | 64      | Depth of every FIFO-A and FIFO-B (power of 2).                 |

**Derived widths** (informational): `E = (H-F)/S+1`, `N_PID = F²`,
`IDX_W = clog2(H)`, `CID_W = clog2(E²)`, `PID_W = clog2(F²)`,
`PTR_W = clog2(N_NZ_MAX+1)`, `ENT_AW = clog2(N_NZ_MAX)`,
`RPTR_AW = clog2(N_ROWS+1)`, `N_CNT_W = clog2(N_ROWS+1)`.

### Sizing hints

| Bound                        | Why                                                                |
|------------------------------|--------------------------------------------------------------------|
| `N_NZ_MAX ≥ #non-zeros`      | Activation SRAM must fit one channel's CSR entries.                |
| `N_ROWS ≥ H`                 | Scanner walks `scan_n_rows`; row_ptr flops sized to `N_ROWS+1`.    |
| `FIFO_D ≥ max(per-PID entries)` | Stage 2 doesn't drain until `s2_start`; FIFO-A fills first.     |

For a 32×32 padded activation at 50% sparsity, ~512 non-zeros × G² = 2048
emissions distributed over G² lanes → ~512 per FIFO-A. Bump `FIFO_D=2048`
for that workload (the `mobilenet` make target does this).

## Reset and clock

- Single positive-edge clock `clk`.
- Active-low **synchronous** reset `rst_n`. Hold low for at least 2 cycles.
- After reset, all storage is zeroed: `row_ptr_q` flops (every row is
  "empty") and FIFO pointers. The activation SRAM is not reset, but unread
  on read. WSP is an external input (from the PE array), not stored here —
  no broadcasts fire until the PEs have loaded weights.

## Activation loading (CSR)

The host loads each input channel into the on-chip activation SRAM in CSR
form, using two write-only fill ports.

### Port: row_ptr flops

| Signal             | Width        | Direction | Role                           |
|--------------------|--------------|-----------|--------------------------------|
| `fill_rptr_we`     | 1            | in        | Write strobe.                  |
| `fill_rptr_addr`   | `RPTR_AW`    | in        | Pointer index, 0..N_ROWS.      |
| `fill_rptr_data`   | `PTR_W`      | in        | Value to write.                |

Write semantics: `if (fill_rptr_we) row_ptr_q[fill_rptr_addr] <= fill_rptr_data`
on the next rising edge. One pointer per cycle. The CSR convention is
`row_ptr[r]` = entry index where row `r` begins; `row_ptr[N_ROWS]` is the
total non-zero count (one past the end of the last row).

### Port: activation SRAM

| Signal               | Width    | Direction | Role                                |
|----------------------|----------|-----------|-------------------------------------|
| `fill_entry_we`      | 1        | in        | Write strobe.                       |
| `fill_entry_addr`    | `ENT_AW` | in        | Entry index, 0..N_NZ_MAX-1.         |
| `fill_entry_value`   | `DATA_W` | in        | Non-zero activation value (signed). |
| `fill_entry_col`     | `IDX_W`  | in        | Column index (= `Y` coordinate).    |

Each write deposits `{value, col_idx}` into one word of the activation
SRAM. Entries must be written in row-major CSR scan order (consecutive
`fill_entry_addr` values from 0 upward), so that the row_ptr pointers
match. Row index (`X`) is implicit — recovered by the scanner FSM from
`row_ptr_q`.

### Fill ordering

Both fill ports are independent and can be driven in either order. The
canonical sequence used by the testbenches is:

1. Write all `N_ROWS+1` `row_ptr` values via the row_ptr port.
2. Write all non-zeros via the entry port.
3. Pulse `scan_start` to begin the scan.

The fill ports are write-only and unrelated to the read path used during
the scan, so writes can occur while Stage 2 is busy on the *previous*
channel — but the model TBs serialize for simplicity.

## WSP (per-PE, from the PE array)

The Weight Sparsity Pattern that routing uses to gate FIFO-B multicasts is
a direct input to the APU, one WSP per PE. There is no on-chip WSP file:
each PE derives its own WSP from its weight bank (`pe_mem.wsp_q`) and drives
it straight into the router. Loading PE weights is all it takes to set up
routing — in `gospa.sv` the PE array's `wsp` output is wired to `apu.wsp`.

| Signal | Width              | Direction | Role                                 |
|--------|--------------------|-----------|--------------------------------------|
| `wsp`  | `[N_PE][N_PID]`    | in        | Per-PE WSP, LSB-first by PID.        |

Bit ordering: `wsp[k][p] = 1` means PE `k` has a non-zero weight at `PID=p`.
So `wsp[k] = N_PID'b0000_0101` for an F=3 PE means "PE k wants activations
whose PID is 0 or 2". This matches the PE weight bank directly (no reversal).

For the **V2** mapping (multiple kernels per PE), the union of the lane WSPs
is what a PE exports; from the router's perspective every PE always has
exactly one WSP.

### Stable-WSP requirement

- Standalone (APU-only tests): drive `wsp` before `s2_start` and hold it.
- In `gospa.sv`: `wsp` is stable once the PEs are armed (`pe_wload_done`),
  which the host does before `s2_start`.
- Changing `wsp` mid-pass (`s2_busy=1`) is **unsafe**: routing samples it
  combinationally each cycle, so a change mid-drain can split one PID's
  contents across two patterns and corrupt FIFO-B.

## Running a channel (scan + Stage 2)

After fill + WSP load, the host runs one channel by pulsing the two
framing controls.

### Scan control

| Signal              | Width      | Direction | Role                                          |
|---------------------|------------|-----------|-----------------------------------------------|
| `scan_start`        | 1          | in        | 1-cycle pulse to begin walking the SRAM.      |
| `scan_n_rows`       | `N_CNT_W`  | in        | How many rows to walk (1..N_ROWS).            |
| `scan_base_row`     | `IDX_W`    | in        | Global row index of SRAM slot 0 (for `out_x`).|
| `scan_busy`         | 1          | out       | High while the scan is in progress.           |
| `scan_done`         | 1          | out       | 1-cycle pulse when the last tuple is accepted.|

Hold `scan_start` for one cycle. The scanner walks SRAM rows
`scan_base_row .. scan_base_row + scan_n_rows - 1`, emitting one
`(Axy, X=row, Y=col)` tuple per accepted non-zero into apu_stage1. Empty
rows are skipped without firing any SRAM read. Steady-state throughput
is ~1 non-zero per cycle.

### Stage 2 framing

| Signal       | Width | Direction | Role                                        |
|--------------|-------|-----------|---------------------------------------------|
| `s2_start`   | 1     | in        | 1-cycle pulse to begin routing drain.       |
| `s2_busy`    | 1     | out       | High while routing is walking FIFO-A lanes. |
| `s2_done`    | 1     | out       | 1-cycle pulse when the last lane drains.    |

The sequence per channel is:

```
   fill_entry_*  ↘
   fill_rptr_*  ─┴→ wsp stable (PEs armed upstream)
                       │
                       ↓
                   scan_start ─→ ... scan_done ─→  [pipeline drain, a few cycles]
                                                       │
                                                       ↓
                                                   s2_start ─→ ... s2_done
                                                       │
                                                       ↓
                                              FIFO-B read by PEs
```

Concurrency note: the model TBs serialize scan and stage 2 (wait
`scan_done` before `s2_start`) so FIFO-A fills before draining begins.
A future revision can overlap them; backpressure (Stage 2 stalls when
FIFO-B full, which freezes FIFO-A drain, which freezes the scanner via
`apu_stage1.in_ready`) already handles that.

## FIFO-B read interface (consumed by PEs)

Standard valid/ready handshake, `N_PE` ports in parallel. Each read is one
**M-wide beat** (`NUM_MULTS` activations that share a PID, distinct CIDs) —
the router pops up to `NUM_MULTS` same-PID entries per cycle from FIFO-A and
packs them into one beat, so one FIFO-B entry = one beat.

| Signal                | Width                          | Direction | Role                                 |
|-----------------------|--------------------------------|-----------|--------------------------------------|
| `fifob_rd_valid`      | `[N_PE]`                       | out       | Per-PE: 1 if a beat is present.      |
| `fifob_rd_pid`        | `[N_PE][PID_W]`                | out       | Beat PID (shared by all lanes).      |
| `fifob_rd_lane_valid` | `[N_PE][NUM_MULTS]`            | out       | Per-lane: this lane carries a real activation. |
| `fifob_rd_act`        | `[N_PE][NUM_MULTS][DATA_W]`    | out       | Per-lane activation `Axy`.           |
| `fifob_rd_cid`        | `[N_PE][NUM_MULTS][CID_W]`     | out       | Per-lane output-pixel `CID`.         |
| `fifob_rd_ready`      | `[N_PE]`                       | in        | Per-PE: pop the beat this cycle.     |

These line up 1:1 with `pe_array`'s `fifob_*` beat inputs. Lane 0 is the
oldest entry in the beat. A partial beat (`< NUM_MULTS` valid lanes) happens
when a PID's remaining run is shorter than `NUM_MULTS`. Each PE pops
independently; backpressure on any single PE only stalls *its* lane (until
routing needs to multicast into a full FIFO-B, then routing stalls).

## Backpressure summary

End-to-end stall path, in order of who-stalls-whom:

1. PE de-asserts `fifob_rd_ready[k]` → that FIFO-B fills → `wr_ready` low.
2. `routing.sv` sees `b_ready[k] = 0` while `sel[k] = 1` → `all_ready = 0` →
   no pop, no push.
3. FIFO-A lane that routing was draining stops popping → eventually fills.
4. `apu_stage1`'s all-or-nothing join (`s1_ready`) goes low.
5. Scanner's `out_ready` goes low → SRAM read isn't issued; pending entry
   data is held on the read port.
6. Fill ports are unaffected (independent write-only path).

## Operating a sequence of input channels

Each input channel is one full reset-free cycle of:

1. (Optional) reload PE weights if the chunking changed — this updates the
   `wsp` the PEs export.
2. Refill `row_ptr` and the entry SRAM with the new channel's CSR.
3. Pulse `scan_start`; wait for `scan_done`.
4. Pulse `s2_start`; wait for `s2_done`; drain FIFO-B in parallel.

WSPs (via resident PE weights) and SRAM contents persist across channels —
only rewrite what changed. The PE-side accumulator persistence (sum over
input channels) is modeled in software; see
`sw/functional.py:goSPA_multichannel`.

## Verification entry points

| Target                            | What it runs                                                                       |
|-----------------------------------|------------------------------------------------------------------------------------|
| `make MODULE=test_apu SIM=verilator`     | 5-test functional suite, default H=8/F=3/S=1/N_PE=4.                   |
| `make sweep_apu`                  | Sweep of 6 layer configs (paper toy through 10×5).                                 |
| `make mobilenet`                  | RTL ↔ functional cosim on MobileNetV2 first conv (red channel, 32 kernels, V2).    |
| `WAVES=1 make mobilenet`          | Same, also dumps `dump.vcd` for waveform viewing.                                  |
| `make MODULE=test_apu H=… F=… …`  | Arbitrary parameter overrides.                                                     |

Golden references all live in `sw/functional.py`. Stage 1 + Stage 2 routing
through `fm.goSPA_route(...)`; the per-test goldens build their expected
`fifo_b_list` from that.

## File map

- `apu.sv` — top-level wiring.
- `act_sram_scanner.sv` — activation SRAM + CSR-to-coordinate FSM.
- `stage1/apu_stage1.sv` — zero_act → position_encode → idgen → FIFO-A bank.
- `stage2/apu_stage2.sv` — routing → FIFO-B bank.
- `stage2/routing.sv` — drain-and-multicast core.
- `stage1/{zero_act,position_encode,idgen}.sv` — Stage 1 substages.
- `stage1/csr_decode.sv` — standalone CSR decoder, no longer instantiated
  inside the APU (the scanner has replaced it). Kept for the unit test
  in `testing/apu/stage1/test_csr_decode.py`.
- `../common/{fifo,sram}.sv` — shared storage primitives.
