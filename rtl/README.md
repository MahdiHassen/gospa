# RTL

The synthesizable SystemVerilog implementation of GoSPA lives in `V2/`
(the directory name is historical — an earlier design iteration was removed):

```
V2/
  apu/          Activation Processing Unit
    stage1/       act_sram_scanner, csr_decode, position_encode, idgen,
                  zero_act, apu_stage1   (CSR scan + ID generation → FIFO-A)
    stage2/       routing, apu_stage2    (WSP-gated multicast → FIFO-B)
    apu.sv        Stage-1 + Stage-2 + FIFOs
  pe/           pe, pe_array, pe_mem, pe_fetch, pe_lane
  common/       fifo, sram, arith/{mult_pipe, mac_pipe, rca_add}
  gospa.sv      top level (APU + PE array)
  synth/        Vivado flow + resource sweep + RESULTS.md
```

The dataflow is "one kernel per PE": a Stage-2 beat carries up to `N_MULTS`
same-PID activations with distinct CIDs; a single Curr weight selected by the
beat PID feeds all lanes, and each PE's WSP is derived in hardware from its
loaded weights.

The FPGA synthesis flow (`V2/synth/`) resolves RTL paths relative to itself;
`V2/synth/RESULTS.md` records the signed-off post-route numbers.

To run the system-level golden test:

```
cd testing/gospa && make MODULE=test_gospa
```

The `.sv` module headers are the source of truth for port lists and
parameters; `testing/gospa/gospa_tb.py` shows the current host sequence.
