# RTL — two architecture versions

Both dataflow interpretations of GoSPA described in the report live here,
each synthesizable and each with a working testbench tree:

| Dir     | Architecture | Testbenches | Provenance |
|---------|--------------|-------------|------------|
| `V1/`   | Multiple kernels per PE: one activation per beat is broadcast to all `N_MULTS` lanes, each lane holds its own kernel and gates on its own WSP; the APU routes on the per-PE **union** WSP. | `testing/V1/` | Snapshot of commit `1c1617f` ("start mini compiler for large input activations"), extracted verbatim. |
| `V2/`   | One kernel per PE ("1 weight × M activations"): a beat carries up to `N_MULTS` same-PID activations with distinct CIDs; a single Curr weight selected by the beat PID feeds all lanes. WSPs are derived in hardware from the loaded weights. | `testing/` (everything except `testing/V1/`) | Current development tree (moved from `rtl/` at the top level). |

The FPGA synthesis flow (`V2/synth/`) resolves RTL relative to itself and
targets the V2 tree; `V2/synth/RESULTS.md` records post-route numbers for
both versions.

To run the system-level golden tests:

```
cd testing/gospa    && make MODULE=test_gospa      # V2
cd testing/V1/gospa && make MODULE=test_gospa      # V1
```

Note the per-version READMEs inside `V1/` and `V2/` predate some interface
changes within their own eras — the `.sv` headers are the source of truth.
