# Software models

Python models of the GoSPA accelerator, used as the golden reference for RTL
verification and for design-space exploration.

- `functional.py` — golden functional model. Reimplements the dataflow (CSR
  decode, PID/CID generation, FIFO-A binning, WSP-gated routing, CID-indexed
  accumulation) bit-exactly; every cocotb testbench checks the RTL against
  it, and it is itself checked against a dense convolution and PyTorch.
- `perf_model.py`, `perf_pe.py`, `sim.py`, `layer.py`, `config.py` — the
  cycle-accounting performance model. Attributes each layer to a Stage-1 /
  Stage-2 / PE / memory bottleneck; this is what motivated widening the
  Stage-2 router.
- `gospa_compile.py` — maps layer tensors onto the accelerator (tiling,
  CSR encoding, per-PE kernel assignment).
- `workloads/` — AlexNet and MobileNetV2 layer descriptors.

```bash
python3 sim.py            # network-level performance model
python3 sim.py --sweep    # activation/weight density sweep
```
