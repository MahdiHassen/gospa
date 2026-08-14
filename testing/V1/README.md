# V1 testbenches

Testbench tree for the V1 architecture (`rtl/V1/` — multiple kernels per
PE, union-WSP routing), extracted from commit `1c1617f` together with the
RTL it verifies. Changes made relative to that snapshot:

- Makefile paths repointed (`rtl` → `rtl/V1`, one directory deeper).
- `sw_v1/functional.py` — the V1-era functional model, vendored here so
  these tests never pick up the current (V2-oriented) `sw/functional.py`.
  Test `sys.path` entries point at it.
- `apu/stage1/Makefile` gained `--Wno-DECLFILENAME` (newer Verilator
  treats that lint as fatal under `-Wall`).
- `pe/test_pe_array.py` was **dropped**: it targeted a single-kernel-per-PE
  fill interface that predates even the `1c1617f` RTL (already broken at
  its own commit). `pe_array` is fully exercised by the `gospa/` system
  tests, which instantiate it inside the top.

- `gospa/test_gospa_perf.py` is **new**: a density-swept performance
  measurement (cycles / useful MACs / utilization, golden-checked) on the
  same workload generator and seed as V2's `test_gospa_dseg.py`, used for
  the paper's RTL-measured V1-vs-V2 comparison. Build it with
  `FIFO_D=1024`: V1's serial scan-then-route flow deadlocks if a per-PID
  FIFO-A fills mid-scan, so FIFO-A must hold a whole channel at H=32.
  Invoked by `artifact/run.sh v1v2`.

All targets pass under Verilator 5.048 / cocotb 2.0:

```
cd gospa      && make MODULE=test_gospa   # 3/3   (+ `make mobilenet` 3/3)
cd pe         && make MODULE=test_pe      # 5/5
cd apu/full   && make                     # 5/5
cd apu/idgen  && make                     # 2/2
cd apu/routing&& make                     # 5/5
cd apu/stage1 && make MODULE=test_apu_stage1   # 5/5 (+ csr_decode 8/8, zero_act 8/8)
```
