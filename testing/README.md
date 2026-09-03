# Testing

All testbenches and simulation scripts for the GoSPA RTL. The directory
structure mirrors `rtl/V2/` — each block has a corresponding test directory
here. Testbenches are cocotb-based (Python co-simulation) except the FIFO,
which has a plain SystemVerilog testbench.

Every system-level test golden-checks the RTL against the Python functional
model (`sw/functional.py`), which is itself checked against a dense
convolution and against PyTorch.

## Dependencies

| Tool | Version | Notes |
|---|---|---|
| Verilator | ≥ 5.0 | Primary simulator |
| Icarus Verilog | 12.0 | Alternative (`SIM=icarus`); default in `arith/` |
| cocotb | 1.9.2 / 2.x | RTL ↔ Python co-simulation |
| Python | 3.12 | numpy, torch for reference tensors |

Activate the repo virtualenv first — the Makefiles set `PYTHONPATH` to
`sw/` and `testing/ref/` automatically.

## Directory structure

```
common/         FIFO (SystemVerilog TB) + sram/ (dual-port SRAM)
arith/          pipelined multiplier and MAC
apu/            idgen/, stage1/, full/  (APU unit + integration tests)
pe/             single PE and PE array
gospa/          full-accelerator tests + performance sweeps
ref/            MobileNetV2 / AlexNet reference tensors
artifact/       one-command reproduction runner (see artifact/README.md)
```

## Running tests

Run from each block's directory. `SIM=verilator|icarus` selects the
simulator; `MODULE=` selects the testbench where a directory has several:

```bash
cd testing/common      && make                          # FIFO (10 tests / 92 checks)
cd testing/common/sram && make                          # dual-port SRAM
cd testing/arith       && make MODULE=test_mult_pipe    # pipelined signed multiplier
cd testing/arith       && make MODULE=test_mac_pipe     # fused multiply-add
cd testing/apu/idgen   && make                          # CID / PID generation
cd testing/apu/stage1  && make MODULE=test_csr_decode   # CSR decode
cd testing/apu/stage1  && make MODULE=test_zero_act     # zero-activation filter
cd testing/apu/stage1  && make MODULE=test_apu_stage1   # Stage-1 chain
cd testing/apu/full    && make                          # full APU
cd testing/apu/full    && make mobilenet                # APU on real MobileNet conv1
cd testing/pe          && make pe                       # single PE
cd testing/pe          && make array                    # PE array
cd testing/gospa       && make test_gospa               # full accelerator
cd testing/gospa       && make mobilenet                # MobileNetV2 first conv
cd testing/gospa       && make apple                    # real 80×80 MobileNetV2 conv1
```

Layer and array geometry are overridable, e.g.
`make test_gospa H=10 F=3 S=2 N_PE=8 N_MULTS=4`. Add `WAVES=1` to dump a
VCD for waveform viewing.

Sweep targets also exist: `make sweep_pe` / `sweep_array` (PE),
`make sweep_gospa` / `conv5` (full accelerator), `make sweep_s1` (Stage-1),
`make sweep_apu`, `make sweep` (idgen, SRAM).
