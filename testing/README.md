# Testing

This directory contains all testbenches and simulation scripts for the GoSPA RTL implementation.
The directory structure mirrors `rtl/` — each IP block under `rtl/<block>/` has a corresponding test directory here.

## Dependencies

| Tool | Version | Notes |
|---|---|---|
| Verilator | 5.x | Primary simulator. `verilator --version` to check. |
| Icarus Verilog | any | Alternative. `iverilog -V` to check. |
| GTKWave | any | Optional, for waveform viewing. |

Install on Ubuntu/Debian:
```
sudo apt install verilator iverilog gtkwave
```

---

## Directory Structure

```
testing/
  common/         -- tests for shared utility modules (fifo, sram_wrapper, ...)
  apu/            -- tests for APU Stage-1 and Stage-2   (to be added)
  pe/             -- tests for Processing Element         (to be added)
```

---

## Running Tests

### FIFO (`testing/common/`)

Tests the parameterized synchronous FIFO used for FIFO-A and FIFO-B in GoSPA.

```bash
cd testing/common

# Run with Verilator (default)
make

# Run with Icarus Verilog
make SIM=iverilog

# Open waveform in GTKWave after simulation
make wave

# Lint only (no simulation)
make lint

# Clean build artifacts
make clean
```

Or use the shell script:
```bash
./run_sim.sh              # Verilator
./run_sim.sh iverilog     # Icarus Verilog
./run_sim.sh wave         # Verilator + open GTKWave
```

**What is tested (10 tests, 92 checks):**
1. Initial state after reset
2. Single push — flags and count
3. Single pop — data integrity
4. Fill to full
5. Overflow protection (push when full)
6. Drain to empty, verify FIFO order
7. Underflow protection (pop when empty)
8. Simultaneous push + pop (count unchanged)
9. GoSPA FIFO-A scenario from paper toy example (F=2, H=3, S=1)
10. Randomized stress — 50 push / 50 pop

Expected output ends with:
```
  ALL TESTS PASSED  (92 checks)
```
