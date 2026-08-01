# GoSPA — FPGA Synthesis Results

Vivado 2024.2, out-of-context post-route implementation of `gospa.sv` on the
Kria KV260 (`xck26-sfvc784-2LV-c`). Config: N_PE = 8, N_MULTS = 4, H = 8, F = 3,
S = 1, DATA_W = 16, ACC_W = 32, FIFO depth = 64.

## V1 architecture

Target clock 250 MHz (4.000 ns).

### Timing (post-route)

| Metric | Value |
|---|---|
| Worst negative slack (WNS) | −0.421 ns |
| Achievable period | 4.421 ns |
| Maximum frequency | ≈ 226 MHz |

### Resource utilisation

| Resource | Used | Available | Utilisation |
|---|---|---|---|
| LUT  | 36 519 | 117 120 | 31.2 % |
| FF   | 59 772 | 234 240 | 25.5 % |
| DSP  | 0      | 1 248   | 0 % |
| BRAM | 0      | 144     | 0 % |

## V2 architecture

Target clock 1 GHz (1.000 ns).

### Timing (post-route)

| Metric | Value |
|---|---|
| Worst negative slack (WNS) | −4.855 ns |
| Achievable period | 5.855 ns |
| Maximum frequency | ≈ 171 MHz |

### Resource utilisation

| Resource | Used | Available | Utilisation |
|---|---|---|---|
| LUT  | 42 698 | 117 120 | 36.5 % |
| FF   | 72 610 | 234 240 | 31.0 % |
| DSP  | 0      | 1 248   | 0 % |
| BRAM | 0      | 144     | 0 % |
