"""
goSPA performance model -- hardware configuration.

A single parameter bag (`HwConfig`) carrying every knob the perf model reads, so
that all hardware parameters are passed in at runtime and nothing is baked into
the timing code (see PERF_MODEL_PLAN.md sec.7/sec.11). Defaults are placeholders;
the ones marked TODO are to be set from the paper's config table (sec.10) and the
RTL testbenches (July calibration -- sec.9).
"""

from dataclasses import dataclass
from typing import Callable, Optional, Union


@dataclass
class HwConfig:
    # --- array geometry ---------------------------------------------------
    N_PE: int = 8            # number of PEs
    M: int = 4               # multipliers (= output channels) per PE

    # --- clock ------------------------------------------------------------
    FREQ_HZ: float = 1e9     # TODO(sec.10): paper clock frequency

    # --- FIFO depths (informational; backpressure not modeled yet, sec.3) -
    FIFO_A_DEPTH: int = 64
    FIFO_B_DEPTH: int = 64

    # --- PE weight reload (passed straight through to perf_pe) ------------
    # W_UPDATE_PENALTY: None -> derive P = f(M); an int -> fixed P; or a
    # callable P = f(M). When None, perf_pe uses W_FETCH_LATENCY / W_FETCH_BW.
    W_UPDATE_PENALTY: Optional[Union[int, Callable[[int], int]]] = None
    W_FETCH_LATENCY: int = 0      # TODO(sec.9): fixed weight-fetch latency
    W_FETCH_BW: int = 1           # weights delivered per cycle
    RELOAD_MODEL: str = "double_buffer"   # "double_buffer" | "simple" | "ideal"

    # --- APU Stage 1 enumerator ------------------------------------------
    # "serial":   stage1 = n_nz + n_pairs   (1 decode + 1 per emitted pair)
    # "unrolled": stage1 = max(n_nz, n_pairs)
    STAGE1_ENUM: str = "serial"

    # --- memory (fixed-latency + bandwidth; the calibration target) ------
    MEM_BW_BYTES: int = 16        # B: bytes moved per cycle
    MEM_LATENCY: int = 0          # L: fixed access latency in cycles
    MEM_PORTS: str = "shared"     # "shared" (act+wgt share B) | "split" (max)
    BYTES_ACT: int = 2            # bytes per activation entry (ACT_W=16 -> 2)
    BYTES_W: int = 2              # bytes per weight
    BYTES_OUT: int = 4            # bytes per output element (store; layer scope)

    # --- data widths (provisional, from the RTL) -------------------------
    ACT_W: int = 16
    CID_W: int = 6
    PID_W: int = 4

    # --- pipeline fill/drain (applied at layer scope by layer.py) --------
    FILL: int = 0                 # TODO(sec.9): calibrate vs RTL
    DRAIN: int = 0                # TODO(sec.9): calibrate vs RTL
