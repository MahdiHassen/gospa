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
    M: int = 4               # multipliers per PE (output channels in "channel"
                             # arch; activation lanes in "act")

    # --- dataflow mode ----------------------------------------------------
    # "channel": M kernels/PE, one activation broadcast to M lanes, per-lane
    #            WSPs union-gated. Multiplier utilization is capped at weight
    #            density (the union over-admission penalty).
    # "act":     one kernel/PE shared by all M multipliers; M activations
    #            consumed per cycle (bundled by PID, <=M per bundle); single
    #            WSP, no union. Utilization is set by PID-group length and is
    #            ~independent of weight density.
    ARCH: str = "channel"

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
    # "parallel": stage1 = n_nz            RTL-faithful. idgen is a purely
    #             combinational G^2 fan-out (PIPE=0) and one activation's pairs
    #             carry distinct PIDs, so they land in G^2 different FIFO-A
    #             lanes in the same cycle -> 1 activation/cycle. Downstream
    #             back-pressure is captured by the max() over stages.
    #             (see rtl/apu/stage1/idgen.sv, apu_stage1.sv)
    # "unrolled": stage1 = max(n_nz, n_pairs)   1-pair/cycle emit (analysis knob)
    # "serial":   stage1 = n_nz + n_pairs       serial emit (pessimistic knob)
    STAGE1_ENUM: str = "parallel"

    # --- APU Stage 2 router width ----------------------------------------
    # B = FIFO-A heads the router pops per cycle, so stage2 = ceil(n_pairs / B).
    # "native": as-built -- B = M in "act" (the FIFO-A -> FIFO-B transfer is
    #           widened to M activations/cycle) and B = 1 in "channel" (the
    #           router drains one entry/cycle and fans it out). This is the
    #           default: it reproduces the RTL, where routing.sv pops exactly
    #           one head per cycle (`go = head_valid && all_ready`).
    # int:      a fixed router width, e.g. 12.
    # "auto":   B = B*, the knee -- the narrowest router that stops being the
    #           bottleneck. Scope is the WHOLE NETWORK (one fixed width for
    #           every layer), taken as the max of the per-pass B* over all
    #           passes of all layers; sim.py resolves it. Below layer scope
    #           there is no network to max over, so simulate_layer/score_pass
    #           fall back to per-pass B*. See stage2_batch_star in PassStats.
    # NOTE: "auto" is an ANALYSIS knob -- it answers "how wide would the router
    # have to be", not "how wide is it". It changes every reported number.
    STAGE2_BATCH: Union[int, str] = "native"

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
