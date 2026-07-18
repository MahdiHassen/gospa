"""
goSPA performance model -- single Processing Element (PE) cycle accounting.

Companion to `functional.py`. Where `pe_process_v2` computes the *values* a PE
produces, this module computes the *timing*: how many cycles that same PE spends
consuming its FIFO-B stream, plus lane-utilization stats. It does NOT recompute
MACs -- it walks the already-routed stream and counts.

Scope (this file): one PE in isolation. Out of scope: front-end routing
(`broadcast_to_fifo_b` hands us `fifo_b` and the per-lane WSPs) and cross-PE
aggregation (`pe_stage = max_k pe_cycles[k]`, FILL/DRAIN) -- those live in the
sim driver.

Corrected v2 microarchitecture this models (see PERF_MODEL_PLAN.md sec.2):
  - The PE holds M output channels, one per multiplier lane. Each lane has its
    own WSP (length F^2), its own sparse weight column, its own CID-indexed
    accumulator (so there is no accumulator write-port stall).
  - FIFO-B admission is gated upstream by the UNION of the M WSPs.
  - One activation (Axy, CID, PID) is consumed per cycle and broadcast to all M
    lanes: lane c does a useful MAC iff WSP_c[PID]==1, else it idles. This
    union over-admission is the "lane under-utilization" effect we report.
  - On a PID change the M-weight column is reloaded. A Curr/Next double buffer
    prefetches the next column, so the reload is hidden when the current PID's
    run is long enough to cover the fetch (the `double_buffer` model below).

All hardware parameters are passed in at runtime; nothing is baked in.
"""

from dataclasses import dataclass, field
from math import ceil


# ---------------------------------------------------------------------------
# Reload penalty (scales with M = the weight-column size)
# ---------------------------------------------------------------------------

def default_w_update_penalty(m, *, w_fetch_latency=0, w_fetch_bw=1):
    """
    Cycles to reload one PID's weight column of `m` weights.

        P = w_fetch_latency + ceil(m / w_fetch_bw)

    A reload fetches the whole column (one weight per lane / output channel),
    so the cost grows with M = the number of multipliers in the PE. With the
    placeholder defaults (latency=0, bw=1) this is simply P = M: M cycles to
    stream M weights at one per cycle. `w_fetch_latency` (fixed access latency)
    and `w_fetch_bw` (weights delivered per cycle) are calibration knobs to fit
    against the PE testbench later.
    """
    return w_fetch_latency + ceil(m / max(1, w_fetch_bw))


def _resolve_penalty(m, w_update_penalty, w_fetch_latency, w_fetch_bw):
    """Turn the `w_update_penalty` argument into a concrete per-reload cycle
    count P. Accepts: None (derive from M), an int (use as-is), or a callable
    P = f(M)."""
    if w_update_penalty is None:
        return default_w_update_penalty(
            m, w_fetch_latency=w_fetch_latency, w_fetch_bw=w_fetch_bw)
    if callable(w_update_penalty):
        return int(w_update_penalty(m))
    return int(w_update_penalty)


# ---------------------------------------------------------------------------
# Result record
# ---------------------------------------------------------------------------

@dataclass
class PEStats:
    # --- timing ---
    pe_cycles: int          # active + stall; this is what the sim's max_k uses
    active_cycles: int      # = len(fifo_b); one activation consumed per cycle
    stall_cycles: int       # reload bubbles not hidden by the double buffer
    num_reloads: int        # column reloads charged (= in-stream PID changes)
    num_pid_groups: int     # contiguous PID runs in the stream
    num_distinct_pids: int  # == num_pid_groups when FIFO-B is PID-ordered

    # --- work / utilization ---
    M: int                  # lanes (output channels) in this PE
    fifo_b_len: int
    useful_macs: int        # sum over entries of (# lanes with WSP[pid]==1)
    overall_lane_util: float  # useful_macs / (pe_cycles  * M)  -- incl. stalls
    compute_lane_util: float  # useful_macs / (active_cycles * M) -- compute only
    useful_per_lane: list = field(default_factory=list)  # MACs done by lane c
    idle_per_lane: list = field(default_factory=list)     # active cycles lane c idled
    pid_occupancy: dict = field(default_factory=dict)     # PID -> entries consumed

    # --- echo of the resolved knobs (handy for calibration/debug) ---
    w_update_penalty: int = 0
    reload_model: str = "double_buffer"


# ---------------------------------------------------------------------------
# Stream entry point
# ---------------------------------------------------------------------------

def pe_perf_from_stream(fifo_b, wsps, *,
                        w_update_penalty=None,
                        w_fetch_latency=0,
                        w_fetch_bw=1,
                        reload_model="double_buffer",
                        count_first_load=False):
    """
    Cycle + utilization accounting for one PE over its FIFO-B stream.

        fifo_b : ordered [(Axy, CID, PID), ...] for THIS PE (e.g.
                 goSPA_run's fifo_b_list[k]). Stage 2 delivers it in PID
                 order, so each PID appears as one contiguous run/group.
        wsps   : list of M WSP bit-arrays (one per lane), each length F^2.

    Penalty (runtime-configurable, scales with M = len(wsps)):
        P = default_w_update_penalty(M, ...) unless `w_update_penalty` overrides.

    reload_model:
        "double_buffer" (default, faithful): a column reload is hidden behind
            the previous PID's run; stall only when that run is shorter than P.
                stall = sum_{i>=1} max(0, P - run_len(group_{i-1}))
        "simple": charge P on every PID change (conservative upper bound).
        "ideal":  weights treated as always resident (stall = 0).

    count_first_load:
        If True, charge P once for filling the very first column (nothing to
        hide it behind). Default False -- the sim rolls that into FILL.

    Returns a PEStats.
    """
    M = len(wsps)
    P = _resolve_penalty(M, w_update_penalty, w_fetch_latency, w_fetch_bw)

    active_cycles = len(fifo_b)

    useful_macs = 0
    useful_per_lane = [0] * M
    pid_occupancy = {}
    group_lens = []          # run length of each contiguous PID group
    prev_pid = None
    cur_len = 0

    for entry in fifo_b:
        pid = entry[2]

        # utilization: how many of the M lanes actually fire at this PID
        for c in range(M):
            if wsps[c][pid]:
                useful_per_lane[c] += 1
                useful_macs += 1
        pid_occupancy[pid] = pid_occupancy.get(pid, 0) + 1

        # grouping (relies on FIFO-B being PID-ordered out of Stage 2)
        if pid != prev_pid:
            if prev_pid is not None:
                group_lens.append(cur_len)
            prev_pid = pid
            cur_len = 0
        cur_len += 1

    if prev_pid is not None:
        group_lens.append(cur_len)

    G = len(group_lens)
    num_reloads = max(0, G - 1)   # = "#PID_changes_in_k" from the plan

    # --- reload stalls ---
    stall_cycles = 0
    if count_first_load and G >= 1:
        stall_cycles += P         # first column fill has nothing to hide behind
    for i in range(1, G):
        if reload_model == "double_buffer":
            stall_cycles += max(0, P - group_lens[i - 1])
        elif reload_model == "simple":
            stall_cycles += P
        elif reload_model == "ideal":
            pass
        else:
            raise ValueError(f"unknown reload_model {reload_model!r}")

    pe_cycles = active_cycles + stall_cycles
    idle_per_lane = [active_cycles - u for u in useful_per_lane]

    overall_lane_util = useful_macs / (pe_cycles * M) if (pe_cycles and M) else 0.0
    compute_lane_util = useful_macs / (active_cycles * M) if (active_cycles and M) else 0.0

    return PEStats(
        pe_cycles=pe_cycles,
        active_cycles=active_cycles,
        stall_cycles=stall_cycles,
        num_reloads=num_reloads,
        num_pid_groups=G,
        num_distinct_pids=len(pid_occupancy),
        M=M,
        fifo_b_len=active_cycles,
        useful_macs=useful_macs,
        overall_lane_util=overall_lane_util,
        compute_lane_util=compute_lane_util,
        useful_per_lane=useful_per_lane,
        idle_per_lane=idle_per_lane,
        pid_occupancy=dict(sorted(pid_occupancy.items())),
        w_update_penalty=P,
        reload_model=reload_model,
    )


# ---------------------------------------------------------------------------
# Activation-parallel PE ("act" arch)
# ---------------------------------------------------------------------------

def pe_perf_actparallel(fifo_b, wsp, m, *, count_first_load=False):
    """
    Cycle + utilization accounting for one PE in the "act" dataflow: one kernel
    per PE shared by all M multipliers, M activations consumed per cycle.

        fifo_b : ordered [(Axy, CID, PID), ...] for THIS PE, PID-grouped out of
                 Stage 2 (each PID is one contiguous run). Every entry already
                 has WSP[PID]==1 (single-kernel gating upstream), so every entry
                 is a real MAC on one multiplier.
        wsp    : the PE's single WSP bit-array (length F^2). Used only to score
                 useful MACs defensively; under correct gating this equals
                 len(fifo_b).
        m      : number of multipliers in the PE (= activation lanes).

    Dataflow (decision B2): the whole F^2 kernel is resident, so there is NO
    per-PID weight reload. Within a PID run of length L the PE consumes up to
    `m` activations per cycle against the single resident weight for that PID,
    taking ceil(L / m) cycles. The only idle lane-slots are the partial last
    cycle of each run (the "tail"):

        pe_cycles   = sum_g ceil(L_g / m)
        useful_macs = sum_g L_g          (= len(fifo_b) under correct gating)
        lane_util   = useful_macs / (pe_cycles * m)   ->  1 - tail_waste

    Utilization here is set by the PID-group lengths L_g relative to m, NOT by
    weight density -- that is the whole point of this arch (see the union
    penalty in pe_perf_from_stream). Long runs (big feature maps, 1x1/FC) push
    util -> 1; runs shorter than m waste up to m-1 lane-slots each.

    count_first_load: charge one whole-kernel load (F^2 weights, m-agnostic)
    up front; default False -- the sim rolls that into FILL.

    Returns a PEStats (M field = m; reload_model = "resident").
    """
    m = max(1, m)

    useful_macs = 0
    pid_occupancy = {}
    group_lens = []
    prev_pid = None
    cur_len = 0

    for entry in fifo_b:
        pid = entry[2]
        if wsp[pid]:
            useful_macs += 1
        pid_occupancy[pid] = pid_occupancy.get(pid, 0) + 1

        if pid != prev_pid:
            if prev_pid is not None:
                group_lens.append(cur_len)
            prev_pid = pid
            cur_len = 0
        cur_len += 1
    if prev_pid is not None:
        group_lens.append(cur_len)

    G = len(group_lens)

    # Bundle each PID run into ceil(L / m) cycles.
    active_cycles = sum((L + m - 1) // m for L in group_lens)

    # Per-lane useful / idle, assuming round-robin fill within each cycle: full
    # cycles busy all m lanes; each run's tail busies (L mod m) lanes, so higher
    # lane indices absorb the tail idling.
    useful_per_lane = [0] * m
    for L in group_lens:
        full, rem = divmod(L, m)
        for c in range(m):
            useful_per_lane[c] += full
        for c in range(rem):
            useful_per_lane[c] += 1

    # Optional one-time whole-kernel fill (F^2 weights); m-agnostic.
    stall_cycles = len(wsp) if (count_first_load and G >= 1) else 0
    pe_cycles = active_cycles + stall_cycles
    idle_per_lane = [pe_cycles - u for u in useful_per_lane]

    slots = pe_cycles * m
    overall_lane_util = useful_macs / slots if slots else 0.0
    compute_lane_util = useful_macs / (active_cycles * m) if active_cycles else 0.0

    return PEStats(
        pe_cycles=pe_cycles,
        active_cycles=active_cycles,
        stall_cycles=stall_cycles,
        num_reloads=0,                 # whole kernel resident -- no reload
        num_pid_groups=G,
        num_distinct_pids=len(pid_occupancy),
        M=m,
        fifo_b_len=len(fifo_b),
        useful_macs=useful_macs,
        overall_lane_util=overall_lane_util,
        compute_lane_util=compute_lane_util,
        useful_per_lane=useful_per_lane,
        idle_per_lane=idle_per_lane,
        pid_occupancy=dict(sorted(pid_occupancy.items())),
        w_update_penalty=0,
        reload_model="resident",
    )


# ---------------------------------------------------------------------------
# Self-test (sec.8 of the design: PE validated in isolation)
# ---------------------------------------------------------------------------

def _selftest():
    ok = True

    def check(name, cond):
        nonlocal ok
        ok = ok and cond
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    # 1) Fig. 13 replay: stream PIDs 1,1,3,3,3; single kernel WSP=[0,1,0,1].
    #    M=1 -> P=1; groups [2,3]; reload hidden (prev run 2 >= 1).
    fig13 = [(2, 1, 1), (-2, 2, 1), (-2, 0, 3), (3, 2, 3), (-3, 3, 3)]
    wsp_fig13 = [[0, 1, 0, 1]]
    s = pe_perf_from_stream(fig13, wsp_fig13)
    print(f"  Fig.13: {s}")
    check("Fig.13 active_cycles == 5", s.active_cycles == 5)
    check("Fig.13 num_reloads == 1", s.num_reloads == 1)
    check("Fig.13 penalty P == M == 1", s.w_update_penalty == 1)
    check("Fig.13 stall hidden (==0)", s.stall_cycles == 0)
    check("Fig.13 pe_cycles == 5", s.pe_cycles == 5)
    check("Fig.13 all lanes useful (util==1)", s.compute_lane_util == 1.0)

    # 2) Roofline: one PID, all M lanes set -> no reload, full utilization.
    M = 4
    stream = [(1, 0, 7)] * 10
    wsps = [[1] * 16 for _ in range(M)]
    s = pe_perf_from_stream(stream, wsps)
    check("roofline num_reloads == 0", s.num_reloads == 0)
    check("roofline stall == 0", s.stall_cycles == 0)
    check("roofline overall_lane_util == 1", s.overall_lane_util == 1.0)
    check("roofline useful_macs == 40", s.useful_macs == 10 * M)

    # 3) Single non-zero -> minimum latency, no reload.
    s = pe_perf_from_stream([(5, 0, 2)], [[0, 0, 1, 0]])
    check("single-nz pe_cycles == 1", s.pe_cycles == 1)
    check("single-nz num_reloads == 0", s.num_reloads == 0)

    # 4) double_buffer exposes a real (unhidden) stall: M=4 -> P=4; a length-1
    #    PID group can't hide the next 4-cycle column fetch -> stall 3.
    short = [(1, 0, 0), (1, 0, 5), (1, 0, 5), (1, 0, 5), (1, 0, 5)]
    wsps4 = [[1] * 16 for _ in range(4)]
    s = pe_perf_from_stream(short, wsps4)            # groups [1, 4], P=4
    check("double_buffer P == M == 4", s.w_update_penalty == 4)
    check("double_buffer stall == 3 (=4-1)", s.stall_cycles == 3)
    check("double_buffer pe_cycles == 8", s.pe_cycles == 8)

    # 5) reload_model monotonicity: simple >= double_buffer >= ideal.
    a = pe_perf_from_stream(short, wsps4, reload_model="simple").stall_cycles
    b = pe_perf_from_stream(short, wsps4, reload_model="double_buffer").stall_cycles
    c = pe_perf_from_stream(short, wsps4, reload_model="ideal").stall_cycles
    check("monotonic simple >= double_buffer >= ideal", a >= b >= c and c == 0)

    # 6) Union under-utilization: 2 lanes, disjoint WSPs -> half the lane-slots
    #    idle even though every activation is admitted.
    wsp_disjoint = [[1, 0], [0, 1]]              # F^2 = 2
    stream2 = [(1, 0, 0), (1, 0, 1)]
    s = pe_perf_from_stream(stream2, wsp_disjoint)
    check("union under-util compute_lane_util == 0.5", s.compute_lane_util == 0.5)
    check("union under-util useful_per_lane == [1,1]", s.useful_per_lane == [1, 1])

    # 7) act arch: single PID run of length 5, m=2 -> ceil(5/2)=3 cycles, all 5
    #    entries useful; one tail slot idles on the high lane -> util = 5/6.
    run5 = [(1, 0, 3)] * 5
    s = pe_perf_actparallel(run5, [1] * 16, 2)
    check("act pe_cycles == ceil(5/2) == 3", s.pe_cycles == 3)
    check("act useful_macs == 5", s.useful_macs == 5)
    check("act num_reloads == 0", s.num_reloads == 0)
    check("act util == 5/6", abs(s.overall_lane_util - 5 / 6) < 1e-9)
    check("act tail idles high lane: useful_per_lane == [3,2]",
          s.useful_per_lane == [3, 2])

    # 8) act divisible runs -> util == 1 (no tail waste).
    two_runs = [(1, 0, 0)] * 4 + [(1, 0, 5)] * 4      # groups [4, 4]
    s = pe_perf_actparallel(two_runs, [1] * 16, 2)
    check("act divisible pe_cycles == 4", s.pe_cycles == 4)
    check("act divisible util == 1", abs(s.overall_lane_util - 1.0) < 1e-9)
    check("act num_pid_groups == 2", s.num_pid_groups == 2)

    # 9) act util is decoupled from weight density: same run, sparse vs dense
    #    single kernel -> identical util (both fire every admitted entry).
    long_run = [(1, 0, 2)] * 10                        # ceil(10/3)=4 -> 10/12
    u_sparse = pe_perf_actparallel(long_run, [0, 0, 1] + [0] * 13, 3).overall_lane_util
    u_dense = pe_perf_actparallel(long_run, [1] * 16, 3).overall_lane_util
    check("act util independent of weight density", u_sparse == u_dense)
    check("act util == 10/12", abs(u_dense - 10 / 12) < 1e-9)

    print(f"\nperf_pe self-test: {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    _selftest()
