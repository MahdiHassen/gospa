"""
goSPA performance model -- per-pass cycle accounting.

This is the per-PASS engine that sits between `perf_pe.py` (one PE) and
`layer.py` (whole layer). A *pass* is one front-end routing run -- one input
channel over one output-channel tile (<= N_PE*M channels in "channel" arch,
<= N_PE in "act"). Given that pass's routed data it:

  - counts APU Stage 1 (ID-gen) and Stage 2 (router/broadcast),
  - calls perf_pe per PE and takes pe_stage = max_k pe_cycles  (load imbalance),
  - estimates memory cycles,
  - returns pass_cycles = max(stage1, stage2, pe_stage, mem) plus the breakdown.

Out of scope (-> layer.py): the input-channel loop, the output-channel pass
loop ceil(Cout / (N_PE*M)), and the FILL/DRAIN bracketing. Input-channel
accumulation is reused from functional.goSPA_multichannel (resident
accumulators), so a layer is just a sequence of passes this module scores.

Routing comes from functional.goSPA_route -- the perf model instruments the real
dataflow (v2 for "channel", v1 for "act"), it does not re-implement it. The
dataflow is chosen by HwConfig.ARCH; all hardware parameters arrive via HwConfig
(config.py). See PERF_MODEL_PLAN.md sec.3/4/5/11.
"""

from dataclasses import dataclass, field, replace
from math import ceil

import functional as fn
from perf_pe import pe_perf_from_stream, pe_perf_actparallel, PEStats
from config import HwConfig


# ---------------------------------------------------------------------------
# Routed-pass bundle + result record
# ---------------------------------------------------------------------------

@dataclass
class RoutedPass:
    """Materialized routing for one pass, built from functional.goSPA_route.
    `pe_lane_wsps[k]` is PE k's list of per-lane WSPs (NOT the union) -- perf_pe
    needs the individual lanes to measure union under-utilization."""
    n_nz: int
    n_pairs: int
    fifo_b_list: list
    pe_lane_wsps: list
    pe_chunks: list
    E: int = 0

    @classmethod
    def from_routing(cls, r):
        pe_lane_wsps = [[r.wsps[k] for k in chunk] for chunk in r.pe_chunks]
        return cls(n_nz=r.n_nz, n_pairs=r.n_pairs, fifo_b_list=r.fifo_b_list,
                   pe_lane_wsps=pe_lane_wsps, pe_chunks=r.pe_chunks, E=r.E)


@dataclass
class PassStats:
    pass_cycles: int          # = max of the four stages below
    stage1_cycles: int
    stage2_cycles: int
    pe_stage_cycles: int      # = max_k pe_cycles
    mem_cycles: int
    bottleneck: str           # which stage set pass_cycles
    per_pe: list = field(default_factory=list)   # PEStats per PE
    slowest_pe: int = -1
    load_imbalance: float = 1.0   # max_k pe_cycles / mean_k pe_cycles
    array_util: float = 0.0       # useful_macs / (pass_cycles * N_PE * M)
    n_nz: int = 0
    n_pairs: int = 0
    stage2_batch: int = 1         # B actually used: stage2 = ceil(n_pairs / B)
    stage2_batch_star: int = 1    # B*, the knee -- always recorded, even when
                                  # B is pinned, so the headroom stays visible


# ---------------------------------------------------------------------------
# Stage counters
# ---------------------------------------------------------------------------

def _stage1_cycles(routed, cfg):
    """APU Stage 1 (ID-gen) cycle count -- see HwConfig.STAGE1_ENUM.

    "parallel" (RTL-faithful default): idgen is a combinational G^2 fan-out that
    pushes one activation's pairs to G^2 distinct FIFO-A lanes in a single
    cycle, so throughput is 1 activation/cycle -> n_nz. Downstream back-pressure
    (FIFO-A full) is captured by score_pass's max() over stages, not here."""
    if cfg.STAGE1_ENUM == "parallel":
        return routed.n_nz
    if cfg.STAGE1_ENUM == "serial":
        return routed.n_nz + routed.n_pairs
    if cfg.STAGE1_ENUM == "unrolled":
        return max(routed.n_nz, routed.n_pairs)
    raise ValueError(f"unknown STAGE1_ENUM {cfg.STAGE1_ENUM!r}")


def _mem_cycles(routed, cfg):
    """Activation + weight streaming for this pass.

    Output store is deliberately omitted: with resident accumulators the result
    is written once per layer (not per pass), so it belongs to layer.py
    (PERF_MODEL_PLAN.md sec.5). Memory is the least-developed stage and the main
    RTL calibration target (sec.4/sec.9)."""
    act_bytes = routed.n_nz * cfg.BYTES_ACT
    nnz_weights = sum(sum(w) for lanes in routed.pe_lane_wsps for w in lanes)
    wgt_bytes = nnz_weights * cfg.BYTES_W
    B = max(1, cfg.MEM_BW_BYTES)
    if cfg.MEM_PORTS == "split":
        return max(ceil(act_bytes / B), ceil(wgt_bytes / B)) + cfg.MEM_LATENCY
    return ceil((act_bytes + wgt_bytes) / B) + cfg.MEM_LATENCY


# ---------------------------------------------------------------------------
# Pass scoring
# ---------------------------------------------------------------------------

def native_batch(cfg):
    """B for STAGE2_BATCH="native" -- the as-built router width.

    "act": the FIFO-A -> FIFO-B transfer is widened to M activations/cycle, so
    Stage 2 drains M pairs/cycle. "channel": the router drains one FIFO-A entry
    per cycle and fans it out to every PE in parallel, so B = 1."""
    return max(1, cfg.M) if cfg.ARCH == "act" else 1


def batch_star(n_pairs, stage1, pe_stage, mem):
    """B*, the knee: the narrowest router that stops being the bottleneck.

    stage2 = ceil(n_pairs / B) is the ONLY stage that depends on B -- widening
    the router does not change how fast a PE chews its FIFO-B (pe_stage), how
    many activations Stage 1 enumerates, or the memory traffic. So the smallest
    B with ceil(n_pairs / B) <= max(stage1, pe_stage, mem) is exactly the point
    past which widening buys nothing, and it is computable in one shot with no
    circular dependency.

    This is the general form of the M/d_w law: for "act" at large F the binding
    stage is the PE and B* -> M/d_w, but on small kernels Stage 1 (n_nz, one
    activation/cycle) takes over first and B* stops short of it."""
    other = max(stage1, pe_stage, mem)
    return max(1, ceil(n_pairs / other)) if other else 1


def resolve_batch(cfg, n_pairs, stage1, pe_stage, mem):
    """Turn cfg.STAGE2_BATCH into a concrete (B, B*) pair.

    "auto" resolves to the PER-PASS B* here. Network-wide "auto" -- one fixed
    width for every layer -- is resolved by sim.run_network, which maxes B*
    over all passes and then re-scores via rescore_at_batch; at pass scope
    there is no network to max over."""
    star = batch_star(n_pairs, stage1, pe_stage, mem)
    spec = cfg.STAGE2_BATCH
    if spec == "native":
        return native_batch(cfg), star
    if spec == "auto":
        return star, star
    if isinstance(spec, int) and spec >= 1:
        return spec, star
    raise ValueError(f"bad STAGE2_BATCH {spec!r}: want 'native', 'auto', or int >= 1")


def rescore_at_batch(ps, cfg, B):
    """Re-derive (pass_cycles, bottleneck, array_util) at router width B.

    Pure recomputation from counters already in `ps` -- no re-routing, no
    functional re-run. Valid because stage1/pe_stage/mem are all independent of
    B, so only stage2 moves. This is what makes network-wide "auto" cheap: the
    expensive functional sim runs once, then every pass is re-derived at the
    single chosen width."""
    stage2 = ceil(ps.n_pairs / B) if ps.n_pairs else 0
    stages = {"stage1": ps.stage1_cycles, "stage2": stage2,
              "pe": ps.pe_stage_cycles, "mem": ps.mem_cycles}
    bottleneck = max(stages, key=stages.get)
    pass_cycles = stages[bottleneck]
    useful = sum(s.useful_macs for s in ps.per_pe)
    slots = pass_cycles * cfg.N_PE * cfg.M
    return replace(ps, pass_cycles=pass_cycles, stage2_cycles=stage2,
                   bottleneck=bottleneck, stage2_batch=B,
                   array_util=useful / slots if slots else 0.0)


def score_pass(routed, cfg):
    """Score one already-routed pass. Pure: no functional re-run, no I/O."""
    stage1 = _stage1_cycles(routed, cfg)

    if cfg.ARCH == "act":
        # One kernel/PE shared by M multipliers. pe_lane_wsps[k] is a
        # single-element list (one WSP) under v1 routing.
        per_pe = [
            pe_perf_actparallel(fb, routed.pe_lane_wsps[k][0], cfg.M)
            for k, fb in enumerate(routed.fifo_b_list)
        ]
    elif cfg.ARCH == "channel":
        # M kernels/PE, one activation broadcast to M lanes.
        per_pe = [
            pe_perf_from_stream(
                fb, routed.pe_lane_wsps[k],
                w_update_penalty=cfg.W_UPDATE_PENALTY,
                w_fetch_latency=cfg.W_FETCH_LATENCY,
                w_fetch_bw=cfg.W_FETCH_BW,
                reload_model=cfg.RELOAD_MODEL,
            )
            for k, fb in enumerate(routed.fifo_b_list)
        ]
    else:
        raise ValueError(f"unknown ARCH {cfg.ARCH!r}")
    pe_cycles = [s.pe_cycles for s in per_pe]
    pe_stage = max(pe_cycles) if pe_cycles else 0

    mem = _mem_cycles(routed, cfg)

    # Stage 2 is the one stage that depends on the router width, so it is
    # counted last: B* is read off the stages that do not.
    B, B_star = resolve_batch(cfg, routed.n_pairs, stage1, pe_stage, mem)
    stage2 = ceil(routed.n_pairs / B) if routed.n_pairs else 0

    stages = {"stage1": stage1, "stage2": stage2, "pe": pe_stage, "mem": mem}
    bottleneck = max(stages, key=stages.get)
    pass_cycles = stages[bottleneck]

    if per_pe:
        slowest = max(range(len(per_pe)), key=lambda i: per_pe[i].pe_cycles)
        mean_pe = sum(pe_cycles) / len(pe_cycles)
        load_imbalance = pe_stage / mean_pe if mean_pe else 1.0
    else:
        slowest, load_imbalance = -1, 1.0

    # Array utilization over WALL-CLOCK: fraction of the full physical array's
    # multiply-slots (pass_cycles * N_PE * M) that did useful work. Because the
    # denominator is pass_cycles -- not each PE's own pe_cycles -- it charges
    # every slot idled when the PE is NOT the bottleneck: front-end/mem stall
    # (pass_cycles > pe_stage), cross-PE load imbalance, and PEs left idle on a
    # partial tile (fewer than N_PE active). It collapses back to the old
    # PE-intrinsic ratio only when the pass is PE-bound, balanced, and full.
    useful = sum(s.useful_macs for s in per_pe)
    slots = pass_cycles * cfg.N_PE * cfg.M
    array_util = useful / slots if slots else 0.0

    return PassStats(
        pass_cycles=pass_cycles, stage1_cycles=stage1, stage2_cycles=stage2,
        pe_stage_cycles=pe_stage, mem_cycles=mem, bottleneck=bottleneck,
        per_pe=per_pe, slowest_pe=slowest, load_imbalance=load_imbalance,
        array_util=array_util, n_nz=routed.n_nz, n_pairs=routed.n_pairs,
        stage2_batch=B, stage2_batch_star=B_star,
    )


def run_pass(activation, kernels_tile, stride, cfg, *, interpretation=None):
    """Drive the functional front end for one output tile, then score it.

    The routing interpretation follows cfg.ARCH unless overridden: "channel" ->
    "v2" (M kernels/PE, union WSP), "act" -> "v1" (one kernel/PE, single WSP).
    Both share the same front end; only the PE mapping differs."""
    if interpretation is None:
        interpretation = "v1" if cfg.ARCH == "act" else "v2"
    r = fn.goSPA_route(activation, kernels_tile, stride,
                       num_pes=cfg.N_PE, num_mults=cfg.M,
                       interpretation=interpretation)
    return score_pass(RoutedPass.from_routing(r), cfg)


# ---------------------------------------------------------------------------
# Self-test (front-end cross-check + the sec.9 invariants)
# ---------------------------------------------------------------------------

def _raises(fn_):
    try:
        fn_()
    except ValueError:
        return True
    return False


def _selftest():
    ok = True

    def check(name, cond):
        nonlocal ok
        ok = ok and cond
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    # STAGE2_BATCH pinned to the as-built width: cases 1-6 assert the native
    # Stage-2 formula, so they must not follow a future change of the default.
    cfg = HwConfig(N_PE=8, M=4, STAGE2_BATCH="native")

    # 1) Functional v2 case: 4x4 activation, three 2x2 kernels in one PE.
    act = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
    kernels = [[[1, 0], [0, 1]], [[0, 2], [-1, 0]], [[1, 1], [1, 1]]]
    r = fn.goSPA_route(act, kernels, 1, num_pes=cfg.N_PE, num_mults=cfg.M,
                       interpretation="v2")
    ps = run_pass(act, kernels, 1, cfg)
    print(f"  case1: {ps.bottleneck} pass={ps.pass_cycles} "
          f"s1={ps.stage1_cycles} s2={ps.stage2_cycles} "
          f"pe={ps.pe_stage_cycles} mem={ps.mem_cycles} util={ps.array_util:.3f}")

    check("stage2 == n_pairs", ps.stage2_cycles == r.n_pairs)
    check("single PE holds all 3 kernels",
          len(ps.per_pe) == 1 and r.pe_chunks == [[0, 1, 2]])
    # pe_stage must equal perf_pe replayed over the real routed streams
    manual = max(
        pe_perf_from_stream(fb, [r.wsps[k] for k in chunk]).pe_cycles
        for chunk, fb in zip(r.pe_chunks, r.fifo_b_list))
    check("pe_stage == max_k perf_pe(routed)", ps.pe_stage_cycles == manual)
    check("pass == max(stages)",
          ps.pass_cycles == max(ps.stage1_cycles, ps.stage2_cycles,
                                ps.pe_stage_cycles, ps.mem_cycles))
    check("bottleneck names the max stage",
          {"stage1": ps.stage1_cycles, "stage2": ps.stage2_cycles,
           "pe": ps.pe_stage_cycles, "mem": ps.mem_cycles}[ps.bottleneck]
          == ps.pass_cycles)
    check("array_util in [0,1]", 0.0 <= ps.array_util <= 1.0)
    # The wall-clock array_util can never exceed the old PE-intrinsic ratio
    # (useful / sum pe_cycles*M): pass_cycles*N_PE >= sum_k pe_cycles_k always,
    # so charging front-end/imbalance/idle-PE slots only lowers it. This is the
    # whole point of the metric -- it stops assuming the PE is the bottleneck.
    pe_only = (sum(s.useful_macs for s in ps.per_pe)
               / sum(s.pe_cycles * s.M for s in ps.per_pe))
    check("array_util <= PE-intrinsic util", ps.array_util <= pe_only + 1e-9)
    check("single PE -> imbalance == 1", abs(ps.load_imbalance - 1.0) < 1e-9)

    # 2) Roofline: dense activation, all-ones kernels, S=F (no window overlap),
    #    and a tile sized to FILL the array (N_PE*M kernels). Every lane fires
    #    every cycle and the pass is PE-bound, so the wall-clock array_util
    #    (full physical array in the denominator) reaches 1.0.
    dense = [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]
    cfg_rl = HwConfig(N_PE=2, M=2)
    ones = [[[1, 1], [1, 1]] for _ in range(cfg_rl.N_PE * cfg_rl.M)]
    r2 = fn.goSPA_route(dense, ones, 2, num_pes=cfg_rl.N_PE, num_mults=cfg_rl.M,
                        interpretation="v2")
    ps2 = run_pass(dense, ones, 2, cfg_rl)
    check("roofline n_pairs == n_nz", r2.n_pairs == r2.n_nz)
    check("roofline pass is PE-bound", ps2.pass_cycles == ps2.pe_stage_cycles)
    check("roofline array_util == 1", abs(ps2.array_util - 1.0) < 1e-9)

    # 3) Load imbalance: one dense kernel vs one near-empty kernel, one kernel
    #    per PE (M=1) -> two PEs with very different FIFO-B depths.
    k_dense = [[1, 1], [1, 1]]
    k_sparse = [[1, 0], [0, 0]]
    cfg2 = HwConfig(N_PE=8, M=1)
    psd = run_pass(dense, [k_dense, k_sparse], 1, cfg2)
    check("two PEs", len(psd.per_pe) == 2)
    check("imbalance > 1 across PEs", psd.load_imbalance > 1.0)

    # 4) STAGE1_ENUM knob: serial >= unrolled, and serial == n_nz + n_pairs.
    cfg_s = HwConfig(STAGE1_ENUM="serial")
    cfg_u = HwConfig(STAGE1_ENUM="unrolled")
    rp = RoutedPass.from_routing(r)
    s_serial = _stage1_cycles(rp, cfg_s)
    s_unroll = _stage1_cycles(rp, cfg_u)
    check("stage1 serial == n_nz + n_pairs", s_serial == r.n_nz + r.n_pairs)
    check("stage1 serial >= unrolled", s_serial >= s_unroll)

    # 5) parallel Stage-1 (the RTL-faithful default) == n_nz.
    cfg_p = HwConfig(STAGE1_ENUM="parallel")
    s_par = _stage1_cycles(rp, cfg_p)
    check("stage1 parallel == n_nz", s_par == r.n_nz)
    check("stage1 serial >= parallel", s_serial >= s_par)

    # 6) act arch on a weight-sparse, large-fmap case: Stage 2 drains M/cycle,
    #    no PE reloads, and lane util beats channel arch (union penalty gone).
    big = [[((i * 7 + j) % 5) for j in range(8)] for i in range(8)]   # ~sparse act
    ksparse = [[[1, 0], [0, 0]], [[0, 1], [0, 0]], [[1, 0], [0, 1]]]  # sparse wgts
    cfg_act = HwConfig(N_PE=8, M=4, ARCH="act", STAGE2_BATCH="native")
    cfg_ch = HwConfig(N_PE=8, M=4, ARCH="channel", STAGE2_BATCH="native")
    r_act = fn.goSPA_route(big, ksparse, 1, num_pes=8, num_mults=4,
                           interpretation="v1")
    ps_act = run_pass(big, ksparse, 1, cfg_act)
    ps_ch = run_pass(big, ksparse, 1, cfg_ch)
    print(f"  arch cmp: act util={ps_act.array_util:.3f} "
          f"ch util={ps_ch.array_util:.3f} | "
          f"act(s2={ps_act.stage2_cycles} pe={ps_act.pe_stage_cycles}) "
          f"ch(s2={ps_ch.stage2_cycles} pe={ps_ch.pe_stage_cycles})")
    check("act stage2 == ceil(n_pairs / M)",
          ps_act.stage2_cycles == ceil(r_act.n_pairs / 4))
    check("act one kernel per PE", len(ps_act.per_pe) == len(ksparse))
    check("act no PE reloads", all(s.num_reloads == 0 for s in ps_act.per_pe))
    check("act array_util in [0,1]", 0.0 <= ps_act.array_util <= 1.0)
    check("ch array_util in [0,1]", 0.0 <= ps_ch.array_util <= 1.0)

    # 7) STAGE2_BATCH knob.
    check("native act B == M", native_batch(cfg_act) == 4)
    check("native channel B == 1", native_batch(cfg_ch) == 1)
    check("native default preserved", HwConfig().STAGE2_BATCH == "native")

    # B=native is just B=1 spelled differently in "channel".
    ps_b1 = run_pass(big, ksparse, 1, HwConfig(N_PE=8, M=4, ARCH="channel",
                                               STAGE2_BATCH=1))
    check("channel B=1 == native", ps_b1.pass_cycles == ps_ch.pass_cycles)

    # Widening the router can only shorten a pass, never lengthen it.
    prev, mono = None, True
    for B in (1, 2, 4, 8, 16, 32, 64):
        c = run_pass(big, ksparse, 1,
                     HwConfig(N_PE=8, M=4, ARCH="act", STAGE2_BATCH=B))
        mono = mono and (prev is None or c.pass_cycles <= prev)
        prev = c.pass_cycles
    check("pass_cycles non-increasing in B", mono)

    # B* is the knee: at B* Stage 2 has stopped binding, and going far past it
    # changes nothing.
    ps_auto = run_pass(big, ksparse, 1, HwConfig(N_PE=8, M=4, ARCH="act",
                                                 STAGE2_BATCH="auto"))
    star = ps_auto.stage2_batch_star
    check("auto uses B*", ps_auto.stage2_batch == star)
    check("auto is not stage2-bound", ps_auto.bottleneck != "stage2")
    ps_wide = run_pass(big, ksparse, 1, HwConfig(N_PE=8, M=4, ARCH="act",
                                                 STAGE2_BATCH=star * 10))
    check("B* saturates (10x B* is no faster)",
          ps_wide.pass_cycles == ps_auto.pass_cycles)
    check("B* is minimal (B*-1 is strictly slower or B*==1)",
          star == 1 or run_pass(big, ksparse, 1,
                                HwConfig(N_PE=8, M=4, ARCH="act",
                                         STAGE2_BATCH=star - 1)
                                ).pass_cycles > ps_auto.pass_cycles)

    # B* is recorded even when B is pinned, so headroom stays visible.
    check("B* recorded under native", ps_act.stage2_batch_star >= 1)

    # rescore_at_batch must reproduce a real run at the same width.
    re_b = rescore_at_batch(ps_act, cfg_act, star)
    check("rescore_at_batch == run_pass at same B",
          re_b.pass_cycles == ps_auto.pass_cycles
          and abs(re_b.array_util - ps_auto.array_util) < 1e-12
          and re_b.bottleneck == ps_auto.bottleneck)

    # The cap we derived: in "act", useful = N_PE*d_w*n_pairs while Stage 2
    # needs >= n_pairs/M cycles, so native array_util <= d_w regardless of
    # which stage actually binds. Here d_w = 4/16 nonzero weights per kernel.
    d_w = sum(sum(row) for k in ksparse for row in k) / (len(ksparse) * 4)
    check("native act array_util <= d_w", ps_act.array_util <= d_w + 1e-9)

    check("bad STAGE2_BATCH rejected",
          _raises(lambda: run_pass(big, ksparse, 1,
                                   HwConfig(ARCH="act", STAGE2_BATCH=0))))

    print(f"\nperf_model self-test: {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    _selftest()
