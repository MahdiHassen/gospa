"""
goSPA performance model -- Layer descriptor and multi-pass expansion.

Sits above perf_model.py (which scores one pass) and config.py (hardware knobs).
Adds the two outer loops perf_model does not own:
  - output-channel tiling: ceil(Cout / tile) tiles, tile = N_PE*M ("channel")
    or N_PE ("act" -- one kernel/PE, so M x more tiles)              (sec.5)
  - input-channel passes:  Cin passes per tile; accumulators persist (sec.5)
Applies FILL / DRAIN bracketing (sec.4) and the once-per-layer output store
(sec.5 -- perf_model.py deliberately omits the output write-back).
Synthetic Bernoulli sparsity is used for phase-1 bring-up (sec.6); real
PyTorch-captured maps slot in without changing this module's interface.
"""

import random
from dataclasses import dataclass, field, replace
from math import ceil

from config import HwConfig
import perf_model as pm


# ---------------------------------------------------------------------------
# Layer descriptor  (PERF_MODEL_PLAN.md sec.5)
# ---------------------------------------------------------------------------

@dataclass
class Layer:
    H: int               # input height (before padding)
    W: int               # input width  (before padding)
    F: int               # kernel spatial size (F x F)
    S: int               # stride
    pad: int             # zero-padding on each side
    Cin: int             # input channels
    Cout: int            # output channels
    type: str = "conv"   # "conv" | "1x1" | "dw" | "fc"
    d_a: float = 1.0     # activation density (1.0 = dense)
    d_w: float = 1.0     # weight density     (1.0 = dense)
    name: str = ""       # optional human-readable label


# ---------------------------------------------------------------------------
# Synthetic sparsity  (phase 1 -- sec.6; replaced by real maps in phase 2)
# ---------------------------------------------------------------------------

def _make_activation(H, W, d_a, rng):
    """Bernoulli H x W activation: zero with prob (1-d_a), small int otherwise."""
    out = []
    for _ in range(H):
        row = []
        for _ in range(W):
            if rng.random() < d_a:
                v = rng.randint(1, 127)
                row.append(v if rng.random() > 0.5 else -v)
            else:
                row.append(0)
        out.append(row)
    return out


def _make_kernel(F, d_w, rng):
    """Bernoulli F x F kernel: zero with prob (1-d_w), small int otherwise."""
    return [
        [
            (rng.randint(1, 3) * (1 if rng.random() > 0.5 else -1))
            if rng.random() < d_w else 0
            for _ in range(F)
        ]
        for _ in range(F)
    ]


# ---------------------------------------------------------------------------
# Result record
# ---------------------------------------------------------------------------

@dataclass
class LayerStats:
    layer_cycles: int
    pass_cycles_total: int   # sum of per-pass pass_cycles (no fill/drain/store)
    fill_cycles: int
    drain_cycles: int
    out_store_cycles: int    # output writeback charged once per layer (sec.5)
    n_out_tiles: int         # ceil(Cout / tile); tile = N_PE*M ("channel") | N_PE ("act")
    n_passes: int            # n_out_tiles * Cin (conv/1x1/fc) | n_out_tiles (dw)
    E: int                   # output spatial dimension (E x E feature map)
    per_pass: list = field(default_factory=list)   # list[PassStats]
    array_util: float = 0.0         # useful_macs / (sum pass_cycles * N_PE * M)
    load_imbalance: float = 1.0     # mean per-pass load-imbalance factor
    bottleneck_hist: dict = field(default_factory=dict)   # stage -> pass count
    latency_ms: float = 0.0         # layer_cycles / FREQ_HZ * 1000
    stage2_batch: int = 1           # router width B used by every pass
    stage2_batch_star: int = 1      # max over passes of B* -- the narrowest
                                    # router that would stop binding ANYWHERE
                                    # in this layer (recorded even when B is
                                    # pinned, so headroom stays visible)


# ---------------------------------------------------------------------------
# Layer simulation
# ---------------------------------------------------------------------------

def simulate_layer(layer: Layer, cfg: HwConfig, seed: int = 0) -> LayerStats:
    """
    Expand one Layer into the sequence of passes perf_model scores, apply
    FILL / DRAIN and the once-per-layer output store, and return LayerStats.

    Pass structure (sec.5):
      conv / 1x1 / fc:  n_out_tiles * Cin passes.  One routing run per
        (output-channel tile, input channel); accumulators stay resident
        across the Cin inner passes, as in goSPA_multichannel.
      dw (depthwise):   n_out_tiles passes.  Each output channel oc uses
        input channel ic = oc.  The hardware would need a different
        activation per PE lane; the perf model uses one representative
        activation per tile (a deliberate approximation for phase 1).

    Synthetic Bernoulli activations / kernels are generated from `seed`
    (deterministic: same seed -> same sparsity pattern -> same cycle counts).
    """
    if layer.type not in ("conv", "1x1", "dw", "fc"):
        raise ValueError(f"unknown layer type {layer.type!r}")
    if layer.type == "dw" and layer.Cin != layer.Cout:
        raise ValueError("depthwise layer requires Cin == Cout")

    H_eff = layer.H + 2 * layer.pad
    W_eff = layer.W + 2 * layer.pad
    E = (H_eff - layer.F) // layer.S + 1
    # "channel" packs N_PE*M output channels per tile (M kernels/PE); "act"
    # packs only N_PE (one kernel/PE), so it needs M x more tiles (and passes).
    tile_size = cfg.N_PE * cfg.M if cfg.ARCH == "channel" else cfg.N_PE
    n_out_tiles = ceil(layer.Cout / tile_size)

    rng = random.Random(seed)
    per_pass = []

    for t in range(n_out_tiles):
        oc_start = t * tile_size
        oc_end = min(oc_start + tile_size, layer.Cout)
        n_oc = oc_end - oc_start

        if layer.type == "dw":
            # One representative activation shared across the tile's lanes.
            act = _make_activation(H_eff, W_eff, layer.d_a, rng)
            kernels_tile = [_make_kernel(layer.F, layer.d_w, rng)
                            for _ in range(n_oc)]
            per_pass.append(pm.run_pass(act, kernels_tile, layer.S, cfg))
        else:
            # One routing run per (output tile, input channel).
            for _ in range(layer.Cin):
                act = _make_activation(H_eff, W_eff, layer.d_a, rng)
                kernels_tile = [_make_kernel(layer.F, layer.d_w, rng)
                                for _ in range(n_oc)]
                per_pass.append(pm.run_pass(act, kernels_tile, layer.S, cfg))

    # Output writeback: resident accumulators -> memory, charged once per layer.
    out_bytes = layer.Cout * E * E * cfg.BYTES_OUT
    out_store_cycles = ceil(out_bytes / max(1, cfg.MEM_BW_BYTES)) + cfg.MEM_LATENCY

    return _aggregate(per_pass, cfg, out_store_cycles, n_out_tiles, E)


def _aggregate(per_pass, cfg, out_store_cycles, n_out_tiles, E) -> LayerStats:
    """Roll a scored pass list up into LayerStats. Shared by simulate_layer and
    rescore_layer_at_batch so the two can never drift apart."""
    pass_cycles_total = sum(ps.pass_cycles for ps in per_pass)
    layer_cycles = cfg.FILL + pass_cycles_total + out_store_cycles + cfg.DRAIN

    # Array utilization over wall-clock: useful MACs / (the full physical array
    # N_PE*M offered over every pass's pass_cycles). Charges front-end/mem
    # stalls, cross-PE load imbalance and idle PEs -- all the slots wasted when
    # the PE array is not the bottleneck. FILL/DRAIN/out_store are excluded, so
    # this is the compute-pass array efficiency (matches sim.py's achieved/ideal).
    useful = sum(s.useful_macs for ps in per_pass for s in ps.per_pe)
    slots  = pass_cycles_total * cfg.N_PE * cfg.M
    array_util = useful / slots if slots else 0.0

    imbalances = [ps.load_imbalance for ps in per_pass if ps.per_pe]
    load_imbalance = sum(imbalances) / len(imbalances) if imbalances else 1.0

    bottleneck_hist = {}
    for ps in per_pass:
        bottleneck_hist[ps.bottleneck] = bottleneck_hist.get(ps.bottleneck, 0) + 1

    latency_ms = layer_cycles / max(1.0, cfg.FREQ_HZ) * 1000.0

    return LayerStats(
        layer_cycles=layer_cycles,
        pass_cycles_total=pass_cycles_total,
        fill_cycles=cfg.FILL,
        drain_cycles=cfg.DRAIN,
        out_store_cycles=out_store_cycles,
        n_out_tiles=n_out_tiles,
        n_passes=len(per_pass),
        E=E,
        per_pass=per_pass,
        load_imbalance=load_imbalance,
        bottleneck_hist=bottleneck_hist,
        latency_ms=latency_ms,
        array_util=array_util,
        stage2_batch=max((ps.stage2_batch for ps in per_pass), default=1),
        stage2_batch_star=max((ps.stage2_batch_star for ps in per_pass),
                              default=1),
    )


def rescore_layer_at_batch(ls: LayerStats, cfg: HwConfig, B: int) -> LayerStats:
    """Re-derive a whole layer at router width B, with no re-simulation.

    Only Stage 2 depends on B, and every other stage counter is already stored
    per pass, so this is pure arithmetic over `ls.per_pass`. This is what lets
    sim.py resolve a network-wide "auto" width: simulate once, then re-derive
    every layer at the single chosen B."""
    per_pass = [pm.rescore_at_batch(ps, cfg, B) for ps in ls.per_pass]
    return _aggregate(per_pass, cfg, ls.out_store_cycles, ls.n_out_tiles, ls.E)


# ---------------------------------------------------------------------------
# Self-test  (sec.9 invariants extended to layer scope)
# ---------------------------------------------------------------------------

def _selftest():
    ok = True

    def check(name, cond):
        nonlocal ok
        ok = ok and cond
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    cfg = HwConfig(N_PE=4, M=2, FREQ_HZ=1e9, FILL=10, DRAIN=5)

    # 1) conv: cycle decomposition and pass-count invariants.
    layer = Layer(H=8, W=8, F=3, S=1, pad=0, Cin=2, Cout=8,
                  type="conv", d_a=0.7, d_w=0.6)
    stats = simulate_layer(layer, cfg, seed=42)
    print(f"  conv: cycles={stats.layer_cycles} passes={stats.n_passes} "
          f"tiles={stats.n_out_tiles} util={stats.array_util:.3f} "
          f"imbalance={stats.load_imbalance:.3f}")
    check("n_out_tiles == ceil(Cout / (N_PE*M))",
          stats.n_out_tiles == ceil(layer.Cout / (cfg.N_PE * cfg.M)))
    check("n_passes == n_out_tiles * Cin",
          stats.n_passes == stats.n_out_tiles * layer.Cin)
    check("layer_cycles == FILL + pass_total + out_store + DRAIN",
          stats.layer_cycles == (cfg.FILL + stats.pass_cycles_total
                                  + stats.out_store_cycles + cfg.DRAIN))
    check("array_util in [0, 1]", 0.0 <= stats.array_util <= 1.0)
    check("load_imbalance >= 1", stats.load_imbalance >= 1.0 - 1e-9)
    check("E == (H - F) // S + 1",
          stats.E == (layer.H - layer.F) // layer.S + 1)

    # 2) Dense roofline: d_a=d_w=1, S=F, Cout=tile_size (array FULL) -> the
    #    wall-clock array_util == 1 (every lane busy, pass PE-bound, no idle PEs).
    #    With S=F=2, each activation maps to exactly one (CID,PID) pair; all-ones
    #    kernels make union WSP all-1 so every activation is admitted and every
    #    lane fires at every entry.
    tile_size = cfg.N_PE * cfg.M
    layer_dense = Layer(H=4, W=4, F=2, S=2, pad=0, Cin=1, Cout=tile_size,
                        type="conv", d_a=1.0, d_w=1.0)
    stats_dense = simulate_layer(layer_dense, cfg, seed=0)
    print(f"  dense roofline: util={stats_dense.array_util:.3f}")
    check("dense -> array_util == 1.0", abs(stats_dense.array_util - 1.0) < 1e-9)

    # 3) 1x1 conv: runs without error, pass count correct.
    layer_1x1 = Layer(H=7, W=7, F=1, S=1, pad=0, Cin=3, Cout=4,
                      type="1x1", d_a=0.8, d_w=1.0)
    stats_1x1 = simulate_layer(layer_1x1, cfg, seed=1)
    check("1x1 n_passes == n_out_tiles * Cin",
          stats_1x1.n_passes == stats_1x1.n_out_tiles * layer_1x1.Cin)

    # 4) Depthwise: n_passes == n_out_tiles (no Cin inner loop).
    layer_dw = Layer(H=7, W=7, F=3, S=1, pad=1, Cin=8, Cout=8,
                     type="dw", d_a=0.6, d_w=0.7)
    stats_dw = simulate_layer(layer_dw, cfg, seed=2)
    print(f"  dw: passes={stats_dw.n_passes} tiles={stats_dw.n_out_tiles}")
    check("dw n_passes == n_out_tiles",
          stats_dw.n_passes == stats_dw.n_out_tiles)

    # 5) Padding: E computed from H_eff = H + 2*pad.
    layer_pad = Layer(H=5, W=5, F=3, S=1, pad=1, Cin=1, Cout=tile_size,
                      type="conv", d_a=1.0, d_w=1.0)
    stats_pad = simulate_layer(layer_pad, cfg, seed=3)
    E_exp = (layer_pad.H + 2 * layer_pad.pad - layer_pad.F) // layer_pad.S + 1
    check("padded E == (H+2p-F)//S+1", stats_pad.E == E_exp)

    # 6) FILL=DRAIN=0: layer_cycles == pass_total + out_store.
    cfg0 = HwConfig(N_PE=4, M=2, FILL=0, DRAIN=0)
    stats0 = simulate_layer(layer, cfg0, seed=42)
    check("FILL=DRAIN=0: layer == pass_total + out_store",
          stats0.layer_cycles == stats0.pass_cycles_total + stats0.out_store_cycles)

    # 7) Bottleneck histogram covers all passes.
    check("bottleneck_hist sums to n_passes",
          sum(stats.bottleneck_hist.values()) == stats.n_passes)

    # 8) act arch: one kernel/PE tiling (M x more tiles). On the wall-clock
    #    array_util the M x re-stream against an un-widened Stage 1 shows up as
    #    idle PE-slots, so act does NOT beat channel on total-cycle utilization
    #    even though its PE-intrinsic lane util is higher (union penalty gone).
    #    This is exactly the effect the PE-cycle-only metric used to hide.
    cfg_ch = HwConfig(N_PE=4, M=2, FILL=0, DRAIN=0, ARCH="channel")
    cfg_ac = HwConfig(N_PE=4, M=2, FILL=0, DRAIN=0, ARCH="act")
    lyr = Layer(H=10, W=10, F=3, S=1, pad=0, Cin=2, Cout=16,
                type="conv", d_a=0.8, d_w=0.4)
    st_ch = simulate_layer(lyr, cfg_ch, seed=7)
    st_ac = simulate_layer(lyr, cfg_ac, seed=7)
    print(f"  arch: ch tiles={st_ch.n_out_tiles} util={st_ch.array_util:.3f} "
          f"cyc={st_ch.pass_cycles_total} | "
          f"act tiles={st_ac.n_out_tiles} util={st_ac.array_util:.3f} "
          f"cyc={st_ac.pass_cycles_total}")
    check("act n_out_tiles == ceil(Cout / N_PE)",
          st_ac.n_out_tiles == ceil(lyr.Cout / cfg_ac.N_PE))
    check("act has M x channel's tiles",
          st_ac.n_out_tiles == cfg_ac.M * st_ch.n_out_tiles)
    check("act n_passes == n_out_tiles * Cin",
          st_ac.n_passes == st_ac.n_out_tiles * lyr.Cin)
    check("both array_util in [0,1]",
          0.0 <= st_ac.array_util <= 1.0 and 0.0 <= st_ch.array_util <= 1.0)
    # On a 3x3 conv the two arches are ~neutral in wall-clock: act's PE-intrinsic
    # utilization edge (union penalty gone) is cancelled by its M x re-stream
    # against an un-widened Stage 1. The PE-cycle-only metric used to report act
    # as the clear winner here; array_util correctly shows the tie.
    check("act no longer beats channel on wall-clock util (3x3 ~neutral)",
          abs(st_ac.array_util - st_ch.array_util) < 0.05)

    # 9) Router-width knob at layer scope.
    lyr9 = Layer(name="c9", H=13, W=13, F=3, S=1, pad=1, Cin=2, Cout=16,
                 type="conv", d_a=0.5, d_w=0.4)
    cfg9 = HwConfig(N_PE=8, M=4, ARCH="act", STAGE2_BATCH="native")
    st9 = simulate_layer(lyr9, cfg9, seed=0)
    Bstar = st9.stage2_batch_star
    check("layer B* == max over passes",
          Bstar == max(ps.stage2_batch_star for ps in st9.per_pass))
    check("layer B* >= native B", Bstar >= st9.stage2_batch)

    # rescore_layer_at_batch must agree with a full re-simulation at the same B.
    re9 = rescore_layer_at_batch(st9, cfg9, Bstar)
    sim9 = simulate_layer(lyr9, replace(cfg9, STAGE2_BATCH=Bstar), seed=0)
    check("rescore_layer_at_batch == simulate_layer at same B",
          re9.pass_cycles_total == sim9.pass_cycles_total
          and re9.layer_cycles == sim9.layer_cycles
          and abs(re9.array_util - sim9.array_util) < 1e-12
          and re9.bottleneck_hist == sim9.bottleneck_hist)
    # Widening to the layer-wide B* must not leave any pass Stage-2-bound.
    check("no pass stage2-bound at layer B*",
          "stage2" not in re9.bottleneck_hist)
    check("widening cannot slow the layer",
          re9.pass_cycles_total <= st9.pass_cycles_total)
    check("native default unchanged by the refactor",
          simulate_layer(lyr9, HwConfig(N_PE=8, M=4, ARCH="act"), seed=0)
          .pass_cycles_total == st9.pass_cycles_total)

    print(f"\nlayer self-test: {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    _selftest()
