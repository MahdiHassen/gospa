"""
goSPA performance model -- network-level simulation driver.

Top driver (PERF_MODEL_PLAN.md sec.7/8/11.7). Runs a list of Layers through
simulate_layer, sums to total cycles, and reports:
  - per-layer and total latency (ms) + FPS
  - lane utilization and per-PE load-imbalance factor
  - speedup vs. dense baseline (headline vs. paper Table -- sec.8)
  - achieved vs. ideal MAC throughput

Entry point:  python sim.py              (AlexNet, d_a=0.5, d_w=0.5)
              python sim.py --no-baseline (skip the dense comparison run)
"""

import sys
import os
from dataclasses import dataclass, replace

from config import HwConfig
from layer import Layer, LayerStats, simulate_layer


# ---------------------------------------------------------------------------
# Network result record
# ---------------------------------------------------------------------------

@dataclass
class NetworkStats:
    per_layer: list           # list[LayerStats] -- sparse run
    per_layer_dense: list     # list[LayerStats] -- dense baseline; [] if skipped
    total_cycles: int
    total_cycles_dense: int   # 0 if baseline not run
    total_latency_ms: float
    fps: float
    speedup: float            # total_cycles_dense / total_cycles (1.0 if no baseline)
    lane_util: float          # useful_macs / (pe_cycles * M) across all passes+layers
    load_imbalance: float     # mean per-layer load-imbalance factor
    achieved_macs_cycle: float   # useful_macs / total_wall_pass_cycles
    ideal_macs_cycle: int        # N_PE * M


# ---------------------------------------------------------------------------
# Network runner
# ---------------------------------------------------------------------------

def run_network(layers, cfg, *, seed=0, baseline=True):
    """
    Run each Layer through simulate_layer and return NetworkStats.

    baseline=True (default): re-runs each layer at d_a=1.0, d_w=1.0 to
    compute speedup vs dense (sec.8). Roughly doubles total runtime.
    """
    per_layer = []
    per_layer_dense = []

    for i, layer in enumerate(layers):
        label = (layer.name or f"layer{i}")
        print(f"  [{i+1}/{len(layers)}] {label:<8}", end=" ", flush=True)

        stats = simulate_layer(layer, cfg, seed=seed + i)
        per_layer.append(stats)

        if baseline:
            ds = simulate_layer(replace(layer, d_a=1.0, d_w=1.0), cfg, seed=seed + i)
            per_layer_dense.append(ds)
            spd = ds.layer_cycles / stats.layer_cycles
            print(f"cycles={stats.layer_cycles:>12,}  util={stats.lane_util:.3f}  speedup={spd:.2f}x")
        else:
            print(f"cycles={stats.layer_cycles:>12,}  util={stats.lane_util:.3f}")

    total_cycles = sum(s.layer_cycles for s in per_layer)
    total_dense  = sum(s.layer_cycles for s in per_layer_dense) if baseline else 0

    freq = max(1.0, cfg.FREQ_HZ)
    total_latency_ms = total_cycles / freq * 1000.0
    fps = 1000.0 / total_latency_ms if total_latency_ms else 0.0
    speedup = total_dense / total_cycles if (baseline and total_cycles) else 1.0

    # Utilization: sum useful MACs / sum (pe_cycles * M) across all layers+passes+PEs
    useful = sum(s.useful_macs
                 for ls in per_layer
                 for ps in ls.per_pass
                 for s  in ps.per_pe)
    slots  = sum(s.pe_cycles * s.M
                 for ls in per_layer
                 for ps in ls.per_pass
                 for s  in ps.per_pe)
    lane_util = useful / slots if slots else 0.0

    # Achieved throughput: useful MACs per wall-clock pass cycle
    wall_cycles = sum(ls.pass_cycles_total for ls in per_layer)
    ideal = cfg.N_PE * cfg.M
    achieved_macs_cycle = useful / wall_cycles if wall_cycles else 0.0

    imbalances = [ls.load_imbalance for ls in per_layer]
    load_imbalance = sum(imbalances) / len(imbalances) if imbalances else 1.0

    return NetworkStats(
        per_layer=per_layer,
        per_layer_dense=per_layer_dense,
        total_cycles=total_cycles,
        total_cycles_dense=total_dense,
        total_latency_ms=total_latency_ms,
        fps=fps,
        speedup=speedup,
        lane_util=lane_util,
        load_imbalance=load_imbalance,
        achieved_macs_cycle=achieved_macs_cycle,
        ideal_macs_cycle=ideal,
    )


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(net, layers, cfg):
    has_baseline = bool(net.per_layer_dense)
    freq = max(1.0, cfg.FREQ_HZ)
    ideal = cfg.N_PE * cfg.M

    col = f"{'Layer':<8}  {'Cycles':>12}  {'Lat(ms)':>8}  {'Util':>6}  {'Imbal':>6}  {'Bottleneck':>10}"
    if has_baseline:
        col += f"  {'Speedup':>7}"
    sep = "-" * len(col)

    print(sep)
    print(col)
    print(sep)

    for i, (ls, layer) in enumerate(zip(net.per_layer, layers)):
        lat = ls.layer_cycles / freq * 1000.0
        bk  = max(ls.bottleneck_hist, key=ls.bottleneck_hist.get) if ls.bottleneck_hist else "-"
        name = (layer.name or f"layer{i}")[:8]
        row = (f"{name:<8}  {ls.layer_cycles:>12,}  {lat:>8.3f}  "
               f"{ls.lane_util:>6.3f}  {ls.load_imbalance:>6.3f}  {bk:>10}")
        if has_baseline:
            spd = net.per_layer_dense[i].layer_cycles / ls.layer_cycles
            row += f"  {spd:>6.2f}x"
        print(row)

    print(sep)
    total_row = (f"{'TOTAL':<8}  {net.total_cycles:>12,}  {net.total_latency_ms:>8.3f}  "
                 f"{net.lane_util:>6.3f}  {net.load_imbalance:>6.3f}  {'':>10}")
    if has_baseline:
        total_row += f"  {net.speedup:>6.2f}x"
    print(total_row)
    print(sep)

    print()
    print(f"FPS                  : {net.fps:.2f}")
    print(f"Achieved throughput  : {net.achieved_macs_cycle:.2f} MACs/cycle  "
          f"(ideal {ideal}  ->  {net.achieved_macs_cycle / ideal * 100:.1f}% array efficiency)")
    if has_baseline:
        print(f"Speedup vs dense     : {net.speedup:.2f}x")
        print(f"  paper target (AlexNet): 1.38x  -- gap explained by synthetic sparsity")
        print(f"  placeholders (d_a={layers[0].d_a}, d_w={layers[0].d_w}); "
              f"replace with real maps for paper replication (sec.9)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(__file__))
    from workloads.alexnet import LAYERS

    no_baseline = "--no-baseline" in sys.argv

    cfg = HwConfig()   # defaults: N_PE=8, M=4, FREQ_HZ=1e9

    print(f"goSPA perf model -- AlexNet")
    print(f"  HW : N_PE={cfg.N_PE}  M={cfg.M}  FREQ={cfg.FREQ_HZ/1e9:.1f} GHz")
    print(f"  Sparsity : d_a={LAYERS[0].d_a}  d_w={LAYERS[0].d_w}  (synthetic phase-1)")
    print(f"  Baseline : {'disabled' if no_baseline else 'enabled (doubles runtime)'}")
    print(f"  Layers   : {len(LAYERS)}")
    print()

    net = run_network(LAYERS, cfg, seed=0, baseline=not no_baseline)

    print()
    print_report(net, LAYERS, cfg)
