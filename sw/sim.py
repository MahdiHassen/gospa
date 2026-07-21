"""
goSPA performance model -- network-level simulation driver.

Top driver (PERF_MODEL_PLAN.md sec.7/8/11.7). Runs a list of Layers through
simulate_layer, sums to total cycles, and reports:
  - per-layer and total latency (ms) + FPS
  - array utilization (wall-clock) and per-PE load-imbalance factor
  - speedup vs. dense baseline (headline vs. paper Table -- sec.8)
  - achieved vs. ideal MAC throughput

Entry point:  python sim.py              (AlexNet, d_a=0.5, d_w=0.5)
              python sim.py --no-baseline (skip the dense comparison run)
              python sim.py --sweep     (act-architecture sparsity sweep)
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
    array_util: float         # useful_macs / (sum pass_cycles * N_PE * M) over layers
    load_imbalance: float     # mean per-layer load-imbalance factor
    achieved_macs_cycle: float   # useful_macs / total_wall_pass_cycles
    ideal_macs_cycle: int        # N_PE * M


@dataclass
class SweepPoint:
    activation_density: float
    weight_density: float
    total_cycles: int
    total_latency_ms: float
    fps: float
    array_util: float
    load_imbalance: float
    achieved_macs_cycle: float
    speedup_vs_dense: float = 0.0


@dataclass
class SweepStats:
    points: list
    dense_cycles: int = 0


# ---------------------------------------------------------------------------
# Network runner
# ---------------------------------------------------------------------------

def run_network(layers, cfg, *, seed=0, baseline=True, verbose=True):
    """
    Run each Layer through simulate_layer and return NetworkStats.

    baseline=True (default): re-runs each layer at d_a=1.0, d_w=1.0 to
    compute speedup vs dense (sec.8). Roughly doubles total runtime.

    verbose=False suppresses per-layer progress output, which is useful when
    this function is called repeatedly by the sparsity sweep.
    """
    per_layer = []
    per_layer_dense = []

    for i, layer in enumerate(layers):
        label = (layer.name or f"layer{i}")
        if verbose:
            print(f"  [{i+1}/{len(layers)}] {label:<8}", end=" ", flush=True)

        stats = simulate_layer(layer, cfg, seed=seed + i)
        per_layer.append(stats)

        if baseline:
            ds = simulate_layer(replace(layer, d_a=1.0, d_w=1.0), cfg, seed=seed + i)
            per_layer_dense.append(ds)
            if verbose:
                spd = ds.layer_cycles / stats.layer_cycles
                print(f"cycles={stats.layer_cycles:>12,}  util={stats.array_util:.3f}  speedup={spd:.2f}x")
        elif verbose:
            print(f"cycles={stats.layer_cycles:>12,}  util={stats.array_util:.3f}")

    total_cycles = sum(s.layer_cycles for s in per_layer)
    total_dense  = sum(s.layer_cycles for s in per_layer_dense) if baseline else 0

    freq = max(1.0, cfg.FREQ_HZ)
    total_latency_ms = total_cycles / freq * 1000.0
    fps = 1000.0 / total_latency_ms if total_latency_ms else 0.0
    speedup = total_dense / total_cycles if (baseline and total_cycles) else 1.0

    # Array utilization over WALL-CLOCK: sum useful MACs / (the full physical
    # array N_PE*M offered over every pass's pass_cycles). This charges the slots
    # idled whenever the PE array is not the bottleneck (front-end/mem stalls,
    # load imbalance, idle PEs) -- unlike a pe_cycles-only ratio. Equals
    # achieved_macs_cycle / ideal below by construction.
    useful = sum(s.useful_macs
                 for ls in per_layer
                 for ps in ls.per_pass
                 for s  in ps.per_pe)
    wall_cycles = sum(ls.pass_cycles_total for ls in per_layer)
    ideal = cfg.N_PE * cfg.M
    achieved_macs_cycle = useful / wall_cycles if wall_cycles else 0.0
    array_util = achieved_macs_cycle / ideal if ideal else 0.0

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
        array_util=array_util,
        load_imbalance=load_imbalance,
        achieved_macs_cycle=achieved_macs_cycle,
        ideal_macs_cycle=ideal,
    )


# ---------------------------------------------------------------------------
# Activation/weight sparsity sweep
# ---------------------------------------------------------------------------

def parse_density_list(text):
    """Parse a comma-separated list of non-zero densities in [0, 1]."""
    values = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            value = float(token)
        except ValueError as exc:
            raise ValueError(f"invalid density {token!r}") from exc
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"density must be in [0, 1], got {value}")
        if value not in values:
            values.append(value)
    if not values:
        raise ValueError("density list must contain at least one value")
    return values


def select_sweep_layers(layers, layer_name=None):
    """Return the full workload or one layer selected by its exact name."""
    if layer_name is None:
        return list(layers)
    selected = [layer for layer in layers if layer.name == layer_name]
    if not selected:
        names = ", ".join(layer.name or "<unnamed>" for layer in layers)
        raise ValueError(f"unknown layer {layer_name!r}; available layers: {names}")
    return selected


def run_sparsity_sweep(layers, cfg, activation_densities, weight_densities,
                       *, seed=0, baseline=True, verbose=True):
    """Sweep synthetic activation/weight densities on the ``act`` dataflow.

    Each grid point receives the same base seed. The dense reference is run at
    most once and reused for every point, including the (1.0, 1.0) grid point.
    Density means the probability of a non-zero value; reported sparsity is
    therefore ``1 - density``.
    """
    if cfg.ARCH != "act":
        raise ValueError("sparsity sweep is defined for the new ARCH='act' path")
    if not layers:
        raise ValueError("sparsity sweep requires at least one layer")

    activation_densities = list(activation_densities)
    weight_densities = list(weight_densities)
    for value in activation_densities + weight_densities:
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"density must be in [0, 1], got {value}")
    if not activation_densities or not weight_densities:
        raise ValueError("both density axes must contain at least one value")

    dense_net = None
    if baseline:
        dense_layers = [replace(layer, d_a=1.0, d_w=1.0) for layer in layers]
        if verbose:
            print("  dense baseline", flush=True)
        dense_net = run_network(
            dense_layers, cfg, seed=seed, baseline=False, verbose=False
        )

    points = []
    total_points = len(activation_densities) * len(weight_densities)
    point_index = 0
    for d_a in activation_densities:
        for d_w in weight_densities:
            point_index += 1
            if verbose:
                print(f"  [{point_index}/{total_points}] d_a={d_a:.3f} "
                      f"d_w={d_w:.3f}", flush=True)

            if dense_net is not None and d_a == 1.0 and d_w == 1.0:
                net = dense_net
            else:
                point_layers = [replace(layer, d_a=d_a, d_w=d_w)
                                for layer in layers]
                net = run_network(
                    point_layers, cfg, seed=seed, baseline=False, verbose=False
                )

            speedup = (dense_net.total_cycles / net.total_cycles
                       if dense_net is not None and net.total_cycles else 0.0)
            points.append(SweepPoint(
                activation_density=d_a,
                weight_density=d_w,
                total_cycles=net.total_cycles,
                total_latency_ms=net.total_latency_ms,
                fps=net.fps,
                array_util=net.array_util,
                load_imbalance=net.load_imbalance,
                achieved_macs_cycle=net.achieved_macs_cycle,
                speedup_vs_dense=speedup,
            ))

    return SweepStats(
        points=points,
        dense_cycles=dense_net.total_cycles if dense_net is not None else 0,
    )


def print_sweep_report(sweep, layers, cfg):
    """Print one compact row per activation/weight density pair."""
    layer_label = ", ".join(layer.name or "<unnamed>" for layer in layers)
    print()
    print(f"goSPA sparsity sweep -- ARCH={cfg.ARCH} -- layers: {layer_label}")
    print("Density is the non-zero fraction; sparsity = 1 - density.")

    header = (f"{'ActDen':>6}  {'ActSparse':>9}  {'WgtDen':>6}  "
              f"{'WgtSparse':>9}  {'Cycles':>12}  {'FPS':>10}  "
              f"{'Util':>6}  {'Imbal':>6}  {'MAC/cyc':>8}")
    if sweep.dense_cycles:
        header += f"  {'Speedup':>7}"
    separator = "-" * len(header)
    print(separator)
    print(header)
    print(separator)

    for point in sweep.points:
        row = (f"{point.activation_density:>6.2f}  "
               f"{1.0 - point.activation_density:>9.2f}  "
               f"{point.weight_density:>6.2f}  "
               f"{1.0 - point.weight_density:>9.2f}  "
               f"{point.total_cycles:>12,}  {point.fps:>10.2f}  "
               f"{point.array_util:>6.3f}  "
               f"{point.load_imbalance:>6.3f}  "
               f"{point.achieved_macs_cycle:>8.2f}")
        if sweep.dense_cycles:
            row += f"  {point.speedup_vs_dense:>6.2f}x"
        print(row)
    print(separator)


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
               f"{ls.array_util:>6.3f}  {ls.load_imbalance:>6.3f}  {bk:>10}")
        if has_baseline:
            spd = net.per_layer_dense[i].layer_cycles / ls.layer_cycles
            row += f"  {spd:>6.2f}x"
        print(row)

    print(sep)
    total_row = (f"{'TOTAL':<8}  {net.total_cycles:>12,}  {net.total_latency_ms:>8.3f}  "
                 f"{net.array_util:>6.3f}  {net.load_imbalance:>6.3f}  {'':>10}")
    if has_baseline:
        total_row += f"  {net.speedup:>6.2f}x"
    print(total_row)
    print(sep)

    print()
    print(f"FPS                  : {net.fps:.2f}")
    print(f"Achieved throughput  : {net.achieved_macs_cycle:.2f} MACs/cycle  "
          f"(ideal {ideal}  ->  {net.achieved_macs_cycle / ideal * 100:.1f}% array efficiency"
          f"  == the Util column)")
    if has_baseline:
        print(f"Speedup vs dense     : {net.speedup:.2f}x")
        print(f"  paper target (AlexNet): 1.38x  -- gap explained by synthetic sparsity")
        print(f"  placeholders (d_a={layers[0].d_a}, d_w={layers[0].d_w}); "
              f"replace with real maps for paper replication (sec.9)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    sys.path.insert(0, os.path.dirname(__file__))

    parser = argparse.ArgumentParser(description="goSPA network-level performance model")
    parser.add_argument("--net", default="alexnet",
                        choices=["alexnet", "mobilenetv2"],
                        help="workload to simulate (default: alexnet)")
    parser.add_argument("--arch", default=None,
                        choices=["channel", "act"],
                        help="dataflow: 'channel' (M kernels/PE, union WSP) or "
                             "'act' (1 kernel/PE, M activations/cycle) "
                             "(default: channel, or act with --sweep)")
    parser.add_argument("--no-baseline", action="store_true",
                        help="skip dense comparison run")
    parser.add_argument("--da", default=0.5,
                        help="activation density", type=float)
    parser.add_argument("--dw", default=0.5,
                        help="weight density", type=float)
    parser.add_argument("--seed", default=0, type=int,
                        help="synthetic sparsity RNG seed (default: 0)")
    parser.add_argument("--sweep", action="store_true",
                        help="run an activation/weight density grid on the new "
                             "act architecture")
    parser.add_argument("--sweep-da", default="0.1,0.5,1.0",
                        help="comma-separated activation densities for --sweep")
    parser.add_argument("--sweep-dw", default="0.1,0.5,1.0",
                        help="comma-separated weight densities for --sweep")
    parser.add_argument("--sweep-layer", default=None,
                        help="sweep one exact layer name instead of the whole network")
    args = parser.parse_args()

    arch = args.arch or ("act" if args.sweep else "channel")
    if args.sweep and arch != "act":
        parser.error("--sweep requires the new --arch act dataflow")

    if args.net == "alexnet":
        from workloads.alexnet import LAYERS
        net_label = "AlexNet"
    else:
        from workloads.mobilenetv2 import LAYERS
        net_label = "MobileNetV2"

    cfg = HwConfig(ARCH=arch)   # defaults: N_PE=8, M=4, FREQ_HZ=1e9

    if args.sweep:
        try:
            sweep_layers = select_sweep_layers(LAYERS, args.sweep_layer)
            activation_densities = parse_density_list(args.sweep_da)
            weight_densities = parse_density_list(args.sweep_dw)
        except ValueError as exc:
            parser.error(str(exc))

        print(f"goSPA perf-model sparsity sweep -- {net_label}")
        print(f"  HW : N_PE={cfg.N_PE}  M={cfg.M}  "
              f"FREQ={cfg.FREQ_HZ/1e9:.1f} GHz")
        print(f"  Arch : {cfg.ARCH}  (Stage-1 enum: {cfg.STAGE1_ENUM})")
        print(f"  Activation densities : {activation_densities}")
        print(f"  Weight densities     : {weight_densities}")
        print(f"  Seed : {args.seed}")
        print(f"  Baseline : {'disabled' if args.no_baseline else 'one shared dense run'}")
        print()

        sweep = run_sparsity_sweep(
            sweep_layers,
            cfg,
            activation_densities,
            weight_densities,
            seed=args.seed,
            baseline=not args.no_baseline,
        )
        print_sweep_report(sweep, sweep_layers, cfg)
        raise SystemExit(0)

    new_layers = []
    if args.da == 0.5 and args.dw == 0.5:
        new_layers = LAYERS
    else:
        for i, layer in enumerate(LAYERS):
            new_layers.append(replace(layer, d_a=args.da, d_w=args.dw))

    print(f"goSPA perf model -- {net_label}")
    print(f"  HW : N_PE={cfg.N_PE}  M={cfg.M}  FREQ={cfg.FREQ_HZ/1e9:.1f} GHz")
    print(f"  Arch : {cfg.ARCH}  (Stage-1 enum: {cfg.STAGE1_ENUM})")
    print(f"  Sparsity : d_a={new_layers[0].d_a}  d_w={new_layers[0].d_w}  (synthetic phase-1)")
    print(f"  Baseline : {'disabled' if args.no_baseline else 'enabled (doubles runtime)'}")
    print(f"  Layers   : {len(new_layers)}")
    print()

    # new_layers = []
    # for i, layer in enumerate(LAYERS):
    #     new_layers.append(replace(layer, d_a=0.3, d_w=1.0))

    net = run_network(new_layers, cfg, seed=args.seed,
                      baseline=not args.no_baseline)

    print()
    print_report(net, new_layers, cfg)
