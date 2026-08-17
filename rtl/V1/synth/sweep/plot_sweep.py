#!/usr/bin/env python3
"""
plot_sweep.py -- LUT/FF resource scaling figures for the IEEE report.
GoSPA Project -- Team 19, ECE 720 (Spring 2026)

Reads sweep_results.csv (produced by run_sweep.sh) and writes, next to it, a
1x3 panel figure of LUT and FF usage against each swept parameter:

    resource_sweep.pdf / .png   (+ per-panel res_vs_<param>.pdf / .png)

Each panel holds the other two parameters at the baseline (N_PE=8, N_MULTS=4,
FIFO_D=64). Sized for a two-column IEEE page: use \\begin{figure*} and
\\includegraphics[width=\\textwidth]{resource_sweep.pdf}.

    python3 plot_sweep.py [path/to/sweep_results.csv] [--version V1|V2]

The version only changes the axis wording: N_MULTS is the number of kernels
held per PE in V1, but the beat width (activations consumed per beat) in V2.
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# --- config ------------------------------------------------------------------
BASE_NPE, BASE_NMULTS, BASE_FIFO = 8, 4, 64

# Okabe-Ito blue / vermillion: a published colour-blind-safe pair, kept far apart
# in both hue and lightness. Line style and marker repeat the distinction, so the
# series stay separable in greyscale print and under any CVD type.
SERIES = [
    ("LUT", "#0072B2", "-",  "o"),
    ("FF",  "#D55E00", "--", "s"),
]
INK, MUTED, GRID = "#1a1a1a", "#555555", "#d9d9d9"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.8,
    "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white",
})


def sweeps(version):
    """(csv column, fixed baseline pairs, x-axis label) per panel.

    N_MULTS is the same CSV column in both versions but means different things:
    in V2 it is the beat width B (activations consumed per beat, and with
    S2_BEATS=1 also the FIFO-A -> FIFO-B drain width), whereas in V1 it is the
    number of kernels a PE holds. The FIFO axis moves FIFO-A and FIFO-B together.
    """
    if version == "V1":
        sym, mults_label = "$M$", "Kernels per PE  $M$"
    else:
        sym, mults_label = "$B$", "Beat width  $B$"
    return [
        ("N_PE",    [("N_MULTS", BASE_NMULTS), ("FIFO_D", BASE_FIFO)],
         f"PE count  ({sym}={BASE_NMULTS}, FIFO={BASE_FIFO})"),
        ("N_MULTS", [("N_PE", BASE_NPE), ("FIFO_D", BASE_FIFO)],
         f"{mults_label}  ({BASE_NPE} PEs, FIFO={BASE_FIFO})"),
        ("FIFO_D",  [("N_PE", BASE_NPE), ("N_MULTS", BASE_NMULTS)],
         f"FIFO depth, A and B  ({BASE_NPE} PEs, {sym}={BASE_NMULTS})"),
    ]


def load(path):
    with open(path) as f:
        return [{k: int(v) for k, v in row.items()} for row in csv.DictReader(f)]


def series(rows, var, fixed, metric):
    (ka, va), (kb, vb) = fixed
    pts = sorted({(r[var], r[metric]) for r in rows if r[ka] == va and r[kb] == vb})
    return [p[0] for p in pts], [p[1] for p in pts]


def draw(ax, rows, var, fixed, xlabel):
    xs_ref = None
    for metric, colour, style, marker in SERIES:
        xs, ys = series(rows, var, fixed, metric)
        xs_ref = xs
        # Categorical x: the swept values are geometric (2,4,8,16), so equal
        # spacing keeps the panels readable instead of crowding the low end.
        pos = range(len(xs))
        ax.plot(pos, [y / 1000.0 for y in ys], style, color=colour, lw=1.6,
                marker=marker, ms=5, mfc=colour, mec="white", mew=0.8,
                label=metric, zorder=3)
    ax.set_xticks(range(len(xs_ref)))
    ax.set_xticklabels([str(x) for x in xs_ref])
    ax.set_xlabel(xlabel, fontsize=9, labelpad=5)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.margins(x=0.10, y=0.18)
    ax.tick_params(labelsize=8.5, length=3)


def main():
    ap = argparse.ArgumentParser(description="LUT/FF resource scaling figures.")
    ap.add_argument("csv", nargs="?", help="sweep_results.csv (default: next to this script)")
    ap.add_argument("--version", choices=["V1", "V2"], default="V2",
                    help="architecture, sets the N_MULTS axis wording")
    opts = ap.parse_args()

    version = opts.version
    csv_path = opts.csv or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "sweep_results.csv")
    outdir = os.path.dirname(os.path.abspath(csv_path))
    rows = load(csv_path)
    panels = sweeps(version)

    # Two-column IEEE figure*: full text width, short enough not to eat a page.
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.35), sharey=True)
    for ax, (var, fixed, xlabel) in zip(axes, panels):
        draw(ax, rows, var, fixed, xlabel)
    axes[0].set_ylabel("Resources (thousands)", fontsize=9, labelpad=5)
    # One legend for the whole figure -- two series, so a legend is mandatory.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.995, 1.02),
               ncol=2, frameon=False, fontsize=9, handlelength=2.4)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(outdir, "resource_sweep.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(outdir, "resource_sweep.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Per-panel versions (single-column width) in case they are wanted separately.
    for var, fixed, xlabel in panels:
        f, ax = plt.subplots(figsize=(3.3, 2.5))
        draw(ax, rows, var, fixed, xlabel)
        ax.set_ylabel("Resources (thousands)", fontsize=9, labelpad=5)
        ax.legend(frameon=False, fontsize=8.5, handlelength=2.4)
        f.tight_layout()
        f.savefig(os.path.join(outdir, f"res_vs_{var}.pdf"), bbox_inches="tight")
        f.savefig(os.path.join(outdir, f"res_vs_{var}.png"), dpi=300, bbox_inches="tight")
        plt.close(f)

    print(f"[{version}] wrote resource_sweep.pdf/.png and res_vs_*.pdf/.png to {outdir}")


if __name__ == "__main__":
    main()
