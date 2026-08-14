#!/usr/bin/env python3
"""
plot_sweep.py -- poster figures of LUT usage vs sweep parameters.
GoSPA Project -- Team 19, ECE 720 (Spring 2026)

Reads sweep_results.csv (produced by run_sweep.sh) and writes, next to it:
  lut_vs_N_PE.png / .pdf, lut_vs_N_MULTS.png / .pdf, lut_vs_FIFO_D.png / .pdf,
  and a combined lut_sweep.png / .pdf.

Each panel holds the other two parameters at the baseline (N_PE=8, N_MULTS=4,
FIFO_D=64). Single series (LUT) -> no legend; the title names it.

    python3 plot_sweep.py [path/to/sweep_results.csv]
"""
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# --- config ------------------------------------------------------------------
BASE_NPE, BASE_NMULTS, BASE_FIFO = 8, 4, 64
BLUE = "#0072B2"    # Okabe-Ito blue -- colourblind-safe, high contrast
INK, MUTED, GRID = "#1a1a1a", "#555555", "#dcdcdc"

SWEEPS = [   # (variable, [(fixed_key, fixed_val), ...], axis label)
    ("N_PE",    [("N_MULTS", BASE_NMULTS), ("FIFO_D", BASE_FIFO)],   "Number of PEs  (N_PE)"),
    ("N_MULTS", [("N_PE", BASE_NPE),        ("FIFO_D", BASE_FIFO)],   "Multiplier lanes per PE  (N_MULTS)"),
    ("FIFO_D",  [("N_PE", BASE_NPE),        ("N_MULTS", BASE_NMULTS)], "FIFO depth  (FIFO-A & FIFO-B)"),
]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 12,
    "axes.edgecolor": MUTED, "axes.linewidth": 1.0,
    "text.color": INK, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    # transparent everywhere -> the figure drops onto any poster surface
    "figure.facecolor": "none", "axes.facecolor": "none", "savefig.facecolor": "none",
})


def load(path):
    with open(path) as f:
        return [{k: int(v) for k, v in row.items()} for row in csv.DictReader(f)]


def series(rows, var, fixed):
    (ka, va), (kb, vb) = fixed
    pts = sorted({(r[var], r["LUT"]) for r in rows if r[ka] == va and r[kb] == vb})
    return [p[0] for p in pts], [p[1] for p in pts]


def draw(ax, xs, ys, xlabel, title):
    ax.plot(xs, ys, "-o", color=BLUE, lw=2.4, ms=10, mfc=BLUE, mec="white", mew=1.4, zorder=3)
    for x, y in zip(xs, ys):
        ax.annotate(f"{y:,}", (x, y), textcoords="offset points", xytext=(0, 11),
                    ha="center", fontsize=10.5, color=INK)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12, color=INK)
    ax.set_xlabel(xlabel, fontsize=12.5, labelpad=8)
    ax.set_xticks(xs)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.grid(axis="y", color=GRID, lw=0.9, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.margins(x=0.14, y=0.22)
    ax.tick_params(labelsize=11)


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "sweep_results.csv")
    outdir = os.path.dirname(os.path.abspath(csv_path))
    rows = load(csv_path)

    # combined 1x3 (shared y so panels are comparable)
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), sharey=True)
    for ax, (var, fixed, xlabel) in zip(axes, SWEEPS):
        xs, ys = series(rows, var, fixed)
        draw(ax, xs, ys, xlabel, f"LUT vs {var}")
    axes[0].set_ylabel("LUT usage", fontsize=13, labelpad=8)
    fig.suptitle("GoSPA resource scaling — LUT usage (Kria KV260)",
                 fontsize=16.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "lut_sweep.png"), dpi=220, bbox_inches="tight", transparent=True)
    fig.savefig(os.path.join(outdir, "lut_sweep.pdf"), bbox_inches="tight", transparent=True)
    plt.close(fig)

    # individual panels
    for var, fixed, xlabel in SWEEPS:
        xs, ys = series(rows, var, fixed)
        f, ax = plt.subplots(figsize=(6.2, 4.8))
        draw(ax, xs, ys, xlabel, f"LUT usage vs {var}")
        ax.set_ylabel("LUT usage", fontsize=13, labelpad=8)
        f.tight_layout()
        f.savefig(os.path.join(outdir, f"lut_vs_{var}.png"), dpi=220, bbox_inches="tight", transparent=True)
        f.savefig(os.path.join(outdir, f"lut_vs_{var}.pdf"), bbox_inches="tight", transparent=True)
        plt.close(f)

    print(f"wrote lut_sweep.png/.pdf and lut_vs_*.png/.pdf to {outdir}")


if __name__ == "__main__":
    main()
