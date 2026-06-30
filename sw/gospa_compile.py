"""
gospa_compile.py -- a mini "compiler" / scheduler that lowers an arbitrary conv
layer (H_img x W_img x C_in  ->  C_out) onto a FIXED GoSPA instance.

The hardware is built for one H x H tile (so E, N_CID=E*E, N_PID=F*F are fixed at
synthesis, and N_PE*N_MULTS output channels run in parallel). Anything bigger is
handled in software here: tile the input into H x H windows (with the F-S conv
halo so tile outputs are exact), group the output channels, and emit the loop
nest of HW invocations -- input channels innermost so the PE banks accumulate
across C_in before a single drain.

compile_layer() returns the schedule (a flat list of Pass records you could
lower to register writes) + a feasibility report. run_schedule() executes the
schedule functionally (integer MACs, exactly what GoSPA computes) so it can be
checked against a full reference conv -- the tiling/scheduling is verified exact.

    python gospa_compile.py        # self-checks several arbitrary shapes
"""
from dataclasses import dataclass, field
import numpy as np


# ----------------------------------------------------------------------------
# Fixed hardware model
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class HW:
    H: int                 # input tile (H x H), fixed at synthesis
    F: int = 3             # kernel
    S: int = 2             # stride
    N_PE: int = 8
    N_MULTS: int = 4
    N_NZ_MAX: int = 1024   # entry SRAM depth (nonzeros / tile)
    N_ROWS: int = 32       # row-ptr depth (input rows / tile)
    FIFO_D: int = 2048     # FIFO-A depth (Stage-1 entries / PID / tile)

    @property
    def E(self):  return (self.H - self.F) // self.S + 1     # output / tile edge
    @property
    def N_CID(self): return self.E * self.E                  # accumulator banks/lane
    @property
    def N_CH(self):  return self.N_PE * self.N_MULTS         # out-channels / pass
    @property
    def G(self):  return (self.F + self.S - 1) // self.S     # ceil(F/S), idgen fanout
    @property
    def tile_stride(self): return self.E * self.S            # input rows between tiles
    @property
    def halo(self): return self.F - self.S                   # tile overlap


@dataclass
class Pass:
    idx: int
    oc_group: int          # output-channel group
    oc_base: int           # first output channel of the group
    n_oc: int              # channels in this group (<= N_CH)
    in_ch: int             # input channel for this scan
    in_r0: int; in_c0: int # top-left of the H x H input window (zero-padded at edges)
    out_r0: int; out_c0: int  # global output offset of this tile
    out_h: int; out_c: int    # valid output rows/cols kept (<= E)
    reload_w: bool         # weights differ from previous pass -> must reload
    drain: bool            # last input channel of the tile -> drain after


# ----------------------------------------------------------------------------
# Compiler
# ----------------------------------------------------------------------------
def compile_layer(hw: HW, C_in: int, C_out: int, H_img: int, W_img: int):
    assert H_img >= hw.F and W_img >= hw.F, "image smaller than the kernel"
    Eh = (H_img - hw.F) // hw.S + 1            # global output size
    Ew = (W_img - hw.F) // hw.S + 1
    n_tr = -(-Eh // hw.E)                      # ceil: output tiles per axis
    n_tc = -(-Ew // hw.E)
    n_g  = -(-C_out // hw.N_CH)               # output-channel groups

    sched, idx = [], 0
    last_w = None
    for g in range(n_g):
        oc_base = g * hw.N_CH
        n_oc = min(hw.N_CH, C_out - oc_base)
        for tr in range(n_tr):
            for tc in range(n_tc):
                out_r0, out_c0 = tr * hw.E, tc * hw.E
                out_h = min(hw.E, Eh - out_r0)
                out_c = min(hw.E, Ew - out_c0)
                for ic in range(C_in):
                    w_id = (g, ic)
                    sched.append(Pass(
                        idx=idx, oc_group=g, oc_base=oc_base, n_oc=n_oc, in_ch=ic,
                        in_r0=tr * hw.tile_stride, in_c0=tc * hw.tile_stride,
                        out_r0=out_r0, out_c0=out_c0, out_h=out_h, out_c=out_c,
                        reload_w=(w_id != last_w), drain=(ic == C_in - 1)))
                    last_w = w_id
                    idx += 1

    # ---- feasibility against the fixed SRAM/FIFO sizes (worst-case dense tile) ----
    worst_nz = hw.H * hw.H
    fifoa_worst = -(-(hw.G * hw.G * worst_nz) // (hw.F * hw.F))   # per-PID entries
    report = {
        "out_size": (Eh, Ew), "spatial_tiles": n_tr * n_tc, "oc_groups": n_g,
        "passes": len(sched), "scans": len(sched),
        "drains": n_g * n_tr * n_tc, "weight_loads": sum(p.reload_w for p in sched),
        "fits_N_ROWS": hw.H <= hw.N_ROWS,
        "fits_N_NZ_MAX(dense)": worst_nz <= hw.N_NZ_MAX,
        "fits_FIFO_D(dense)": fifoa_worst <= hw.FIFO_D,
        "worst_nz_per_tile": worst_nz, "worst_fifoa_per_pid": fifoa_worst,
    }
    return sched, report


# ----------------------------------------------------------------------------
# Functional execution (mirrors what GoSPA computes: integer MACs)
# ----------------------------------------------------------------------------
def _conv2d_int(act, ker, S):
    F = ker.shape[0]
    win = np.lib.stride_tricks.sliding_window_view(act, (F, F))[::S, ::S]
    return np.tensordot(win, ker, axes=([2, 3], [0, 1])).astype(np.int64)


def run_schedule(hw, sched, img, W):
    """img: (C_in,H_img,W_img) int ; W: (C_out,C_in,F,F) int -> (C_out,Eh,Ew)."""
    C_out = W.shape[0]
    Eh = (img.shape[1] - hw.F) // hw.S + 1
    Ew = (img.shape[2] - hw.F) // hw.S + 1
    out = np.zeros((C_out, Eh, Ew), np.int64)
    acc = None                                  # current tile's PE banks (n_oc,E,E)
    for p in sched:
        if acc is None:
            acc = np.zeros((p.n_oc, hw.E, hw.E), np.int64)
        # extract the H x H input window for this channel, zero-padded at edges
        tile = np.zeros((hw.H, hw.H), np.int64)
        r1, c1 = min(p.in_r0 + hw.H, img.shape[1]), min(p.in_c0 + hw.H, img.shape[2])
        tile[:r1 - p.in_r0, :c1 - p.in_c0] = img[p.in_ch, p.in_r0:r1, p.in_c0:c1]
        for k in range(p.n_oc):                 # one lane per output channel
            acc[k] += _conv2d_int(tile, W[p.oc_base + k, p.in_ch], hw.S)
        if p.drain:
            out[p.oc_base:p.oc_base + p.n_oc,
                p.out_r0:p.out_r0 + p.out_h,
                p.out_c0:p.out_c0 + p.out_c] = acc[:, :p.out_h, :p.out_c]
            acc = None
    return out


def reference_conv(img, W, S):
    C_out, C_in = W.shape[0], W.shape[1]
    Eh = (img.shape[1] - W.shape[2]) // S + 1
    Ew = (img.shape[2] - W.shape[3]) // S + 1
    out = np.zeros((C_out, Eh, Ew), np.int64)
    for oc in range(C_out):
        for ic in range(C_in):
            out[oc] += _conv2d_int(img[ic].astype(np.int64), W[oc, ic], S)
    return out


# ----------------------------------------------------------------------------
# Self-check: arbitrary sizes / channel counts on a fixed HW
# ----------------------------------------------------------------------------
def _demo(hw, C_in, C_out, H_img, W_img, rng, label):
    img = rng.integers(-128, 128, size=(C_in, H_img, W_img), dtype=np.int64)
    img[rng.random((C_in, H_img, W_img)) < 0.5] = 0           # ~50% sparse
    W = rng.integers(-8, 8, size=(C_out, C_in, hw.F, hw.F), dtype=np.int64)

    sched, rep = compile_layer(hw, C_in, C_out, H_img, W_img)
    got = run_schedule(hw, sched, img, W)
    ref = reference_conv(img, W, hw.S)
    ok = np.array_equal(got, ref)
    print(f"[{label}] in {C_in}x{H_img}x{W_img} -> out {C_out}x{rep['out_size'][0]}x{rep['out_size'][1]} "
          f"| tiles={rep['spatial_tiles']} oc_groups={rep['oc_groups']} "
          f"passes={rep['passes']} drains={rep['drains']} wloads={rep['weight_loads']} "
          f"| fits[rows={rep['fits_N_ROWS']},nz={rep['fits_N_NZ_MAX(dense)']},"
          f"fifo={rep['fits_FIFO_D(dense)']}] | verify={'EXACT' if ok else 'MISMATCH'}")
    assert ok, "tiled schedule != full conv"


if __name__ == "__main__":
    rng = np.random.default_rng(0xC0FFEE)
    # Fixed GoSPA: H=16 tile, 4x4=16 output channels/pass, modest SRAM.
    hw = HW(H=16, F=3, S=2, N_PE=4, N_MULTS=4, N_NZ_MAX=256, N_ROWS=16, FIFO_D=512)
    print(f"fixed HW: H={hw.H} F={hw.F} S={hw.S} -> E={hw.E} N_CID={hw.N_CID} "
          f"N_CH={hw.N_CH}/pass  SRAM(nz={hw.N_NZ_MAX},rows={hw.N_ROWS},fifo={hw.FIFO_D})\n")
    _demo(hw, C_in=3,  C_out=32, H_img=80, W_img=80, rng=rng, label="apple-shape  ")
    _demo(hw, C_in=1,  C_out=8,  H_img=28, W_img=28, rng=rng, label="mnist-ish    ")
    _demo(hw, C_in=5,  C_out=10, H_img=45, W_img=60, rng=rng, label="odd HxW, 5ch ")
    _demo(hw, C_in=64, C_out=128,H_img=33, W_img=33, rng=rng, label="deep layer   ")
    print("\nall schedules verified exact vs. full conv.")
