#!/usr/bin/env python3
"""V1 (channel, union-WSP) vs V2 (act, one kernel/PE) on the AlexNet conv
stack, per layer, at the RTL-measured per-layer densities (torchvision
AlexNet, apple-image activations -- same tensors as testing/gospa's
test_gospa_alexnet.py). N_PE=8 x M=4, native router widths, 100 MHz.

Usage (from sw/):  python alexnet_arch_compare.py
"""
import sys

from config import HwConfig
from layer import Layer
from sim import run_network

# (name, H, F, S, pad, Cin, Cout, da, dw) -- da/dw measured from the
# integer-quantized apple-image tensors used by the RTL testbench.
CONVS = [
    ("conv1", 224, 11, 4, 2,   3,  64, 0.996, 0.948),
    ("conv2",  27,  5, 1, 2,  64, 192, 0.513, 0.762),
    ("conv3",  13,  3, 1, 1, 192, 384, 0.539, 0.909),
    ("conv4",  13,  3, 1, 1, 384, 256, 0.145, 0.953),
    ("conv5",  13,  3, 1, 1, 256, 256, 0.163, 0.972),
]

FREQ = 100e6


def run(arch):
    out = []
    for (name, h, f, s, pad, cin, cout, da, dw) in CONVS:
        lay = Layer(H=h, W=h, F=f, S=s, pad=pad, Cin=cin, Cout=cout,
                    type="conv", d_a=da, d_w=dw, name=name)
        cfg = HwConfig(ARCH=arch, N_PE=8, M=4, FREQ_HZ=FREQ)
        net = run_network([lay], cfg, seed=0, baseline=False, verbose=False)
        ls = net.per_layer[0]
        out.append((name, ls.layer_cycles, 100.0 * ls.array_util))
    return out


def main():
    v1 = run("channel")
    v2 = run("act")
    print(f"{'layer':>6} {'V1 cycles':>12} {'V1 util%':>9} "
          f"{'V2 cycles':>12} {'V2 util%':>9} {'V2/V1 speedup':>14}")
    t1 = t2 = 0
    for (n, c1, u1), (_, c2, u2) in zip(v1, v2):
        t1 += c1
        t2 += c2
        print(f"{n:>6} {c1:>12,} {u1:>9.1f} {c2:>12,} {u2:>9.1f} "
              f"{c1 / c2:>14.2f}")
    print(f"{'TOTAL':>6} {t1:>12,} {'':>9} {t2:>12,} {'':>9} {t1/t2:>14.2f}")
    print(f"\nconv-stack fps @100 MHz:  V1 = {FREQ / t1:6.1f}   "
          f"V2 = {FREQ / t2:6.1f}")


if __name__ == "__main__":
    main()
