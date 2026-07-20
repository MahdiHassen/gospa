"""
MobileNetV2 layer list for the GoSPA perf model.

Architecture: Sandler et al., CVPR 2018 (Table 2), width_mult=1.0, input 224x224x3.

Each inverted-residual "bottleneck" block is (expansion t, output channels c,
repeat count n, first-repeat stride s; remaining repeats use stride 1) and
lowers to up to three convs:
  1. 1x1 pointwise "expand"    Cin -> Cin*t      (type "1x1"; skipped when t==1,
                                                   matching the reference impl)
  2. 3x3 depthwise             per-channel, stride s, pad=1   (type "dw")
  3. 1x1 pointwise "project"   Cin*t -> c, linear (type "1x1")

Average pooling and the residual (skip) adds are not modeled -- GoSPA targets
conv/FC compute, same convention as workloads/alexnet.py. The final 1x1 conv
(320 -> 1280) and classifier (1280 -> 1000) are modeled as in alexnet.py's
fc7/fc8: F=H=W=1, treated as a plain matrix-vector op on the pooled 1x1 map.

Sparsity values (d_a, d_w) are phase-1 placeholders (sec.6 of PERF_MODEL_PLAN.md).
Replace with PyTorch-captured post-ReLU activation maps and pruned weight masks
in phase 2.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from layer import Layer

# Phase-1 sparsity placeholders -- swap per-layer once real maps are available.
_D_A = 0.5   # post-ReLU activation density
_D_W = 0.5   # pruned weight density

# (expansion t, output channels c, repeat count n, first-repeat stride s)
_BOTTLENECKS = [
    (1,  16, 1, 1),
    (6,  24, 2, 2),
    (6,  32, 3, 2),
    (6,  64, 4, 2),
    (6,  96, 3, 1),
    (6, 160, 3, 2),
    (6, 320, 1, 1),
]


def _build_layers():
    layers = []

    # conv1: 224x224x3 -> 112x112x32 (F=3, S=2, pad=1)
    H, Cin = 224, 3
    layers.append(Layer(H=H, W=H, F=3, S=2, pad=1, Cin=Cin, Cout=32,
                         type="conv", d_a=_D_A, d_w=_D_W, name="conv1"))
    H = (H + 2 * 1 - 3) // 2 + 1
    Cin = 32

    for group, (t, c, n, s) in enumerate(_BOTTLENECKS):
        for rep in range(n):
            stride = s if rep == 0 else 1
            hidden = Cin * t
            tag = f"bneck{group}_{rep}"

            if t != 1:
                layers.append(Layer(H=H, W=H, F=1, S=1, pad=0, Cin=Cin, Cout=hidden,
                                     type="1x1", d_a=_D_A, d_w=_D_W, name=f"{tag}_expand"))

            layers.append(Layer(H=H, W=H, F=3, S=stride, pad=1, Cin=hidden, Cout=hidden,
                                 type="dw", d_a=_D_A, d_w=_D_W, name=f"{tag}_dw"))
            H = (H + 2 * 1 - 3) // stride + 1

            layers.append(Layer(H=H, W=H, F=1, S=1, pad=0, Cin=hidden, Cout=c,
                                 type="1x1", d_a=_D_A, d_w=_D_W, name=f"{tag}_project"))
            Cin = c

    # conv2 (1x1): 7x7x320 -> 7x7x1280
    layers.append(Layer(H=H, W=H, F=1, S=1, pad=0, Cin=Cin, Cout=1280,
                         type="1x1", d_a=_D_A, d_w=_D_W, name="conv2"))

    # classifier: 1280 -> 1000, applied on the pooled 1x1 map (1x1 conv)
    layers.append(Layer(H=1, W=1, F=1, S=1, pad=0, Cin=1280, Cout=1000,
                         type="fc", d_a=_D_A, d_w=_D_W, name="classifier"))

    return layers


LAYERS = _build_layers()
