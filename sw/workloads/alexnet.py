"""
AlexNet layer list for the GoSPA perf model.

Architecture: Krizhevsky et al., NIPS 2012.
Input: 227×227×3 (the actual size that satisfies the paper's stride/kernel math;
       commonly cited as 224×224 but 227 is needed for conv1 with F=11, S=4, pad=0
       to produce the 55×55 output map stated in the paper).

Pooling layers are not modeled -- GoSPA targets conv/FC compute, not pooling.

FC layers are modeled as degenerate convolutions:
  FC6: F=H=W=6 -> one spatial output position (E=1), Cin=256, Cout=4096
  FC7/8: F=H=W=1 -> 1x1 conv; standard matrix-vector interpretation

Sparsity values (d_a, d_w) are phase-1 placeholders (sec.6 of PERF_MODEL_PLAN.md).
Replace with PyTorch-captured post-ReLU activation maps and pruned weight masks
in phase 2.  Typical empirical values for AlexNet after ReLU: ~50% activation
zeros; after magnitude pruning: ~50% weight zeros.

Runtime note: conv1 (227×227 activation, Cin=3) generates ~25k non-zeros per
channel and will be the slowest layer to route through the functional front end.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from layer import Layer

# Phase-1 sparsity placeholders -- swap per-layer once real maps are available.
_D_A = 0.5   # post-ReLU activation density
_D_W = 0.5   # pruned weight density

LAYERS = [
    # conv1: 227x227x3 -> 55x55x96  (F=11, S=4, pad=0)
    Layer(H=227, W=227, F=11, S=4, pad=0, Cin=3,    Cout=96,   type="conv", d_a=_D_A, d_w=_D_W, name="conv1"),
    # pool1 (3x3, s=2): 55x55 -> 27x27
    # conv2: 27x27x96  -> 27x27x256 (F=5,  S=1, pad=2)
    Layer(H=27,  W=27,  F=5,  S=1, pad=2, Cin=96,   Cout=256,  type="conv", d_a=_D_A, d_w=_D_W, name="conv2"),
    # pool2 (3x3, s=2): 27x27 -> 13x13
    # conv3: 13x13x256 -> 13x13x384 (F=3,  S=1, pad=1)
    Layer(H=13,  W=13,  F=3,  S=1, pad=1, Cin=256,  Cout=384,  type="conv", d_a=_D_A, d_w=_D_W, name="conv3"),
    # conv4: 13x13x384 -> 13x13x384 (F=3,  S=1, pad=1)
    Layer(H=13,  W=13,  F=3,  S=1, pad=1, Cin=384,  Cout=384,  type="conv", d_a=_D_A, d_w=_D_W, name="conv4"),
    # conv5: 13x13x384 -> 13x13x256 (F=3,  S=1, pad=1)
    Layer(H=13,  W=13,  F=3,  S=1, pad=1, Cin=384,  Cout=256,  type="conv", d_a=_D_A, d_w=_D_W, name="conv5"),
    # pool5 (3x3, s=2): 13x13 -> 6x6
    # fc6:  6x6x256 -> 4096  (modeled as conv F=6; E=(6-6)//1+1=1)
    Layer(H=6,   W=6,   F=6,  S=1, pad=0, Cin=256,  Cout=4096, type="fc",   d_a=_D_A, d_w=_D_W, name="fc6"),
    # fc7:  4096 -> 4096     (1x1 conv)
    Layer(H=1,   W=1,   F=1,  S=1, pad=0, Cin=4096, Cout=4096, type="fc",   d_a=_D_A, d_w=_D_W, name="fc7"),
    # fc8:  4096 -> 1000     (1x1 conv)
    Layer(H=1,   W=1,   F=1,  S=1, pad=0, Cin=4096, Cout=1000, type="fc",   d_a=_D_A, d_w=_D_W, name="fc8"),
]
