"""
alexnet_layers.py -- the five conv layers of pretrained torchvision AlexNet
on the team's apple image, quantized for the goSPA integer datapath.

For conv layer i (features[0,3,6,8,10]) we capture the layer's REAL input
activations from a forward pass on apples.jpg (224x224 pipeline), then
quantize symmetrically per tensor:
    acts   : round(x * 127 / max|x|)   (conv1 input has negatives; later
                                        inputs are post-ReLU >= 0)
    weights: round(w * 127 / max|w|)   int8 range
Golden for the RTL is integer conv on these tensors (self-consistent, same
convention as the MobileNet flow). Cached as alexnet_layer<i>_apple.npz.

    get_layer(i) -> dict(acts, weights, stride, pad, act_scale, wgt_scale)
"""
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
CONV_IDX = [0, 3, 6, 8, 10]          # features[] indices of the 5 convs


def _npz(i):
    return os.path.join(_HERE, f"alexnet_layer{i}_apple.npz")


def _quant(x):
    m = float(np.abs(x).max())
    s = (m / 127.0) if m > 0 else 1.0
    return np.round(x / s).astype(np.int32), s


def _extract_all():
    import torch
    from torchvision import models
    from PIL import Image

    w = models.AlexNet_Weights.IMAGENET1K_V1
    model = models.alexnet(weights=w).eval()
    img = Image.open(os.path.join(_HERE, "apples.jpg")).convert("RGB")
    x = w.transforms()(img).unsqueeze(0)

    caps = {}
    hooks = []
    for li, fi in enumerate(CONV_IDX, start=1):
        def mk(li):
            def fn(module, inputs, output):
                caps[li] = (inputs[0][0].detach().numpy(),
                            module.weight.detach().numpy(),
                            int(module.stride[0]), int(module.padding[0]))
            return fn
        hooks.append(model.features[fi].register_forward_hook(mk(li)))
    with torch.inference_mode():
        model(x)
    for h in hooks:
        h.remove()

    for li, (act, wgt, stride, pad) in caps.items():
        qa, sa = _quant(act)
        qw, sw = _quant(wgt)
        np.savez_compressed(_npz(li), acts=qa.astype(np.int16),
                            weights=qw.astype(np.int16),
                            stride=np.int32(stride), pad=np.int32(pad),
                            act_scale=np.float64(sa), wgt_scale=np.float64(sw))


def get_layer(i):
    if not os.path.exists(_npz(i)):
        _extract_all()
    z = np.load(_npz(i))
    return dict(acts=z["acts"], weights=z["weights"],
                stride=int(z["stride"]), pad=int(z["pad"]),
                act_scale=float(z["act_scale"]),
                wgt_scale=float(z["wgt_scale"]))


def golden_conv(acts_padded, weights, stride):
    """Integer valid conv via numpy sliding windows.
    acts_padded: (Cin, Hp, Hp) int; weights: (Cout, Cin, F, F) int.
    Returns (Cout, E, E) int64."""
    cin, hp, _ = acts_padded.shape
    cout, _, f, _ = weights.shape
    e = (hp - f) // stride + 1
    win = np.lib.stride_tricks.sliding_window_view(
        acts_padded, (f, f), axis=(1, 2))[:, ::stride, ::stride]  # (Cin,E,E,F,F)
    out = np.tensordot(weights.astype(np.int64),
                       win.astype(np.int64), axes=([1, 2, 3], [0, 3, 4]))
    return out                                       # (Cout, E, E)


if __name__ == "__main__":
    for i in range(1, 6):
        L = get_layer(i)
        a, w = L["acts"], L["weights"]
        da = float((a != 0).mean())
        dw = float((w != 0).mean())
        print(f"conv{i}: act {a.shape} da={da:.3f}  wgt {w.shape} "
              f"dw={dw:.3f}  S={L['stride']} P={L['pad']}")
