"""
apple_input.py -- real verification vector for the GoSPA functional tests.

The accelerator implements MobileNetV2's first conv (F=3, S=2). This module
serves that layer's *real* input + golden so a gospa run can be checked against
PyTorch (instead of only synthetic random vectors):

  load_channels()      -> list of 3 signed [80x80] int activations (R,G,B),
                          ready for fill_activation_csr(). numpy-only, no torch.
  first_conv_kernels() -> int8 kernels[in_ch][out_ch] (FxF). needs torch.
  golden_output()      -> reference first-conv output per output channel, summed
                          over the 3 input channels. needs torch + sw/functional.
  regenerate()         -> rebuild apples_80_centered.npy from apples.jpg.

apples_80_centered.npy is the (3,80,80) zero-point-subtracted activation
(a_uint8 - 114): signed/zero-centered so the signed datapath, zero_act sparsity,
and quantized-conv math are all correct. 80x80 is the smallest input MobileNetV2
still classifies the apple at (~96% conf). Run `python apple_input.py` to
regenerate the .npy and self-check the prediction.
"""
import os
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_NPY  = os.path.join(_HERE, "apples_80_centered.npy")

RES = 80       # input map (RES x RES)
F   = 3        # first-conv kernel
S   = 2        # first-conv stride
ZP  = 114      # MobileNetV2 input QuantStub zero_point (fixed by calibration)


def load_channels(path=_NPY):
    """3 signed (RES x RES) int matrices (R,G,B) for fill_activation_csr()."""
    arr = np.load(path)                                  # (3,RES,RES) int16
    return [arr[c].astype(int).tolist() for c in range(arr.shape[0])]


def first_conv_kernels(n_out=32):
    """int8 first-conv weights as kernels[in_ch][out_ch], each FxF."""
    import sys
    if _HERE not in sys.path:
        sys.path.insert(0, _HERE)
    from mobilenet import get_first_conv
    _, conv0 = get_first_conv()
    iw = conv0.weight().int_repr().numpy()               # (32, 3, F, F) int8
    return [[[[int(v) for v in row] for row in iw[f, c]]
             for f in range(min(n_out, iw.shape[0]))]
            for c in range(iw.shape[1])]


def golden_output(n_out=32):
    """Reference first-conv output per output channel = sum over the 3 input
    channels of conv2d_reference(channel_act, channel_kernel, S)."""
    import sys
    sw = os.path.join(_HERE, "..", "..", "sw")
    if sw not in sys.path:
        sys.path.insert(0, sw)
    import functional as fm
    chans = load_channels()
    kers  = first_conv_kernels(n_out)
    E = (RES - F) // S + 1
    out = [[[0] * E for _ in range(E)] for _ in range(n_out)]
    for c in range(len(chans)):
        for oc in range(n_out):
            part = fm.conv2d_reference(chans[c], kers[c][oc], S)
            for i in range(E):
                for j in range(E):
                    out[oc][i][j] += part[i][j]
    return out


def regenerate(jpg=None, out=_NPY):
    """Rebuild apples_80_centered.npy from apples.jpg; returns (path, label, conf)."""
    import torch
    from PIL import Image
    from torchvision.transforms import functional as TF
    from torchvision.models.quantization import (
        mobilenet_v2, MobileNet_V2_QuantizedWeights)

    jpg = jpg or os.path.join(_HERE, "apples.jpg")
    w = MobileNet_V2_QuantizedWeights.DEFAULT
    model = mobilenet_v2(weights=w, quantize=True).eval()
    tfm = w.transforms()
    zp, scale = int(model.quant.zero_point), float(model.quant.scale)
    assert zp == ZP, f"zero_point changed: {zp} != {ZP}"

    img = Image.open(jpg).convert("RGB")
    x = TF.center_crop(
        TF.resize(img, round(RES / 0.875), interpolation=tfm.interpolation),
        [RES, RES])
    inp = TF.normalize(TF.to_tensor(x), tfm.mean, tfm.std).unsqueeze(0)
    with torch.inference_mode():
        centered = (model.quant(inp).int_repr()[0].to(torch.int32) - zp).to(torch.int16)
        p, idx = model((centered.float() * scale).unsqueeze(0)).softmax(1)[0].max(0)
    np.save(out, centered.numpy().astype(np.int16))
    return out, w.meta["categories"][int(idx)], float(p)


if __name__ == "__main__":
    path, label, conf = regenerate()
    arr = np.array(load_channels())
    print(f"wrote {os.path.basename(path)}  shape=(3,{RES},{RES})  "
          f"range=[{arr.min()},{arr.max()}]  nonzero={int((arr != 0).sum())}/{arr.size}")
    print(f"round-trip prediction: {label} {100 * conf:.1f}%")
