"""
mobilenet_sparsity.py -- per-layer weight/activation sparsity of quantized
MobileNetV2 on the team's apple image.

Answers "what does sparsity typically look like on a real workload" so the PE
utilization sweep (testing/pe/test_pe.py::test_perf) can be read against real
(act_density, wgt_density) operating points instead of only synthetic grids.

Zeros are counted the way the goSPA hardware sees them:
  - activation zero  = quantized value == zero_point (dequantizes to exactly 0;
    what apu_stage1's zero_act filter would drop)
  - weight zero      = int_repr() == 0 (per-channel symmetric, zp = 0)

Runs the full INT8 model twice: on the 80x80 centered apple tensor (the HW
verification vector, apples_80_centered.npy) and on the standard 224x224
pipeline from apples.jpg. Writes one CSV per run
(mobilenet_sparsity_<tag>.csv) and prints a summary table.

Usage:  python mobilenet_sparsity.py
"""
import os
import sys

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF
from torchvision.models.quantization import (
    mobilenet_v2,
    MobileNet_V2_QuantizedWeights,
)

_HERE = os.path.dirname(os.path.abspath(__file__))


def _is_qconv_or_qlinear(m):
    """Duck-type quantized Conv2d / ConvReLU2d / Linear: packed weight()
    accessor + output scale/zero_point."""
    return (hasattr(m, "weight") and callable(m.weight)
            and hasattr(m, "scale") and hasattr(m, "zero_point"))


def _weight_stats(m):
    w = m.weight().int_repr()
    nz = int((w != 0).sum())
    return nz, w.numel(), tuple(w.shape)


def _tensor_density(t):
    """Fraction of elements that are NOT hardware-zero."""
    if t.is_quantized:
        zp = t.q_zero_point()
        return float((t.int_repr() != zp).float().mean()), t.int_repr().numel()
    return float((t != 0).float().mean()), t.numel()


def profile(model, x, tag):
    rows = []

    def hook(name, mod):
        def fn(module, inputs, output):
            t = inputs[0]
            act_d, n_in = _tensor_density(t)
            w_nz, w_n, w_shape = _weight_stats(module)
            if isinstance(module, torch.nn.modules.conv._ConvNd) or len(w_shape) == 4:
                cout, cin_g, kh, kw = w_shape
                groups = getattr(module, "groups", 1)
                stride = getattr(module, "stride", (1,))[0]
                oh, ow = output.shape[-2], output.shape[-1]
                macs = oh * ow * cout * cin_g * kh * kw   # cin_g = Cin/groups
                if groups == 1 and (kh, kw) == (1, 1):
                    ltype = "pw1x1"
                elif groups == cout and cin_g == 1:
                    ltype = f"dw{kh}x{kw}s{stride}"
                else:
                    ltype = f"conv{kh}x{kw}s{stride}"
            else:
                cout, cin_g = w_shape
                kh = kw = 1
                groups, stride = 1, 1
                macs = cout * cin_g
                ltype = "linear"
            rows.append(dict(
                name=name, type=ltype, cout=cout, cin_g=cin_g, k=kh,
                stride=stride, groups=groups,
                in_hw=f"{t.shape[-2]}x{t.shape[-1]}" if t.dim() == 4 else "-",
                act_density=act_d, n_act=n_in,
                wgt_density=w_nz / w_n, n_wgt=w_n, macs=macs,
            ))
        return fn

    handles = [m.register_forward_hook(hook(n, m))
               for n, m in model.named_modules() if _is_qconv_or_qlinear(m)]
    with torch.inference_mode():
        out = model(x)
    for h in handles:
        h.remove()

    p, idx = out.softmax(1)[0].max(0)
    cats = MobileNet_V2_QuantizedWeights.DEFAULT.meta["categories"]
    print(f"\n=== {tag}: prediction '{cats[int(idx)]}' {100 * float(p):.1f}%  "
          f"({len(rows)} quantized conv/linear layers) ===")

    hdr = (f"{'#':>3} {'layer':<34} {'type':<9} {'in HxW':>8} "
           f"{'Cin/g':>6} {'Cout':>5} {'actDens%':>9} {'wgtDens%':>9} {'MMACs':>8}")
    print(hdr)
    print("-" * len(hdr))
    for i, r in enumerate(rows):
        print(f"{i:>3} {r['name']:<34} {r['type']:<9} {r['in_hw']:>8} "
              f"{r['cin_g']:>6} {r['cout']:>5} {100 * r['act_density']:>9.1f} "
              f"{100 * r['wgt_density']:>9.1f} {r['macs'] / 1e6:>8.2f}")
    print("-" * len(hdr))

    tot_macs = sum(r["macs"] for r in rows)
    aw_act = sum(r["act_density"] * r["macs"] for r in rows) / tot_macs
    aw_wgt = sum(r["wgt_density"] * r["macs"] for r in rows) / tot_macs
    print(f"MAC-weighted mean density: activations {100 * aw_act:.1f}%   "
          f"weights {100 * aw_wgt:.1f}%   (total {tot_macs / 1e6:.1f} MMACs)")
    for lt in ("conv3x3s2", "dw3x3s1", "dw3x3s2", "pw1x1", "linear"):
        sel = [r for r in rows if r["type"] == lt]
        if not sel:
            continue
        m = sum(r["macs"] for r in sel)
        a = sum(r["act_density"] * r["macs"] for r in sel) / m
        w = sum(r["wgt_density"] * r["macs"] for r in sel) / m
        print(f"  {lt:<9}: {len(sel):>2} layers, {100 * m / tot_macs:>5.1f}% of MACs, "
              f"act {100 * a:>5.1f}%  wgt {100 * w:>5.1f}%")

    csv = os.path.join(_HERE, f"mobilenet_sparsity_{tag}.csv")
    with open(csv, "w") as fh:
        fh.write("idx,name,type,in_hw,cin_per_group,cout,kernel,stride,groups,"
                 "act_density,wgt_density,n_act,n_wgt,macs\n")
        for i, r in enumerate(rows):
            fh.write(f"{i},{r['name']},{r['type']},{r['in_hw']},{r['cin_g']},"
                     f"{r['cout']},{r['k']},{r['stride']},{r['groups']},"
                     f"{r['act_density']:.6f},{r['wgt_density']:.6f},"
                     f"{r['n_act']},{r['n_wgt']},{r['macs']}\n")
    print(f"wrote {csv}")
    return rows


def apple_80_input(model):
    """The HW verification vector: apples_80_centered.npy -> model input.
    Same reconstruction as apple_input.regenerate()'s round-trip check."""
    arr = np.load(os.path.join(_HERE, "apples_80_centered.npy"))   # int16, centered
    scale = float(model.quant.scale)
    return (torch.from_numpy(arr.astype(np.float32)) * scale).unsqueeze(0)


def apple_224_input():
    """Standard eval pipeline (resize 256 / center-crop 224) on apples.jpg."""
    w = MobileNet_V2_QuantizedWeights.DEFAULT
    tfm = w.transforms()
    img = Image.open(os.path.join(_HERE, "apples.jpg")).convert("RGB")
    x = TF.center_crop(TF.resize(img, 256, interpolation=tfm.interpolation), [224, 224])
    return TF.normalize(TF.to_tensor(x), tfm.mean, tfm.std).unsqueeze(0)


if __name__ == "__main__":
    model = mobilenet_v2(
        weights=MobileNet_V2_QuantizedWeights.DEFAULT, quantize=True).eval()
    profile(model, apple_80_input(model), "apple80")
    profile(model, apple_224_input(), "apple224")
