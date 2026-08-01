"""
mobilenet_layers.py -- real per-layer tensors of quantized MobileNetV2 on the
80x80 apple vector, for layer-by-layer RTL bring-up.

For a given execution-order layer index (same numbering as
mobilenet_sparsity_apple80.csv), capture the layer's INPUT activations as
hardware-signed ints (int_repr - zero_point: exactly what the datapath
computes on) and its int8 weights. Cached to an .npz so cocotb runs do not
need torch after the first extraction.

    get_layer(idx) -> dict:
        type    : 'conv3x3s2' | 'dw3x3s1' | 'pw1x1' | ...
        stride  : int
        groups  : int
        acts    : [Cin][H][H] python ints (signed)
        weights : [Cout][Cin_g][kh][kw] python ints (signed int8)
"""
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))


def _npz_path(idx):
    return os.path.join(_HERE, f"mobilenet_layer{idx}_apple80.npz")


def _extract_many(idxs):
    """One forward pass capturing every requested layer index."""
    import torch
    from torchvision.models.quantization import (
        mobilenet_v2, MobileNet_V2_QuantizedWeights)
    from mobilenet_sparsity import _is_qconv_or_qlinear, apple_80_input

    model = mobilenet_v2(
        weights=MobileNet_V2_QuantizedWeights.DEFAULT, quantize=True).eval()

    want = set(idxs)
    cap = {}
    order = []

    def hook(mod):
        def fn(module, inputs, output):
            i = len(order)
            order.append(module)
            if i in want:
                t = inputs[0]
                cap[i] = dict(
                    acts=(t.int_repr().to(torch.int32)
                          - t.q_zero_point())[0].numpy().astype(np.int16),
                    weights=module.weight().int_repr().numpy(),
                    stride=int(getattr(module, "stride", (1,))[0]),
                    groups=int(getattr(module, "groups", 1)))
        return fn

    handles = [m.register_forward_hook(hook(m))
               for _, m in model.named_modules() if _is_qconv_or_qlinear(m)]
    with torch.inference_mode():
        model(apple_80_input(model))
    for h in handles:
        h.remove()

    missing = want - set(cap)
    assert not missing, f"layer idxs {missing} not reached ({len(order)} layers)"
    for i, c in cap.items():
        np.savez_compressed(
            _npz_path(i), acts=c["acts"], weights=c["weights"],
            stride=np.int32(c["stride"]), groups=np.int32(c["groups"]))


def _extract(idx):
    _extract_many([idx])


def list_layers():
    """Meta for every 4-D quantized conv layer (skips the linear classifier):
    [{idx, type, H, F, S, groups, cin, cout}] in execution order."""
    import torch
    from torchvision.models.quantization import (
        mobilenet_v2, MobileNet_V2_QuantizedWeights)
    from mobilenet_sparsity import _is_qconv_or_qlinear, apple_80_input

    model = mobilenet_v2(
        weights=MobileNet_V2_QuantizedWeights.DEFAULT, quantize=True).eval()
    rows = []
    order = []

    def hook(mod):
        def fn(module, inputs, output):
            i = len(order)
            order.append(module)
            w = module.weight().int_repr()
            if w.dim() != 4:
                return                      # linear classifier: not mapped
            cout, cin_g, kh, kw = w.shape
            groups = int(getattr(module, "groups", 1))
            stride = int(getattr(module, "stride", (1,))[0])
            t = inputs[0]
            if groups == 1 and (kh, kw) == (1, 1):
                ltype = "pw1x1"
            elif groups == cout and cin_g == 1:
                ltype = f"dw{kh}x{kw}s{stride}"
            else:
                ltype = f"conv{kh}x{kw}s{stride}"
            rows.append(dict(idx=i, type=ltype, H=int(t.shape[-1]), F=int(kh),
                             S=stride, groups=groups,
                             cin=int(t.shape[1]), cout=int(cout)))
        return fn

    handles = [m.register_forward_hook(hook(m))
               for _, m in model.named_modules() if _is_qconv_or_qlinear(m)]
    with torch.inference_mode():
        model(apple_80_input(model))
    for h in handles:
        h.remove()
    return rows


def ensure_extracted(idxs):
    todo = [i for i in idxs if not os.path.exists(_npz_path(i))]
    if todo:
        _extract_many(todo)


def get_layer(idx):
    if not os.path.exists(_npz_path(idx)):
        _extract(idx)
    z = np.load(_npz_path(idx))
    acts = z["acts"]                       # (Cin, H, W) int16
    w = z["weights"]                       # (Cout, Cin_g, kh, kw) int8
    stride, groups = int(z["stride"]), int(z["groups"])
    cout, cin_g, kh, kw = w.shape
    if groups == 1 and (kh, kw) == (1, 1):
        ltype = "pw1x1"
    elif groups == cout and cin_g == 1:
        ltype = f"dw{kh}x{kw}s{stride}"
    else:
        ltype = f"conv{kh}x{kw}s{stride}"
    return dict(
        type=ltype, stride=stride, groups=groups,
        acts=[acts[c].astype(int).tolist() for c in range(acts.shape[0])],
        weights=[[[[int(v) for v in row] for row in w[o, g]]
                  for g in range(cin_g)] for o in range(cout)],
    )


if __name__ == "__main__":
    for i in (1, 2, 3):
        L = get_layer(i)
        nnz = [sum(1 for r in ch for v in r if v) for ch in L["acts"]]
        print(f"layer {i}: {L['type']} groups={L['groups']} "
              f"Cin={len(L['acts'])} Cout={len(L['weights'])} "
              f"H={len(L['acts'][0])} nnz[:4]={nnz[:4]}")
