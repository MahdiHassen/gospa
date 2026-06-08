import torch
from torchvision.models.quantization import (
    mobilenet_v2,
    MobileNet_V2_QuantizedWeights,
)


def get_first_conv():
    # Pretrained, post-training quantized MobileNetV2 (INT8 weights).
    model = mobilenet_v2(
        weights=MobileNet_V2_QuantizedWeights.DEFAULT,
        quantize=True,
    )
    model.eval()

    # features[0] is a ConvBNActivation block; first child is the Conv2d.
    first_block = model.features[0]
    first_conv = next(m for m in first_block.modules()
                      if hasattr(m, "weight") and m.weight().dim() == 4)
    return model, first_conv


def describe_quantized_weight(conv):
    # In quantized convs the weight is a packed qtensor accessor: conv.weight()
    w_q = conv.weight()
    print("Quantized weight dtype:", w_q.dtype)
    print("Quantized weight shape:", tuple(w_q.shape))
    print("Quantization scheme:", w_q.qscheme())

    # int representation (the raw int8 values stored in HW-friendly form)
    int_w = w_q.int_repr()
    print("int_repr dtype:", int_w.dtype)
    print("int_repr min/max:", int_w.min().item(), int_w.max().item())

    # Per-channel scales/zero-points (MobileNet uses per-channel symmetric).
    if w_q.qscheme() in (torch.per_channel_affine, torch.per_channel_symmetric):
        scales = w_q.q_per_channel_scales()
        zps = w_q.q_per_channel_zero_points()
        axis = w_q.q_per_channel_axis()
        print(f"Per-channel along axis {axis}:  "
              f"{scales.numel()} scales, scale range "
              f"[{scales.min().item():.3e}, {scales.max().item():.3e}], "
              f"zero-points unique: {zps.unique().tolist()}")
    else:
        print(f"Per-tensor: scale={w_q.q_scale():.3e}, zero_point={w_q.q_zero_point()}")

    return int_w


def channel_sparsity(int_w: torch.Tensor):
    # int_w shape: (out, in, kH, kW). Sparsity = fraction of exact-zero ints per slice.
    elems = int_w.shape[-1] * int_w.shape[-2]
    sp = (int_w == 0).sum(dim=(-1, -2)).float() / elems  # (out, in)
    print(f"\nINT8 exact-zero sparsity (out={int_w.shape[0]}, in={int_w.shape[1]}, "
          f"k={int_w.shape[-2]}x{int_w.shape[-1]}):")
    print(f"  overall:      {sp.mean().item():.4f}")
    print(f"  per-filter:   min={sp.mean(1).min().item():.4f}  "
          f"max={sp.mean(1).max().item():.4f}  mean={sp.mean(1).mean().item():.4f}")

    flat = sp.flatten()
    topk = torch.topk(flat, k=min(5, flat.numel()))
    print("  top-5 sparsest (filter, in-channel) slices:")
    for v, idx in zip(topk.values.tolist(), topk.indices.tolist()):
        f_idx, c_idx = divmod(idx, sp.shape[1])
        print(f"    filter {f_idx:>3d}, in-ch {c_idx}: {v:.4f}")


if __name__ == "__main__":
    model, first_conv = get_first_conv()
    print("First conv module:", first_conv)
    int_w = describe_quantized_weight(first_conv)
    channel_sparsity(int_w)
