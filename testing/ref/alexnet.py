import torch
from torchvision import models
from torchvision.models import AlexNet_Weights


def get_first_conv_weights():
    model = models.alexnet(weights=AlexNet_Weights.DEFAULT)
    model.eval()

    first_conv = model.features[0]
    assert isinstance(first_conv, torch.nn.Conv2d)

    weights = first_conv.weight.detach().clone()
    bias = first_conv.bias.detach().clone() if first_conv.bias is not None else None
    return first_conv, weights, bias


def channel_sparsity(weights: torch.Tensor, threshold: float = 0.0):
    # weights: (out_channels, in_channels, kH, kW)
    # Returns sparsity per (filter, channel) slice as a (out, in) tensor.
    if threshold == 0.0:
        mask = weights == 0
    else:
        mask = weights.abs() <= threshold
    elems_per_slice = weights.shape[-1] * weights.shape[-2]
    return mask.sum(dim=(-1, -2)).float() / elems_per_slice


def print_sparsity_report(weights: torch.Tensor, threshold: float):
    label = "exact zeros" if threshold == 0.0 else f"|w| <= {threshold:g}"
    sp = channel_sparsity(weights, threshold)  # (64, 3)
    overall = sp.mean().item()
    per_channel = sp.mean(dim=0)  # mean over filters, per input channel (R,G,B)

    print(f"\n--- Sparsity ({label}) ---")
    print(f"Overall mean sparsity: {overall:.4f}")
    print(f"Per input-channel mean (R,G,B): "
          f"{per_channel[0].item():.4f}, {per_channel[1].item():.4f}, {per_channel[2].item():.4f}")

    per_filter = sp.mean(dim=1)  # mean over channels, per filter
    print(f"Per-filter sparsity — min: {per_filter.min().item():.4f}, "
          f"max: {per_filter.max().item():.4f}, mean: {per_filter.mean().item():.4f}")

    # Show the top-5 sparsest (filter, channel) slices.
    flat = sp.flatten()
    topk = torch.topk(flat, k=5)
    print("Top-5 sparsest (filter, channel) slices:")
    for v, idx in zip(topk.values.tolist(), topk.indices.tolist()):
        f_idx, c_idx = divmod(idx, sp.shape[1])
        print(f"  filter {f_idx:>2d}, channel {c_idx} ({'RGB'[c_idx]}): {v:.4f}")


if __name__ == "__main__":
    layer, weights, bias = get_first_conv_weights()

    print("First conv layer:", layer)
    print("Weight shape:", tuple(weights.shape))
    print("Weight dtype:", weights.dtype)
    print("Weight min/max/mean:",
          weights.min().item(), weights.max().item(), weights.mean().item())
    if bias is not None:
        print("Bias shape:", tuple(bias.shape))

    # Pretrained dense weights have ~0 exact zeros — also check near-zero thresholds.
    print_sparsity_report(weights, threshold=0.0)
    print_sparsity_report(weights, threshold=1e-3)
    print_sparsity_report(weights, threshold=1e-2)
