"""
BD-YOLOv8s practical reproduction modules for Ultralytics.

Implements the three main components reported for BD-YOLOv8s:
- ODConv: omni-dimensional dynamic convolution, used to replace the second Conv layer.
- CBAM: channel and spatial attention, inserted after the first two C2f blocks.
- CARAFE: content-aware feature reassembly, used to replace nearest-neighbor upsampling.

This is a clean-room practical implementation intended for comparative experiments.
Place this file at: ultralytics/nn/modules/BDYOLO.py
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from ultralytics.nn.modules.conv import Conv
except Exception:  # fallback when imported relatively inside ultralytics
    from .conv import Conv


def autopad(k, p=None, d=1):
    """Pad to 'same' shape outputs, compatible with Ultralytics Conv.autopad absence."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class HSigmoid(nn.Module):
    def __init__(self, inplace: bool = True):
        super().__init__()
        self.inplace = inplace

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu6(x + 3.0, inplace=self.inplace) / 6.0


class ODConv(nn.Module):
    """Omni-dimensional dynamic convolution.

    Args are intentionally close to Ultralytics Conv: c1, c2, k, s, p, g, d, act.
    The additional kernel_num and reduction args can be left as defaults.
    """

    default_act = nn.SiLU()

    def __init__(
        self,
        c1: int,
        c2: int,
        k: int = 3,
        s: int = 1,
        p=None,
        g: int = 1,
        d: int = 1,
        act: bool | nn.Module = True,
        kernel_num: int = 4,
        reduction: float = 0.0625,
    ):
        super().__init__()
        assert c1 % g == 0, f"in_channels {c1} must be divisible by groups {g}"
        self.c1 = c1
        self.c2 = c2
        self.k = k if isinstance(k, int) else k[0]
        self.s = s
        self.p = autopad(k, p, d)
        self.g = g
        self.d = d
        self.kernel_num = kernel_num

        hidden = max(int(c1 * reduction), 16)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(c1, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
        )

        self.channel_fc = nn.Conv2d(hidden, c1, 1, bias=True)
        self.filter_fc = nn.Conv2d(hidden, c2, 1, bias=True)
        self.spatial_fc = nn.Conv2d(hidden, self.k * self.k, 1, bias=True)
        self.kernel_fc = nn.Conv2d(hidden, kernel_num, 1, bias=True)

        self.hsigmoid = HSigmoid()
        self.weight = nn.Parameter(torch.randn(kernel_num, c2, c1 // g, self.k, self.k))
        self.bias = None

        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        self.reset_parameters()

    def reset_parameters(self):
        for i in range(self.kernel_num):
            nn.init.kaiming_uniform_(self.weight[i], a=math.sqrt(5))
        nn.init.zeros_(self.channel_fc.bias)
        nn.init.zeros_(self.filter_fc.bias)
        nn.init.zeros_(self.spatial_fc.bias)
        nn.init.zeros_(self.kernel_fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        att = self.fc(self.avg_pool(x))

        channel_att = self.hsigmoid(self.channel_fc(att))  # B,C,1,1
        filter_att = self.hsigmoid(self.filter_fc(att))    # B,Cout,1,1
        spatial_att = self.hsigmoid(self.spatial_fc(att)).view(b, 1, 1, 1, self.k, self.k)
        kernel_att = F.softmax(self.kernel_fc(att).view(b, self.kernel_num, 1, 1, 1, 1), dim=1)

        x = x * channel_att

        # Aggregate dynamic kernels per sample.
        weight = self.weight.unsqueeze(0)  # 1,K,Cout,Cin/groups,k,k
        aggregate_weight = (kernel_att * spatial_att * weight).sum(dim=1)
        aggregate_weight = aggregate_weight.view(b * self.c2, self.c1 // self.g, self.k, self.k)

        x = x.view(1, b * self.c1, h, w)
        y = F.conv2d(x, aggregate_weight, bias=None, stride=self.s, padding=self.p, dilation=self.d, groups=b * self.g)
        y = y.view(b, self.c2, y.shape[-2], y.shape[-1])
        y = y * filter_att
        return self.act(self.bn(y))


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = self.mlp(F.adaptive_avg_pool2d(x, 1))
        mx = self.mlp(F.adaptive_max_pool2d(x, 1))
        return torch.sigmoid(avg + mx)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        assert kernel_size in (3, 7)
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        return torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))


class CBAM(nn.Module):
    """Convolutional Block Attention Module. Keeps input and output channels unchanged."""

    def __init__(self, c1: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        self.ca = ChannelAttention(c1, reduction)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


class CARAFE(nn.Module):
    """CARAFE-like content-aware upsampling.

    Args:
        c1: channels
        scale: upsampling scale, usually 2
        k: reassembly kernel size
        compressed_channels: intermediate channels for kernel generation
    """

    def __init__(self, c1: int, scale: int = 2, k: int = 5, compressed_channels: int = 64):
        super().__init__()
        self.scale = scale
        self.k = k
        self.comp = nn.Conv2d(c1, compressed_channels, 1, bias=False)
        self.enc = nn.Conv2d(
            compressed_channels,
            (scale * scale) * (k * k),
            kernel_size=3,
            padding=1,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        mask = self.enc(self.comp(x))
        mask = F.pixel_shuffle(mask, self.scale)  # B,k*k,H*s,W*s
        mask = F.softmax(mask, dim=1)

        x_up = F.interpolate(x, scale_factor=self.scale, mode="nearest")
        h2, w2 = x_up.shape[-2:]
        patches = F.unfold(x_up, kernel_size=self.k, padding=self.k // 2)
        patches = patches.view(b, c, self.k * self.k, h2, w2)
        out = (patches * mask.unsqueeze(1)).sum(dim=2)
        return out


__all__ = ["ODConv", "CBAM", "CARAFE"]
