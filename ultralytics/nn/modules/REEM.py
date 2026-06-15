"""
Optimized REEM module for Ultralytics YOLO-style models.

This version keeps the original residual edge-enhancement idea but makes it more
useful in detection heads:
1. multi-scale local/directional depthwise edge branches;
2. a fixed Sobel prior to emphasize real gradient responses;
3. channel and spatial gates to suppress background noise;
4. bounded residual scaling for stable training from pretrained YOLO weights.

Expected YAML usage with a custom parse_model rule:
    - [-1, 1, REEM, []]
The parser should pass the current input channel count as the first argument.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, groups=1, bias=False):
        super().__init__()
        if p is None:
            p = k // 2 if isinstance(k, int) else tuple(kk // 2 for kk in k)
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class REEM(nn.Module):
    """Residual Edge Enhancement Module. Input and output channels are identical."""

    def __init__(self, channels, reduction=4, init_scale=0.1):
        super().__init__()
        if not isinstance(channels, int) or channels <= 0:
            raise ValueError(f"REEM expects a positive channel integer, got {channels!r}")

        self.channels = channels

        self.local_dw = ConvBNAct(channels, channels, k=3, groups=channels)
        self.dilate_dw = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=2, dilation=2, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )
        self.strip_h = ConvBNAct(channels, channels, k=(1, 5), p=(0, 2), groups=channels)
        self.strip_v = ConvBNAct(channels, channels, k=(5, 1), p=(2, 0), groups=channels)

        self.edge_norm = nn.BatchNorm2d(channels)
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 5, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )

        hidden = max(channels // reduction, 8)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=7, padding=3, bias=True),
            nn.Sigmoid(),
        )

        self.alpha = nn.Parameter(torch.tensor(float(init_scale)))

        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x, persistent=False)
        self.register_buffer("sobel_y", sobel_y, persistent=False)

    def _sobel_magnitude(self, x):
        dtype = x.dtype
        x_float = x.float()
        weight_x = self.sobel_x.to(device=x.device).expand(self.channels, 1, 3, 3)
        weight_y = self.sobel_y.to(device=x.device).expand(self.channels, 1, 3, 3)
        grad_x = F.conv2d(x_float, weight_x, padding=1, groups=self.channels)
        grad_y = F.conv2d(x_float, weight_y, padding=1, groups=self.channels)
        edge = torch.sqrt(grad_x.square() + grad_y.square() + 1e-6)
        return edge.to(dtype=dtype)

    def forward(self, x):
        e_local = self.local_dw(x)
        e_dilate = self.dilate_dw(x)
        e_h = self.strip_h(x)
        e_v = self.strip_v(x)
        e_sobel = self.edge_norm(self._sobel_magnitude(x))

        edge_feat = self.fuse(torch.cat([e_local, e_dilate, e_h, e_v, e_sobel], dim=1))
        edge_feat = edge_feat * self.channel_gate(edge_feat) * self.spatial_gate(edge_feat)
        return x + torch.tanh(self.alpha) * edge_feat
