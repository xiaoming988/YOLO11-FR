"""
Optimized FFCM module for Ultralytics YOLO-style models.

Design goals:
1. Preserve the original feature as a residual path so the plug-in does not destroy
   pretrained YOLO features at the beginning of training.
2. Run FFT/IFFT in float32 for AMP stability.
3. Fix the two-branch local mixer so the 5x5 branch consumes the second split.
4. Add lightweight channel/spatial gates to suppress harmful frequency noise.

Expected YAML usage with a custom parse_model rule:
    - [-1, 1, Fused_Fourier_Conv_Mixer, []]
The parser should pass the current input channel count as the first argument.
"""

from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    """Small Conv-BN-GELU block used inside FFCM."""

    def __init__(self, c1, c2, k=1, s=1, p=None, groups=1, bias=False):
        super().__init__()
        if p is None:
            p = k // 2 if isinstance(k, int) else tuple(kk // 2 for kk in k)
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


def _autocast_off(x):
    """Disable autocast only around FFT math; fall back safely on older torch builds."""

    try:
        return torch.amp.autocast(device_type=x.device.type, enabled=False)
    except Exception:
        return nullcontext()


class FourierUnit(nn.Module):
    """Frequency-domain 1x1 mixing over real and imaginary FFT components."""

    def __init__(self, in_channels, out_channels, groups=1, norm="ortho"):
        super().__init__()
        self.groups = groups
        self.norm = norm
        self.conv_layer = nn.Conv2d(
            in_channels=in_channels * 2,
            out_channels=out_channels * 2,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels * 2)
        self.act = nn.GELU()

    def _conv_bn_float(self, x):
        """Supports inference after model.half() by evaluating this branch in fp32."""

        if self.conv_layer.weight.dtype == torch.float32:
            return self.bn(self.conv_layer(x))

        y = F.conv2d(
            x,
            self.conv_layer.weight.float(),
            bias=None,
            stride=self.conv_layer.stride,
            padding=self.conv_layer.padding,
            dilation=self.conv_layer.dilation,
            groups=self.groups,
        )
        running_mean = self.bn.running_mean.float() if self.bn.running_mean is not None else None
        running_var = self.bn.running_var.float() if self.bn.running_var is not None else None
        weight = self.bn.weight.float() if self.bn.weight is not None else None
        bias = self.bn.bias.float() if self.bn.bias is not None else None
        return F.batch_norm(y, running_mean, running_var, weight, bias, self.bn.training, self.bn.momentum, self.bn.eps)

    def forward(self, x):
        dtype = x.dtype
        _, _, h, w = x.shape

        with _autocast_off(x):
            x_float = x.float()
            ffted = torch.fft.rfft2(x_float, norm=self.norm)
            ffted = torch.cat([ffted.real, ffted.imag], dim=1).contiguous()

            ffted = self.act(self._conv_bn_float(ffted))

            real, imag = torch.chunk(ffted, 2, dim=1)
            ffted = torch.complex(real.contiguous(), imag.contiguous())
            out = torch.fft.irfft2(ffted, s=(h, w), norm=self.norm)

        return out.to(dtype=dtype)


class Freq_Fusion(nn.Module):
    """Stable frequency fusion block used by Fused_Fourier_Conv_Mixer."""

    def __init__(
        self,
        dim,
        kernel_size=(1, 3, 5, 7),
        se_ratio=8,
        local_size=8,
        scale_ratio=2,
        spilt_num=4,
    ):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.local_size = local_size
        self.scale_ratio = scale_ratio
        self.spilt_num = spilt_num

        self.conv_init_1 = ConvBNAct(dim, dim, k=1)
        self.conv_init_2 = ConvBNAct(dim, dim, k=1)
        self.ffc = FourierUnit(dim * 2, dim * 2)

        hidden = max((dim * 2) // se_ratio, 8)
        self.freq_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim * 2, hidden, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden, dim * 2, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.out_norm = nn.BatchNorm2d(dim * 2)
        self.out_act = nn.GELU()
        self.gamma = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        x_1, x_2 = torch.split(x, self.dim, dim=1)
        x_1 = self.conv_init_1(x_1)
        x_2 = self.conv_init_2(x_2)
        x0 = torch.cat([x_1, x_2], dim=1)

        freq = self.ffc(x0)
        freq = freq * self.freq_gate(freq)
        out = x0 + torch.tanh(self.gamma) * freq
        return self.out_act(self.out_norm(out))


class Fused_Fourier_Conv_Mixer(nn.Module):
    """
    FFCM plug-in block. Input and output channels are identical.

    Args:
        dim: input channel count. For YOLO parse_model, pass ch[f].
    """

    def __init__(
        self,
        dim,
        token_mixer_for_gloal=Freq_Fusion,
        mixer_kernel_size=(1, 3, 5, 7),
        local_size=8,
        reduction=4,
    ):
        super().__init__()
        if not isinstance(dim, int) or dim <= 0:
            raise ValueError(f"Fused_Fourier_Conv_Mixer expects a positive channel integer, got {dim!r}")

        self.dim = dim
        self.conv_init = ConvBNAct(dim, dim * 2, k=1)

        self.dw_conv_1 = ConvBNAct(dim, dim, k=3, groups=dim)
        self.dw_conv_2 = ConvBNAct(dim, dim, k=5, groups=dim)

        self.mixer_gloal = token_mixer_for_gloal(
            dim=dim,
            kernel_size=mixer_kernel_size,
            se_ratio=8,
            local_size=local_size,
        )

        self.proj = nn.Sequential(
            ConvBNAct(dim * 2, dim, k=1),
            ConvBNAct(dim, dim, k=3, groups=dim),
        )

        hidden = max(dim // reduction, 8)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, hidden, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden, dim, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(dim, 1, kernel_size=7, padding=3, bias=True),
            nn.Sigmoid(),
        )
        self.beta = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        identity = x

        x = self.conv_init(x)
        x_1, x_2 = torch.split(x, self.dim, dim=1)

        x_local_1 = self.dw_conv_1(x_1)
        # Important bug fix: the 5x5 branch must use the second split, not x_1 again.
        x_local_2 = self.dw_conv_2(x_2)

        x_global = self.mixer_gloal(torch.cat([x_local_1, x_local_2], dim=1))
        out = self.proj(x_global)
        out = out * self.channel_gate(out) * self.spatial_gate(out)

        return identity + torch.tanh(self.beta) * out


# Optional aliases for different import styles in custom YOLO projects.
FFCM = Fused_Fourier_Conv_Mixer
