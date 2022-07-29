import math
import torch
import torch.nn as nn
from SE_Module import SE
from SE_Module import _make_divisible

__all__ = ['ghost_net']

def depthwise_conv(in_channels, out_channels, kernel_size, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size//2, groups=in_channels, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(inplace=True),
    )

class GhostModule(nn.Module):
    def __init__(self, input, output, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.output = output
        init_channels = math.ceil(output / ratio)
        new_channels = init_channels * (ratio - 1)

        self.operation1 = nn.Sequential(
            nn.Conv2d(input, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.operation2 = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.operation1(x)
        x2 = self.operation2(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.output, :, :]


class BottleneckStructure(nn.Module):
    def __init__(self, in_chs, mid_chs, out_chs, dw_kernel_size=3, stride=1, se_ratio=0.):
        super(BottleneckStructure, self).__init__()
        se_layer = se_ratio is not None and se_ratio > 0.
        self.stride = stride

        self.ghost1 = GhostModule(in_chs, mid_chs, relu=True)

        if self.stride > 1:
            self.conv_dw = nn.Conv2d(mid_chs, mid_chs, dw_kernel_size, stride=stride,
                                     padding=(dw_kernel_size - 1) // 2,
                                     groups=mid_chs, bias=False)
            self.bn_dw = nn.BatchNorm2d(mid_chs)

        if se_layer:
            self.se = SE(mid_chs, se_ratio=se_ratio)
        else:
            self.se = None

        self.ghost2 = GhostModule(mid_chs, out_chs, relu=False)

        if (in_chs == out_chs and self.stride == 1):
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_chs, in_chs, dw_kernel_size, stride=stride,
                          padding=(dw_kernel_size - 1) // 2, groups=in_chs, bias=False),
                nn.BatchNorm2d(in_chs),
                nn.Conv2d(in_chs, out_chs, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_chs),
            )

    def forward(self, x):
        initial = x
        x = self.ghost1(x)

        if self.stride > 1:
            x = self.conv_dw(x)
            x = self.bn_dw(x)

        if self.se is not None:
            x = self.se(x)

        x = self.ghost2(x)
        x += self.shortcut(initial)
        return x


class GhostNet(nn.Module):
    def __init__(self, cfgs, num_classes=1000, width=1.0):
        super(GhostNet, self).__init__()

        self.cfgs = cfgs

        output_channel = _make_divisible(16 * width, 4)
        self.conv_stem = nn.Conv2d(3, output_channel, 3, 2, 1, bias=False)
        self.bn = nn.BatchNorm2d(output_channel)
        self.leakyrelu = nn.LeakyReLU(inplace=True)
        input_channel = output_channel

        stages = []
        block = BottleneckStructure
        for cfg in self.cfgs:
            layers = []
            for k, exp_size, c, se_ratio, s in cfg:
                output_channel = _make_divisible(c * width, 4)
                hidden_channel = _make_divisible(exp_size * width, 4)
                layers.append(block(input_channel, hidden_channel, output_channel, k, s,
                                    se_ratio=se_ratio))
                input_channel = output_channel
            stages.append(nn.Sequential(*layers))

        output_channel = _make_divisible(exp_size * width, 4)
        stages.append(nn.Sequential(depthwise_conv(input_channel, output_channel, 1)))
        input_channel = output_channel

        self.blocks = nn.Sequential(*stages)
        output_channel = 1280
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.conv = nn.Conv2d(input_channel, output_channel, 1, 1, 0, bias=True)
        self.leakyrelu2 = nn.LeakyReLU(inplace=True)
        self.classifier = nn.Linear(output_channel, num_classes)

    def forward(self, x):
        x = self.conv_stem(x)
        x = self.bn(x)
        x = self.leakyrelu(x)

        x = self.blocks(x)

        x = self.pooling(x)

        x = self.conv(x)
        x = self.leakyrelu2(x)
        x = x.view(x.size(0), -1)

        x = self.classifier(x)
        return x


def ghostnet(**kwargs):  # k, t, c, se, s
    cfgs = [
        [[3, 16, 16, 0, 1]],
        [[3, 48, 24, 0, 2]],
        [[3, 72, 24, 0, 1]],
        [[5, 72, 40, 0.25, 2]],
        [[5, 120, 40, 0.25, 1]],
        [[3, 240, 80, 0, 2]],
        [[3, 200, 80, 0, 1],
         [3, 184, 80, 0, 1],
         [3, 184, 80, 0, 1],
         [3, 480, 112, 0.25, 1],
         [3, 672, 112, 0.25, 1]],
        [[5, 672, 160, 0.25, 2]],
        [[5, 960, 160, 0, 1],
         [5, 960, 160, 0.25, 1],
         [5, 960, 160, 0, 1],
         [5, 960, 160, 0.25, 1]]
    ]
    return GhostNet(cfgs, **kwargs)
