import torch.nn as nn
from torch.nn import functional as F

#   *************************************ResNet50*********************************************      
class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, groups=1, activation=True):
        super(Conv, self).__init__()
        padding = kernel_size // 2 if padding is None else padding
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                              padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True) if activation else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, down_sample=False, groups=1):
        super(Bottleneck, self).__init__()
        stride = 2 if down_sample else 1
        mid_channels = out_channels // 4
        self.shortcut = Conv(in_channels, out_channels, kernel_size=1, stride=stride, activation=False) \
            if in_channels != out_channels else nn.Identity()
        self.conv = nn.Sequential(*[
            Conv(in_channels, mid_channels, kernel_size=1, stride=1),
            Conv(mid_channels, mid_channels, kernel_size=3, stride=stride, groups=groups),
            Conv(mid_channels, out_channels, kernel_size=1, stride=1, activation=False)
        ])

    def forward(self, x):
        y = self.conv(x) + self.shortcut(x)
        return F.relu(y, inplace=True)

class ResNet50(nn.Module):
    def __init__(self, ):
        super(ResNet50, self).__init__()
        self.stem = nn.Sequential(*[
            Conv(3, 64, kernel_size=7, stride=2),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        ])
        self.stages0 = nn.Sequential(*[
            self._make_stage(64, 256, down_sample=False, num_blocks=3),
        ])
        self.stages1 = nn.Sequential(*[
            self._make_stage(256, 512, down_sample=True, num_blocks=4),
        ])
        self.stages2 = nn.Sequential(*[
            self._make_stage(512, 1024, down_sample=True, num_blocks=6),

        ])
        self.stages3 = nn.Sequential(*[
            self._make_stage(1024, 2048, down_sample=True, num_blocks=3),
        ])

    @staticmethod
    def _make_stage(in_channels, out_channels, down_sample, num_blocks):
        layers = [Bottleneck(in_channels, out_channels, down_sample=down_sample)]
        for _ in range(1, num_blocks):
            layers.append(Bottleneck(out_channels, out_channels, down_sample=False))
        return nn.Sequential(*layers)

    def forward(self, x):           #input: bs, 3, 640, 640
        x = self.stem(x)            #bs, 64, 160, 160
        out0 = self.stages0(x)      #bs, 64, 160, 160
        out1 = self.stages1(out0)   #bs, 512, 80, 80
        out2 = self.stages2(out1)   #bs, 1024, 40, 40
        out3 = self.stages3(out2)   #bs, 2048, 20, 20

        return out0, out1, out2, out3
