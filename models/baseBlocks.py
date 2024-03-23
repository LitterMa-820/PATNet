import torch
from torch import nn
from models.p2t import PoolingAttention, IRB, p2t_base, p2t_small, p2t_tiny
from timm.models.layers import DropPath
from torch.nn import functional as F


class CBR(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding, bias=True, group=1, dilation=1,
                 act=nn.ReLU()):
        super(CBR, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, padding=padding, bias=bias, groups=group,
                      dilation=dilation),
            nn.BatchNorm2d(out_channel),
            act)

    def forward(self, x):
        # print(x.shape)
        return self.conv(x)


class P2tBackbone(nn.Module):
    def __init__(self, p2t_path):
        super(P2tBackbone, self).__init__()
        self.p2t_backbone1 = p2t_base()
        self.p2t_backbone2 = p2t_base()
        if p2t_path is not None:
            self.p2t_backbone1.init_weights(p2t_path)
            self.p2t_backbone2.init_weights(p2t_path)

    def forward(self, x, y):
        x_out = self.p2t_backbone1(x)
        y_out = self.p2t_backbone2(y)
        return x_out, y_out


class P2tBackboneSmall(nn.Module):
    def __init__(self, p2t_path):
        super(P2tBackboneSmall, self).__init__()
        self.p2t_backbone1 = p2t_small()
        self.p2t_backbone2 = p2t_small()
        if p2t_path is not None:
            self.p2t_backbone1.init_weights(p2t_path)
            self.p2t_backbone2.init_weights(p2t_path)

    def forward(self, x, y):
        x_out = self.p2t_backbone1(x)
        y_out = self.p2t_backbone2(y)
        return x_out, y_out


class P2tBackboneTiny(nn.Module):
    def __init__(self, p2t_path):
        super(P2tBackboneTiny, self).__init__()
        self.p2t_backbone1 = p2t_tiny()
        self.p2t_backbone2 = p2t_tiny()
        if p2t_path is not None:
            self.p2t_backbone1.init_weights(p2t_path)
            self.p2t_backbone2.init_weights(p2t_path)

    def forward(self, x, y):
        x_out = self.p2t_backbone1(x)
        y_out = self.p2t_backbone2(y)
        return x_out, y_out


class P2tBackboneBaseAndTiny(nn.Module):
    def __init__(self, p2t_path1, p2t_path2):
        super(P2tBackboneBaseAndTiny, self).__init__()
        self.p2t_backbone1 = p2t_base()
        self.p2t_backbone2 = p2t_tiny()
        if p2t_path1 is not None:
            self.p2t_backbone1.init_weights(p2t_path1)
        if p2t_path2 is not None:
            self.p2t_backbone2.init_weights(p2t_path2)

    def forward(self, x, y):
        x_out = self.p2t_backbone1(x)
        y_out = self.p2t_backbone2(y)
        return x_out, y_out


class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y
