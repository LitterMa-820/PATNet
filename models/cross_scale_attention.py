import torch
from torch import nn
from timm.models.layers import DropPath
from torch.nn import functional as F


class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads, ):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        # q线性投射
        self.q = nn.Sequential(nn.Linear(dim, dim))
        self.k = nn.Sequential(nn.Linear(dim, dim))
        self.v = nn.Sequential(nn.Linear(dim, dim))
        self.proj = nn.Linear(dim, dim)
        # self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v)
        x = x.transpose(1, 2).contiguous().reshape(B, N, C)
        x = self.proj(x)
        return x


class IRB(nn.Module):
    """
    act_layer=nn.Hardswish
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, ksize=3, act_layer=nn.Hardswish, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, 1, 0)
        self.act = act_layer()
        self.conv = nn.Conv2d(hidden_features, hidden_features, kernel_size=ksize, padding=ksize // 2, stride=1,
                              groups=hidden_features)
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, 1, 0)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        B, N, C = x.shape
        # XI att = Seq2Image(Xatt) 转为二维特征图
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        # 1*1扩充通道
        x = self.fc1(x)
        # hardwish 激活函数
        x = self.act(x)
        # 3*3采样
        x = self.conv(x)
        # 激活函数
        x = self.act(x)
        # 把通道还原
        x = self.fc2(x)
        # 变回1维
        return x.reshape(B, C, -1).permute(0, 2, 1)



class CrossScaleAttention(nn.Module):
    """
    这次把一维线性投射换成二维卷积投射
    """

    def __init__(self, dim, last_dim, num_heads, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.q = nn.Sequential(nn.Conv2d(dim, dim, 3, 1, 1, bias=True))
        self.k = nn.Sequential(nn.Conv2d(dim, dim, 3, 1, 1, bias=True))
        self.v = nn.Sequential(nn.Conv2d(dim, dim, 3, 1, 1, bias=True))

        self.norm_x = nn.LayerNorm(dim)
        self.norm_y = nn.LayerNorm(dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(dim, dim, 3, 1, 1, bias=True)
        # self.norm = nn.LayerNorm(dim)

    def forward(self, x, y):
        """
        :param x: shape of x is B C1 H1 W1 ,patches1=H1*W1
        :param y: low resolution features B C2 H2 W2,patches = H2*W2
        :return:att_fuse(x,y)
        """
        B, C, H1, W1 = x.shape
        _, _, H2, W2 = y.shape
        N1 = H1 * W1
        N2 = H2 * W2

        x = self.norm_x(x.reshape(B, C, -1).permute(0, 2, 1))
        x = x.permute(0, 2, 1).reshape(B, C, H1, W1)
        y = self.norm_y(y.reshape(B, C, -1).permute(0, 2, 1))
        y = y.permute(0, 2, 1).reshape(B, C, H2, W2)

        q = self.q(x)
        k = self.k(y)
        v = self.v(y)
        q = q.reshape(B, C, -1).permute(0, 2, 1)
        q = q.reshape(B, N1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        k = k.reshape(B, C, -1).permute(0, 2, 1)
        k = k.reshape(B, N2, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        v = v.reshape(B, C, -1).permute(0, 2, 1)
        v = v.reshape(B, N2, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        # print('attn:', attn.shape)
        out = (attn @ v)
        out = out.transpose(1, 2).contiguous().reshape(B, N1, C).permute(0, 2, 1).reshape(B, C, H1, W1)
        # print('out:', out.shape)
        out = self.proj(out)
        out = out.reshape(B, N1, C)
        return out


class CrossScaleAttnBlock(nn.Module):
    def __init__(self, dim, last_dim, num_heads, mlp_ratio=4., drop_path=0.):
        super().__init__()
        self.channel_conv = nn.Conv2d(last_dim, dim, 1, 1, 0)
        self.norm1_3 = nn.LayerNorm(dim)
        self.norm2_1 = nn.LayerNorm(dim)
        self.norm2_2 = nn.LayerNorm(dim)
        self.attn1 = CrossScaleAttention(dim, last_dim, num_heads)
        self.attn2 = SelfAttention(dim, num_heads)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp1 = IRB(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=nn.Hardswish, drop=0., ksize=3)
        self.mlp2 = IRB(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=nn.Hardswish, drop=0., ksize=3)

    def forward(self, x, y):
        B, C, H, W = x.shape
        # y_C2->C1
        y = self.channel_conv(y)
        y_up = F.interpolate(y, (H, W), mode='bilinear').reshape(B, C, -1).permute(0, 2, 1)
        fuse = y_up + self.drop_path(self.attn1(x, y))
        fuse = fuse + self.drop_path(self.mlp1(self.norm1_3(fuse), H, W))

        fuse = fuse + self.drop_path(self.attn2(self.norm2_1(fuse)))
        out = fuse + self.drop_path(self.mlp2(self.norm2_2(fuse), H, W))
        return out


