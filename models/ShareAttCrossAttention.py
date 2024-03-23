import torch
from torch import nn
from models.p2t import PoolingAttention, p2t_base, IRB
from timm.models.layers import DropPath
from torch.nn import functional as F
from models.baseBlocks import CBR


class ShareAttCrossAttention(nn.Module):
    """
    rgb:q_r,v_r
    depth:k_d,v_d
    attn_r: q_r@k_d
    attn_d: attn_r.T
    """

    def __init__(self, dim, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., ):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        # rgb线性投射
        self.q_r = nn.Sequential(nn.Linear(dim, dim, bias=qkv_bias))
        # self.k_r = nn.Sequential(nn.Linear(dim, dim, bias=qkv_bias))
        self.v_r = nn.Sequential(nn.Linear(dim, dim, bias=qkv_bias))
        # depth线性投射
        # self.q_d = nn.Sequential(nn.Linear(dim, dim, bias=qkv_bias))
        self.k_d = nn.Sequential(nn.Linear(dim, dim, bias=qkv_bias))
        self.v_d = nn.Sequential(nn.Linear(dim, dim, bias=qkv_bias))
        self.norm_r = nn.LayerNorm(dim)
        self.norm_d = nn.LayerNorm(dim)

        self.proj_r = nn.Linear(dim, dim)
        self.proj_d = nn.Linear(dim, dim)

    def forward(self, x, y):
        B, N, C = x.shape
        # N = H * W
        # layer norm
        x = self.norm_r(x)
        y = self.norm_d(y)

        # proj and multi head
        q_r = self.q_r(y).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        # k_r = self.k_r(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v_r = self.v_r(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        # q_d = self.q_d(y).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k_d = self.k_d(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v_d = self.v_d(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        # share cross attn
        attn_r = (q_r @ k_d.transpose(-2, -1)) * self.scale
        attn_d = attn_r.transpose(-2, -1)
        attn_r = attn_r.softmax(dim=-1)
        attn_d = attn_d.softmax(dim=-1)

        out_r = (attn_r @ v_r)
        out_r = out_r.transpose(1, 2).contiguous().reshape(B, N, C)
        out_r = self.proj_r(out_r)

        out_d = (attn_d @ v_d).transpose(1, 2).contiguous().reshape(B, N, C)
        out_d = out_d.transpose(1, 2).contiguous().reshape(B, N, C)
        out_d = self.proj_d(out_d)

        return out_r, out_d


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., ):
        super().__init__()
        self.cross_attention = ShareAttCrossAttention(dim, num_heads)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp = IRB(in_features=dim * 2, hidden_features=int(dim * mlp_ratio), act_layer=nn.Hardswish, drop=drop,
                       ksize=3)
        self.norm = nn.LayerNorm(dim * 2)
        self.reduce_conv = nn.Conv2d(dim * 2, dim, 3, 1, 1)

    def forward(self, x, y):
        B, C, H, W = x.shape
        x = x.reshape(B, C, -1).permute(0, 2, 1)
        y = y.reshape(B, C, -1).permute(0, 2, 1)
        out_r, out_d = self.cross_attention(x, y)
        x = x + self.drop_path(out_r)
        # x = x.permute(0, 2, 1).reshape(B, C, H, W)
        y = y + self.drop_path(out_d)
        # y = y.permute(0, 2, 1).reshape(B, C, H, W)
        fuse = torch.cat((x, y), dim=-1)
        fuse = fuse + self.drop_path(self.mlp(self.norm(fuse), H, W))
        fuse = fuse.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        fuse = self.reduce_conv(fuse)
        return fuse
