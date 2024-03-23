from os import sep
from pickle import TRUE
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.jit as jit
from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg

from mmseg.models.builder import BACKBONES
from mmcv.runner import load_checkpoint
from mmseg.utils import get_root_logger

import numpy as np
from time import time

__all__ = [
    'p2t_tiny', 'p2t_small', 'p2t_base', 'p2t_large', 'PoolingAttention', 'IRB'
]


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


class PoolingAttention(nn.Module):
    def __init__(self, dim, num_heads=2, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 pool_ratios=[1, 2, 3, 6]):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        # 1*1 2*2 3*3 6*6
        self.num_elements = np.array([t * t for t in pool_ratios]).sum()

        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        # q线性投射
        self.q = nn.Sequential(nn.Linear(dim, dim, bias=qkv_bias))
        # kv用金字塔
        self.kv = nn.Sequential(nn.Linear(dim, dim * 2, bias=qkv_bias))

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.pool_ratios = pool_ratios
        self.pools = nn.ModuleList()

        self.norm = nn.LayerNorm(dim)

    # def forward(self, x, H, W, d_convs=None):
    #     # batch，patch_n,channel
    #     B, N, C = x.shape
    #     # q = x->B N Heads C -> B Heads N C
    #     # print(x.shape)
    #     # print("after linear",self.num_heads,self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).shape)
    #     q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    #     pools = []
    #     # x->B C N(H*W) -> B C H W
    #     x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
    #     for (pool_ratio, l) in zip(self.pool_ratios, d_convs):
    #         # 从x_中自适应池化到对应比率
    #         pool = F.adaptive_avg_pool2d(x_, (round(H / pool_ratio), round(W / pool_ratio)))
    #         # Penc i = DWConv(Pi) + Pi, i = 1, 2, · · · , n,
    #         pool = pool + l(pool)
    #         # print("---pool----:",pool.view(B, C, -1).shape)
    #         pools.append(pool.view(B, C, -1))
    #     # P = LayerNorm(Concat(Penc 1 , Penc 2 , ..., Penc n )).
    #     # 在patch数量维上做拼接 pools为B C N(所有pool size的总和)
    #     pools = torch.cat(pools, dim=2)
    #     pools = self.norm(pools.permute(0, 2, 1))
    #     # print("---pools---:",pools.shape)
    #     # 先用kv映射到两倍，pools->B,patch_n,2,heads_num,c//heads_num，再单独把2提出来,再把heads放在前头
    #     kv = self.kv(pools).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    #     k, v = kv[0], kv[1]
    #     # 自注意力
    #     attn = (q @ k.transpose(-2, -1)) * self.scale
    #     attn = attn.softmax(dim=-1)
    #     x = (attn @ v)
    #     # x_shape -> B H N C -> B N (H H_C) -> B N C
    #     # .contiguous()原地操作
    #     x = x.transpose(1, 2).contiguous().reshape(B, N, C)
    #
    #     x = self.proj(x)
    #
    #     return x

    def forward(self, x, H, W, y=None, d_convs=None):
        B, N, C = x.shape
        # q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        if y is not None:
            q = self.q(y).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        else:
            q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        pools = []
        # join y in attention
        # if y is not None:
        #     x_ = y.permute(0, 2, 1).reshape(B, C, H, W)
        # else:
        #     x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
        x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
        for (pool_ratio, l) in zip(self.pool_ratios, d_convs):
            # 从x_中自适应池化到对应比率
            pool = F.adaptive_avg_pool2d(x_, (round(H / pool_ratio), round(W / pool_ratio)))
            # Penc i = DWConv(Pi) + Pi, i = 1, 2, · · · , n,
            pool = pool + l(pool)
            # print("---pool----:",pool.view(B, C, -1).shape)
            pools.append(pool.view(B, C, -1))
        pools = torch.cat(pools, dim=2)
        pools = self.norm(pools.permute(0, 2, 1))
        kv = self.kv(pools).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        # testing
        # q = q.div((q.norm(dim=-1, keepdim=True)))
        # k = k.div((k.norm(dim=-1, keepdim=True)))
        # testing over
        attn = (q @ k.transpose(-2, -1)) * self.scale
        # print('qk atten:', q.shape, k.shape, attn.shape)
        # print(attn.shape)
        # print(q.div((q.norm(dim=-1, keepdim=True))).shape, k.div((k.norm(dim=-1, keepdim=True))).shape)
        # print(attn)
        # print(torch.argmax(attn,dim=-1))
        attn = attn.softmax(dim=-1)

        x = (attn @ v)
        # print('v:',x.shape,v.shape)
        # x_shape -> B H N C -> B N (H H_C) -> B N C
        # .contiguous()原地操作
        x = x.transpose(1, 2).contiguous().reshape(B, N, C)

        x = self.proj(x)

        return x


class Block(nn.Module):
    """
    一个基础块包含金字塔自注意力机制
    """

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, pool_ratios=[12, 16, 20, 24]):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = PoolingAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, pool_ratios=pool_ratios)
        # 一种正则手段，随机删除分支路径
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = IRB(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=nn.Hardswish, drop=drop,
                       ksize=3)

    def forward(self, x, H, W, d_convs=None):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W, d_convs=d_convs))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, kernel_size=3, in_chans=3, embed_dim=768, overlap=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        # 能否整除
        assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, \
            f"img_size {img_size} should be divided by patch_size {patch_size}."
        # patch_num
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        # 使用卷积拿出padding,这样效果会比直接展开成768要好吗？
        if not overlap:
            self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        else:
            self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=patch_size,
                                  padding=kernel_size // 2)

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x_shape b c h w
        B, C, H, W = x.shape
        # patch_size=4,224/4=56
        # proj(x)->x_shape b c(768) p_h,p_w(224->56)
        # flatten(x)->x_shape b c p_h*p_w(56*56) ->transpose(x) x_shape b p_h*p_w c 符合vit
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        H, W = H // self.patch_size[0], W // self.patch_size[1]
        # print('H W',H,W)
        # 返回embedding后的x,p_h和p_w
        return x, (H, W)


class PyramidPoolingTransformer(nn.Module):
    """
    注：这些参数都是根据P2T是tiny、small、large在后面指定的，这里的参数只是参考
    其中Embedding_dim是论文中的C，mlp_ratios是论文中的E（IRB扩张比例），depths是每个阶段有多少个transformer block
    具体参考表一。
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000, embed_dims=[64, 128, 320, 512],
                 num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0.1, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 9, 3],
                 **kwargs):  #
        super().__init__()
        print("loading p2t")
        self.num_classes = num_classes
        self.depths = depths

        self.embed_dims = embed_dims

        # pyramid pooling ratios for each stage
        pool_ratios = [[12, 16, 20, 24], [6, 8, 10, 12], [3, 4, 5, 6], [1, 2, 3, 4]]

        self.patch_embed1 = PatchEmbed(img_size=img_size, patch_size=4, kernel_size=7, in_chans=in_chans,
                                       embed_dim=embed_dims[0], overlap=True)

        self.patch_embed2 = PatchEmbed(img_size=img_size // 4, patch_size=2, in_chans=embed_dims[0],
                                       embed_dim=embed_dims[1], overlap=True)
        self.patch_embed3 = PatchEmbed(img_size=img_size // 8, patch_size=2, in_chans=embed_dims[1],
                                       embed_dim=embed_dims[2], overlap=True)
        self.patch_embed4 = PatchEmbed(img_size=img_size // 16, patch_size=2, in_chans=embed_dims[2],
                                       embed_dim=embed_dims[3], overlap=True)

        self.d_convs1 = nn.ModuleList(
            # group 代表分组卷积不做全卷积
            [nn.Conv2d(embed_dims[0], embed_dims[0], kernel_size=3, stride=1, padding=1, groups=embed_dims[0]) for temp
             in pool_ratios[0]])
        self.d_convs2 = nn.ModuleList(
            [nn.Conv2d(embed_dims[1], embed_dims[1], kernel_size=3, stride=1, padding=1, groups=embed_dims[1]) for temp
             in pool_ratios[1]])
        self.d_convs3 = nn.ModuleList(
            [nn.Conv2d(embed_dims[2], embed_dims[2], kernel_size=3, stride=1, padding=1, groups=embed_dims[2]) for temp
             in pool_ratios[2]])
        self.d_convs4 = nn.ModuleList(
            [nn.Conv2d(embed_dims[3], embed_dims[3], kernel_size=3, stride=1, padding=1, groups=embed_dims[3]) for temp
             in pool_ratios[3]])

        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        ksize = 3

        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            pool_ratios=pool_ratios[0])
            for i in range(depths[0])])

        cur += depths[0]
        self.block2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            pool_ratios=pool_ratios[1])
            for i in range(depths[1])])

        cur += depths[1]

        self.block3 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            pool_ratios=pool_ratios[2])
            for i in range(depths[2])])

        cur += depths[2]

        self.block4 = nn.ModuleList([Block(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            pool_ratios=pool_ratios[3])
            for i in range(depths[3])])
        # print(self.block3)
        # classification head, usually not used in dense prediction tasks
        self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()
        self.gap = nn.AdaptiveAvgPool1d(1)

        self.apply(self._init_weights)

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, map_location='cpu', strict=False, logger=logger)

    def reset_drop_path(self, drop_path_rate):
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        cur = 0
        for i in range(self.depths[0]):
            self.block1[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[0]
        for i in range(self.depths[1]):
            self.block2[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[1]
        for i in range(self.depths[2]):
            self.block3[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[2]
        for i in range(self.depths[3]):
            self.block4[i].drop_path.drop_prob = dpr[cur + i]

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        # return {'pos_embed', 'cls_token'} # has pos_embed may be better
        return {'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        outs = []

        B = x.shape[0]

        # stage 1
        x, (H, W) = self.patch_embed1(x)

        for idx, blk in enumerate(self.block1):
            x = blk(x, H, W, self.d_convs1)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        outs.append(x)

        # stage 2
        x, (H, W) = self.patch_embed2(x)

        for idx, blk in enumerate(self.block2):
            x = blk(x, H, W, self.d_convs2)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        outs.append(x)

        x, (H, W) = self.patch_embed3(x)

        for idx, blk in enumerate(self.block3):
            x = blk(x, H, W, self.d_convs3)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        outs.append(x)

        # stage 4
        x, (H, W) = self.patch_embed4(x)

        for idx, blk in enumerate(self.block4):
            x = blk(x, H, W, self.d_convs4)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        outs.append(x)

        return outs

    def forward_CM(self, x):
        pass

    def forward(self, x):
        x = self.forward_features(x)

        return x


def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v

    return out_dict


'''
@BACKBONES.register_module()应该就是在mmcv里面注册成为一个backbone
'''


class p2t_tiny(PyramidPoolingTransformer):
    def __init__(self, **kwargs):
        super().__init__(
            patch_size=4, embed_dims=[48, 96, 240, 384], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 6, 3],
            drop_rate=0.0, drop_path_rate=0.1, **kwargs)


class p2t_small(PyramidPoolingTransformer):
    def __init__(self, **kwargs):
        super().__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 9, 3], mlp_ratios=[8, 8, 4, 4],
            drop_rate=0.0, drop_path_rate=0.1, **kwargs)


class p2t_base(PyramidPoolingTransformer):
    def __init__(self, **kwargs):
        super().__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 18, 3], mlp_ratios=[8, 8, 4, 4],
            drop_rate=0.0, drop_path_rate=0.3, **kwargs)


class p2t_large(PyramidPoolingTransformer):
    def __init__(self, **kwargs):
        super().__init__(
            patch_size=4, embed_dims=[64, 128, 320, 640], num_heads=[1, 2, 5, 8],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 8, 27, 3], mlp_ratios=[8, 8, 4, 4],
            drop_rate=0.0, drop_path_rate=0.3, **kwargs)


if __name__ == '__main__':
    x = torch.rand(1, 3, 224, 224)
    p2t = p2t_base()
    p2t.init_weights('../saved_models/p2t_base.pth')
    x = p2t(x)
    print(x[0].shape)
    # print(p2t)
