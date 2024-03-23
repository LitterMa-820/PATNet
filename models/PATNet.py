from models.baseBlocks import CBR, P2tBackbone
from models.ShareAttCrossAttention import *
from models.cross_scale_attention import *


class p3Net(nn.Module):
    def __init__(self, path=None):
        super().__init__()
        layer_scale_init_value = 1e-6
        self.backbone = P2tBackbone(path)
        self.CrossAttention4 = CrossAttentionBlock(512, 8)

        self.CrossAttention3 = CrossAttentionBlock(320, 5)
        self.conv3_2 = CBR(320, 320, 3, 1, 1, act=nn.PReLU())
        self.csf3 = CrossScaleAttnBlock(320, 512, 5)

        self.CrossAttention2 = CrossAttentionBlock(128, 2)
        self.conv2_2 = CBR(128, 128, 3, 1, 1, act=nn.PReLU())
        self.csf2 = CrossScaleAttnBlock(128, 320, 2)

        self.CrossAttention1 = CrossAttentionBlock(64, 1)
        self.conv1_2 = CBR(64, 64, 3, 1, 1, act=nn.PReLU())
        self.csf1 = CrossScaleAttnBlock(64, 128, 1)

        self.side_out4 = CBR(512, 1, 3, 1, 1, act=nn.PReLU())
        self.side_out3 = CBR(320, 1, 3, 1, 1, act=nn.PReLU())
        self.side_out2 = CBR(128, 1, 3, 1, 1, act=nn.PReLU())
        self.ps_out = nn.PixelShuffle(4)
        self.out_conv = CBR(64, 16, 3, 1, 1, act=nn.PReLU())
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((64)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None

    def forward(self, x, y):
        x_out, y_out = self.backbone(x, y)
        x1, x2, x3, x4 = x_out
        y1, y2, y3, y4 = y_out
        # layer 4
        fuse4 = self.CrossAttention4(x4, y4)
        side_out4 = self.side_out4(fuse4)

        # layer3
        B, C, H, W = x3.shape
        fuse3 = self.CrossAttention3(x3, y3)
        cs_fuse3 = self.csf3(fuse3, fuse4)
        cs_fuse3 = cs_fuse3.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        fuse3 = self.conv3_2(fuse3 * cs_fuse3)
        side_out3 = self.side_out3(fuse3)

        # layer 2
        B, C, H, W = x2.shape
        fuse2 = self.CrossAttention2(x2, y2)
        cs_fuse2 = self.csf2(fuse2, fuse3)
        cs_fuse2 = cs_fuse2.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        fuse2 = self.conv2_2(fuse2 * cs_fuse2)
        side_out2 = self.side_out2(fuse2)

        # layer1
        B, C, H, W = x1.shape
        fuse1 = self.CrossAttention1(x1, y1)
        cs_fuse1 = self.csf1(fuse1, fuse2)
        cs_fuse1 = cs_fuse1.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        fuse1 = self.conv1_2(fuse1 * cs_fuse1)
        fuse1 = fuse1.reshape(B, C, -1).permute(0, 2, 1)
        fuse1 = self.gamma * fuse1
        fuse1 = fuse1.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        side_out1 = self.out_conv(fuse1)
        side_out1 = self.ps_out(side_out1)
        return side_out4, side_out3, side_out2, side_out1
