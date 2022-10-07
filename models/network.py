import torch.nn as nn
import torch
from .basic import ConvBlock, DownsampleBlock, ResidualBlock, SkipConnection, TransConnection, UpsampleBlock


class HourGlass3(nn.Module):
    def __init__(self, inChannel=6, outChannel=3, resNum=4):
        super(HourGlass3, self).__init__()
        self.inConv = ConvBlock(inChannel, 64, convNum=2)
        self.down1 = nn.Sequential(*[DownsampleBlock(64, 128, withConvRelu=False), ConvBlock(128, 128, convNum=2)])
        self.down2 = nn.Sequential(
            *[DownsampleBlock(128, 256, withConvRelu=False), ConvBlock(256, 256, convNum=2)])
        self.down3 = nn.Sequential(
            *[DownsampleBlock(256, 512, withConvRelu=False), ConvBlock(512, 512, convNum=4)])
        self.residual = nn.Sequential(*[ResidualBlock(512) for _ in range(resNum)])
        self.up3 = nn.Sequential(*[UpsampleBlock(512, 256), ConvBlock(256, 256, convNum=4)])
        self.skip3 = SkipConnection(256)
        self.up2 = nn.Sequential(*[UpsampleBlock(256, 128), ConvBlock(128, 128, convNum=2)])
        self.skip2 = SkipConnection(128)
        self.up1 = nn.Sequential(*[UpsampleBlock(128, 64), ConvBlock(64, 64, convNum=2)])
        self.skip1 = SkipConnection(64)
        self.outConv = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, outChannel, kernel_size=1, padding=0)
        )

    def forward(self, x):
        f1 = self.inConv(x)
        f2 = self.down1(f1)
        f3 = self.down2(f2)
        f4 = self.down3(f3)
        r4 = self.residual(f4)
        r3 = self.skip3(self.up3(r4), f3)
        r2 = self.skip2(self.up2(r3), f2)
        r1 = self.skip1(self.up1(r2), f1)
        y = self.outConv(r1)
        return y


class SlimHourGlass3(nn.Module):
    def __init__(self, inChannel=3, outChannel=3, resNum=3):
        super(SlimHourGlass3, self).__init__()
        self.inConv = ConvBlock(inChannel, 64, convNum=2)
        self.down1 = nn.Sequential(*[DownsampleBlock(64, 128, withConvRelu=False), ConvBlock(128, 128, convNum=2)])
        self.down2 = nn.Sequential(
            *[DownsampleBlock(128, 256, withConvRelu=False), ConvBlock(256, 256, convNum=2)])
        self.down3 = nn.Sequential(
            *[DownsampleBlock(256, 512, withConvRelu=False), ConvBlock(512, 512, convNum=4)])
        self.residual = nn.Sequential(*[ResidualBlock(512) for _ in range(resNum)])
        self.up3 = nn.Sequential(*[UpsampleBlock(512, 256), ConvBlock(256, 256, convNum=4)])
        self.up2 = nn.Sequential(*[UpsampleBlock(256, 128), ConvBlock(128, 128, convNum=2)])
        self.up1 = nn.Sequential(*[UpsampleBlock(128, 64), ConvBlock(64, 64, convNum=2)])
        self.outConv = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, outChannel, kernel_size=1, padding=0)
        )

    def forward(self, x):
        f1 = self.inConv(x)
        f2 = self.down1(f1)
        f3 = self.down2(f2)
        f4 = self.down3(f3)
        r4 = self.residual(f4)
        r3 = self.up3(r4)
        r2 = self.up2(r3)
        r1 = self.up1(r2)
        y = self.outConv(r1)
        return y


class ArmedHourGlass2(nn.Module):
    def __init__(self, inChannel=3, outChannel=3, resNum=5):
        super(ArmedHourGlass2, self).__init__()
        self.inConv = nn.Conv2d(inChannel, 64, kernel_size=3, padding=1)
        self.residualBefore = nn.Sequential(*[ResidualBlock(64) for _ in range(2)])
        self.down1 = nn.Sequential(
            *[DownsampleBlock(64, 128, withConvRelu=False), ConvBlock(128, 128, convNum=2)])
        self.down2 = nn.Sequential(
            *[DownsampleBlock(128, 256, withConvRelu=False), ConvBlock(256, 256, convNum=2)])
        self.residual = nn.Sequential(*[ResidualBlock(256) for _ in range(resNum)])
        self.up2 = nn.Sequential(*[UpsampleBlock(256, 128), ConvBlock(128, 128, convNum=2)])
        self.skip2 = SkipConnection(128)
        self.up1 = nn.Sequential(*[UpsampleBlock(128, 64), ConvBlock(64, 64, convNum=2)])
        self.skip1 = SkipConnection(64)
        self.residualAfter = nn.Sequential(*[ResidualBlock(64) for _ in range(2)])
        self.outConv = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, outChannel, kernel_size=1, padding=0)
        )

    def forward(self, x):
        f1 = self.inConv(x)
        f1 = self.residualBefore(f1)
        f2 = self.down1(f1)
        f3 = self.down2(f2)
        r3 = self.residual(f3)
        r2 = self.skip2(self.up2(r3), f2)
        r1 = self.skip1(self.up1(r2), f1)
        f4 = self.residualAfter(r1)
        y = self.outConv(f4)
        return y

class HourGlass2(nn.Module):
    def __init__(self, inChannel=3, outChannel=3, resNum=5):
        super(HourGlass2, self).__init__()
        self.inConv = nn.Conv2d(inChannel, 64, kernel_size=3, padding=1)
        self.down1 = nn.Sequential(
            *[DownsampleBlock(64, 128, withConvRelu=False), ConvBlock(128, 128, convNum=2)])
        self.down2 = nn.Sequential(
            *[DownsampleBlock(128, 256, withConvRelu=False), ConvBlock(256, 256, convNum=2)])
        self.residual = nn.Sequential(*[ResidualBlock(256) for _ in range(resNum)])
        self.up2 = nn.Sequential(*[UpsampleBlock(256, 128), ConvBlock(128, 128, convNum=2)])
        self.skip2 = SkipConnection(128)
        self.up1 = nn.Sequential(*[UpsampleBlock(128, 64), ConvBlock(64, 64, convNum=2)])
        self.skip1 = SkipConnection(64)
        self.outConv = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, outChannel, kernel_size=1, padding=0)
        )

    def forward(self, x):
        f1 = self.inConv(x)
        f2 = self.down1(f1)
        f3 = self.down2(f2)
        r3 = self.residual(f3)
        r2 = self.skip2(self.up2(r3), f2)
        r1 = self.skip1(self.up1(r2), f1)
        y = self.outConv(r1)
        return y

class HourGlass2R(nn.Module):
    def __init__(self, inChannel=3, outChannel=3, resNum=5):
        super(HourGlass2R, self).__init__()
        self.inConv = nn.Conv2d(inChannel, 64, kernel_size=3, padding=1)
        self.down1 = nn.Sequential(
            *[DownsampleBlock(64, 128, withConvRelu=False), ConvBlock(128, 128, convNum=2)])
        self.down2 = nn.Sequential(
            *[DownsampleBlock(128, 256, withConvRelu=False), ConvBlock(256, 256, convNum=2)])
        self.residual = nn.Sequential(*[ResidualBlock(256) for _ in range(resNum)])
        self.up2 = nn.Sequential(*[UpsampleBlock(256, 128), ConvBlock(128, 128, convNum=2)])
        self.skip2 = SkipConnection(128)
        self.up1 = nn.Sequential(*[UpsampleBlock(128, 64), ConvBlock(64, 64, convNum=2)])
        self.skip1 = SkipConnection(64)
        self.outConv = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, outChannel, kernel_size=1, padding=0)
        )

    def forward(self, x):
        f1 = self.inConv(x)
        f2 = self.down1(f1)
        f3 = self.down2(f2)
        r3 = self.residual(f3)
        r2 = self.skip2(self.up2(r3), f2)
        r1 = self.skip1(self.up1(r2), f1)
        y = self.outConv(r1)
        return y

        
class ResNet(nn.Module):
    def __init__(self, inChannel=3, outChannel=3, resNum=5):
        super(ResNet, self).__init__()
        self.inConv = nn.Conv2d(inChannel, 64, kernel_size=3, padding=1)
        self.residual = nn.Sequential(*[ResidualBlock(64) for _ in range(resNum)])
        self.outConv = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, outChannel, kernel_size=1, padding=0)
        )

    def forward(self, x):
        f1 = self.inConv(x)
        f2 = self.residual(f1)
        y = self.outConv(f2)
        return y


class WarpNet(nn.Module):
    def __init__(self, inChannel=1, outChannel=3, resNum=5):
        super(WarpNet, self).__init__()
        self.inConv = nn.Conv2d(inChannel, 64, kernel_size=3, padding=1)
        self.dispResidual = nn.Sequential(*[ResidualBlock(64) for _ in range(resNum)])
        self.occResidual = nn.Sequential(*[ResidualBlock(64) for _ in range(2)])
        self.disp_out = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, kernel_size=1, padding=0),
            nn.Tanh()
        )
        self.occ_out = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1, padding=0),
            nn.Tanh()
        )

    def forward(self, meta_map):
        f1 = self.inConv(meta_map)
        f2 = self.dispResidual(f1)
        disp_map = self.disp_out(f2)
        occ_map = self.occ_out(self.occResidual(f2))
        return disp_map, occ_map

        
class DispOccNet(nn.Module):
    def __init__(self, inChannel=3, outChannel=3, resNum=3):
        super(DispOccNet, self).__init__()
        self.residul_trans = ResNet(256, 256, resNum=2)
        #! decoder for DispNet
        self.disp_up2 = UpsampleBlock(256, 128)
        self.disp_skip2 = SkipConnection(128)
        self.disp_trans2 = ConvBlock(128, 128, convNum=2)
        self.disp_up1 = UpsampleBlock(128, 64)
        self.disp_skip1 = SkipConnection(64)
        self.disp_trans1 = ConvBlock(64, 64, convNum=2)
        self.disp_out = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, kernel_size=1, padding=0),
            nn.Tanh()
        )
        #! decoder for OccNet
        self.occ_up2 = UpsampleBlock(256, 128)
        self.occ_skip2 = TransConnection(128)
        self.occ_up1 = UpsampleBlock(128, 64)
        self.occ_skip1 = TransConnection(64)
        self.occ_out = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1, padding=0),
            nn.Tanh()
        )

    def forward(self, f64, f128, f256):
        f1 = self.residul_trans(f256)
        f2_a = self.disp_skip2(self.disp_up2(f1), f128)
        f2_b = self.disp_trans2(f2_a)
        f3_a = self.disp_skip1(self.disp_up1(f2_b), f64)
        f3_b = self.disp_trans1(f3_a)
        disp_map = self.disp_out(f3_b)

        occ_f2 = self.occ_skip2(self.occ_up2(f1), f2_a)
        occ_f3 = self.occ_skip1(self.occ_up1(occ_f2), f3_a)
        occ_map = self.occ_out(occ_f3)
        return disp_map, occ_map

        
class FillNet(nn.Module):
    def __init__(self, inChannel=3, outChannel=3, resNum=3):
        super(FillNet, self).__init__()
        self.inConv_img = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.down1 = DownsampleBlock(64, 128, withConvRelu=False)
        self.down2 = DownsampleBlock(128, 256, withConvRelu=False)
        #! decoder for FillNet
        self.combine = SkipConnection(256)
        self.residul_trans = ResNet(256, 256, resNum=2)
        self.up2 = UpsampleBlock(256, 128)
        self.skip2 = TransConnection(128)
        self.up1 = UpsampleBlock(128, 64)
        self.skip1 = TransConnection(64)
        self.post_trans = ConvBlock(64, 64, convNum=2)
        self.conv_out = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=1, padding=0),
            nn.Tanh()
        )

    def forward(self, i_warped, f64, f128, f256):
        f64_in = self.inConv_img(i_warped)
        f128_in = self.down1(f64_in)
        f256_in = self.down2(f128_in)
        f1 = self.residul_trans(self.combine(f256, f256_in))
        f2 = self.skip2(self.up2(f1), f128)
        f3 = self.skip1(self.up1(f2), f64)
        f4 = self.post_trans(f3)
        disocc_map = self.conv_out(f4)
        return disocc_map


class RefineNet(nn.Module):
    def __init__(self, inChannel=3, outChannel=3, resNum=3):
        super(RefineNet, self).__init__()
        self.inConv1 = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=1, padding=0)
        )
        self.inConv2 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.ConvA = nn.Conv2d(128, 64, kernel_size=1, padding=0)
        self.down1 = nn.Sequential(
            *[DownsampleBlock(64, 128, withConvRelu=False), ConvBlock(128, 128, convNum=2)])
        self.down2 = nn.Sequential(
            *[DownsampleBlock(128, 256, withConvRelu=False), ConvBlock(256, 256, convNum=2)])
        self.residual = nn.Sequential(*[ResidualBlock(256) for _ in range(resNum)])
        self.up2 = nn.Sequential(*[UpsampleBlock(256, 128), ConvBlock(128, 128, convNum=2)])
        self.skip2 = SkipConnection(128)
        self.up1 = nn.Sequential(*[UpsampleBlock(128, 64), ConvBlock(64, 64, convNum=2)])
        self.skip1 = SkipConnection(64)
        self.residual_at = ResidualBlock(256)
        self.up2_at = nn.Sequential(*[UpsampleBlock(256, 128), ConvBlock(128, 128, convNum=2)])
        self.skip2_at = SkipConnection(128)
        self.up1_at = nn.Sequential(*[UpsampleBlock(128, 64), ConvBlock(64, 64, convNum=2)])
        self.skip1_at = SkipConnection(64)
        self.outConv = nn.Conv2d(64, outChannel, kernel_size=1, padding=0)

    def forward(self, c_view, c_edit, meta_map):
        f1_a= self.inConv1(torch.cat((c_view, c_edit), dim=1))
        f1_b = self.inConv2(meta_map)
        f1 = self.convA(torch.cat((f1_a, f1_b), dim=1))
        #! feature learning branch
        d1 = self.down1(f1)
        d2 = self.down2(d1)
        r3 = self.residual(d2)
        u2 = self.skip2(self.up2(r3), d1)
        u1 = self.skip1(self.up1(u2), f1)
        #! attention learning branch
        r3_at = self.residual(d2)
        u2_at = self.skip2_at(self.up2_at(r3_at), d1)
        u1_at = self.skip1_at(self.up1_at(u2_at), f1)
        y = self.outConv(u1*u1_at)
        return y