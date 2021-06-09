"""
 Depth + Confidence estimation network
 DL model for U-NET refer: https://arxiv.org/pdf/1505.04597.pdf
"""

import torch
import torch.nn as nn


class UNET(nn.Module):

    def __init__(self, bn=True):
        super(UNET, self).__init__()
        self.bn = bn

        self.max_pool = nn.MaxPool2d(2)

        self.down1 = self.single_u_net_down_module(1, 64)
        self.down2 = self.single_u_net_down_module(64, 128)
        self.down3 = self.single_u_net_down_module(128, 256)
        self.down4 = self.single_u_net_down_module(256, 512)
        self.down5 = self.single_u_net_down_module(512, 1024)

        self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.up4 = self.single_u_net_down_module(1024 + 512, 512)
        self.up3 = self.single_u_net_down_module(512 + 256, 256)
        self.up2 = self.single_u_net_down_module(256 + 128, 128)
        self.up1 = self.single_u_net_down_module(128 + 64, 64)

        self.final_layer = nn.Conv2d(64, 2, kernel_size=1, stride=1, padding=0)

    def single_u_net_down_module(self, in_channels, out_channels):
        if self.bn:
            t = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )
        else:
            t = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.ReLU()
            )

        return t

    def single_u_net_up_module(self, in_channels, out_channels):
        if self.bn:
            t = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )
        else:
            t = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.ReLU()
            )

        return t

    def forward(self, input):
        #### Down modules ####
        d1 = self.down1(input)
        d1_mp = self.max_pool(d1)
        d2 = self.down2(d1_mp)
        d2_mp = self.max_pool(d2)
        d3 = self.down3(d2_mp)
        d3_mp = self.max_pool(d3)
        d4 = self.down4(d3_mp)
        d4_mp = self.max_pool(d4)
        d5 = self.down5(d4_mp)

        #### Up modules ####
        u4_up = self.up_sample(d5)
        u4 = self.up4(torch.cat([u4_up, d4], dim=1))
        u3_up = self.up_sample(u4)
        u3 = self.up3(torch.cat([u3_up, d3], dim=1))
        u2_up = self.up_sample(u3)
        u2 = self.up2(torch.cat([u2_up, d2], dim=1))
        u1_up = self.up_sample(u2)
        u1 = self.up1(torch.cat([u1_up, d1], dim=1))

        output = self.final_layer(u1)

        print(output.shape)



if __name__ == "__main__":
    i = torch.zeros((1, 1, 640, 640)).to(torch.device("cuda"))
    m = UNET().to(torch.device("cuda"))
    m(i)
