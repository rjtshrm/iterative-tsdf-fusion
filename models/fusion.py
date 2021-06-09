import torch
import torch.nn as nn

class Fusion3D(nn.Module):
    
    def __init__(self, bn=True):
        super(Fusion3D, self).__init__()

        self.bn = bn

        self.block1 = self.encoder(in_channels=1)
        self.block2 = self.encoder(in_channels=3) # Feature: input = 2 + 1, output = 9
        self.block3 = self.encoder(in_channels=9) # Feature: input = 6 + 3, output = 27
        self.block4 = self.encoder(in_channels=27) # Feature: input = 18 + 9, output = 81

        # feature reducer block
        self.block5 = self.decoder(81) # Feature: input = 81, output = 27
        self.block6 = self.decoder(27) # iFeature: nput = 27, output = 9
        self.block7 = self.decoder(9)  # Feature: input = 9, output = 3

        self.final_block = self.final_decoder(3) # Feature: input = 3, output = 2 (global tsdf, global weights)

    def encoder(self, in_channels):
        if self.bn:
            t = nn.Sequential(
                nn.Conv3d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm3d(in_channels),
                nn.LeakyReLU(),
                nn.Dropout3d(p=0.2),
                nn.Conv3d(in_channels, in_channels * 2, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm3d(in_channels * 2),
                nn.LeakyReLU(),
                nn.Dropout3d(p=0.2)
            )
        else:
            t = nn.Sequential(
                nn.Conv3d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(),
                nn.Conv3d(in_channels, in_channels * 2, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(),
            )

        return t

    def decoder(self, in_channels):
        if self.bn:
            t = nn.Sequential(
                nn.Conv3d(in_channels, in_channels // 3, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm3d(in_channels // 3),
                nn.LeakyReLU(),
                nn.Dropout3d(p=0.2),
                nn.Conv3d(in_channels // 3, in_channels // 3, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm3d(in_channels // 3),
                nn.LeakyReLU(),
                nn.Dropout3d(p=0.2)
            )
        else:
            t = nn.Sequential(
                nn.Conv3d(in_channels, in_channels // 3, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(),
                nn.Conv3d(in_channels // 3, in_channels // 3, kernel_size=1, stride=1, padding=0),
                nn.LeakyReLU(),
            )

        return t

    def final_decoder(self, in_channels):
        if self.bn:
            t = nn.Sequential(
                nn.Conv3d(in_channels, in_channels // 3, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm3d(in_channels // 3),
                nn.LeakyReLU(),
                nn.Dropout3d(p=0.2),
                nn.Conv3d(in_channels // 3, in_channels // 3, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm3d(in_channels // 3),
                nn.LeakyReLU(),
                nn.Conv3d(in_channels // 3, 1 + in_channels // 3, kernel_size=1, stride=1, padding=0),
                nn.Tanh()
            )
        else:
            t = nn.Sequential(
                nn.Conv3d(in_channels, in_channels // 3, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(),
                nn.Conv3d(in_channels // 3, in_channels // 3, kernel_size=1, stride=1, padding=0),
                nn.LeakyReLU(),
                nn.Conv3d(in_channels // 3, 1 + in_channels // 3, kernel_size=1, stride=1, padding=0),
                nn.Tanh()
            )

        return t

    def feature_extractor(self, tsdf_vol):
        x1 = self.block1(tsdf_vol)
        x1 = torch.cat([x1, tsdf_vol], dim=1)
        x2 = self.block2(x1)
        x2 = torch.cat([x2, x1], dim=1)
        x3 = self.block3(x2)
        x3 = torch.cat([x3, x2], dim=1)
        x4 = self.block4(x3)
        x4 = torch.cat([x4, x3], dim=1)

        return x4

    def forward(self, input_tsdf_vol, input_tsdf_weight, global_tsdf_vol, global_tsdf_weights):

        input_tsdf_feat = self.feature_extractor(input_tsdf_vol)
        global_tsdf_feat = self.feature_extractor(global_tsdf_vol)

        total_weight = input_tsdf_weight + global_tsdf_weights
        accumlated_features = (input_tsdf_feat * input_tsdf_weight + global_tsdf_feat * global_tsdf_weights) / total_weight

        d1 = self.block5(accumlated_features)
        d2 = self.block6(d1)
        d3 = self.block7(d2)

        o = self.final_block(d3)
        print(o.shape)



if __name__ == "__main__":
    i = torch.zeros((1, 1, 25, 29, 21))
    m = Fusion3D(bn=True)
    m(i, i, i, i)
