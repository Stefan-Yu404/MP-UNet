import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self, in_channel=1, out_channel=2, training=True):
        super(UNet, self).__init__()
        self.training = training
        self.encoder1 = nn.Conv3d(in_channel, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.encoder2 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.encoder3 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.encoder4 = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))

        self.decoder1 = nn.Conv3d(256, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.decoder2 = nn.Conv3d(128, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.decoder3 = nn.Conv3d(64, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.decoder4 = nn.Conv3d(32, 2, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        
        self.map4 = nn.Sequential(
            nn.Conv3d(2, out_channel, kernel_size=(1, 1, 1), stride=(1, 1, 1)),
            nn.Upsample(scale_factor=(1, 1, 1), mode='trilinear', align_corners=False),
            nn.Softmax(dim=1)
        )

        # Mapping at 128*128 scale.
        self.map3 = nn.Sequential(
            nn.Conv3d(64, out_channel, kernel_size=(1, 1, 1), stride=(1, 1, 1)),
            nn.Upsample(scale_factor=(4, 4, 4), mode='trilinear', align_corners=False),
            nn.Softmax(dim=1)
        )

        # Mapping at 64*64 scale.
        self.map2 = nn.Sequential(
            nn.Conv3d(128, out_channel, kernel_size=(1, 1, 1), stride=(1, 1, 1)),
            nn.Upsample(scale_factor=(8, 8, 8), mode='trilinear', align_corners=False),
            nn.Softmax(dim=1)
        )

        # Mapping at 32*32 scale.
        self.map1 = nn.Sequential(
            nn.Conv3d(256, out_channel, kernel_size=(1, 1, 1), stride=(1, 1, 1)),
            nn.Upsample(scale_factor=(16, 16, 16), mode='trilinear', align_corners=False),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        out = F.relu(F.max_pool3d(self.encoder1(x), 2, 2))
        t1 = out
        out = F.relu(F.max_pool3d(self.encoder2(out), 2, 2))
        t2 = out
        out = F.relu(F.max_pool3d(self.encoder3(out), 2, 2))
        t3 = out
        out = F.relu(F.max_pool3d(self.encoder4(out), 2, 2))

        output1 = self.map1(out)

        out = F.relu(F.interpolate(self.decoder1(out), scale_factor=(2, 2, 2), mode='trilinear', align_corners=False))
        out = torch.add(out, t3)
        output2 = self.map2(out)

        out = F.relu(F.interpolate(self.decoder2(out), scale_factor=(2, 2, 2), mode='trilinear', align_corners=False))
        out = torch.add(out, t2)
        output3 = self.map3(out)

        out = F.relu(F.interpolate(self.decoder3(out), scale_factor=(2, 2, 2), mode='trilinear', align_corners=False))
        out = torch.add(out, t1)
        out = F.relu(F.interpolate(self.decoder4(out), scale_factor=(2, 2, 2), mode='trilinear', align_corners=False))
        output4 = self.map4(out)

        if self.training is True:
            return output1, output2, output3, output4
        else:
            return output4
