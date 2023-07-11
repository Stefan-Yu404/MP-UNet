import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class AttentionBlock(nn.Module):
    def __init__(self, in_channels_x, in_channels_g, int_channels):
        super(AttentionBlock, self).__init__()
        self.Wx = nn.Sequential(nn.Conv3d(in_channels_x, int_channels, kernel_size=(1, 1, 1)),
                                nn.BatchNorm3d(int_channels))
        self.Wg = nn.Sequential(nn.Conv3d(in_channels_g, int_channels, kernel_size=(1, 1, 1)),
                                nn.BatchNorm3d(int_channels))
        self.psi = nn.Sequential(nn.Conv3d(int_channels, 1, kernel_size=(1, 1, 1)),
                                 nn.BatchNorm3d(1),
                                 nn.Sigmoid())

    def forward(self, x, g):
        """"
        Apply the Wx to the skip connection
        """
        x1 = self.Wx(x)
        # after applying Wg to the input, upsample to the size of the skip connection
        g1 = nn.functional.interpolate(self.Wg(g), x1.shape[2:], mode='trilinear', align_corners=False)
        out = self.psi(nn.ReLU()(x1 + g1))
        return out * x


class PositionalEncoding3D(nn.Module):
    def __init__(self, channels):
        """
        param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding3D, self).__init__()
        channels = int(np.ceil(channels / 2))
        self.channels = channels
        inv_freq = 1. / (10000**(torch.arange(0, channels, 2).float() / channels))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, tensor):
        """
        param tensor: A 5d tensor of size (batch_size, ch, x, y, z)
        return: Positional Encoding Matrix of size (batch_size, ch, x, y, z)
        """
        if len(tensor.shape) != 5:
            raise RuntimeError("The input tensor has to be 5d!")
        batch_size, x, y, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        pos_y = torch.arange(y, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1).unsqueeze(1)
        emb_y = torch.cat((sin_inp_y.sin(), sin_inp_y.cos()), dim=-1)
        emb = torch.zeros((x, y, self.channels * 2),
                          device=tensor.device).type(tensor.type())
        emb[:, :, :self.channels] = emb_x
        emb[:, :, self.channels:2 * self.channels] = emb_y

        return emb[None, :, :, :orig_ch].repeat(batch_size, 1, 1, 1)


class PositionalEncodingPermute3D(nn.Module):
    def __init__(self, channels):
        """
        Accepts (batchsize, ch, x, y, z) instead of (batchsize, x, y, z, ch)
        """
        super(PositionalEncodingPermute3D, self).__init__()
        self.penc = PositionalEncoding3D(channels)

    def forward(self, tensor):
        tensor = tensor.permute(0, 2, 3, 1)
        enc = self.penc(tensor)
        return enc.permute(0, 3, 1, 2)


class PamCell(nn.Module):
    """
    Position attention module
    """
    def __init__(self, in_dim):
        super(PamCell, self).__init__()
        self.channel_in = in_dim

        self.query_conv = nn.Conv3d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=(1, 1, 1))
        self.key_conv = nn.Conv3d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=(1, 1, 1))
        self.value_conv = nn.Conv3d(in_channels=in_dim, out_channels=in_dim, kernel_size=(1, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        inputs :
            x : input feature maps( B x C x H x W x L)
        returns :
            out : attention value + input feature
            attention: B x (HxWxL) x (HxWxL)
        """
        m_batchsize, channel, height, width, length = x.size()

        # Equivalent to calling PositionalEncoding3D.
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height * length).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height * length)

        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height * length)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, channel, height, width, length)

        out = self.gamma * out + x
        return out


class PamLayer(nn.Module):
    """
    Apply PAM attention
    input:
        in_ch : input channels
        use_pam : Boolean value whether to use PAM_Module or CAM_Module
    output:
        returns the attention map
    """
    def __init__(self, in_ch):
        super(PamLayer, self).__init__()

        self.attn = nn.Sequential(
            nn.Conv3d(in_ch * 2, in_ch, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(in_ch),
            nn.PReLU(),
            # nn.ELU(),

            PamCell(in_ch),
            nn.Conv3d(in_ch, in_ch, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(in_ch),
            nn.PReLU(),
            # nn.ELU(),
        )

    def forward(self, x):
        return self.attn(x)


class Encoder(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.encoder_stage = nn.Sequential(
            nn.Conv3d(in_channel, out_channel, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.PReLU(out_channel),

            nn.Conv3d(out_channel, out_channel, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.PReLU(out_channel),
        )

    def forward(self, x):
        return self.encoder_stage(x)


class Decoder(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.decoder_stage = nn.Sequential(
            nn.Conv3d(in_channel, out_channel, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.PReLU(out_channel),

            nn.Conv3d(out_channel, out_channel, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.PReLU(out_channel),

            nn.Conv3d(out_channel, out_channel, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.PReLU(out_channel),
        )

    def forward(self, x):
        return self.decoder_stage(x)


class Down(nn.Module):
    def __init__(self, in_channel, out_channel, k, s, p):
        super().__init__()
        self.down_conv = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.Conv3d(in_channel, out_channel, kernel_size=(k, k, k), stride=(s, s, s), padding=(p, p, p)),
            nn.BatchNorm3d(out_channel),
            # nn.PReLU(out_channel),
            nn.ELU(out_channel),
        )

    def forward(self, x):
        return self.down_conv(x)


class Up(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.up_conv = nn.Sequential(
            nn.ConvTranspose3d(in_channel, out_channel, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            nn.PReLU(out_channel)
        )

    def forward(self, x):
        return self.up_conv(x)


class Map(nn.Module):
    def __init__(self, in_channel, out_channel, a, b, c):
        super().__init__()
        self.map_conv = nn.Sequential(
            nn.Conv3d(in_channel, out_channel, kernel_size=(1, 1, 1), stride=(1, 1, 1)),
            nn.Upsample(scale_factor=(a, b, c), mode='trilinear', align_corners=False),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.map_conv(x)


class MPUNet(nn.Module):
    def __init__(self, in_channel=1, out_channel=2, training=True):
        super().__init__()

        self.training = training
        self.dorp_rate = 0.2

        self.encoder1 = Encoder(in_channel, 16)
        self.encoder2 = Encoder(32, 32)
        self.encoder3 = Encoder(64, 64)
        self.encoder4 = Encoder(128, 128)
        self.encoder5 = Encoder(256, 256)

        self.decoder1 = Decoder(128, 256)
        self.decoder2 = Decoder(128 + 64, 128)
        self.decoder3 = Decoder(64 + 32, 64)
        self.decoder4 = Decoder(32 + 16, 32)
        self.decoder5 = Decoder(8, 2)

        self.down1 = Down(16, 32, 2, 2, 0)
        self.down2 = Down(32, 64, 2, 2, 0)
        self.down3 = Down(64, 128, 2, 2, 0)
        self.down4 = Down(128, 256, 3, 1, 1)
        self.down5 = Down(256, 512, 3, 1, 1)

        self.up1 = Up(512, 256)
        self.up2 = Up(256, 128)
        self.up3 = Up(128, 64)
        self.up4 = Up(64, 32)

        factor = 1
        self.map5 = Map(2 * factor, out_channel, 1, 1, 1)  # The final large-scale mapping (256*256).
        self.map4 = Map(32 * factor, out_channel, 1, 1, 1)  # Mapping (256*256).
        self.map3 = Map(64 * factor, out_channel, 2, 2, 2)  # Mapping (128*128).
        self.map2 = Map(128 * factor, out_channel, 4, 4, 4)  # Mapping (64*64).
        self.map1 = Map(256 * factor, out_channel, 8, 8, 8)  # Mapping (32*32).

        self.Att4 = AttentionBlock(256, 256, 128)
        self.Att3 = AttentionBlock(128, 128, 64)
        self.Att2 = AttentionBlock(64, 64, 32)
        self.Att1 = AttentionBlock(32, 32, 16)

        self.PCL = PamLayer(256)

    def forward(self, inputs):
        long_range1 = self.encoder1(inputs) + inputs
        short_range1 = self.down1(long_range1)

        long_range2 = self.encoder2(short_range1) + short_range1
        long_range2 = F.dropout(long_range2, self.dorp_rate, self.training)
        short_range2 = self.down2(long_range2)

        long_range3 = self.encoder3(short_range2) + short_range2
        long_range3 = F.dropout(long_range3, self.dorp_rate, self.training)
        short_range3 = self.down3(long_range3)

        long_range4 = self.encoder4(short_range3) + short_range3
        long_range4 = F.dropout(long_range4, self.dorp_rate, self.training)
        short_range4 = self.down4(long_range4)

        long_range5 = self.encoder5(short_range4) + short_range4
        long_range5 = F.dropout(long_range5, self.dorp_rate, self.training)
        short_range5 = self.down5(long_range5)
        short_range5 = self.PCL(short_range5)

        outputs = self.decoder1(long_range4) + short_range5
        outputs = self.Att4(x=outputs, g=short_range5)
        outputs = F.dropout(outputs, self.dorp_rate, self.training)
        output1 = self.map1(outputs)

        short_range6 = self.up2(outputs)
        outputs = self.decoder2(torch.cat([short_range6, long_range3], dim=1)) + short_range6
        outputs = self.Att3(x=outputs, g=short_range6)
        outputs = F.dropout(outputs, self.dorp_rate, self.training)
        output2 = self.map2(outputs)

        short_range7 = self.up3(outputs)
        outputs = self.decoder3(torch.cat([short_range7, long_range2], dim=1)) + short_range7
        outputs = self.Att2(x=outputs, g=short_range7)
        outputs = F.dropout(outputs, self.dorp_rate, self.training)
        output3 = self.map3(outputs)

        short_range8 = self.up4(outputs)
        outputs = self.decoder4(torch.cat([short_range8, long_range1], dim=1)) + short_range8
        outputs = self.Att1(x=outputs, g=short_range8)
        output4 = self.map4(outputs)

        output4 = (output1 + output2 + output3 + output4) / 4

        if self.training is True:
            return output1, output2, output3, output4
        else:
            return output4
