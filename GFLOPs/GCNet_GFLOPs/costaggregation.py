import torch
import torch.nn as nn
import torch.nn.functional as F


class CostAggregation(nn.Module):
    """
    inputs --- [B, 64, 1/2D, 1/2H, 1/2W]
    """
    def __init__(self, max_disp):
        super().__init__()
        self.net19 = Conv3dBn(64, 32, 3, 1, 1, 1, True)
        self.net20 = Conv3dBn(32, 32, 3, 1, 1, 1, True)
        self.net21 = Conv3dBn(64, 64, 3, 2, 1, 1, True)  # down sample
        self.net22 = Conv3dBn(64, 64, 3, 1, 1, 1, True)
        self.net23 = Conv3dBn(64, 64, 3, 1, 1, 1, True)
        self.net24 = Conv3dBn(64, 64, 3, 2, 1, 1, True)  # down sample
        self.net25 = Conv3dBn(64, 64, 3, 1, 1, 1, True)
        self.net26 = Conv3dBn(64, 64, 3, 1, 1, 1, True)
        self.net27 = Conv3dBn(64, 64, 3, 2, 1, 1, True)  # down sample
        self.net28 = Conv3dBn(64, 64, 3, 1, 1, 1, True)
        self.net29 = Conv3dBn(64, 64, 3, 1, 1, 1, True)
        self.net30 = Conv3dBn(64, 128, 3, 2, 1, 1, True)  # down sample
        self.net31 = Conv3dBn(128, 128, 3, 1, 1, 1, True)
        self.net32 = Conv3dBn(128, 128, 3, 1, 1, 1, True)
        self.net33 = ConvTranspose3dBn(128, 64, 3, 2, 1, 1, True)  # up sample
        self.net34 = ConvTranspose3dBn(64, 64, 3, 2, 1, 1, True)  # up sample
        self.net35 = ConvTranspose3dBn(64, 64, 3, 2, 1, 1, True)  # up sample
        self.net36 = ConvTranspose3dBn(64, 32, 3, 2, 1, 1, True)  # up sample
        self.net37 = nn.ConvTranspose3d(in_channels=32, out_channels=1, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.regression = DisparityRegression(max_disp)

    def forward(self, inputs):
        out21 = self.net21(inputs)
        out24 = self.net24(out21)
        out27 = self.net27(out24)
        out37 = self.net37(self.net36(self.net35(self.net34(self.net33(self.net32(self.net31(self.net30(out27)))) + self.net29(self.net28(out27))) + self.net26(self.net25(out24))) + self.net23(self.net22(out21))) + self.net20(self.net19(inputs)))
        # [B, 1, D, H, W]
        cost = out37.squeeze(1)  # [B, D, H, W]
        prob = F.softmax(-cost, dim=1)  # [B, D, H, W]
        disp = self.regression(prob)  # [B, H, W]
        return disp


class DisparityRegression(nn.Module):

    def __init__(self, max_disp):
        super().__init__()

        self.disp_score = torch.range(0, max_disp - 1)  # [D]
        self.disp_score = self.disp_score.unsqueeze(0).unsqueeze(2).unsqueeze(3)  # [1, D, 1, 1]

    def forward(self, prob):
        disp_score = self.disp_score.expand_as(prob).type_as(prob)  # [B, D, H, W]
        out = torch.sum(disp_score * prob, dim=1)  # [B, H, W]
        return out


class Conv3dBn(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1, use_relu=True):
        super().__init__()

        net = [nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=False),
               nn.BatchNorm3d(out_channels)]
        if use_relu:
            net.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*net)

    def forward(self, inputs):
        out = self.net(inputs)
        return out


class ConvTranspose3dBn(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, output_padding=0, use_relu=True):
        super().__init__()

        net = [nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=False),
               nn.BatchNorm3d(out_channels)]
        if use_relu:
            net.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*net)

    def forward(self, inputs):
        out = self.net(inputs)
        return out


if __name__ == '__main__':
    input = torch.randn(1, 64, 256, 128, 256)  # batch size=1, channel=64, disparity=256, height=128, width=256
    cost_aggregation = CostAggregation(256)
    output = cost_aggregation(input)
    print(output.shape)
