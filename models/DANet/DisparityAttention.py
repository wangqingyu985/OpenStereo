import torch
from torch import nn
from torch.nn import init


class DisparityAttention(nn.Module):
    def __init__(self, disparity, reduction=16):
        super().__init__()
        self.maxpool = nn.AdaptiveMaxPool3d(1)
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.da = nn.Sequential(
            nn.Conv3d(in_channels=disparity, out_channels=disparity // reduction, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv3d(in_channels=disparity // reduction, out_channels=disparity, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.transpose(input=x, dim0=1, dim1=2)  # [B, D, C, H, W]
        max_out = self.da(self.maxpool(x))  # [B, D, 1, 1, 1]
        avg_out = self.da(self.avgpool(x))  # [B, D, 1, 1, 1]
        output = self.sigmoid(max_out + avg_out)  # [B, D, 1, 1, 1]
        output = torch.transpose(input=output, dim0=1, dim1=2)  # [B, 1, D, 1, 1]
        return output


class DABlock(nn.Module):
    def __init__(self, disparity=256, reduction=16):
        super().__init__()
        self.da = DisparityAttention(disparity=disparity, reduction=reduction)

    def forward(self, x):
        out = x*self.da(x)  # [B, C, D, H, W]
        return out+x

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)


if __name__ == '__main__':
    input = torch.randn(50, 512, 256, 7, 7)  # batch size=50, channel=512, disparity=256, height=7, width=7
    da = DABlock(disparity=256, reduction=16)
    output = da(input)
    print(output.shape)
