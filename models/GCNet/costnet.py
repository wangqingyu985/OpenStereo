import torch
import torch.nn as nn


class CostNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = Conv2dBn(in_channels=3, out_channels=32, kernel_size=5, stride=2, padding=2, dilation=1, use_relu=True)
        self.conv2 = nn.Sequential(
                StackedBlocks(n_blocks=8, in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, dilation=1)
            )
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, dilation=1, bias=False)

    def forward(self, inputs):
        output = self.conv1(inputs)
        output = self.conv2(output)
        out = self.conv3(output)
        return out


class StackedBlocks(nn.Module):

    def __init__(self, n_blocks, in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, dilation=1):
        super().__init__()
        net = []
        for i in range(n_blocks):
            net.append(ResidualBlock(in_channels=in_channels,
                                     out_channels=out_channels,
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     padding=padding,
                                     dilation=dilation
                                     )
                       )
        self.net = nn.Sequential(*net)

    def forward(self, inputs):
        out = self.net(inputs)
        return out


class ResidualBlock(nn.Module):

    def __init__(self, in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, dilation=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.net1 = Conv2dBn(in_channels=self.in_channels,
                             out_channels=self.out_channels,
                             kernel_size=self.kernel_size,
                             stride=self.stride,
                             padding=self.padding,
                             dilation=self.dilation,
                             use_relu=True
                             )
        self.net2 = self.net1

    def forward(self, inputs):
        residual = inputs
        out = self.net1(inputs)
        out = self.net2(out)
        out = out + residual
        return out


class Conv2dBn(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, use_relu=True):
        super().__init__()

        net = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=False),
               nn.BatchNorm2d(out_channels)]
        if use_relu:
            net.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*net)

    def forward(self, inputs):
        out = self.net(inputs)
        return out


if __name__ == '__main__':
    input = torch.randn(1, 3, 256, 512)  # batch size=1, channel=3, height=256, width=512
    costnet = CostNet()
    output = costnet(input)
    print(output.shape)
