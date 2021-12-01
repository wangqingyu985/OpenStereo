import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def CostVolume(input_feature, candidate_feature, position="left", method="subtract", k=4, batch_size=4, channel=32, D=256, H=256, W=512):
    """
    Some parameters:
        position: means whether the input feature img is left or right
        k: the conv counts of the first stage, the feature extraction stage
    """
    origin = input_feature  # img shape: [bs, c, H // 2**k, W // 2**k]
    candidate = candidate_feature
    if position == "left":
        leftMinusRightMove_List = []
        for disparity in range(D // 2**k):
            if disparity == 0:
                if method == "subtract":
                    leftMinusRightMove = origin - candidate  # subtract method
                else:
                    leftMinusRightMove = torch.cat((origin, candidate), dim=1)  # concat mathod
                leftMinusRightMove_List.append(leftMinusRightMove)
            else:
                zero_padding = np.zeros((origin.shape[0], channel, origin.shape[2], disparity))
                zero_padding = torch.from_numpy(zero_padding).float()
                zero_padding = zero_padding.cuda()
                left_move = torch.cat((origin, zero_padding), dim=3)
                if method == "subtract":
                    leftMinusRightMove = left_move[:, :, :, :origin.shape[3]] - candidate
                else:
                    leftMinusRightMove = torch.cat((left_move[:, :, :, :origin.shape[3]], candidate), 1)  # concat the channels
                leftMinusRightMove_List.append(leftMinusRightMove)
        cost_volume = torch.stack(leftMinusRightMove_List, dim=1)  # [batch_size, count(disparitys), channel, H, W]
        return cost_volume


class StereoNet(nn.Module):
    def __init__(self, batch_size, cost_volume_method):
        super(StereoNet, self).__init__()

        self.batch_size = batch_size
        self.cost_volume_method = cost_volume_method

        cost_volume_channel = 32
        if cost_volume_method == "subtract":
            cost_volume_channel = 32
        elif cost_volume_method == "concat":
            cost_volume_channel = 64
        else:
            print("cost_volume_method is not right")

        self.downsampling = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=2, padding=2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=2, padding=2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=2, padding=2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=2, padding=2),
        )
        self.res = nn.Sequential(
            ResBlock(in_channel=32, out_channel=32),
            ResBlock(in_channel=32, out_channel=32),
            ResBlock(in_channel=32, out_channel=32),
            ResBlock(in_channel=32, out_channel=32),
            ResBlock(in_channel=32, out_channel=32),
            ResBlock(in_channel=32, out_channel=32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
        )
        """ using 3d conv to instead the Euclidean distance"""
        self.cost_volume_filter = nn.Sequential(
            MetricBlock(in_channel=cost_volume_channel, out_channel=32),
            MetricBlock(in_channel=32, out_channel=32),
            MetricBlock(in_channel=32, out_channel=32),
            MetricBlock(in_channel=32, out_channel=32),
            nn.Conv3d(in_channels=32, out_channels=1, kernel_size=3, padding=1),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=3, padding=1),
            ResBlock(in_channel=32, out_channel=32, dilation=1),
            ResBlock(32, 32, dilation=2),
            ResBlock(32, 32, dilation=4),
            ResBlock(32, 32, dilation=8),
            ResBlock(32, 32, dilation=1),
            ResBlock(32, 32, dilation=1),
            nn.Conv2d(32, 1, 3, padding=1),
        )

    def forward_once_1(self, x):
        output = self.downsampling(x)
        output = self.res(output)
        return output

    def forward_stage1(self, input_l, input_r):
        output_l = self.forward_once_1(input_l)
        output_r = self.forward_once_1(input_r)
        return output_l, output_r

    def forward_once_2(self, cost_volume):
        """the index cost volume's dimension is not right for conv3d here, so we change it"""
        cost_volume = cost_volume.permute([0, 2, 1, 3, 4])
        output = self.cost_volume_filter(cost_volume)  # [batch_size, channel, disparity, h, w]
        disparity_low = output
        return disparity_low  # low resolution disparity map

    def forward_stage2(self, feature_l, feature_r):
        cost_v_l = CostVolume(feature_l, feature_r, "left", method=self.cost_volume_method, k=4, batch_size=self.batch_size)
        disparity_low = self.forward_once_2(cost_v_l)
        disparity_low = torch.squeeze(disparity_low, dim=1)
        return disparity_low

    def forward_stage3(self, disparity_low, left):
        """upsample and concatenate"""
        d_high = nn.functional.interpolate(disparity_low, [left.shape[2], left.shape[3]], mode='bilinear', align_corners=True)
        d_high = soft_argmin(d_high)
        d_concat = torch.cat([d_high, left], dim=1)
        d_refined = self.refine(d_concat)
        return d_refined

    def forward(self, left, right):
        left_feature, right_feature = self.forward_stage1(left, right)
        disparity_low_l = self.forward_stage2(left_feature, right_feature)
        d_initial_l = nn.functional.interpolate(disparity_low_l, [left.shape[2], left.shape[3]], mode='bilinear', align_corners=True)
        d_initial_l = soft_argmin(d_initial_l)
        d_refined_l = self.forward_stage3(disparity_low_l, left)
        d_final_l = d_initial_l + d_refined_l
        d_final_l = nn.ReLU()(d_final_l)
        d_final_l = d_final_l.squeeze(dim=0)
        return d_final_l


class MetricBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1):
        super(MetricBlock, self).__init__()
        self.conv3d_1 = nn.Conv3d(in_channel, out_channel, 3, 1, 1)
        self.bn1 = nn.BatchNorm3d(out_channel)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        out = self.conv3d_1(x)
        out = self.bn1(out)
        out = self.relu(out)
        return out


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, dilation=1, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        padding = dilation
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=padding, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu1 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=stride, padding=padding, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.relu2 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.in_ch = in_channel
        self.out_ch = out_channel
        self.p = padding
        self.d = dilation

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu2(out)
        return out


def soft_argmin(cost_volume):
    """
        Remove single-dimensional entries from the shape of an array.
    """
    softmax = nn.Softmax(dim=1)
    disparity_softmax = softmax(-cost_volume)
    d_grid = torch.arange(cost_volume.shape[1], dtype=torch.float)
    d_grid = d_grid.reshape(-1, 1, 1)
    d_grid = d_grid.repeat((cost_volume.shape[0], 1, cost_volume.shape[2], cost_volume.shape[3]))  # [batchSize, 1, h, w]
    d_grid = d_grid.to('cuda')
    tmp = disparity_softmax * d_grid
    arg_soft_min = torch.sum(tmp, dim=1, keepdim=True)
    return arg_soft_min
