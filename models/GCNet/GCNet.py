import math
import torch
import torch.nn as nn
from models.GCNet.costnet import CostNet
from models.GCNet.costaggregation import CostAggregation


class GCNet(nn.Module):

    def __init__(self, max_disp):
        super().__init__()
        self.D = max_disp
        self.__init_params()
        self.cost_net = CostNet()
        self.cost_aggregation = CostAggregation(max_disp)

    def forward(self, left_img, right_img):
        original_size = [self.D, left_img.size(2), left_img.size(3)]  # [D, H, W]
        left_cost = self.cost_net(left_img)  # [B, 32, 1/2H, 1/2W]
        right_cost = self.cost_net(right_img)  # [B, 32, 1/2H, 1/2W]
        B, C, H, W = left_cost.size()
        cost_volume = torch.zeros(B, 64, self.D // 2, H, W).type_as(left_cost)  # [B, 64, 1/2D, 1/2H, 1/2W]
        for i in range(self.D // 2):
            if i > 0:
                cost_volume[:, :C, i, :, i:] = left_cost[:, :, :, i:]
                cost_volume[:, C:, i, :, i:] = right_cost[:, :, :, :-i]
            else:
                cost_volume[:, :C, i, :, :] = left_cost
                cost_volume[:, C:, i, :, :] = right_cost
        disp = self.cost_aggregation(cost_volume)
        return disp

    def __init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
