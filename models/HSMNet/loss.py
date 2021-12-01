import torch.nn.functional as F
import torch.nn as nn


class SmoothL1LossHSM(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, stacked, target, mask):
        loss = (64. / 85) * F.smooth_l1_loss(stacked[0][mask], target[mask], size_average=True) + \
               (16. / 85) * F.smooth_l1_loss(stacked[1][mask], target[mask], size_average=True) + \
               (4. / 85) * F.smooth_l1_loss(stacked[2][mask], target[mask], size_average=True) + \
               (1. / 85) * F.smooth_l1_loss(stacked[3][mask], target[mask], size_average=True)
        return loss
