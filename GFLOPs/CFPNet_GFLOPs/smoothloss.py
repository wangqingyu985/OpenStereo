import torch.nn.functional as F
import torch.nn as nn


class SmoothL1LossC(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, disp, target):
        loss = F.smooth_l1_loss(disp, target)
        return loss
