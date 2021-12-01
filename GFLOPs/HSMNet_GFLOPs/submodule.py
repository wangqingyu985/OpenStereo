from __future__ import print_function
import math
import numpy as np

import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F


class sepConv3dBlock(nn.Module):
    """
    Separable 3d convolution block as 2 separable convolutions and a projection
    layer
    """
    def __init__(self, in_planes, out_planes, stride=(1, 1, 1)):
        super(sepConv3dBlock, self).__init__()
        if in_planes == out_planes and stride == (1, 1, 1):
            self.downsample = None
        else:
            self.downsample = projfeat3d(in_planes=in_planes, out_planes=out_planes, stride=stride)
        self.conv1 = sepConv3d(in_planes=in_planes, out_planes=out_planes, kernel_size=3, stride=stride, pad=1)
        self.conv2 = sepConv3d(in_planes=out_planes, out_planes=out_planes, kernel_size=3, stride=(1, 1, 1), pad=1)

    def forward(self, x):
        x = x.cpu()
        out = F.relu(self.conv1(x), inplace=True)
        if self.downsample:
            x = self.downsample(x)
        out = F.relu(x + self.conv2(out), inplace=True)
        return out


class projfeat3d(nn.Module):
    """
    Turn 3d projection into 2d projection
    """
    def __init__(self, in_planes, out_planes, stride):
        super(projfeat3d, self).__init__()
        self.stride = stride
        self.conv1 = nn.Conv2d(in_planes, out_planes, (1, 1), padding=(0, 0), stride=stride[:2], bias=False)
        self.bn = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        b, c, d, h, w = x.size()
        x = self.conv1(x.view(b, c, d, h*w))
        x = self.bn(x)
        x = x.view(b, -1, d//self.stride[0], h, w)
        return x


# original conv3d block
def sepConv3d(in_planes, out_planes, kernel_size, stride, pad, bias=False):
    if bias:
        return nn.Sequential(nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, padding=pad, stride=stride, bias=bias))
    else:
        return nn.Sequential(nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, padding=pad, stride=stride, bias=bias),
                             nn.BatchNorm3d(out_planes))
    

class disparityregression(nn.Module):
    def __init__(self, maxdisp, divisor):
        super(disparityregression, self).__init__()
        maxdisp = int(maxdisp/divisor)
        self.register_buffer('disp', torch.Tensor(np.reshape(np.array(range(maxdisp)), [1, maxdisp, 1, 1])))
        self.divisor = divisor

    def forward(self, x, ifent=False):
        disp = self.disp.repeat(x.size()[0], 1, x.size()[2], x.size()[3])
        out = torch.sum(x*disp, 1) * self.divisor

        if ifent:
            # entropy
            x = x+1e-12
            ent = (-x*x.log()).sum(dim=1)
            return out, ent
        else:
            return out


class decoderBlock(nn.Module):
    def __init__(self, nconvs, inchannelF, channelF, stride=(1, 1, 1), up=False, nstride=1, pool=False):
        super(decoderBlock, self).__init__()
        self.pool = pool
        stride = [stride]*nstride + [(1, 1, 1)] * (nconvs-nstride)
        self.convs = [sepConv3dBlock(inchannelF, channelF, stride=stride[0])]
        for i in range(1, nconvs):
            self.convs.append(sepConv3dBlock(in_planes=channelF, out_planes=channelF, stride=stride[i]))
        self.convs = nn.Sequential(*self.convs)
        self.classify = nn.Sequential(sepConv3d(in_planes=channelF, out_planes=channelF, kernel_size=3, stride=(1, 1, 1), pad=1),
                                      nn.ReLU(inplace=True),
                                      sepConv3d(in_planes=channelF, out_planes=1, kernel_size=3, stride=(1, 1, 1), pad=1, bias=True)
                                      )
        self.up = False
        if up:
            self.up = True
            self.up = nn.Sequential(nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear'),
                                    sepConv3d(channelF, channelF//2, 3, (1, 1, 1), 1, bias=False),
                                    nn.ReLU(inplace=True)
                                    )

        if pool:
            self.pool_convs = torch.nn.ModuleList([sepConv3d(channelF, channelF, 1, (1, 1, 1), 0),
                                                   sepConv3d(channelF, channelF, 1, (1, 1, 1), 0),
                                                   sepConv3d(channelF, channelF, 1, (1, 1, 1), 0),
                                                   sepConv3d(channelF, channelF, 1, (1, 1, 1), 0)]
                                                  )

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1]*m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if hasattr(m.bias, 'data'):
                    m.bias.data.zero_()

    def forward(self, fvl):
        # left
        fvl = self.convs(fvl)
        # pooling
        if self.pool:
            for i in range(0, 4):
                fvl = self.pool_convs[i](fvl)
        # TODO cost aggregation
        if self.training:
            # classification
            costl = self.classify(fvl)
            if self.up:
                fvl = self.up(fvl)
        else:
            # classification
            if self.up:
                fvl = self.up(fvl)
                costl = fvl
            else:
                costl = self.classify(fvl)

        return fvl, costl.squeeze(1)
