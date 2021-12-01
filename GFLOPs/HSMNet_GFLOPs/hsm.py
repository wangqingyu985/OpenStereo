from __future__ import print_function
import torch.utils.data
from submodule import *
from utils import unet
from torch.autograd import Variable


class HSMNet(nn.Module):
    def __init__(self, maxdisp, clean, level=1):
        super(HSMNet, self).__init__()
        self.maxdisp = maxdisp
        self.clean = clean
        self.feature_extraction = unet()
        self.level = level

        # block 4
        self.decoder6 = decoderBlock(nconvs=6, inchannelF=32, channelF=32, up=True, pool=True)
        if self.level > 2:
            self.decoder5 = decoderBlock(6, 32, 32, up=False, pool=True)
        else:
            self.decoder5 = decoderBlock(6, 32, 32, up=True, pool=True)
            if self.level > 1:
                self.decoder4 = decoderBlock(6, 32, 32, up=False)
            else:
                self.decoder4 = decoderBlock(6, 32, 32, up=True)
                self.decoder3 = decoderBlock(5, 32, 32, stride=(1, 1, 1), up=False, nstride=1)
        # reg
        self.disp_reg8 = disparityregression(maxdisp=self.maxdisp, divisor=8)
        self.disp_reg16 = disparityregression(maxdisp=self.maxdisp, divisor=16)
        self.disp_reg32 = disparityregression(maxdisp=self.maxdisp, divisor=32)
        self.disp_reg64 = disparityregression(maxdisp=self.maxdisp, divisor=64)

    def feature_vol(self, refimg_fea, targetimg_fea, maxdisp, leftview=True):
        """
        diff feature volume
        """
        width = refimg_fea.shape[-1]
        cost = Variable(torch.cuda.FloatTensor(refimg_fea.size()[0], refimg_fea.size()[1], maxdisp, refimg_fea.size()[2], refimg_fea.size()[3]).fill_(0.))
        for i in range(min(maxdisp, width)):
            feata = refimg_fea[:, :, :, i:width]
            featb = targetimg_fea[:, :, :, :width-i]
            # concat
            if leftview:
                cost[:, :refimg_fea.size()[1], i, :, i:] = torch.abs(feata-featb)
            else:
                cost[:, :refimg_fea.size()[1], i, :, :width-i] = torch.abs(featb-feata)
        cost = cost.contiguous()
        return cost

    def forward(self, left):
        right = left
        nsample = left.shape[0]  # batch_size
        conv4, conv3, conv2, conv1 = self.feature_extraction(torch.cat([left, right], dim=0))
        # conv4, conv3, conv2, conv1: [2*batch_size, channel=32, H/64, W/64], [2*batch_size, channel=16, H/32, W/32], [2*batch_size, channel=16, H/16, W/16], [2*batch_size, channel=16, H/8, W/8]
        conv40, conv30, conv20, conv10 = conv4[:nsample], conv3[:nsample], conv2[:nsample], conv1[:nsample]
        # conv40, conv30, conv20, conv10: [batch_size, channel=32, H/64, W/64], [batch_size, channel=16, H/32, W/32], [batch_size, channel=16, H/16, W/16], [batch_size, channel=16, H/8, W/8]
        conv41, conv31, conv21, conv11 = conv4[nsample:], conv3[nsample:], conv2[nsample:], conv1[nsample:]
        # conv41, conv31, conv21, conv11: [batch_size, channel=32, H/64, W/64], [batch_size, channel=16, H/32, W/32], [batch_size, channel=16, H/16, W/16], [batch_size, channel=16, H/8, W/8]
        feat6 = self.feature_vol(refimg_fea=conv40, targetimg_fea=conv41, maxdisp=self.maxdisp//64)
        # fea6: [batch_size, channel=32, D/64, H/64, W/64]
        feat5 = self.feature_vol(refimg_fea=conv30, targetimg_fea=conv31, maxdisp=self.maxdisp//32)
        # fea5: [batch_size, channel=16, D/32, H/32, W/32]
        feat4 = self.feature_vol(refimg_fea=conv20, targetimg_fea=conv21, maxdisp=self.maxdisp//16)
        # fea4: [batch_size, channel=16, D/16, H/16, W/16]
        feat3 = self.feature_vol(refimg_fea=conv10, targetimg_fea=conv11, maxdisp=self.maxdisp//8)
        # fea3: [batch_size, channel=16, D/8, H/8, W/8]

        feat6_2x, cost6 = self.decoder6(feat6)
        # feat6_2x: [batch_size, channel=16, D/32, H/32, W/32]
        # cost6: [batch_size, D/64, H/64, W/64]
        feat6_2x = feat6_2x.cpu()
        feat5 = feat5.cpu()
        feat5 = torch.cat((feat6_2x, feat5), dim=1)
        # fea5: [batch_size, channel=32, D/32, H/32, W/32]

        feat5_2x, cost5 = self.decoder5(feat5)
        # feat5_2x: [batch_size, channel=32, D/16, H/16, W/16]
        # cost5: [batch_size, D/32, H/32, W/32]
        if self.level > 2:
            cost3 = F.upsample(cost5, [left.size()[2], left.size()[3]], mode='bilinear')
        else:
            feat5_2x = feat5_2x.cpu()
            feat4 = feat4.cpu()
            feat4 = torch.cat((feat5_2x, feat4), dim=1)
            # fea4: [batch_size, channel=32, D/16, H/16, W/16]

            feat4_2x, cost4 = self.decoder4(feat4)
            # feat4_2x: [batch_size, channel=16, D/8, H/8, W/8]
            # cost4: [batch_size, D/16, H/16, W/16]
            if self.level > 1:
                cost3 = F.upsample((cost4).unsqueeze(1), [self.disp_reg8.disp.shape[1], left.size()[2], left.size()[3]], mode='trilinear').squeeze(1)
            else:
                feat3 = feat3.cpu()
                feat4_2x = feat4_2x.cpu()
                feat3 = torch.cat((feat4_2x, feat3), dim=1)
                # fea3: [batch_size, channel=32, D/8, H/8, W/8]
                feat3_2x, cost3 = self.decoder3(feat3)
                # feat3_2x: [batch_size, channel=32, D/16, H/8, W/8]
                # cost3: [batch_size, D/8, H/8, W/8]
                # cost3 = F.upsample(cost3, [left.size()[2], left.size()[3]], mode='bilinear')
                cost3 = F.upsample((cost3).unsqueeze(1), [self.disp_reg8.disp.shape[1], left.size()[2], left.size()[3]], mode='trilinear').squeeze(1)
                # cost3: [batch_size, D/8, H, W]
        if self.level > 2:
            final_reg = self.disp_reg32
        else:
            final_reg = self.disp_reg8
        # final_reg: [D/8]

        if self.training or self.clean == -1:
            pred3 = final_reg(x=F.softmax(cost3, dim=1), ifent=False)
            # pred3: [batch_size, H, W]
            entropy = pred3  # to save memory
        else:
            cost3 = cost3.cpu()
            pred3 = final_reg(x=F.softmax(cost3, dim=1), ifent=False)
            # pred3[entropy > self.clean] = np.inf

        if self.training:
            cost6 = F.upsample((cost6).unsqueeze(1), [self.disp_reg64.disp.shape[1], left.size()[2], left.size()[3]], mode='trilinear').squeeze(1)
            # cost6: [batch_size, D/64, H, W]
            cost5 = F.upsample((cost5).unsqueeze(1), [self.disp_reg32.disp.shape[1], left.size()[2], left.size()[3]], mode='trilinear').squeeze(1)
            # cost5: [batch_size, D/32, H, W]
            cost4 = F.upsample((cost4).unsqueeze(1), [self.disp_reg16.disp.shape[1], left.size()[2], left.size()[3]], mode='trilinear').squeeze(1)
            # cost4: [batch_size, D/16, H, W]
            pred6 = self.disp_reg64(F.softmax(cost6, 1))
            pred5 = self.disp_reg32(F.softmax(cost5, 1))
            pred4 = self.disp_reg16(F.softmax(cost4, 1))
            stacked = [pred3, pred4, pred5, pred6]
            return stacked, entropy
        else:
            return pred3
