import argparse
import time as t
from os.path import join

import cv2
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader

from models.HSMNet.hsm import HSMNet
from models.GCNet.GCNet import GCNet
from models.DANet.danet import DANet
from models.PSMNet.PSMnet import PSMNet
from models.GwcNet.gwcnet import GwcNet
from models.CFPNet.basic import CFPNet_b
from models.StereoNet.stereonet import StereoNet
from models.CFPNet.stackhourglass import CFPNet_s
from dataloader.PLantStereo2021 import PlantStereo2021, ToTensor, Normalize, Pad


parser = argparse.ArgumentParser(description='PSMNet inference')
parser.add_argument('--maxdisp', type=int, default=256, help='max diparity')
parser.add_argument('--model', default='GC', help='choose which model to use, GC: GCNet, P: PSMNet, Stereo: StereoNet, CFP: CFPNet, HSM: HSMNet, Gwc: GwcNet, D: DANet')
parser.add_argument('--datadir', default='/home/wangqingyu/PlantStereo/PlantStereo2021/pumpkin', help='data directory')
parser.add_argument('--dispacc', type=bool, default=True, help='high or low accuracy disparity. False<--uint8, True<--float64')
parser.add_argument('--output', default='/home/wangqingyu/PlantStereo/PlantStereo2021/pumpkin/testing/output-GCNet', help='path to the output disparity map')
parser.add_argument('--erroroutput', default='/home/wangqingyu/PlantStereo/PlantStereo2021/pumpkin/testing/erroroutput-GCNet', help='path to the output disparity error map')
parser.add_argument('--model-path', default='/home/wangqingyu/桌面/PlantStereo/checkpoint/best_model.ckpt', help='path to the model')
parser.add_argument('--num-workers', type=int, default=8, help='num workers in loading data')
args = parser.parse_args()


mean = [0.406, 0.456, 0.485]
std = [0.225, 0.224, 0.229]
device_ids = [0]
device = torch.device('cuda:{}'.format(device_ids[0]))
print(torch.cuda.is_available())


def main():

    epe_list = []
    error_1pixel_list = []
    error_3pixel_list = []
    error_5pixel_list = []
    rmse_list = []
    time_list = []

    test_transform = T.Compose([Normalize(mean, std), ToTensor(), Pad(640, 1088)])
    test_dataset = PlantStereo2021(args.datadir, mode='test', transform=test_transform, high_acc=args.dispacc)
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=args.num_workers)

    if args.model == 'GC':
        model = GCNet(max_disp=args.maxdisp).to(device)
    elif args.model == 'P':
        model = PSMNet(max_disp=args.maxdisp).to(device)
    elif args.model == 'Stereo':
        cost_volume_method = "subtract"
        # cost_volume_method = "concat"
        model = StereoNet(batch_size=1, cost_volume_method=cost_volume_method).to(device)
    elif args.model == 'CFP':
        model = CFPNet_s(maxdisp=args.maxdisp).to(device)
    elif args.model == 'HSM':
        model = HSMNet(maxdisp=args.maxdisp, clean=False, level=1).to(device)
    elif args.model == 'Gwc':
        model = GwcNet(maxdisp=args.maxdisp, use_concat_volume=False).to(device)
        # GwcNet_G: use_concat_volune=False
        # GwcNet_GC: use_concat_volune=True
    elif args.model == 'D':
        model = DANet(maxdisp=args.maxdisp, use_concat_volume=False).to(device)

    if len(device_ids) > 1:
        model = nn.DataParallel(model, device_ids=device_ids)
    state = torch.load(args.model_path)
    if len(device_ids) == 1:
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state['state_dict'].items():
            namekey = k[7:]  # remove `module.`
            new_state_dict[namekey] = v
        state['state_dict'] = new_state_dict

    model.load_state_dict(state['state_dict'])
    print('load model from {}'.format(args.model_path))
    print('epoch: {}'.format(state['epoch']))
    print('1px-error: {}%'.format(state['error_1']))
    print('3px-error: {}%'.format(state['error_3']))
    print('5px-error: {}%'.format(state['error_5']))
    print('best epe: {}'.format(state['epe']))
    print('best rmse: {}'.format(state['rmse']))
    model.eval()

    for i, batch in enumerate(test_loader):
        left_img = batch['left'].to(device)
        right_img = batch['right'].to(device)
        target_disp = batch['disp'].to(device)
        target_disp = F.pad(target_disp, pad=(0, 1088-1024, 0, 640-571))

        mask = (target_disp > 0) & (target_disp < args.maxdisp)
        mask = mask.detach_()

        if args.model == 'GC':
            with torch.inference_mode():
                dawn = t.time()
                disp = model(left_img, right_img)
                dusk = t.time()
                time_consuming = dusk - dawn
        elif args.model == 'P':
            with torch.inference_mode():
                dawn = t.time()
                _, _, disp = model(left_img, right_img)
                dusk = t.time()
                time_consuming = dusk - dawn
        elif args.model == 'Stereo':
            with torch.inference_mode():
                dawn = t.time()
                disp = model(left_img, right_img)
                dusk = t.time()
                time_consuming = dusk - dawn
        elif args.model == 'CFP':
            with torch.inference_mode():
                dawn = t.time()
                disp = model(left_img, right_img)
                dusk = t.time()
                time_consuming = dusk - dawn
        elif args.model == 'HSM':
            with torch.inference_mode():
                dawn = t.time()
                disp = model(left_img, right_img)
                dusk = t.time()
                time_consuming = dusk - dawn
        elif args.model == 'Gwc' or args.model == 'D':
            with torch.inference_mode():
                dawn = t.time()
                disp = model(left_img, right_img)
                dusk = t.time()
                time_consuming = dusk - dawn
        print('time consuming:{:.2f}s'.format(time_consuming))
        delta = torch.abs(disp[mask] - target_disp[mask])
        error_mat_1 = (delta > 1.0)
        error_mat_3 = (delta > 3.0)
        error_mat_5 = (delta > 5.0)
        error_1 = (torch.sum(error_mat_1).item() / torch.numel(disp[mask]) * 100)
        error_3 = (torch.sum(error_mat_3).item() / torch.numel(disp[mask]) * 100)
        error_5 = (torch.sum(error_mat_5).item() / torch.numel(disp[mask]) * 100)
        epe = F.l1_loss(input=disp[mask], target=target_disp[mask], size_average=True).detach().cpu()
        rmse = ((F.mse_loss(input=disp[mask], target=target_disp[mask], size_average=True)) ** 0.5).detach().cpu()
        error_1pixel_list.append(error_1)
        error_3pixel_list.append(error_3)
        error_5pixel_list.append(error_5)
        epe_list.append(epe)
        rmse_list.append(rmse)
        time_list.append(time_consuming)
        print('1px-error: {:.5}%'.format(error_1))
        print('3px-error: {:.5}%'.format(error_3))
        print('5px-error: {:.5}%'.format(error_5))
        print('epe: {:.5}'.format(epe))
        print('rmse: {:.5}'.format(rmse))

        disp = disp.squeeze(0).detach().cpu().numpy()
        disp = disp.astype(np.uint8)
        target_disp = target_disp.squeeze(0).detach().cpu().numpy()
        target_disp = target_disp.astype(np.uint8)

        error_map = np.zeros(shape=[disp.shape[0], disp.shape[1]], dtype=np.uint8)
        for v in range(disp.shape[0]):
            for u in range(disp.shape[1]):
                if target_disp[v, u] != 0:
                    error_map[v, u] = abs(int(target_disp[v, u]) - int(disp[v, u]))
                else:
                    error_map[v, u] = 0
        error_map = cv2.applyColorMap(error_map, cv2.COLORMAP_TURBO)
        fmt = '{:06}.png'
        error_filename = join(args.erroroutput, fmt.format(i))
        cv2.imwrite(filename=error_filename, img=error_map, params=None)

        max = np.max(disp)
        for v in range(disp.shape[0]):
            for u in range(disp.shape[1]):
                disp[v, u] = ((disp[v, u] - 158) / (max - 158)) * 255
        color_map = cv2.applyColorMap(disp, cv2.COLORMAP_TURBO)
        filename = join(args.output, fmt.format(i))
        cv2.imwrite(filename=filename, img=color_map, params=None)

    final_epe_list = pd.DataFrame(data=epe_list)
    final_rmse_list = pd.DataFrame(data=rmse_list)
    final_error1_list = pd.DataFrame(data=error_1pixel_list)
    final_error3_list = pd.DataFrame(data=error_3pixel_list)
    final_error5_list = pd.DataFrame(data=error_5pixel_list)
    final_time_list = pd.DataFrame(data=time_list)
    if args.model == 'GC':
        final_epe_list.to_csv('/home/wangqingyu/PlantStereo/PlantStereo2021/spinach/testing/gcnet_epe.csv')
        final_rmse_list.to_csv('/home/wangqingyu/PlantStereo/PlantStereo2021/spinach/testing/gcnet_rmse.csv')
        final_error1_list.to_csv('/home/wangqingyu/PlantStereo/PlantStereo2021/spinach/testing/gcnet_error1.csv')
        final_error3_list.to_csv('/home/wangqingyu/PlantStereo/PlantStereo2021/spinach/testing/gcnet_error3.csv')
        final_error5_list.to_csv('/home/wangqingyu/PlantStereo/PlantStereo2021/spinach/testing/gcnet_error5.csv')
        final_time_list.to_csv('/home/wangqingyu/PlantStereo/PlantStereo2021/spinach/testing/gcnet_time.csv')
    elif args.model == 'P':
        final_epe_list.to_csv('/home/wangqingyu/PlantStereo/PlantStereo2021/pumpkin/testing/psmnet_epe.csv')
        final_rmse_list.to_csv('/home/wangqingyu/PlantStereo/PlantStereo2021/pumpkin/testing/psmnet_rmse.csv')
        final_error1_list.to_csv('/home/wangqingyu/PlantStereo/PlantStereo2021/pumpkin/testing/psmnet_error1.csv')
        final_error3_list.to_csv('/home/wangqingyu/PlantStereo/PlantStereo2021/pumpkin/testing/psmnet_error3.csv')
        final_error5_list.to_csv('/home/wangqingyu/PlantStereo/PlantStereo2021/pumpkin/testing/psmnet_error5.csv')
        final_time_list.to_csv('/home/wangqingyu/PlantStereo/PlantStereo2021/pumpkin/testing/psmnet_time.csv')
    elif args.model == 'Stereo':
        final_epe_list.to_csv('/home/wangqingyu/PlantStereo/PlantStereo2021/pumpkin/testing/stereonet_epe.csv')
        final_rmse_list.to_csv('/home/wangqingyu/PlantStereo/PlantStereo2021/pumpkin/testing/stereonet_rmse.csv')
        final_error1_list.to_csv('/home/wangqingyu/PlantStereo/PlantStereo2021/pumpkin/testing/stereonet_error1.csv')
        final_error3_list.to_csv('/home/wangqingyu/PlantStereo/PlantStereo2021/pumpkin/testing/stereonet_error3.csv')
        final_error5_list.to_csv('/home/wangqingyu/PlantStereo/PlantStereo2021/pumpkin/testing/stereonet_error5.csv')
        final_time_list.to_csv('/home/wangqingyu/PlantStereo/PlantStereo2021/pumpkin/testing/stereonet_time.csv')
    elif args.model == 'CFP':
        final_epe_list.to_csv('/home/wangqingyu/PlantStereo/PlantStereo2021/pumpkin/testing/cfpnet_epe.csv')
        final_rmse_list.to_csv('/home/wangqingyu/PlantStereo/PlantStereo2021/pumpkin/testing/cfpnet_rmse.csv')
        final_error1_list.to_csv('/home/wangqingyu/PlantStereo/PlantStereo2021/pumpkin/testing/cfpnet_error1.csv')
        final_error3_list.to_csv('/home/wangqingyu/PlantStereo/PlantStereo2021/pumpkin/testing/cfpnet_error3.csv')
        final_error5_list.to_csv('/home/wangqingyu/PlantStereo/PlantStereo2021/pumpkin/testing/cfpnet_error5.csv')
        final_time_list.to_csv('/home/wangqingyu/PlantStereo/PlantStereo2021/pumpkin/testing/cfpnet_time.csv')
    elif args.model == 'HSM':
        final_epe_list.to_csv('/home/wangqingyu/PlantStereo/PlantStereo2021/pumpkin/testing/hsmnet_epe.csv')
        final_rmse_list.to_csv('/home/wangqingyu/PlantStereo/PlantStereo2021/pumpkin/testing/hsmnet_rmse.csv')
        final_error1_list.to_csv('/home/wangqingyu/PlantStereo/PlantStereo2021/pumpkin/testing/hsmnet_error1.csv')
        final_error3_list.to_csv('/home/wangqingyu/PlantStereo/PlantStereo2021/pumpkin/testing/hsmnet_error3.csv')
        final_error5_list.to_csv('/home/wangqingyu/PlantStereo/PlantStereo2021/pumpkin/testing/hsmnet_error5.csv')
        final_time_list.to_csv('/home/wangqingyu/PlantStereo/PlantStereo2021/pumpkin/testing/hsmnet_time.csv')
    elif args.model == 'Gwc':
        final_epe_list.to_csv('/home/wangqingyu/PlantStereo/PlantStereo2021/pumpkin/testing/gwcnet_epe.csv')
        final_rmse_list.to_csv('/home/wangqingyu/PlantStereo/PlantStereo2021/pumpkin/testing/gwcnet_rmse.csv')
        final_error1_list.to_csv('/home/wangqingyu/PlantStereo/PlantStereo2021/pumpkin/testing/gwcnet_error1.csv')
        final_error3_list.to_csv('/home/wangqingyu/PlantStereo/PlantStereo2021/pumpkin/testing/gwcnet_error3.csv')
        final_error5_list.to_csv('/home/wangqingyu/PlantStereo/PlantStereo2021/pumpkin/testing/gwcnet_error5.csv')
        final_time_list.to_csv('/home/wangqingyu/PlantStereo/PlantStereo2021/pumpkin/testing/gwcnet_time.csv')
    elif args.model == 'D':
        final_epe_list.to_csv('/home/wangqingyu/PlantStereo/PlantStereo2021/spinach/testing/danet_epe.csv')
        final_rmse_list.to_csv('/home/wangqingyu/PlantStereo/PlantStereo2021/spinach/testing/danet_rmse.csv')
        final_error1_list.to_csv('/home/wangqingyu/PlantStereo/PlantStereo2021/spinach/testing/danet_error1.csv')
        final_error3_list.to_csv('/home/wangqingyu/PlantStereo/PlantStereo2021/spinach/testing/danet_error3.csv')
        final_error5_list.to_csv('/home/wangqingyu/PlantStereo/PlantStereo2021/spinach/testing/danet_error5.csv')
        final_time_list.to_csv('/home/wangqingyu/PlantStereo/PlantStereo2021/spinach/testing/danet_time.csv')


class Pad():
    def __init__(self, H, W):
        self.w = W
        self.h = H

    def __call__(self, sample):
        pad_h = self.h - sample['left'].size(1)
        pad_w = self.w - sample['left'].size(2)

        left = sample['left'].unsqueeze(0)  # [1, 3, H, W]
        left = F.pad(left, pad=(0, pad_w, 0, pad_h))
        right = sample['right'].unsqueeze(0)  # [1, 3, H, W]
        right = F.pad(right, pad=(0, pad_w, 0, pad_h))

        sample['left'] = left.squeeze()
        sample['right'] = right.squeeze()

        return sample


if __name__ == '__main__':
    main()
