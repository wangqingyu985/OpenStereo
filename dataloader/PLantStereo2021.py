from os.path import join

import cv2
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset


class PlantStereo2021(Dataset):

    def __init__(self, directory, mode, validate_size=20, transform=None, high_acc=False):
        super().__init__()

        self.mode = mode
        self.transform = transform
        self.high_acc = high_acc

        if mode == 'train' or mode == 'validate':
            self.dir = join(directory, 'training')
        elif mode == 'test':
            self.dir = join(directory, 'testing')

        left_dir = join(self.dir, 'left_view')
        right_dir = join(self.dir, 'right_view')
        left_imgs = list()
        right_imgs = list()

        if mode == 'train':
            imgs_range = range(100 - validate_size)
        elif mode == 'validate':
            imgs_range = range(100 - validate_size, 100)
        elif mode == 'test':
            imgs_range = range(50)

        fmt = '{:06}.png'

        for i in imgs_range:
            left_imgs.append(join(left_dir, fmt.format(i)))
            right_imgs.append(join(right_dir, fmt.format(i)))

        self.left_imgs = left_imgs
        self.right_imgs = right_imgs

        if mode == 'train' or mode == 'validate':
            disp_imgs = list()
            if self.high_acc:
                disp_dir = join(self.dir, 'disp_high_acc')
                disp_fmt = '{:06}.tiff'
            else:
                disp_dir = join(self.dir, 'disp')
                disp_fmt = '{:06}.png'
            for i in imgs_range:
                disp_imgs.append(join(disp_dir, disp_fmt.format(i)))
        elif mode == 'test':
            disp_imgs = list()
            disp_dir = join(self.dir, 'disp_high_acc')
            disp_fmt = '{:06}.tiff'
            for i in imgs_range:
                disp_imgs.append(join(disp_dir, disp_fmt.format(i)))

        self.disp_imgs = disp_imgs

    def __len__(self):
        return len(self.left_imgs)

    def __getitem__(self, idx):
        data = {}

        # bgr mode
        data['left'] = cv2.imread(self.left_imgs[idx])
        data['right'] = cv2.imread(self.right_imgs[idx])
        if self.mode != 'test':
            if self.high_acc:
                data['disp'] = cv2.imread(self.disp_imgs[idx], -1)
            else:
                data['disp'] = cv2.imread(self.disp_imgs[idx])[:, :, 0]
        else:
            data['disp'] = cv2.imread(self.disp_imgs[idx], -1)

        if self.transform:
            data = self.transform(data)

        return data


class RandomCrop():

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        new_h, new_w = self.output_size
        h, w, _ = sample['left'].shape
        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        for key in sample:
            sample[key] = sample[key][top: top + new_h, left: left + new_w]

        return sample


class Normalize():
    """
    RGB mode
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        sample['left'] = sample['left'] / 255.0
        sample['right'] = sample['right'] / 255.0

        sample['left'] = self.__normalize(sample['left'])
        sample['right'] = self.__normalize(sample['right'])

        return sample

    def __normalize(self, img):
        for i in range(3):
            img[:, :, i] = (img[:, :, i] - self.mean[i]) / self.std[i]
        return img


class ToTensor():

    def __call__(self, sample):
        left = sample['left']
        right = sample['right']

        # H x W x C ---> C x H x W
        sample['left'] = torch.from_numpy(left.transpose([2, 0, 1])).type(torch.FloatTensor)
        sample['right'] = torch.from_numpy(right.transpose([2, 0, 1])).type(torch.FloatTensor)

        if 'disp' in sample:
            sample['disp'] = torch.from_numpy(sample['disp']).type(torch.FloatTensor)

        return sample


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
        disp = sample['disp'].unsqueeze(0).unsqueeze(1)  # [1, 1, H, W]
        disp = F.pad(disp, pad=(0, pad_w, 0, pad_h))

        sample['left'] = left.squeeze()
        sample['right'] = right.squeeze()
        sample['disp'] = disp.squeeze()

        return sample
