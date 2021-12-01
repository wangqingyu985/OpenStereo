import os
import shutil
import argparse
import time as t

import torch
import numpy as np
import torch.nn as nn
import tensorboardX as tX
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader
from dataloader.KITTI2015_loader import KITTI2015
from dataloader.PLantStereo2021 import PlantStereo2021, RandomCrop, ToTensor, Normalize, Pad

from models.GCNet.GCNet import GCNet
from models.GCNet.loss import L1Loss

from models.PSMNet.PSMnet import PSMNet
from models.PSMNet.smoothloss import SmoothL1Loss

from models.StereoNet.stereonet import StereoNet

from models.CFPNet.basic import CFPNet_b
from models.CFPNet.stackhourglass import CFPNet_s
from models.CFPNet.smoothloss import SmoothL1LossC

from models.HSMNet.hsm import HSMNet
from models.HSMNet.loss import SmoothL1LossHSM

from models.GwcNet.gwcnet import GwcNet
from models.GwcNet.loss import model_loss

from models.DANet.danet import DANet
from models.DANet.loss import model_loss

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='stereo_matching_with_deep_learning')
parser.add_argument('--model', default='P', help='choose which model to use, GC: GCNet, P: PSMNet, Stereo: StereoNet, CFP: CFPNet, HSM: HSMNet, Gwc: GwcNet, D: DANet')
parser.add_argument('--maxdisp', type=int, default=256, help='max diparity')
parser.add_argument('--dispacc', type=bool, default=True, help='high or low accuracy disparity. False<--uint8, True<--float64')
parser.add_argument('--logdir', default='log/runs', help='log directory')
parser.add_argument('--dataset', default='KITTI2015', help='dataset to use: PlantStereo or KITTI2015')
parser.add_argument('--subset', default='pumpkin', help='subset of the PlantStereo: pumpkin, pepper, spinach or tomato')
parser.add_argument('--cuda', type=int, default=1, help='gpu number')
parser.add_argument('--batch-size', type=int, default=1, help='batch size')
parser.add_argument('--validate-batch-size', type=int, default=1, help='batch size')
parser.add_argument('--log-per-step', type=int, default=1, help='log per step')
parser.add_argument('--save-per-epoch', type=int, default=1, help='save model per epoch')
parser.add_argument('--model-dir', default='checkpoint', help='directory where save model checkpoint')
parser.add_argument('--model-path', default=None, help='path of model to load')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--num-epochs', type=int, default=500, help='number of training epochs')
parser.add_argument('--num-workers', type=int, default=8, help='num workers in loading data')

args = parser.parse_args()


mean = [0.406, 0.456, 0.485]
std = [0.225, 0.224, 0.229]
device_ids = [0]

writer = tX.SummaryWriter(log_dir=args.logdir, comment='stereo_matching_with_deep_learning')
device = torch.device('cuda')
print(device)


def main(args):

    if args.dataset == 'PlantStereo':
        if args.subset == 'pumpkin':
            datadir = '/home/wangqingyu/PlantStereo/PlantStereo2021/pumpkin'
        elif args.subset == 'pepper':
            datadir = '/home/wangqingyu/PlantStereo/PlantStereo2021/pepper'
        elif args.subset == 'spinach':
            datadir = '/home/wangqingyu/PlantStereo/PlantStereo2021/spinach'
        elif args.subset == 'tomato':
            datadir = '/home/wangqingyu/PlantStereo/PlantStereo2021/tomato'
        train_transform = T.Compose([RandomCrop([256, 512]), Normalize(mean, std), ToTensor()])
        train_dataset = PlantStereo2021(datadir, mode='train', transform=train_transform, high_acc=args.dispacc)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        validate_transform = T.Compose([Normalize(mean, std), ToTensor(), Pad(640, 1088)])
        validate_dataset = PlantStereo2021(datadir, mode='validate', transform=validate_transform, high_acc=args.dispacc)
        validate_loader = DataLoader(validate_dataset, batch_size=args.validate_batch_size, num_workers=args.num_workers)
    elif args.dataset == 'KITTI2015':
        datadir = '/home/wangqingyu/KITTI/2015/'
        train_transform = T.Compose([RandomCrop([256, 512]), Normalize(mean, std), ToTensor()])
        train_dataset = KITTI2015(datadir, mode='train', transform=train_transform)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        validate_transform = T.Compose([Normalize(mean, std), ToTensor(), Pad(384, 1248)])
        validate_dataset = KITTI2015(datadir, mode='validate', transform=validate_transform)
        validate_loader = DataLoader(validate_dataset, batch_size=args.validate_batch_size, num_workers=args.num_workers)

    step = 0
    best_1pixel_error = 100.0
    best_3pixel_error = 100.0
    best_5pixel_error = 100.0
    best_epe = 100.0
    best_rmse = 100.0

    if args.model == 'GC':
        model = GCNet(max_disp=args.maxdisp).to(device)
    elif args.model == 'P':
        model = PSMNet(max_disp=args.maxdisp).to(device)
    elif args.model == 'Stereo':
        batch_size = args.batch_size
        cost_volume_method = "subtract"
        # cost_volume_method = "concat"
        model = StereoNet(batch_size=batch_size, cost_volume_method=cost_volume_method)
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

    model = nn.DataParallel(model, device_ids=device_ids)
    if args.model == 'GC':
        criterion = L1Loss().to(device)
    elif args.model == 'P':
        criterion = SmoothL1Loss().to(device)
    elif args.model == 'Stereo':
        criterion = SmoothL1LossC().to(device)
    elif args.model == 'CFP':
        criterion = SmoothL1LossC().to(device)
    elif args.model == 'HSM':
        criterion = SmoothL1LossHSM().to(device)
    elif args.model == 'Gwc' or args.model == 'D':
        criterion = SmoothL1Loss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if args.model_path is not None:
        state = torch.load(args.model_path)
        model.load_state_dict(state['state_dict'])
        optimizer.load_state_dict(state['optimizer'])
        step = state['step']
        best_1pixel_error = state['error_1']
        best_3pixel_error = state['error_3']
        best_5pixel_error = state['error_5']
        best_epe = state['epe']
        best_rmse = state['rmse']
        print('load model from {}'.format(args.model_path))

    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    for epoch in range(1, args.num_epochs + 1):

        dawn = t.time()

        model.train()
        step = train(model, train_loader, optimizer, criterion, step)
        adjust_lr(optimizer, epoch)

        if epoch % args.save_per_epoch == 0:
            model.eval()
            error_1, error_3, error_5, epe, rmse = validate(model, validate_loader, epoch)
            best_1pixel_error, best_3pixel_error, best_5pixel_error, best_epe, best_rmse = save(model, optimizer, epoch, step, error_1, error_3, error_5, epe, rmse, best_1pixel_error, best_3pixel_error, best_5pixel_error, best_epe, best_rmse)

        dusk = t.time()
        time_consuming = dusk - dawn
        print('time consuming in this epoch:{:.2f}s'.format(time_consuming))


def validate(model, validate_loader, epoch):
    """
    validate 40 image pairs
    """
    num_batches = len(validate_loader)
    idx = np.random.randint(num_batches)

    avg_error_1 = 0.0
    avg_error_3 = 0.0
    avg_error_5 = 0.0
    avg_epe = 0.0
    avg_rmse = 0.0
    for i, batch in enumerate(validate_loader):
        left_img = batch['left'].to(device)
        right_img = batch['right'].to(device)
        target_disp = batch['disp'].to(device)

        mask = (target_disp < args.maxdisp) & (target_disp > 0)
        mask = mask.detach_()

        if args.model == 'GC':
            with torch.inference_mode():
                disp = model(left_img, right_img)
        elif args.model == 'P':
            with torch.inference_mode():
                _, _, disp = model(left_img, right_img)
        elif args.model == 'Stereo':
            with torch.inference_mode():
                disp = model(left_img, right_img)
        elif args.model == 'CFP':
            with torch.inference_mode():
                disp = model(left_img, right_img)
        elif args.model == 'HSM':
            with torch.inference_mode():
                disp = model(left_img, right_img)
        elif args.model == 'Gwc':
            with torch.inference_mode():
                disp = model(left_img, right_img)
        elif args.model == 'D':
            with torch.inference_mode():
                disp = model(left_img, right_img)

        delta = torch.abs(disp[mask] - target_disp[mask])
        error_mat_1pixel = (delta > 1.0)
        error_mat_3pixel = (delta > 3.0)
        error_mat_5pixel = (delta > 5.0)
        error_1pixel = torch.sum(error_mat_1pixel).item() / torch.numel(disp[mask]) * 100
        error_3pixel = torch.sum(error_mat_3pixel).item() / torch.numel(disp[mask]) * 100
        error_5pixel = torch.sum(error_mat_5pixel).item() / torch.numel(disp[mask]) * 100

        epe = F.l1_loss(input=disp[mask], target=target_disp[mask], size_average=True)
        rmse = (F.mse_loss(input=disp[mask], target=target_disp[mask], size_average=True)) ** 0.5

        avg_error_1 += error_1pixel
        avg_error_3 += error_3pixel
        avg_error_5 += error_5pixel
        avg_epe += epe
        avg_rmse += rmse
        if i == idx:
            left_save = left_img
            disp_save = disp

    avg_error_1 = avg_error_1 / num_batches
    avg_error_3 = avg_error_3 / num_batches
    avg_error_5 = avg_error_5 / num_batches
    avg_epe = avg_epe / num_batches
    avg_rmse = avg_rmse / num_batches
    print('epoch: {:04} | 1px-error: {:.5}%'.format(epoch, avg_error_1))
    print('epoch: {:04} | 3px-error: {:.5}%'.format(epoch, avg_error_3))
    print('epoch: {:04} | 5px-error: {:.5}%'.format(epoch, avg_error_5))
    print('epoch: {:04} | epe: {:.5}'.format(epoch, avg_epe))
    print('epoch: {:04} | rmse: {:.5}'.format(epoch, avg_rmse))
    writer.add_scalar('error/1px', avg_error_1, epoch)
    writer.add_scalar('error/3px', avg_error_3, epoch)
    writer.add_scalar('error/5px', avg_error_5, epoch)
    writer.add_scalar('epe', avg_epe, epoch)
    writer.add_scalar('rmse', avg_rmse, epoch)
    save_image(left_save[0], disp_save[0], epoch)

    return avg_error_1, avg_error_3, avg_error_5, avg_epe, avg_rmse


def save_image(left_image, disp, epoch):
    for i in range(3):
        left_image[i] = left_image[i] * std[i] + mean[i]
    b, r = left_image[0], left_image[2]
    left_image[0] = r  # BGR --> RGB
    left_image[2] = b

    disp_img = disp.detach().cpu().numpy()
    fig = plt.figure(figsize=(12.84, 3.84))
    plt.axis('off')  # hide axis
    plt.imshow(disp_img)
    plt.colorbar()

    writer.add_figure('image/disp', fig, global_step=epoch)
    writer.add_image('image/left', left_image, global_step=epoch)


def train(model, train_loader, optimizer, criterion, step):
    """
    train one epoch
    """
    for batch in train_loader:
        step += 1
        optimizer.zero_grad()

        left_img = batch['left'].to(device)
        right_img = batch['right'].to(device)
        target_disp = batch['disp'].to(device)

        mask = (target_disp > 0)
        mask = mask.detach_()

        if args.model == 'GC':
            disp = model(left_img, right_img)
            loss = criterion(disp[mask], target_disp[mask])
            loss.backward()
            optimizer.step()
            if step % args.log_per_step == 0:
                writer.add_scalar('loss/total_loss', loss, step)
                print('step: {:07} | loss: {:.5}'.format(step, loss.item()))
        elif args.model == 'P':
            disp1, disp2, disp3 = model(left_img, right_img)
            loss1, loss2, loss3 = criterion(disp1[mask], disp2[mask], disp3[mask], target_disp[mask])
            total_loss = 0.5 * loss1 + 0.7 * loss2 + 1.0 * loss3
            total_loss.backward()
            optimizer.step()
            if step % args.log_per_step == 0:
                writer.add_scalar('loss/loss1', loss1, step)
                writer.add_scalar('loss/loss2', loss2, step)
                writer.add_scalar('loss/loss3', loss3, step)
                writer.add_scalar('loss/total_loss', total_loss, step)
                print('step: {:07} | total loss: {:.5} | loss1: {:.5} | loss2: {:.5} | loss3: {:.5}'.format(step, total_loss.item(), loss1.item(), loss2.item(), loss3.item()))
        elif args.model == 'Stereo':
            disp = model(left_img, right_img)
            total_loss = criterion(disp[mask], target_disp[mask])
            total_loss.backward()
            optimizer.step()
            if step % args.log_per_step == 0:
                writer.add_scalar('loss/total_loss', total_loss, step)
                print('step: {:07} | total loss: {:.5}'.format(step, total_loss.item()))
        elif args.model == 'CFP':
            disp = model(left_img, right_img)
            total_loss = criterion(disp[mask], target_disp[mask])
            total_loss.backward()
            optimizer.step()
            if step % args.log_per_step == 0:
                writer.add_scalar('loss/total_loss', total_loss, step)
                print('step: {:07} | total loss: {:.5}'.format(step, total_loss.item()))
        elif args.model == 'HSM':
            stacked, entropy = model(left_img, right_img)
            total_loss = criterion(stacked, target_disp, mask)
            total_loss.backward()
            optimizer.step()
            if step % args.log_per_step == 0:
                writer.add_scalar('loss/total_loss', total_loss, step)
                print('step: {:07} | total loss: {:.5}'.format(step, total_loss.item()))
        elif args.model == 'Gwc' or args.model == 'D':
            disp_ests = model(left_img, right_img)
            total_loss = model_loss(disp_ests, target_disp, mask)
            total_loss.backward()
            optimizer.step()
            if step % args.log_per_step == 0:
                writer.add_scalar('loss/total_loss', total_loss, step)
                print('step: {:07} | total loss: {:.5}'.format(step, total_loss.item()))
    return step


def adjust_lr(optimizer, epoch):
    if epoch == 200:
        lr = 0.0001
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    if epoch == 800:
        lr = 0.00001
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def save(model, optimizer, epoch, step, error_1, error_3, error_5, epe, rmse, best_1pixel_error, best_3pixel_error, best_5pixel_error, best_epe, best_rmse):
    path = os.path.join(args.model_dir, '{:04}.ckpt'.format(epoch))

    state = {}
    state['state_dict'] = model.state_dict()
    state['optimizer'] = optimizer.state_dict()
    state['error_1'] = error_1
    state['error_3'] = error_3
    state['error_5'] = error_5
    state['epe'] = epe
    state['rmse'] = rmse
    state['epoch'] = epoch
    state['step'] = step

    # save model trained in this epoch
    torch.save(state, path)
    print('save model at epoch{}'.format(epoch))

    # save best model:
    if error_3 <= best_3pixel_error and epe <= best_epe:
        best_1pixel_error = error_1
        best_3pixel_error = error_3
        best_5pixel_error = error_5
        best_epe = epe
        best_rmse = rmse
        best_path = os.path.join(args.model_dir, 'best_model.ckpt'.format(epoch))
        shutil.copyfile(path, best_path)
        print('best model in epoch {}'.format(epoch))

    # save best error model:
    if error_3 <= best_3pixel_error:
        best_1pixel_error = error_1
        best_3pixel_error = error_3
        best_5pixel_error = error_5
        best_error_path = os.path.join(args.model_dir, 'best_error_model.ckpt'.format(epoch))
        shutil.copyfile(path, best_error_path)
        print('best error model in epoch {}'.format(epoch))

    # save best epe model:
    if epe <= best_epe:
        best_epe = epe
        best_rmse = rmse
        best_epe_path = os.path.join(args.model_dir, 'best_epe_model.ckpt'.format(epoch))
        shutil.copyfile(path, best_epe_path)
        print('best epe model in epoch {}'.format(epoch))

    return best_1pixel_error, best_3pixel_error, best_5pixel_error, best_epe, best_rmse


if __name__ == '__main__':
    main(args)
    writer.close()
