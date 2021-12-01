import numpy as np
import torch
import argparse
from torchstat import stat

from models_HSMNet_GFLOPs.hsm import HSMNet
from models_GCNet_GFLOPs.GCNet import GCNet
from models_PSMNet_GFLOPs.PSMnet import PSMNet
from models_GwcNet_GFLOPs.gwcnet import GwcNet
from models_CFPNet_GFLOPs.basic import CFPNet_b
from models_StereoNet_GFLOPs.stereonet import StereoNet
from models_CFPNet_GFLOPs.stackhourglass import CFPNet_s

parser = argparse.ArgumentParser(description='model GFLOPS')
parser.add_argument('--maxdisp', type=int, default=256, help='max diparity')
parser.add_argument('--model', default='HSM', help='choose which model to use, GC: GCNet, P: PSMNet, Stereo: StereoNet, CFP: CFPNet, HSM: HSMNet, Gwc: GwcNet, D: DANet')
args = parser.parse_args()


def main():
    if args.model == 'GC':
        model = GCNet(max_disp=args.maxdisp)
    elif args.model == 'P':
        model = PSMNet(max_disp=args.maxdisp)
    elif args.model == 'Stereo':
        model = StereoNet(batch_size=1, cost_volume_method='subtract').cpu()
    elif args.model == 'CFP':
        model = CFPNet_b(maxdisp=args.maxdisp).cpu()
    elif args.model == 'HSM':
        model = HSMNet(maxdisp=args.maxdisp, clean=False, level=1).cpu()
    elif args.model == 'Gwc':
        model = GwcNet(maxdisp=args.maxdisp, use_concat_volume=False)
    stat(model=model, input_size=(3, 256, 512))


if __name__ == '__main__':
    main()
