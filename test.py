import argparse
from cgi import test
import cv2
import glob
import numpy as np
from collections import OrderedDict
import os
import torch
import requests
import time
import sys
import utils.utils_image as util
from data.test_dataloder import Dataset as D
from torch.utils.data import DataLoader
from models.network_fnet import FNet

def test(save_dir,a_dir,b_dir,in_channelA,in_channelB,model):
    print(a_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    test_set = D(a_dir, b_dir, in_channelA, in_channelB)
    test_loader = DataLoader(test_set, batch_size=1,
                                     shuffle=False, num_workers=1,drop_last=False, pin_memory=True)
    for i, test_data in enumerate(test_loader):
        imgname = test_data['A_path'][0]
        img_a = test_data['A'].to(device)
        img_b = test_data['B'].to(device)
        if in_channelB==3:
            ycbcr=util.tensor2uint(img_b.detach()[0].float().cpu())
            img_b=img_b[:,0,:,:]
            img_b=img_b.unsqueeze(0)
        # inference
        with torch.no_grad():
            output = model(img_a, img_b)
            output = output.detach()[0].float().cpu()
        output = util.tensor2uint(output)
        if in_channelB==3:
            ycbcr[:,:,0] = output
            output = util.ycbcr2rgb(ycbcr)
        save_name = os.path.join(save_dir, os.path.basename(imgname))
        util.imsave(output, save_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str,
                        default='checkpoints/FNet.pth')
    parser.add_argument('--root_path', type=str, default="datasets/test_img",
                        help='input test image root folder')
    parser.add_argument('--dataset', type=str, default='TNO')
    parser.add_argument('--A_dir', type=str, default='ir',
                        help='modality 2 image name (ir,nir,MRI)')
    parser.add_argument('--B_dir', type=str, default='vi',
                        help='modality 1 image name (vi,CT,PET,SPECT)')
    parser.add_argument('--save_dir', type=str, default='results',
                        help='output image root folder')
    parser.add_argument('--color', action='store_true')
    args = parser.parse_args()
    a_dir = os.path.join(args.root_path,args.dataset,args.A_dir)
    b_dir = os.path.join(args.root_path,args.dataset,args.B_dir)
    save_dir = os.path.join(args.save_dir,args.dataset)
    os.makedirs(save_dir,exist_ok=True)
    in_channelA = 1
    if args.color:
        in_channelB = 3
    else:
        in_channelB = 1
    model = FNet()
    model.load_state_dict(torch.load(args.model_path))
    model.eval()
    test(save_dir,a_dir,b_dir,in_channelA,in_channelB,model)
