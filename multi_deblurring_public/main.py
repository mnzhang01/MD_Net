# -*- coding: utf-8 -*-
"""
@author: Meina Zhang
"""

from __future__ import print_function
import matplotlib.pyplot as plt
import argparse
import os
import numpy as np
from networks.multi_net import multi_attention
import cv2
import torch
import torch.optim
from torch.autograd import Variable
import glob
import torch.nn as nn
from PIL import Image
import warnings
from tqdm import tqdm
from torch.optim.lr_scheduler import MultiStepLR
from utils.common_utils import *
from SSIM import SSIM
from skimage.transform import pyramid_gaussian
import math


parser = argparse.ArgumentParser()
parser.add_argument('--num_iter', type=int, default=1500, help='number of epochs of training')
parser.add_argument('--kernel_size', type=int, default=[79, 79], help='size of blur kernel [height, width]')
parser.add_argument('--img_size', type=int, default=[256, 256], help='size of each image dimension')
parser.add_argument('--img_size1', type=int, default=[630, 518], help='size of each image dimension')
parser.add_argument('--img_size2',  type=int, default=[630, 518], help='size of each image dimension')
parser.add_argument('--data_path', type=str, default="datasets/", help='path to blurry image')
parser.add_argument('--save_path', type=str, default="results/", help='path to save results')
parser.add_argument('--save_frequency', type=int, default=10, help='frequency to save results')
opt = parser.parse_args()
# print(opt)

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
dtype = torch.cuda.FloatTensor

warnings.filterwarnings("ignore") 

files_source = glob.glob(os.path.join(opt.data_path, 'wall.jpg'))

files_source.sort()
save_path = opt.save_path
os.makedirs(save_path, exist_ok=True)

# start #image
for f in files_source:
    INPUT = 'noise'
    pad = 'reflection'
    LR = 0.01
    num_iter = opt.num_iter
    reg_noise_std = 0.001
    p0 = 0.1
    delta_t = 0.1
    path_to_image = f
    imgname = os.path.basename(f)
    imgname = os.path.splitext(imgname)[0]

    _, imgs = get_image(path_to_image, -1) # load image and convert to np.
    y_color = np_to_torch(imgs).type(dtype)
    
    img_gray = readimg_gray(path_to_image)
    
    img_gray = np.float32(img_gray / 255.0)
    y = np.expand_dims(img_gray, 0)
    
    
    img_size = y.shape
    n_scales = 3
    imgs_trans = imgs.transpose(1,2,0)
    pyramid = list(pyramid_gaussian(imgs_trans, n_scales-1, multichannel=True))
    for l in range(len(pyramid)):
        pyramid[l] = pyramid[l].transpose(2,0,1)
        
    pyramid_size0 = pyramid[0].shape
    pyramid_size1 = pyramid[1].shape
    pyramid_size2 = pyramid[2].shape
    pyramid0 = np_to_torch(pyramid[0]).type(dtype)    
    pyramid1 = np_to_torch(pyramid[1]).type(dtype)
    pyramid2 = np_to_torch(pyramid[2]).type(dtype)  
    
    ker_size1 = math.ceil(opt.kernel_size[0]/2)
    ker_size2 = math.ceil(ker_size1/2)
    
    padh1, padw1 = ker_size1-1, ker_size1-1
    padh2, padw2 = ker_size2-1, ker_size2-1

    opt.img_size1[0], opt.img_size1[1] = pyramid_size1[1]+padh1, pyramid_size1[2]+padw1
    opt.img_size2[0], opt.img_size2[1] = pyramid_size2[1]+padh2, pyramid_size2[2]+padw2
    
    padh, padw = opt.kernel_size[0]-1, opt.kernel_size[1]-1
    opt.img_size[0], opt.img_size[1] = img_size[1]+padh, img_size[2]+padw
   
    img_size = imgs.shape

    print(imgname)
    # ######################################################################

    padw, padh = opt.kernel_size[0]-1, opt.kernel_size[1]-1
    opt.img_size[0], opt.img_size[1] = img_size[1]+padw, img_size[2]+padh
    input_depth = 8
    net_input = get_noise(input_depth, INPUT, (opt.img_size2[0], opt.img_size2[1])).type(dtype)

    n_k = 200
    net_input_kernel = get_noise(ker_size2*ker_size2, INPUT, (1, 1)).type(dtype)
    net_input_kernel.squeeze_()
    
    net = multi_attention(ker_size2,ker_size1,opt.kernel_size[0],channel=3).type(dtype)
    lossL1 = nn.L1Loss().type(dtype)
    lossL2 = nn.MSELoss().type(dtype)


    mse = torch.nn.MSELoss().type(dtype)
    ssim = SSIM().type(dtype)

    params_dict = [{'params': net.skip2.parameters(), 'lr': LR},
                   {'params': net.skip1.parameters(), 'lr': LR},
                   {'params': net.FEblock.parameters(), 'lr': 1e-2},
                    {'params': net.fcn2.parameters(), 'lr': 1e-4},
                    {'params': net.fcn1.parameters(), 'lr': 1e-4},
                    {'params': net.fcn0.parameters(), 'lr': 1e-4}]
    

    optimizer = torch.optim.Adam(params_dict)
    scheduler = MultiStepLR(optimizer, milestones=[2000, 3000, 4000], gamma=0.5)  # learning rates

    # initilization inputs
    net_input_saved = net_input.detach().clone()
    net_input_kernel_saved = net_input_kernel.detach().clone()

    for step in tqdm(range(num_iter)):

        scheduler.step(step)
        optimizer.zero_grad()



        skip_out2,kernel2,skip_out1,kernel1,skip_out0,kernel0 = net(net_input,net_input_kernel,\
                                                                    (pyramid_size1[1], pyramid_size1[2]), \
                                                                        (pyramid_size0[1], pyramid_size0[2]),\
                                                                            ker_size1,opt.kernel_size[0])
        

        out_k_m2 = kernel2.view(-1,1,ker_size2,ker_size2)
        out_k_m2 = out_k_m2.repeat(3,1,1,1)
        
        out_k_m1 = kernel1.view(-1,1,ker_size1,ker_size1)
        out_k_m1 = out_k_m1.repeat(3,1,1,1)
        
        out_k_m0 = kernel0.view(-1,1,opt.kernel_size[0],opt.kernel_size[1])
        out_k_m0 = out_k_m0.repeat(3,1,1,1)
    
        


        out_y2 = nn.functional.conv2d(skip_out2, out_k_m2, padding=0, groups =3,bias=None)
        out_y1 = nn.functional.conv2d(skip_out1, out_k_m1, padding=0, groups =3,bias=None)
        out_y0 = nn.functional.conv2d(skip_out0, out_k_m0, padding=0, groups =3,bias=None)
        
        

        if step < 300:
      
            total_loss = mse(out_y2, pyramid2) +mse(out_y1, pyramid1) + 2*mse(out_y0, pyramid0) 

        else:
            total_loss =  1-ssim(pyramid2,out_y2)+ 1-ssim(pyramid1,out_y1) + 2*(1-ssim(pyramid0,out_y0)) 


        total_loss.backward()
        optimizer.step()


        if (step + 1) % opt.save_frequency == 0:
            save_path = os.path.join(opt.save_path, '%s_x.png'%imgname)
            out_x_np = torch_to_np(skip_out0)

            out_x_np = out_x_np[:,padh//2:padh//2+img_size[1], padw//2:padw//2+img_size[2]]

            channel_out = out_x_np.transpose(1,2,0)
            channel_out = (channel_out*255).astype(np.uint8)
            channel_out = Image.fromarray(channel_out).convert('RGB')
            channel_out.save(save_path)
     

            save_path = os.path.join(opt.save_path, '%s_k.png'%imgname)
            out_k_np = out_k_m0.permute(1,0,2,3)
            out_k_np = torch_to_np(out_k_np)
            out_k_np /= np.max(out_k_np)

            channel_out = out_k_np.transpose(1,2,0)
            channel_out = (channel_out*255).astype(np.uint8)
            channel_out = Image.fromarray(channel_out).convert('RGB')
            channel_out.save(save_path)

        torch.save(net, os.path.join(opt.save_path, "%s_net.pth" % imgname))

        
            

