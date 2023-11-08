# -*- coding: utf-8 -*-
"""
@author: Meina Zhang
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.skip import skip
import numpy as np
from networks.networks_res import ResnetBlock, get_norm_layer

def fcn(num_input_channels=200, num_output_channels=21, num_hidden=1000):
    

    model = nn.Sequential()
    model.add(nn.Linear(num_input_channels, num_hidden,bias=True))
    model.add(nn.ReLU6())

    model.add(nn.Linear(num_hidden, num_output_channels))

    model.add(nn.Softmax())
    return model




class ChannelAttention(nn.Module):
    ## channel attention block
    def __init__(self, in_planes, ratio=16): 
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    ## spatial attention block
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
    


class skip_kernel(nn.Module):
    def __init__(self,kernel_size2,kernel_size1,kernel_size0,channel):
        super(skip_kernel, self).__init__()

        self.skip = skip(channel, channel,
                    num_channels_down = [128, 128, 128, 128, 128],
                    num_channels_up   = [128, 128, 128, 128, 128],
                    num_channels_skip = [16, 16, 16, 16, 16],
                    upsample_mode='bilinear',
                    need_sigmoid=True, need_bias=True, pad='reflection', act_fun='LeakyReLU')
        # self.skip = skip(8, 3)
        self.fcn2 = fcn(kernel_size2*kernel_size2,kernel_size2*kernel_size2) 
        self.fcn1 = fcn(kernel_size2*kernel_size2,kernel_size1*kernel_size1) 
        self.fcn0 = fcn(kernel_size1*kernel_size1,kernel_size0*kernel_size0) 


        
    def forward(self, skip_input,fcn_input,input_size1,input_size0,kernel_size1,kernel_size0):
        
        skip_out2 = self.skip(skip_input)
        kernel2 = self.fcn2(fcn_input)
        skip_input1 = F.interpolate(skip_out2, [input_size1[0]-1+kernel_size1,input_size1[1]-1+kernel_size1], mode = 'nearest')
        
        skip_out1 = self.skip(skip_input1)
        kernel1 = self.fcn1(kernel2)
        skip_input0 = F.interpolate(skip_out1, [input_size0[0]-1+kernel_size0,input_size0[1]-1+kernel_size0], mode = 'nearest')
        
        
        skip_out0 = self.skip(skip_input0)
        kernel0 = self.fcn0(kernel1)
        
        
        

        return skip_out2,kernel2,skip_out1,kernel1,skip_out0,kernel0

class FEblock(nn.Module):
    def __init__(self,channel):
        super(FEblock, self).__init__()
        
        self.conv0 = nn.Conv2d(channel, 32, kernel_size=3, padding=1, bias=None)
        self.res_block = ResnetBlock(32,'zero',get_norm_layer('none'), False, True)
        self.ca = ChannelAttention(32)
        self.sa = SpatialAttention()
    def forward(self, skip_input):
        mid_2 = self.conv0(skip_input)
        mid_2 = self.res_block(mid_2)
        mid_2 = self.ca(mid_2)*mid_2
        mid_2 = self.sa(mid_2)*mid_2
        
        return mid_2
        
        
class multi_attention(nn.Module):
    def __init__(self,kernel_size2,kernel_size1,kernel_size0,channel):
        super(multi_attention, self).__init__()

        self.skip2 = skip(8, channel,
                    num_channels_down = [128, 128, 128, 128, 128],
                    num_channels_up   = [128, 128, 128, 128, 128],
                    num_channels_skip = [16, 16, 16, 16, 16],
                    upsample_mode='bilinear',
                    need_sigmoid=True, need_bias=True, pad='reflection', act_fun='LeakyReLU')
        
        self.skip1 = skip(32+channel, channel,
                    num_channels_down = [128, 128, 128, 128, 128],
                    num_channels_up   = [128, 128, 128, 128, 128],
                    num_channels_skip = [16, 16, 16, 16, 16],
                    upsample_mode='bilinear',
                    need_sigmoid=True, need_bias=True, pad='reflection', act_fun='LeakyReLU')
        # self.skip = skip(8, 3)
        self.fcn2 = fcn(kernel_size2*kernel_size2,kernel_size2*kernel_size2) 
        self.fcn1 = fcn(kernel_size2*kernel_size2,kernel_size1*kernel_size1) 
        self.fcn0 = fcn(kernel_size1*kernel_size1,kernel_size0*kernel_size0) 
        self.FEblock = FEblock(channel)
        
    def forward(self, skip_input,fcn_input,input_size1,input_size0,kernel_size1,kernel_size0):
        
        skip_out2 = self.skip2(skip_input)
        kernel2 = self.fcn2(fcn_input)
        mid_2 = self.FEblock(skip_out2)
        skip_input1 = torch.cat((mid_2,skip_out2),1)
        skip_input1 = F.interpolate(skip_input1, [input_size1[0]-1+kernel_size1,input_size1[1]-1+kernel_size1], mode = 'nearest')

        skip_out1 = self.skip1(skip_input1)
        kernel1 = self.fcn1(kernel2)
        mid_1 = self.FEblock(skip_out1)
        skip_input0 = torch.cat((mid_1,skip_out1),1)
        skip_input0 = F.interpolate(skip_input0, [input_size0[0]-1+kernel_size0,input_size0[1]-1+kernel_size0], mode = 'nearest')
        

        skip_out0 = self.skip1(skip_input0)
        kernel0 = self.fcn0(kernel1)
        
        

        return skip_out2,kernel2,skip_out1,kernel1,skip_out0,kernel0
    
    
 