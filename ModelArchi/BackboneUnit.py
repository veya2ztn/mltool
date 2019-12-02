from __future__ import division
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class MLPlayer(torch.nn.Module):
    def __init__(self,channel_list):
        super().__init__()
        in_out_list = [[channel_list[i],channel_list[i+1]]for i in range(len(channel_list)-1)]
        layer = []
        for _in,_out in in_out_list:
            layer.append(torch.nn.Linear(_in,_out))
            layer.append(torch.nn.ReLU(inplace=True))
        self.layers = torch.nn.Sequential(*layer)
        
    def forward(self,x):
        x = self.layers(x)
        return x


class TransposeBottleneck(nn.Module):
    """ Adapted from torchvision.models.resnet """

    def __init__(self, in_channels, output_channels,mid_channel=None,stride=1, norm_layer=nn.BatchNorm2d, strategy='resnet'):
        super(TransposeBottleneck, self).__init__()
        if mid_channel is None:
            planes=(output_channels+in_channels)//2
        else:
            planes=mid_channel
        optpad = stride-1
        self.stride = stride
        self.strategy = strategy

        self.conv1 = nn.ConvTranspose2d(in_channels, planes, kernel_size=1, bias=False)
        self.bn1   = norm_layer(planes)
        self.conv2 = nn.ConvTranspose2d(planes, planes, kernel_size=3, stride=stride,padding=1,output_padding=optpad, bias=False)
        self.bn2   = norm_layer(planes)
        self.conv3 = nn.ConvTranspose2d(planes, output_channels, kernel_size=1, bias=False)
        self.bn3   = norm_layer(output_channels)
        self.relu  = nn.ReLU(inplace=True)
        self.upsample =None

        if (output_channels!=in_channels or stride >1) and strategy == 'resnet':
            self.upsample = nn.Sequential(
                    nn.ConvTranspose2d(in_channels,
                                       output_channels,
                                       kernel_size=3,
                                       stride=stride,
                                       padding=1,
                                       output_padding=optpad,
                                       bias=False) ,
                    norm_layer(output_channels),
                )

    def forward(self, x):
        res = 0
        if self.upsample is not None:
            res = self.upsample(x)
        x = self.conv1(x)
        x =   self.bn1(x)
        x =  self.relu(x)

        x = self.conv2(x)
        x =   self.bn2(x)
        x =  self.relu(x)

        x = self.conv3(x)
        x =   self.bn3(x)

        x += res
        x = self.relu(x)
        return x

class PositiveBottleneck(nn.Module):
    """
    Adapted from torchvision.models.resnet
    The output size basicly is (B,C,W//stride,H//stride)
    """

    def __init__(self, in_channels, output_channels,mid_channel=None, stride=1, dilation=1, norm_layer=nn.BatchNorm2d, strategy='resnet'):
        super(PositiveBottleneck, self).__init__()

        if mid_channel is None:
            planes=(output_channels+in_channels)//2
        else:
            planes=mid_channel
        optpad = stride-1
        self.stride = stride

        self.conv1 = nn.Conv2d(in_channels, planes, kernel_size=1, bias=False, dilation=dilation)
        self.bn1   = norm_layer(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,padding=dilation, bias=False, dilation=dilation)
        self.bn2   = norm_layer(planes)
        self.conv3 = nn.Conv2d(planes, output_channels, kernel_size=1, bias=False, dilation=dilation)
        self.bn3   = norm_layer(output_channels)

        self.relu  = nn.ReLU(inplace=True)
        self.downsample =None

        if (output_channels!=in_channels or stride >1) and strategy == 'resnet':
            self.downsample = nn.Sequential(
                    nn.Conv2d(in_channels,
                              output_channels,
                               kernel_size=3,
                               stride=stride,
                               padding=dilation,
                               bias=False) ,
                    norm_layer(output_channels),
                )

    def forward(self, x):
        res = 0
        if self.downsample is not None:
            res = self.downsample(x)
        x = self.conv1(x)
        x =   self.bn1(x)
        x =  self.relu(x)

        x = self.conv2(x)
        x =   self.bn2(x)
        x =  self.relu(x)

        x = self.conv3(x)
        x =   self.bn3(x)

        x += res
        x = self.relu(x)

        return x


class OneBottleneck(nn.Module):
    """
    Adapted from torchvision.models.resnet
    The output size basicly is (B,C,W//stride,H//stride)
    ---Two Conv Model---
        x = bn1<-conv1<-x
        x = relu(x+res)
    """

    def __init__(self, in_channels, output_channels, stride=1, dilation=1, norm_layer=nn.BatchNorm2d,bias=False, strategy='resnet'):
        super().__init__()

        planes=output_channels
        self.stride = stride
        
        self.ConvBlock1= nn.Sequential(
                        nn.Conv2d(in_channels, planes, kernel_size=1, bias=bias, dilation=dilation),
                        norm_layer(planes),
                        #nn.ReLU(inplace=True),
                        )

        self.relu  = nn.ReLU(inplace=True)
        self.downsample =None

        if (output_channels!=in_channels or stride >1) and strategy == 'resnet':
            self.downsample = nn.Sequential(
                    nn.Conv2d(in_channels,
                              output_channels,
                               kernel_size=3,
                               stride=stride,
                               padding=dilation,
                               bias=bias) ,
                    norm_layer(output_channels),
                )

    def forward(self, x):
        res = 0
        if self.downsample is not None:
            res = self.downsample(x)      
        x = self.ConvBlock1(x)
        x += res
        x = self.relu(x)

        return x    

class TwoBottleneck(nn.Module):
    """
    Adapted from torchvision.models.resnet
    The output size basicly is (B,C,W//stride,H//stride)
    ---Two Conv Model---
        x = relu<-bn1<-conv1<-x
        x = bn2<-conv2<-x
        x = relu(x+res)
    """

    def __init__(self, in_channels, output_channels,mid_channel=None, stride=1, dilation=1, norm_layer=nn.BatchNorm2d,bias=False, strategy='resnet'):
        super().__init__()

        if mid_channel is None:
            planes=(output_channels+in_channels)//2
        else:
            planes=mid_channel
        optpad = stride-1
        self.stride = stride
        
        self.ConvBlock1= nn.Sequential(
                        nn.Conv2d(in_channels, planes, kernel_size=1, dilation=dilation,bias=bias),
                        norm_layer(planes),
                        nn.ReLU(inplace=True),
                        )
        self.ConvBlock2=nn.Sequential(
                        nn.Conv2d(planes, planes, kernel_size=3, stride=stride,padding=dilation, bias=bias, dilation=dilation),
                        norm_layer(planes),
                        #nn.ReLU(inplace=True),
                       )

        self.relu  = nn.ReLU(inplace=True)
        self.downsample =None

        if (output_channels!=in_channels or stride >1) and strategy == 'resnet':
            self.downsample = nn.Sequential(
                    nn.Conv2d(in_channels,
                              output_channels,
                               kernel_size=3,
                               stride=stride,
                               padding=dilation,
                               bias=bias) ,
                    norm_layer(output_channels),
                )

    def forward(self, x):
        res = 0
        if self.downsample is not None:
            res = self.downsample(x)
            
        x = self.ConvBlock1(x)
        x = self.ConvBlock2(x)

        x += res
        x = self.relu(x)

        return x
    
class TriBottleneck(nn.Module):
    """
    Adapted from torchvision.models.resnet
    The output size basicly is (B,C,W//stride,H//stride)
    ---Three Conv Model---
        x = relu<-bn1<-conv1<-x
        x = relu<-bn2<-conv2<-x
        x = bn3<-conv3<-x
        x = relu(x+res)
    """

    def __init__(self, in_channels, output_channels,mid_channel=None, stride=1, dilation=1, norm_layer=nn.BatchNorm2d,bias=False, strategy='resnet'):
        super().__init__()

        if mid_channel is None:
            planes=(output_channels+in_channels)//2
        else:
            planes=mid_channel
        optpad = stride-1
        self.stride = stride
        
        self.ConvBlock1= nn.Sequential(
                        nn.Conv2d(in_channels, planes, kernel_size=1, bias=bias, dilation=dilation),
                        norm_layer(planes),
                        nn.ReLU(inplace=True),
                        )
        self.ConvBlock2=nn.Sequential(
                        nn.Conv2d(planes, planes, kernel_size=3, stride=stride,padding=dilation, bias=bias, dilation=dilation),
                        norm_layer(planes),
                        nn.ReLU(inplace=True),
                        )

        self.ConvBlock3 = nn.Sequential(
                        nn.Conv2d(planes, output_channels, kernel_size=1,bias=bias, dilation=dilation),
                        norm_layer(output_channels),
                        #nn.ReLU(inplace=True),
                        )
        
        self.relu  = nn.ReLU(inplace=True)
        self.downsample =None

        if (output_channels!=in_channels or stride >1) and strategy == 'resnet':
            self.downsample = nn.Sequential(
                    nn.Conv2d(in_channels,
                              output_channels,
                               kernel_size=3,
                               stride=stride,
                               padding=dilation,
                               bias=bias) ,
                    norm_layer(output_channels),
                )

    def forward(self, x):
        res = 0
        if self.downsample is not None:
            res = self.downsample(x)
            
        x = self.ConvBlock1(x)
        x = self.ConvBlock2(x)
        x = self.ConvBlock3(x)
        
        x += res
        x = self.relu(x)

        return x