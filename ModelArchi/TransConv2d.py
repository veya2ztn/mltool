import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class TransposeBottleneck(nn.Module):
    """ Adapted from torchvision.models.resnet """

    def __init__(self, in_channels, output_channels, stride,mid_channels=None,upsample=None, norm_layer=nn.BatchNorm2d):
        super().__init__()
        if mid_channels is None:mid_channels=in_channels
        optpad = stride-1
        self.conv1    = nn.ConvTranspose2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1      = norm_layer(mid_channels)
        self.conv2    = nn.ConvTranspose2d(mid_channels, mid_channels, kernel_size=3, stride=stride,padding=1,output_padding=optpad, bias=False)
        self.bn2      = norm_layer(mid_channels)
        self.conv3    = nn.ConvTranspose2d(mid_channels, output_channels, kernel_size=1, bias=False)
        self.bn3      = norm_layer(output_channels)
        self.relu     = nn.ReLU(inplace=True)
        #self.relu  = nn.LeakyReLU(negative_slope=0.6, inplace=True)
        self.stride   = stride
        self.upsample = upsample
        if self.upsample is None:
            if output_channels!=in_channels or stride >1:
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
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.upsample is not None:
            residual = self.upsample(x)

        out += residual
        out = self.relu(out)

        return out

class TransposeBottleneckV0(TransposeBottleneck):
    def __init__(self, in_channels, output_channels, stride):
        mid_channels  = in_channels
        super().__init__(in_channels, output_channels, stride,mid_channels=mid_channels)
class TransposeBottleneckV1(TransposeBottleneck):
    def __init__(self, in_channels, output_channels, stride):
        mid_channels  = output_channels
        super().__init__(in_channels, output_channels, stride,mid_channels=mid_channels)
class TransposeBottleneckV2(TransposeBottleneck):
    def __init__(self, in_channels, output_channels, stride):
        mid_channels  = (in_channels+output_channels)//2
        super().__init__(in_channels, output_channels, stride,mid_channels=mid_channels)


class UpSampleResNet(nn.Module):
    def __init__(self, block,layerconfig):
        super().__init__()
        block=block
        channel_start,_,_ = layerconfig[0]
        self.conv1   = nn.ConvTranspose2d(layerconfig[0][0], layerconfig[1][0], kernel_size=7)
        self.bn1     = nn.BatchNorm2d(layerconfig[1][0])
        self.relu    = nn.ReLU(inplace=True)
        self.firstpool = nn.AdaptiveAvgPool2d(2)
        self.inplanes= layerconfig[1][0]
        self.layers  = nn.ModuleList()
        for i in range(1,len(layerconfig)):
            channel,layers,stride = layerconfig[i]
            self.layers.append(self._make_layer(block,channel,  layers, stride=stride))
        self.finalpool = nn.AdaptiveAvgPool2d(16)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        layers = []
        layers.append(block(self.inplanes, out_channels, stride))
        self.inplanes = out_channels
        for i in range(1, blocks):
            layers.append(block(self.inplanes, out_channels, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.firstpool(x)
        for layer in self.layers:x = layer(x)
        x = self.finalpool(x)

        return x
