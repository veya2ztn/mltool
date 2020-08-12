import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class Bottleneck(nn.Module):
    """ Adapted from torchvision.models.resnet """

    def __init__(self, in_channels, output_channels, stride,mid_channels=None,upsample=None, norm_layer=nn.BatchNorm2d):
        super().__init__()
        if mid_channels is None:mid_channels=in_channels
        optpad = stride-1
        self.conv1    = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1      = norm_layer(mid_channels)
        self.conv2    = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride,padding=1, bias=False)
        self.bn2      = norm_layer(mid_channels)
        self.conv3    = nn.Conv2d(mid_channels, output_channels, kernel_size=1, bias=False)
        self.bn3      = norm_layer(output_channels)
        self.relu     = nn.ReLU(inplace=True)
        #self.relu  = nn.LeakyReLU(negative_slope=0.6, inplace=True)
        self.stride   = stride
        self.upsample = upsample
        if self.upsample is None:
            if output_channels!=in_channels or stride >1:
                self.upsample = nn.Sequential(
                        nn.Conv2d(in_channels,
                                           output_channels,
                                           kernel_size=3,
                                           stride=stride,
                                           padding=1,
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

class BottleneckV0(Bottleneck):
    def __init__(self, in_channels, output_channels, stride):
        mid_channels  = in_channels
        super().__init__(in_channels, output_channels, stride,mid_channels=mid_channels)
class BottleneckV1(Bottleneck):
    def __init__(self, in_channels, output_channels, stride):
        mid_channels  = output_channels
        super().__init__(in_channels, output_channels, stride,mid_channels=mid_channels)
class BottleneckV2(Bottleneck):
    def __init__(self, in_channels, output_channels, stride):
        mid_channels  = (in_channels+output_channels)//2
        super().__init__(in_channels, output_channels, stride,mid_channels=mid_channels)


class ResNetConfig(nn.Module):
    def __init__(self, block,layerconfig):
        super().__init__()
        block=block
        channel_start,_,_ = layerconfig[0]
        self.conv1   = nn.Conv2d(layerconfig[0][0], layerconfig[1][0], kernel_size=7)
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


class FPNUpSample(nn.Module):
    def __init__(self, block,layerconfig):
        super().__init__()
        block=block
        channel_start,_,_ = layerconfig[0]
        self.conv1   = nn.Conv2d(layerconfig[0][0], layerconfig[1][0], kernel_size=7)
        self.bn1     = nn.BatchNorm2d(layerconfig[1][0])
        self.relu    = nn.ReLU(inplace=True)
        self.firstpool = nn.AdaptiveAvgPool2d(2)
        self.inplanes = layerconfig[1][0]


        # Bottom-up layers
        self.layer2 = self._make_layer(block, layerconfig[1][0], layerconfig[1][1], stride=layerconfig[1][2])
        self.layer3 = self._make_layer(block, layerconfig[2][0], layerconfig[2][1], stride=layerconfig[2][2])
        self.layer4 = self._make_layer(block, layerconfig[3][0], layerconfig[3][1], stride=layerconfig[3][2])
        self.layer5 = self._make_layer(block, layerconfig[4][0], layerconfig[4][1], stride=layerconfig[4][2])

        self.conv6 = nn.Conv2d(layerconfig[4][0], layerconfig[1][0], kernel_size=3, stride=2, padding=1)
        self.conv7 = nn.Conv2d(layerconfig[1][0], layerconfig[1][0], kernel_size=3, stride=2, padding=1)

        # Top layer
        self.toplayer = nn.Conv2d(layerconfig[4][0], layerconfig[1][0], kernel_size=1, stride=1, padding=0)  # Reduce channels

        # Smooth layers
        self.smooth1 = nn.Conv2d(layerconfig[1][0], layerconfig[1][0], kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(layerconfig[1][0], layerconfig[1][0], kernel_size=3, stride=1, padding=1)

        # Lateral layers
        self.latlayer1 = nn.Conv2d(layerconfig[3][0], layerconfig[1][0], kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(layerconfig[2][0], layerconfig[1][0], kernel_size=1, stride=1, padding=0)

        self.finalpool = nn.AdaptiveAvgPool2d(16)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        layers = []
        layers.append(block(self.inplanes, out_channels, stride))
        self.inplanes = out_channels
        for i in range(1, blocks):
            layers.append(block(self.inplanes, out_channels, 1))
        return nn.Sequential(*layers)

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _,_,H,W = y.size()
        return F.interpolate(x, size=(H,W), mode='bilinear',align_corners=True) + y

    def forward(self, x):
        # Bottom-up
        c1 = self.firstpool(self.relu(self.bn1(self.conv1(x))))
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        c5 = self.layer5(c4)
        p6 = self.conv6(c5)
        p7 = self.conv7(F.relu(p6))
        # Top-down
        p5 = self.toplayer(c5)
        p4 = self._upsample_add(p5, self.latlayer1(c4))
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        # Smooth
        p4 = self.smooth1(p4)
        p3 = self.smooth2(p3)

        return p3, p4, p5, p6, p7
