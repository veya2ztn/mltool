import math
import torch
import torch.nn as nn
import torch.nn.init as init
from . import complex_layer as nc

__all__ = ['SqueezeNet', 'squeezenet1_0', 'squeezenet1_1']


class Fire(nn.Module):

    def __init__(self, inplanes, squeeze_planes,
                 expand1x1_planes, expand3x3_planes):
        super(Fire, self).__init__()
        self.inplanes = inplanes
        self.squeeze = nc.Conv1d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nc.ReLU(inplace=True)
        self.expand1x1 = nc.Conv1d(squeeze_planes, expand1x1_planes,
                                   kernel_size=1)
        self.expand1x1_activation = nc.ReLU(inplace=True)
        self.expand3x3 = nc.Conv1d(squeeze_planes, expand3x3_planes,
                                   kernel_size=3, padding=1)
        self.expand3x3_activation = nc.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)


class SqueezeNet(nn.Module):

    def __init__(self, version=1.0):
        super(SqueezeNet, self).__init__()
        if version not in [1.0, 1.1]:
            raise ValueError("Unsupported SqueezeNet version {version}:"
                             "1.0 or 1.1 expected".format(version=version))
        #self.num_classes = num_classes
        if version == 1.0:
            self.features = nn.Sequential(
                nc.Conv1d(3, 96, kernel_size=7, stride=2),
                nc.ReLU(inplace=True),
                nc.AvgPool1d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(96, 16, 64, 64),
                Fire(128, 16, 64, 64),
                Fire(128, 32, 128, 128),
                nc.AvgPool1d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 32, 128, 128),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                nc.AvgPool1d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(512, 64, 256, 256),
            )
        else:
            self.features = nn.Sequential(
                nc.Conv1d(3, 64, kernel_size=3, stride=2),
                nc.ReLU(inplace=True),
                nc.AvgPool1d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(64, 16, 64, 64),
                Fire(128, 16, 64, 64),
                nc.AvgPool1d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(128, 32, 128, 128),
                Fire(256, 32, 128, 128),
                nc.AvgPool1d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                Fire(512, 64, 256, 256),
            )
        # Final convolution is initialized differently form the rest
        #final_conv = nc.Conv1d(512, self.num_classes, kernel_size=1)
        self.classifier = nc.Linear(in_features=1024, out_features=80)
        self.final_pool= nc.AvgPool1d(kernel_size=3,stride=2)
        for m in self.modules():
            if isinstance(m, nc.Conv1d):
                init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x = x.reshape(x.size(0),x.size(1),-1,2)
        x = self.features(x)
        x = self.final_pool(x)
        x = x.reshape(x.size(0),-1,2)
        x = self.classifier(x)
        return x


def squeezenet1_0(pretrained=False, **kwargs):
    r"""SqueezeNet model architecture from the `"SqueezeNet: AlexNet-level
    accuracy with 50x fewer parameters and <0.5MB model size"
    <https://arxiv.org/abs/1602.07360>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SqueezeNet(version=1.0, **kwargs)
    if pretrained:raise NotImplementedError
    return model


def squeezenet1_1(pretrained=False, **kwargs):
    r"""SqueezeNet 1.1 model from the `official SqueezeNet repo
    <https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.1>`_.
    SqueezeNet 1.1 has 2.4x less computation and slightly fewer parameters
    than SqueezeNet 1.0, without sacrificing accuracy.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SqueezeNet(version=1.1, **kwargs)
    if pretrained:raise NotImplementedError
    return model
