import torch
import torch.nn as nn
import numpy as np

from ...cplxmodule.cplxmodule import nn as nc
from ...cplxmodule.cplxmodule.cplx import Cplx
from ...cplxmodule.cplxmodule.resnet import Real2Complex_V0

class ExpandZero(nn.Module):
    def forward(self,x):
        return torch.stack([x,torch.zeros_like(x)],dim=-1)
class ExpandAngle(nn.Module):
    '''
    make sure this layer after a BN and ReLU
    x -> x*e^(ix)
    '''
    def forward(self,x):

        return Cplx(x*torch.cos(np.pi*x),x*torch.sin(np.pi*x))
class TakeNorm(nn.Module):
    def forward(self,x):
        x=abs(x)
        return x
class CplxReLUConvBN(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True,**kargs):
        super().__init__()
        self.op = nn.Sequential(
            Real2Complex_V0(C_in),
            nc.CplxConv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
            nc.CplxBatchNorm2d(C_out, affine=affine),
            TakeNorm(),
        )

    def forward(self, x):
        return self.op(x)
class CplxDilConv(nn.Module):
    def __init__(
        self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True,**kargs
    ):
        super().__init__()
        self.op = nn.Sequential(
            Real2Complex_V0(C_in),
            nc.CplxConv2d(
                C_in,
                C_in,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=C_in,
                bias=False,
            ),
            nc.CplxConv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nc.CplxBatchNorm2d(C_out, affine=affine),
            TakeNorm(),
        )

    def forward(self, x):
        return self.op(x)
class CplxSepConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True,**kargs):
        super().__init__()
        self.op = nn.Sequential(
            Real2Complex_V0(C_in),
            nc.CplxConv2d(
                C_in,
                C_in,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=C_in,
                bias=False,
            ),
            nc.CplxConv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
            nc.CplxBatchNorm2d(C_in, affine=affine),
            nc.CplxAdaptiveModReLU(),
            nc.CplxConv2d(
                C_in,
                C_in,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                groups=C_in,
                bias=False,
            ),
            nc.CplxConv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nc.CplxBatchNorm2d(C_out, affine=affine),
            TakeNorm(),
        )

    def forward(self, x):
        return self.op(x)
