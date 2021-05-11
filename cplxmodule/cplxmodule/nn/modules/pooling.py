import math
import torch

from torch.nn.modules.utils import _single, _pair, _triple
from .base import CplxToCplx
from ... import Cplx


class RealImagePool:
    def forward(self,input):
        return Cplx(self.engineer(input.real), self.engineer(input.imag))

class ComplexAdaptiveAvgPool1d(RealImagePool,CplxToCplx):
    def __init__(self,output_size,**kargs):
        super().__init__()
        self.engineer = torch.nn.AdaptiveAvgPool1d(output_size,**kargs)

class ComplexAdaptiveAvgPool2d(RealImagePool,CplxToCplx):
    def __init__(self,output_size,**kargs):
        super().__init__()
        self.engineer = torch.nn.AdaptiveAvgPool2d(output_size,**kargs)

class ComplexAdaptiveAvgPool3d(RealImagePool,CplxToCplx):
    def __init__(self,output_size,**kargs):
        super().__init__()
        self.engineer = torch.nn.AdaptiveAvgPool3d(output_size,**kargs)

class ComplexAvgPool1d(RealImagePool,CplxToCplx):
    def __init__(self,kernel_size,**kargs):
        super().__init__()
        self.engineer = torch.nn.AvgPool1d(kernel_size,**kargs)

class ComplexAvgPool2d(RealImagePool,CplxToCplx):
    def __init__(self,kernel_size,**kargs):
        super().__init__()
        self.engineer = torch.nn.AvgPool2d(kernel_size,**kargs)

class ComplexAvgPool3d(RealImagePool,CplxToCplx):
    def __init__(self,kernel_size,**kargs):
        super().__init__()
        self.engineer = torch.nn.AvgPool3d(kernel_size,**kargs)
