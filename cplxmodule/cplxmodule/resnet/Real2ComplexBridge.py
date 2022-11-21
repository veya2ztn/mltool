from ..cplx import Cplx
import torch.nn as nn
import torch
import math

class Real2Complex_V1(nn.Module):
    """
    basic follow the Paper BN->ReLU->Conv->BN->ReLU->Conv
    """
    def __init__(self,channels):
        super().__init__()
        self.get_imag_from_real=torch.nn.Sequential(
            # real: batch x (n_features * 2)
            torch.nn.BatchNorm2d(channels),
            torch.nn.ReLU(),
            torch.nn.Conv2d(channels,channels,3,stride=1,padding=1),
            torch.nn.BatchNorm2d(channels),
            torch.nn.ReLU(),
            torch.nn.Conv2d(channels,channels,3,stride=1,padding=1),
        )

    def forward(self,x):
        real = x
        imag = self.get_imag_from_real(real)
        return Cplx(real,imag)

class Real2Complex_V0(nn.Module):
    """
    basic follow the Paper BN->ReLU->Conv->BN->ReLU->Conv
    """
    def __init__(self,channels):
        super().__init__()
        self.get_imag_from_real=torch.nn.Sequential(
            # real: batch x (n_features * 2)
            torch.nn.BatchNorm2d(channels),
            torch.nn.ReLU(),
            torch.nn.Conv2d(channels,channels,3,stride=1,padding=1),
        )

    def forward(self,x):
        real = x
        imag = self.get_imag_from_real(real)
        return Cplx(real,imag)
