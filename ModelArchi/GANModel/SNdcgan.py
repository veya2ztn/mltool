import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import autograd
import time as t
import os
from itertools import chain
from torchvision import utils
from .spectral_normalization import SpectralNorm

class WassersteinLoss(torch.nn.Module):
    def forward(self, x , target):
        loss = -target.mean()*x.mean()
        return loss

class Generator(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.main_module = nn.Sequential(

            nn.ConvTranspose2d(in_channels=100, out_channels=1024, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(num_features=1024),
            nn.ReLU(True),


            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(True),


            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(True),


            nn.ConvTranspose2d(in_channels=256, out_channels=channels, kernel_size=4, stride=2, padding=1))


        self.output = nn.Tanh()

    def forward(self, x):
        x = self.main_module(x)
        return self.output(x)

class Discriminator(torch.nn.Module):
    def __init__(self, channels,version="BCEwithlogit"):
        super().__init__()
        self.version = version
        self.main_module = nn.Sequential(

            SpectralNorm(nn.Conv2d(in_channels=1, out_channels=256, kernel_size=3, stride=1, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),


            SpectralNorm(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1)),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),


            SpectralNorm(nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1)),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),


            )

        if version   == "DCGAN_L":
            self.output = nn.Sequential(SpectralNorm(nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=4, stride=1, padding=0)))
            self.metric = torch.nn.BCEWithLogitsLoss()
        elif version == "WGAN_GP":
            self.output = nn.Sequential(SpectralNorm(nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=4, stride=1, padding=0)))
            self.metric = WassersteinLoss()
        else:
            self.output = nn.Sequential(SpectralNorm(nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=4, stride=1, padding=0)),
                                        nn.Sigmoid())
            if   version == "DCGAN":self.metric = torch.nn.BCELoss()
            elif version == "DCGAN_M":self.metric = torch.nn.MSELoss()
            else:
                raise NotImplementedError
    def forward(self, x, target=None):
        x = self.main_module(x)
        x = self.output(x)
        return x.reshape(x.size(0),x.size(1)) #(b,1)

    def calculate_gradient_penalty(self, real_images, fake_images,GP_lambda= 10):
        batch_size = len(real_images)
        device     = next(self.parameters()).device

        eta = torch.FloatTensor(batch_size,1,1,1).uniform_(0,1)
        eta = eta.expand(batch_size, real_images.size(1), real_images.size(2), real_images.size(3))
        eta = eta.to(device)

        interpolated = eta * real_images + ((1 - eta) * fake_images)
        interpolated = interpolated.to(device)
        interpolated = eta * real_images + ((1 - eta) * fake_images)

        interpolated      = Variable(interpolated, requires_grad=True)
        prob_interpolated = self(interpolated)
        gradients = autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                                   grad_outputs=torch.ones(prob_interpolated.size()).to(device),
                                   create_graph=True, retain_graph=True)[0]
        grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * GP_lambda
        return grad_penalty

class Binary_Checker(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(1))
    def forward(self,x):

        shape=tuple(range(1,len(x.shape)))
        return (x**2).mean(shape).unsqueeze(1)

class DCGAN_MODEL(object):
    def __init__(self, args):
        print("DCGAN model initalization.")
        self.G = Generator(args.channels)
        if args.GAN_TYPE == "ForceBINARY":
            self.D = Binary_Checker()
        else:
            self.D = Discriminator(args.channels,args.GAN_TYPE)
        self.D.version = args.GAN_TYPE
        self.C = args.channels
        self.check_cuda(True)

    def check_cuda(self, cuda_flag=False):
        print(cuda_flag)
        if cuda_flag:
            self.cuda_index = 0
            self.cuda = True
            self.D.cuda(self.cuda_index)
            self.G.cuda(self.cuda_index)
            print("Cuda enabled flag: {}".format(self.cuda))
        else:
            self.cuda = False

    def save_to(self,path,mode="full"):
        checkpoint = self.all_state_dict(mode=mode)
        torch.save(checkpoint,path)

    def all_state_dict(self,epoch=None,mode="full"):
        checkpoint={}
        checkpoint['epoch'] = epoch
        checkpoint['D_state_dict']    = self.D.state_dict()
        checkpoint['G_state_dict']    = self.G.state_dict()
        if mode != "light":
            if hasattr(self,"I2C"):checkpoint['C_state_dict']            = self.I2C.state_dict()
            if hasattr(self,"D_optimizer"):checkpoint['D_optimizer']     = self.d_optimizer.state_dict()
            if hasattr(self,"G_optimizer"):checkpoint['G_optimizer']     = self.g_optimizer.state_dict()
            if hasattr(self,"C_optimizer"):checkpoint['C_optimizer']     = self.c_optimizer.state_dict()
        return checkpoint
