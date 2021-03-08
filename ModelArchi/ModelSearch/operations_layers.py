import torch
import torch.nn as nn


class ReLUcclConvBN(nn.Module):
    def __init__(self, C_in, C_out, k, s, d, circularQ=True,affine=True):
        super(ReLUcclConvBN, self).__init__()
        pp = ((k-1)*d+1-s)//2
        pp = 2*pp if (float(torch.__version__[:3]) < 1.5 and circularQ) else pp

        self.op = nn.Sequential(
          nn.ReLU(inplace=False),
          nn.Conv2d(C_in, C_out, k, stride=s, padding=pp,dilation=d, bias=False,padding_mode="circular" if circularQ else 'zeros'),
          nn.BatchNorm2d(C_out, affine=affine)
        )

    def forward(self, x):
        return self.op(x)
class ReLUcclConvBNWrapper:
    def __init__(self,k,s,d,Q):
        self.k      = k
        self.s      = s
        self.d      = d
        self.Q      = Q

    def __call__(self,C,stride,affine):
        return ReLUcclConvBN(C,C,self.k,self.s,self.d,circularQ=self.Q,affine=affine)


class ReLUConvBN(nn.Module):

  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
    super(ReLUConvBN, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
      nn.BatchNorm2d(C_out, affine=affine)
    )

  def forward(self, x):
    return self.op(x)
class DilConv(nn.Module):

  def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
    super(DilConv, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=C_in, bias=False),
      nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm2d(C_out, affine=affine),
      )

  def forward(self, x):
    return self.op(x)
class SepConv(nn.Module):

  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
    super(SepConv, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
      nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm2d(C_in, affine=affine),
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
      nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm2d(C_out, affine=affine),
      )

  def forward(self, x):
    return self.op(x)
class Identity(nn.Module):
  def __init__(self):
    super(Identity, self).__init__()
  def forward(self, x):
    return x
class Zero(nn.Module):
  def __init__(self, stride):
    super(Zero, self).__init__()
    self.stride = stride
  def forward(self, x):
    if self.stride == 1:
      return x.mul(0.)
    return x[:,:,::self.stride,::self.stride].mul(0.)
class FactorizedReduce(nn.Module):

  def __init__(self, C_in, C_out, affine=True):
    super(FactorizedReduce, self).__init__()
    assert C_out % 2 == 0
    self.relu = nn.ReLU(inplace=False)
    self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
    self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
    self.bn = nn.BatchNorm2d(C_out, affine=affine)

  def forward(self, x):
    x = self.relu(x)
    out = torch.cat([self.conv_1(x), self.conv_2(x[:,:,1:,1:])], dim=1)
    out = self.bn(out)
    return out
