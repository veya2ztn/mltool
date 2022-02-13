import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm

class ReLUcclConvBN(nn.Module):
    def __init__(self, C_in, C_out, k, s, d, CNNModule=nn.Conv2d,Nonlinear=nn.ReLU,circularQ=True,affine=True):
        super(ReLUcclConvBN, self).__init__()
        pp = ((k-1)*d+1-s)//2
        pp = 2*pp if (float(torch.__version__[:3]) < 1.5 and circularQ) else pp

        self.op = nn.Sequential(
          Nonlinear(inplace=False),
          CNNModule(C_in, C_out, k, stride=s, padding=pp,dilation=d, bias=False,padding_mode="circular" if circularQ else 'zeros'),
          AdaptiveBatchNorm2d(C_out, affine=affine)
        )

    def forward(self, x):
        return self.op(x)
class ReLUcclConvBNWrapper:
    def __init__(self,k,s,d,Q,CNNModule=nn.Conv2d,Nonlinear=nn.ReLU):
        self.k      = k
        self.s      = s
        self.d      = d
        self.Q      = Q
        self.CNNModule=CNNModule
        self.Nonlinear=Nonlinear
    def __call__(self,C,stride,affine):
        return ReLUcclConvBN(C,C,self.k,self.s,self.d, CNNModule=self.CNNModule,Nonlinear=self.Nonlinear,circularQ=self.Q,affine=affine)


class AdaptiveBatchNorm2d(_BatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)' .format(input.dim()))
    def forward(self, input):
        self._check_input_dim(input)
        #!!!below is the only different
        if input.shape[-2]==1:return input

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:exponential_average_factor = 0.0
        else:exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        r"""
        Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:bn_training = True
        else:bn_training = (self.running_mean is None) and (self.running_var is None)

        r"""
        Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """
        return F.batch_norm(
            input,
            # If buffers are not to be tracked, ensure that they won't be updated
            self.running_mean if not self.training or self.track_running_stats else None,
            self.running_var if not self.training or self.track_running_stats else None,
            self.weight, self.bias, bn_training, exponential_average_factor, self.eps)
class ReLUConvBN(nn.Module):
  def __init__(self, C_in, C_out, kernel_size, stride, padding, CNNModule=nn.Conv2d,Nonlinear=nn.ReLU, affine=True,padding_mode='zeros'):
    super(ReLUConvBN, self).__init__()
    self.op = nn.Sequential(
      Nonlinear(inplace=False),
      CNNModule(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False,padding_mode=padding_mode),
      AdaptiveBatchNorm2d(C_out, affine=affine)
    )
  def forward(self, x):
    return self.op(x)
class DilConv(nn.Module):
  def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation=2, CNNModule=nn.Conv2d,Nonlinear=nn.ReLU, affine=True,padding_mode='zeros'):
    super(DilConv, self).__init__()
    self.op = nn.Sequential(
      Nonlinear(inplace=False),
      CNNModule(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=C_in, bias=False,padding_mode=padding_mode),
      CNNModule(C_in, C_out, kernel_size=1, padding=0, bias=False,padding_mode=padding_mode),
      AdaptiveBatchNorm2d(C_out, affine=affine),
      )
  def forward(self, x):
    return self.op(x)
class SepConv(nn.Module):
  def __init__(self, C_in, C_out, kernel_size, stride, padding, CNNModule=nn.Conv2d,Nonlinear=nn.ReLU, affine=True,padding_mode='zeros'):
    super(SepConv, self).__init__()
    self.op = nn.Sequential(
      Nonlinear(inplace=False),
      CNNModule(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False,padding_mode=padding_mode),
      CNNModule(C_in, C_in, kernel_size=1, padding=0, bias=False,padding_mode=padding_mode),
      AdaptiveBatchNorm2d(C_in, affine=affine),
      Nonlinear(inplace=False),
      CNNModule(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False,padding_mode=padding_mode),
      CNNModule(C_in, C_out, kernel_size=1, padding=0, bias=False,padding_mode=padding_mode),
      AdaptiveBatchNorm2d(C_out, affine=affine),
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
  def __init__(self, C_in, C_out, CNNModule=nn.Conv2d,Nonlinear=nn.ReLU, affine=True,padding_mode='zeros'):
    super(FactorizedReduce, self).__init__()
    assert C_out % 2 == 0
    self.relu = Nonlinear(inplace=False)
    self.conv_1 = CNNModule(C_in, C_out // 2, 1, stride=2, padding=0, bias=False,padding_mode=padding_mode)
    self.conv_2 = CNNModule(C_in, C_out // 2, 1, stride=2, padding=0, bias=False,padding_mode=padding_mode)
    self.bn = AdaptiveBatchNorm2d(C_out, affine=affine)
  def forward(self, x):
    x = self.relu(x)
    out = torch.cat([self.conv_1(x), self.conv_2(x[:,:,1:,1:])], dim=1)
    out = self.bn(out)
    return out
