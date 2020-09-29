from . import complex_operation as C
from .complex_tensor import ComplexTensor
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np
from torch.nn.modules.conv import _single, _pair, _triple, _reverse_repeat_tuple
from torch.nn.modules.conv import _size_1_t, _size_2_t, _size_3_t
from torch.nn.modules.conv import Optional, List, Tuple,init
from torch import Tensor
import math
def view_as_real(x):
    if x.shape[-1]==2 and x.dtype != torch.complex64:
        return x
    else:
        return torch.view_as_real(x)
class ComplexLinear(torch.nn.Linear):
    '''
    Applies a linear transformation to the incoming data:
    `y = x*A + b`
    (B,I)*(I,O)+(O,)
    cplx  cplx  cplx

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``
    '''
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features,out_features, bias=bias)
        self.weight  = Parameter(torch.randn(in_features,out_features, dtype=torch.cfloat))
        if bias:
            self.bias = Parameter(torch.randn(out_features, dtype=torch.cfloat))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
    def forward(self, x):
        x = view_as_real(x)
        x = C.complex_mm(x,view_as_real(self.weight)) +view_as_real(self.bias)
        x = torch.view_as_complex(x)
        return x
class _ComplexConvNd(torch.nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: _size_1_t,
                 stride: _size_1_t,
                 padding: _size_1_t,
                 dilation: _size_1_t,
                 transposed: bool,
                 output_padding: _size_1_t,
                 groups: int,
                 bias: Optional[Tensor],
                 padding_mode: str) -> None:
        super().__init__()
        kernel_size = self.size_unit(kernel_size)
        stride      = self.size_unit(stride)
        padding     = self.size_unit(padding)
        dilation    = self.size_unit(dilation)
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        valid_padding_modes = {'zeros', 'reflect', 'replicate', 'circular'}
        if padding_mode not in valid_padding_modes:
            raise ValueError("padding_mode must be one of {}, but got padding_mode='{}'".format(
                valid_padding_modes, padding_mode))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        self.padding_mode = padding_mode
        # `_reversed_padding_repeated_twice` is the padding to be passed to
        # `F.pad` if needed (e.g., for non-zero padding types that are
        # implemented as two ops: padding + conv). `F.pad` accepts paddings in
        # reverse order than the dimension.
        self._reversed_padding_repeated_twice = _reverse_repeat_tuple(self.padding, 2)
        if transposed:
            self.weight = Parameter(torch.randn(
                in_channels, out_channels // groups, *kernel_size,dtype=torch.cfloat))
        else:
            self.weight = Parameter(torch.randn(
                out_channels, in_channels // groups, *kernel_size,dtype=torch.cfloat))
        if bias:
            self.bias = Parameter(torch.randn(out_channels,dtype=torch.cfloat))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        return s.format(**self.__dict__)

    def __setstate__(self, state):
        super(_ConvNd, self).__setstate__(state)
        if not hasattr(self, 'padding_mode'):
            self.padding_mode = 'zeros'
class ComplexConv1d(_ComplexConvNd):
    def __init__(self,in_channels: int,out_channels: int,kernel_size: _size_1_t,
        stride: _size_1_t = 1,padding: _size_1_t = 0,dilation: _size_1_t = 1,
        groups: int = 1,bias: bool = True,padding_mode: str = 'zeros'
    ):
        self.size_unit=_single
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _single(0), groups, bias, padding_mode)
    def forward(self, x: Tensor) -> Tensor:
        x     = view_as_real(x)
        weight= view_as_real(self.weight)
        bias  = view_as_real(self.bias) if self.bias is not None else None
        pad_num=self.padding
        if self.padding_mode == 'circular':
            expanded_padding = [0,0]
            for pad in self.padding[::-1]:
                expanded_padding+=[(pad+1)//2,(pad+1)//2]

            expanded_padding = tuple(expanded_padding)
            x = F.pad(x, expanded_padding, mode='circular')
            pad_num = 0
        x= C.complex_conv1d(x, weight,bias=bias,
                         stride=self.stride,padding=pad_num,
                         dilation=self.dilation,groups=self.groups)
        x = torch.view_as_complex(x)
        return x
class ComplexConv2d(_ComplexConvNd):
    def __init__(self,in_channels: int,out_channels: int,kernel_size: _size_2_t,
        stride: _size_2_t = 1,padding: _size_2_t = 0,dilation: _size_2_t = 1,
        groups: int = 1,bias: bool = True,padding_mode: str = 'zeros'
    ):
        self.size_unit=_pair
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation,
                         False, _pair(0), groups, bias, padding_mode)

    def forward(self, x: Tensor) -> Tensor:
        x     = view_as_real(x)
        weight= view_as_real(self.weight)
        bias  = view_as_real(self.bias) if self.bias is not None else None
        pad_num=self.padding
        if self.padding_mode == 'circular':
            expanded_padding = [0,0]
            for pad in self.padding[::-1]:
                expanded_padding+=[(pad+1)//2,(pad+1)//2]

            expanded_padding = tuple(expanded_padding)
            x = F.pad(x, expanded_padding, mode='circular')
            pad_num = 0
        x= C.complex_conv2d(x, weight,bias=bias,
                         stride=self.stride,padding=pad_num,
                         dilation=self.dilation,groups=self.groups)
        x = torch.view_as_complex(x)
        return x
class ComplexTanh(torch.nn.Tanh):
    def forward(self, x):
        x = x.tanh()
        return x

class _ComplexReImNorm(torch.nn.Module):
    '''
    batch norm real and imag individual
    resize real to (0,r_size) Gauss distribution
    reeize imag to (0,i_size) Gauss distribution
    '''
    def forward(self,x):
        real = self.r_norm(x.real)*self.r_size
        imag = self.i_norm(x.imag)*self.i_size
        x    = torch.stack([real,imag],-1)
        x    = torch.view_as_complex(x)
        return x
class ComplexReImNorm1d(_ComplexReImNorm):
    def __init__(self,num_features,size=[1,1], **kargs):
        super().__init__()
        self.r_norm=torch.nn.BatchNorm1d(num_features, **kargs)
        self.i_norm=torch.nn.BatchNorm1d(num_features, **kargs)
        self.r_size,self.i_size = size
class ComplexReImNorm2d(_ComplexReImNorm):
    def __init__(self,num_features,size=[1,1], **kargs):
        super().__init__()
        self.r_norm=torch.nn.BatchNorm2d(num_features, **kargs)
        self.i_norm=torch.nn.BatchNorm2d(num_features, **kargs)
        self.r_size,self.i_size = size

class _ComplexWrapper(torch.nn.Module):
    def forward(self,x):
        x = view_as_real(x)
        x = self.real_layer(x)
        x = torch.view_as_complex(x)
        return x
class ComplexAvgPool1d(_ComplexWrapper):
    def __init__(self,kernel_size,**kargs):
        super().__init__()
        kernel_size = _pair(kernel_size)
        kernel_size = tuple(list(kernel_size)+[1])
        self.real_layer   = torch.nn.AvgPool2d(kernel_size,**kargs)
class ComplexAdaptiveAvgPool1d(_ComplexWrapper):
    def __init__(self,kernel_size,**kargs):
        super().__init__()
        kernel_size = _pair(kernel_size)
        kernel_size = tuple(list(kernel_size)+[2])
        self.real_layer   = torch.nn.AdaptiveAvgPool2d(kernel_size,**kargs)
class ComplexAvgPool2d(_ComplexWrapper):
    def __init__(self,kernel_size,**kargs):
        super().__init__()
        kernel_size = _pair(kernel_size)
        kernel_size = tuple(list(kernel_size)+[1])
        self.real_layer   = torch.nn.AvgPool3d(kernel_size,**kargs)
class ComplexAdaptiveAvgPool2d(_ComplexWrapper):
    def __init__(self,kernel_size,**kargs):
        super().__init__()
        kernel_size = _pair(kernel_size)
        kernel_size = tuple(list(kernel_size)+[2])
        self.real_layer   = torch.nn.AdaptiveAvgPool3d(kernel_size,**kargs)
class ComplexBatchNorm1d(_ComplexWrapper):
    def __init__(self,size,**kargs):
        super().__init__()
        self.real_layer   = torch.nn.BatchNorm2d(size,**kargs)
class ComplexBatchNorm2d(_ComplexWrapper):
    def __init__(self,size,**kargs):
        super().__init__()
        self.real_layer   = torch.nn.BatchNorm3d(size,**kargs)

class ComplexMLPlayer(torch.nn.Module):
    def __init__(self,channel_list):
        super().__init__()
        in_out_list = [[channel_list[i],channel_list[i+1]]for i in range(len(channel_list)-1)]
        layer = []
        for _in,_out in in_out_list:
            layer.append(ComplexLinear(_in,_out))
            layer.append(ComplexReLU(inplace=True))
        self.layers = torch.nn.Sequential(*layer)

    def forward(self,x):
        x = self.layers(x)
        return x
