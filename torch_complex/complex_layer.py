from . import complex_operation as C
from .complex_tensor import ComplexTensor
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np
from torch.nn.modules.utils import _pair,_single,_triple
#from .layer_transformer import SemiToReal ,SemiToReal_Conv2d_first_layer
# Realization = SemiToReal
# Realization_Conv2d_first_layer=SemiToReal_Conv2d_first_layer
NEW_TORCH_FLAG =  (int(torch.__version__.split('.')[0])>=1) and (int(torch.__version__.split('.')[1])>=6)
NEW_TORCH_FLAG = False #we wont use new torch complex tensor, since it is not compelete 

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
        super(ComplexLinear, self).__init__(in_features,out_features, bias=bias)
        self.weight  = Parameter(torch.Tensor(in_features,out_features,2))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features,2))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()


    def forward(self, x):
        x = C.complex_mm(x,self.weight) +self.bias
        x.__class__ = ComplexTensor
        return x
class RealizedLinear(torch.nn.Linear):
    '''
    input size : (batch, size_source,2)
    do flatten ->(batch, 2*size_source)
    do liner   ->(batch, 2*size_target)
    do split   ->(batch, size_target,2)
    '''
    def forward(self,x):
        assert x.shape[-1]==2
        return F.linear(x.flatten(start_dim=-2), self.weight, self.bias).reshape(tuple(list(x.shape)[:-2]+[-1,2]))
class GroupedLinear(torch.nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features,out_features, bias=bias)
        self.weight  = Parameter(torch.Tensor(out_features,in_features,2))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features,2))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
    def forward(self,x):
        assert x.shape[-1]==2
        x1 = F.linear(x[...,0], self.weight[...,0], self.bias[...,0])
        x2 = F.linear(x[...,1], self.weight[...,1], self.bias[...,1])

        return torch.stack([x1,x2],-1)
class ComplexConv1d(torch.nn.Conv1d):
    def __init__(self,in_channels, out_channels, kernel_size, bias=True,**kargs):
        kernel_size = _single(kernel_size)
        kernel_size = tuple(list(kernel_size)+[2])
        super(ComplexConv1d,self).__init__(
            in_channels, out_channels, kernel_size, bias=bias,**kargs)
        if bias:
            self.bias = Parameter(torch.randn(len(self.bias),2))

    def forward(self, x):
        pad_num=self.padding
        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[1] + 1) // 2, self.padding[1] // 2,
                                (self.padding[0] + 1) // 2, self.padding[0] // 2,
                                                         0,                    0)
            x = F.pad(x, expanded_padding, mode='circular')
            pad_num = _single(0)

        x= C.complex_conv1d(x, self.weight,bias=self.bias,
                         stride=self.stride,padding=pad_num,
                         dilation=self.dilation,groups=self.groups)
        x.__class__ = ComplexTensor
        return x
class ComplexConv2d(torch.nn.Conv2d):
    def __init__(self,in_channels, out_channels, kernel_size, bias=True,**kargs):
        kernel_size = _pair(kernel_size)
        kernel_size = tuple(list(kernel_size)+[2])
        super(ComplexConv2d,self).__init__(
            in_channels, out_channels, kernel_size, bias=bias,**kargs)
        if bias:
            self.bias = Parameter(torch.randn(len(self.bias),2))
    def forward(self, x):
        pad_num=self.padding
        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[1] + 1) // 2, self.padding[1] // 2,
                                (self.padding[0] + 1) // 2, self.padding[0] // 2,
                                                         0,                    0)
            x = F.pad(x, expanded_padding, mode='circular')
            pad_num = _pair(0)
        x= C.complex_conv2d(x, self.weight,bias=self.bias,
                         stride=self.stride,padding=pad_num,
                         dilation=self.dilation,groups=self.groups)
        x.__class__ = ComplexTensor
        return x
class RealizedConv2d(torch.nn.Conv2d):

    def forward(self,x):
        return F.conv3d(x, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)[...,:2]
class ComplexConv3d(torch.nn.Conv3d):pass
class ComplexReLU(torch.nn.ReLU):
    __constants__ = ['inplace']
    def __init__(self, inplace=False):
        super(ComplexReLU, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        real = F.relu(x[...,0], inplace=self.inplace)
        imag = x[...,1]
        x=torch.stack([real,imag],-1)
        x.__class__ = ComplexTensor
        return x

    def extra_repr(self):
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str
class ComplexTanh(torch.nn.Tanh):
    def forward(self, x):
        assert x.shape[-1]==2
        if NEW_TORCH_FLAG:

            # torch 1.6 has complex operation support
            x = torch.view_as_complex(x)
            x = x.tanh()
            x = torch.view_as_real(x)
        else:
            x = C.complex_tanh(x)
        x.__class__ = ComplexTensor
        return x

    # def extra_repr(self):
    #     inplace_str = 'inplace=True' if self.inplace else ''
    #     return inplace_str

class ComplexReImNorm1d(torch.nn.Module):pass
class ComplexReImNorm2d(torch.nn.Module):
    '''
    batch norm real and imag individual
    resize real to (0,1) Gauss distribution
    reeize imag to (0,2*pi) Gauss distribution
    '''
    def __init__(self,num_features, **kargs):
        super().__init__()
        self.r_norm=torch.nn.BatchNorm2d(num_features, **kargs)
        self.i_norm=torch.nn.BatchNorm2d(num_features, **kargs)
    def forward(self,x):
        real = x[...,0]
        imag = x[...,1]
        pi   = np.pi
        real = self.r_norm(real)
        imag = self.i_norm(imag)*2*np.pi
        x    = torch.stack([real,imag],-1)
        x.__class__ = ComplexTensor
        return x
class ComplexReImNorm3d(torch.nn.Module):pass

class ComplexBatchNorm1d(torch.nn.BatchNorm2d):
    '''
    batch norm real and imag together
    resize (W,H,2) data form to (0,1) Gauss distribution
    '''
    pass
class ComplexBatchNorm2d(torch.nn.BatchNorm3d):
    '''
    batch norm real and imag together
    resize (W,H,2) data form to (0,1) Gauss distribution
    '''
    pass
class ComplexBatchNorm3d(torch.nn.BatchNorm3d):pass

class ComplexAvgPool1d(torch.nn.AvgPool2d):
    def __init__(self,kernel_size,**kargs):
        kernel_size = _single(kernel_size)
        kernel_size = tuple(list(kernel_size)+[1])
        super().__init__(kernel_size,**kargs)
class ComplexMaxPool1d(torch.nn.MaxPool2d):
    def __init__(self,kernel_size,**kargs):
        kernel_size = _single(kernel_size)
        kernel_size = tuple(list(kernel_size)+[1])
        super().__init__(kernel_size,**kargs)
class ComplexAdaptiveAvgPool1d(torch.nn.AdaptiveAvgPool2d):
    def __init__(self,kernel_size,**kargs):
        kernel_size = _single(kernel_size)
        kernel_size = tuple(list(kernel_size)+[2])
        super().__init__(kernel_size,**kargs)

class ComplexAvgPool2d(torch.nn.AvgPool3d):
    def __init__(self,kernel_size,**kargs):
        kernel_size = _pair(kernel_size)
        kernel_size = tuple(list(kernel_size)+[1])
        super().__init__(kernel_size,**kargs)
class ComplexMaxPool2d(torch.nn.MaxPool3d):
    def __init__(self,kernel_size,**kargs):
        kernel_size = _pair(kernel_size)
        kernel_size = tuple(list(kernel_size)+[1])
        super().__init__(kernel_size,**kargs)
class ComplexAdaptiveAvgPool2d(torch.nn.AdaptiveAvgPool3d):
    def __init__(self,kernel_size,**kargs):
        kernel_size = _pair(kernel_size)
        kernel_size = tuple(list(kernel_size)+[2])
        super().__init__(kernel_size,**kargs)

# class _ComplexAvgPoolNd(torch.nn.modules.pooling._AvgPoolNd):
#     def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False,
#                  count_include_pad=True):
#         super(_ComplexAvgPoolNd, self).__init__()
#         self.kernel_size = kernel_size
#         self.stride = stride or kernel_size
#         self.padding = padding
#         self.ceil_mode = ceil_mode
#         self.count_include_pad = count_include_pad
#
#     def forward(self,x):
#         real = x[...,0]
#         imag = x[...,1]
#         real = self.pool(real, self.kernel_size, self.stride, self.padding, self.ceil_mode,
#                             self.count_include_pad)
#         imag = self.pool(imag, self.kernel_size, self.stride, self.padding, self.ceil_mode,
#                             self.count_include_pad)
#         x    = torch.stack([real,imag],-1)
#         x.__class__ = ComplexTensor
#         return x
# class ComplexAvgPool1d(_ComplexAvgPoolNd):pool = F.avg_pool1d
# class ComplexAvgPool2d(_ComplexAvgPoolNd):pool = F.avg_pool2d
# class ComplexAvgPool3d(_ComplexAvgPoolNd):pool = F.avg_pool3d
#
# class ComplexMaxPool1d(torch.nn.MaxPool1d):pass
# class ComplexMaxPool2d(torch.nn.MaxPool2d):
#     '''
#     basicly, there is no Max Pool for complex number system
#     '''
#     def forward(self,x):
#         real = x[...,0]
#         imag = x[...,1]
#         real = F.max_pool2d(real, self.kernel_size, self.stride, self.padding, self.dilation, self.ceil_mode,self.return_indices)
#         imag = F.max_pool2d(imag, self.kernel_size, self.stride, self.padding, self.dilation, self.ceil_mode,self.return_indices)
#         x    = torch.stack([real,imag],-1)
#         x.__class__ = ComplexTensor
#         return x
# class ComplexMaxPool3d(torch.nn.MaxPool3d):pass

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

Linear= ComplexLinear

Conv1d= ComplexConv1d
Conv2d= ComplexConv2d
Conv3d= ComplexConv3d

ReLU= ComplexReLU

BatchNorm1d= ComplexBatchNorm1d
BatchNorm2d= ComplexBatchNorm2d
BatchNorm3d= ComplexBatchNorm3d

ReImNorm1d= ComplexReImNorm1d
ReImNorm2d= ComplexReImNorm2d
ReImNorm3d= ComplexReImNorm3d

AvgPool1d= ComplexAvgPool1d
AvgPool2d= ComplexAvgPool2d

MaxPool1d= ComplexMaxPool1d
MaxPool2d= ComplexMaxPool2d

AdaptiveAvgPool1d= ComplexAdaptiveAvgPool1d
AdaptiveAvgPool2d= ComplexAdaptiveAvgPool2d
