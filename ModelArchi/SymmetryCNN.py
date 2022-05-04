import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import torch
import math
from torch.nn.modules.utils import _pair
#from ..groupy.gconv.make_gconv_indices import *
from torch.nn import Conv2d
import numpy as np

class fliprot90:
    def __init__(self,rot_times,flip):
        self.rot_times = rot_times
        self.flip      = flip
    def __call__(self,data):
        data = np.flip(data,-2) if self.flip else data
        data = np.rot90(data,self.rot_times,(-2,-1))
        return data
    def __repr__(self):
        return f"do flip {self.flip} times and rotate 90 degree {self.rot_times} times"
GroupV2  = [fliprot90(0,0) , fliprot90(0,1)]
GroupH2  = [fliprot90(0,0) , fliprot90(2,1)]
GroupZ2  = [fliprot90(0,0) , fliprot90(0,1), fliprot90(2,0), fliprot90(2,1)]
GroupP4  = [fliprot90(i,0) for i in range(4)]
GroupP4Z2= [fliprot90(i,flipQ) for i in range(4) for flipQ in [0,1]]

def object2slice(x):
    return np.unravel_index(np.argsort(x, axis=None), x.shape)

def get_slice_for_Group(ksize,Group):
    ksize       = _pair(ksize)
    the_object  = np.arange(np.prod(ksize)).reshape(*ksize)
    inds        = np.concatenate([object2slice(group(the_object)) for group in Group],1)
    return inds


def trans_filter(w, inds_reshape):
    w_indexed = w[..., inds_reshape[0].tolist(), inds_reshape[1].tolist()]
    w_indexed = w_indexed.view(w.size(0), w.size(1),-1, w.size(-2), w.size(-1))
    w_indexed = w_indexed.permute(0, 2, 1, 3, 4).contiguous()
    w_indexed = w_indexed.view(w_indexed.size(0)*w_indexed.size(1),w_indexed.size(2),w_indexed.size(3),w_indexed.size(4))
    return w_indexed

def trans_filter_old(w, inds):
    # for groupy usage
    inds_reshape = inds.reshape((-1, inds.shape[-1])).astype(np.int64)
    if len(w.shape)==4:
        w_indexed = w[..., inds_reshape[:, 1].tolist(), inds_reshape[:, 2].tolist()]
    elif len(w.shape)==5:
        w_indexed = w[..., inds_reshape[:, 0].tolist(), inds_reshape[:, 1].tolist(), inds_reshape[:, 2].tolist()]
    else:
        raise
    w_indexed = w_indexed.view(w_indexed.size()[0], w_indexed.size()[1],inds.shape[0], inds.shape[1], inds.shape[2], inds.shape[3])
    w_indexed = w_indexed.permute(0, 2, 1, 3, 4, 5).contiguous()
    w_indexed = w_indexed.view(w_indexed.size()[0]*inds.shape[0],w_indexed.size()[2]* inds.shape[1], inds.shape[2], inds.shape[3])
    return w_indexed


class Symmetry_Conv2d(Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True,**kwargs):
        super().__init__(in_channels, out_channels, kernel_size,bias=bias,**kwargs)
        if bias:self.bias = Parameter(torch.Tensor(out_channels*self.group_num))
        else:
            self.bias = None
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.inds  = get_slice_for_Group(kernel_size,self.Group)
    def forward(self, x):
        weight = trans_filter(self.weight, self.inds)
        x = F.conv2d(x, weight, self.bias, self.stride,self.padding, self.dilation, self.groups) if self.padding_mode == 'zeros' else \
            F.conv2d(F.pad(x, self._reversed_padding_repeated_twice, mode=self.padding_mode),weight, self.bias, self.stride,_pair(0), self.dilation, self.groups)
        x  = x.view(x.shape[0],self.out_channels,-1,x.shape[-2],x.shape[-1])
        x  = x.mean(2)
        return x

class P4_Conv2d(Symmetry_Conv2d):
    group_num = 4
    Group     = GroupP4

class P4Z2_Conv2d(Symmetry_Conv2d):
    group_num = 8
    Group     = GroupP4Z2

class V2_Conv2d(Symmetry_Conv2d):
    group_num = 2
    Group     = GroupV2

class Z2_Conv2d(Symmetry_Conv2d):
    group_num = 4
    Group     = GroupZ2

class H2_Conv2d(Symmetry_Conv2d):
    group_num = 2
    Group     = GroupH2

def symmetry_config_fix(kernel_size,stride,padding,dilation):
    # when stride = 2, the k=3,s=2,p=1 CNN layer for even size input like (16,16) would destroy symmetry passing.
    # we will force this case to right way, like (k,s,p) = (3,2,1) --> (2,2,0)
    if not isinstance(kernel_size,int):
        assert  kernel_size[0] == kernel_size[1]
        kernel_size = kernel_size[0]
    if not isinstance(stride,int):
        assert  stride[0] == stride[1]
        stride = stride[0]
    if not isinstance(padding,int):
        assert  padding[0] == padding[1]
        padding = padding[0]
    if not isinstance(dilation,int):
        assert  dilation[0] == dilation[1]
        dilation = dilation[0]
    if dilation>1:
        return symmetry_config_fix(kernel_size,stride,padding,1)
    if  (kernel_size-1)*dilation != 2*padding + stride -1:
        if padding %2 !=0:
            padding+=1
            return symmetry_config_fix(kernel_size,stride,padding,dilation)
        kernel_size = (2*padding + stride -1)//dilation + 1
    if kernel_size==1 and stride==2 and padding == 0:
        kernel_size =2
    return kernel_size,stride,padding,dilation
    # raise NotImplementedError(f'''please check your CNN config kernel, stride and padding:
    #     kernel_size ={kernel_size },
    #     stride      ={stride      },
    #     padding     ={padding     },
    #     dilation    ={dilation    }
    #     ''')

SymmetryCNNPool={"P4":P4_Conv2d,'P4Z2':P4Z2_Conv2d,'V2':V2_Conv2d,'H2':H2_Conv2d,'Z2':Z2_Conv2d}
def cnn2symmetrycnn(module,type='P4Z2',active_symmetry_fix=False):
    module_output = module

    if isinstance(module, Conv2d):
        in_channels =module.in_channels
        out_channels=module.out_channels
        kernel_size =module.kernel_size
        stride      =module.stride
        padding     =module.padding
        dilation    =module.dilation
        bias        =False if module.bias is None else True
        if active_symmetry_fix:
            assert active_symmetry_fix == 'even' # not only for even input
            kernel_size,stride,padding,dilation = symmetry_config_fix(kernel_size,stride,padding,dilation)
        module_output = SymmetryCNNPool[type](in_channels =in_channels ,
                                             out_channels=out_channels,
                                             kernel_size =kernel_size ,
                                             stride      =stride      ,
                                             padding     =padding     ,
                                             dilation    =dilation    ,
                                             bias        =bias        )
    for name, child in module.named_children():
        module_output.add_module(name, cnn2symmetrycnn(child,type,active_symmetry_fix=active_symmetry_fix))
    del module
    return module_output
