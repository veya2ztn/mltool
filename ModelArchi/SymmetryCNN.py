import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import torch
import math
from torch.nn.modules.utils import _pair
from groupy.gconv.make_gconv_indices import *
from torch.nn import Conv2d

make_indices_functions = {(1, 4): make_c4_z2_indices,
                          (4, 4): make_c4_p4_indices,
                          (1, 8): make_d4_z2_indices,
                          (8, 8): make_d4_p4m_indices}


def trans_filter(w, inds):
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


class D4_Conv2d(Conv2d):
    group_num = 4
    def __init__(self, in_channels, out_channels, kernel_size, bias=True,**kwargs):
        super().__init__(in_channels, out_channels, kernel_size,bias=bias,**kwargs)
        self.inds  = make_indices_functions[(1,self.group_num)](kernel_size)

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels*self.group_num))
        else:
            self.bias = None
            self.register_parameter('bias', None)
        self.reset_parameters()
    def forward(self, x):
        weight = trans_filter(self.weight, self.inds)
        x = F.conv2d(x, weight, self.bias, self.stride,self.padding, self.dilation, self.groups) if self.padding_mode == 'zeros' else \
            F.conv2d(F.pad(x, self._reversed_padding_repeated_twice, mode=self.padding_mode),weight, self.bias, self.stride,_pair(0), self.dilation, self.groups)
        x  = x.view(x.shape[0],self.out_channels,self.group_num,x.shape[-2],x.shape[-1])
        x  = x.mean(2)
        return x
