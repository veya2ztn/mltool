import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
import numpy as np

def get_conv_patch(img,kernel_size,stride,padding):
    img_b,img_d,img_h, img_w=img.shape
    ker_h,ker_w = _pair(kernel_size)
    str_h,str_w = _pair(stride)

    pad_l=pad_u=pad_r=pad_d=None
    if isinstance(padding,int):pad_l=pad_u=pad_r=pad_d=padding
    elif isinstance(padding,list) or isinstance(padding,tuple):
        if   len(padding)==1:pad_l=pad_u=pad_r=pad_d=padding[0]
        elif len(padding)==2:pad_l=pad_r=padding[0];pad_u=pad_d=padding[1]
        elif len(padding)==4:pad_l,pad_u,pad_r,pad_d = padding
    assert pad_l is not None
    #img_h, img_w=img.shape[-2:]
    # assert not (img_h-ker_h+pad_l+pad_r)%str_h
    # assert not (img_w-ker_w+pad_u+pad_d)%str_w
    out_h = (img_h-ker_h+pad_l+pad_r)//str_h +1
    out_w = (img_w-ker_w+pad_u+pad_d)//str_w +1
    i0 = np.repeat(np.arange(ker_h), ker_w).reshape(-1,1)
    j0 = np.tile(np.arange(ker_w), ker_h).reshape(-1,1)


    i1 = np.repeat(str_h*np.arange(out_h), out_w).reshape(1,-1)
    j1 = np.tile(  str_w*np.arange(out_w), out_h).reshape(1,-1)
    i  = i0+i1
    j  = j0+j1
    i_g= i.transpose(1,0)
    j_g= j.transpose(1,0)

    pad_img=torch.nn.functional.pad(img, (pad_l,pad_r,pad_u,pad_d),mode='constant')
    #pad_img=np.pad(img, ((0,0),(0,0),(1,1),(1,1)),mode='constant')
    patches = pad_img[...,i_g,j_g].reshape(img_b,img_d,out_h, out_w,ker_h,ker_w)
    return patches

def get_chain_contraction(tensor):
    size   = int(tensor.shape[0])
    while size > 1:
        half_size = size // 2
        nice_size = 2 * half_size
        leftover  = tensor[nice_size:]
        tensor    = torch.einsum("mbik,mbkj->mbij",tensor[0:nice_size:2], tensor[1:nice_size:2])
        #(k/2,NB,D,D),(k/2,NB,D,D) <-> (k/2,NB,D,D)
        tensor   = torch.cat([tensor, leftover], axis=0)
        size     = half_size + int(size % 2 == 1)
    return tensor.squeeze(0)

class Conv2dAMPS(nn.Module):
    '''
    input  is (bs,d,w,h). The bs*n conv patch
    output is (bs,d,w,h)
    weight is (k_h,k_w,out_channels,out_channels, in_channels)
    # nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
    '''
    def __init__(self,in_channels,out_channels,kernel_size=2,stride=1,padding=0,init_std=1e-9,fixed_bias=True,**kargs):
        super().__init__()
        self.k_h,self.k_w = _pair(kernel_size)
        self.kernel_size  = kernel_size
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.stride       = stride
        self.padding      = padding
        self.fixed_bias   = fixed_bias



        # bias_mat = torch.eye(self.out_channels)#[D,D]
        # if fixed_bias>0:
        #     self.register_buffer(name='bias_mat', tensor=bias_mat)
        # else:
        #     print('fixed_bias==False')
        #     self.register_parameter(name='bias_mat', param=nn.Parameter(bias_mat))

        shape        = [self.k_h*self.k_w, self.out_channels, self.out_channels, self.in_channels]
        bias_mat     = torch.eye(self.out_channels).unsqueeze(-1).repeat(1,1,self.in_channels)
        self.tensors = nn.Parameter(init_std * torch.randn(shape)+ bias_mat)

    def forward(self, input_data):
        # the input data shape is (B,C,W,H)
        # expand to convolution patch
        embedded_data = get_conv_patch(input_data,self.kernel_size,self.stride,self.padding)# (B,C,W,H,k_w,k_h)
        B,C , W, H ,k_w,k_h = embedded_data.shape
        embedded_data = embedded_data.permute(0,2,3,4,5,1).reshape(B*W*H,k_w*k_h,C)#i.e.(NB,k,C)
        # the embed data is considered as NB MPS chains. every MPS chains get k length and physics dim is P
        # the weight is (k,D,D,C) and the final result is
        # the contraction result along P and along WH
        # Contraction along P
        tensor  = torch.einsum('wijp,nwp->wnij',self.tensors,embedded_data)
        # (k,D,D,C) <-> (NB,k,C) -> (k,NB,D,D)
        #tensor  = tensor + self.bias_mat #:: the bias will act on every matrix
        # (k,NB,D,D) + (D,D)
        # Contraction along k
        tensor  = get_chain_contraction(tensor)
        tensor  = tensor[:,0]#(NB,D,D)->(NB,D)
        #tensor  = torch.einsum("mbii->mbi",tensor)
        tensor  = tensor.view(B, W, H, self.out_channels).permute(0,3,1,2)#(B,C,W,H)
        return tensor

class embedConv2dMPS_odd(nn.Module):
    '''
    input  is (bs,d,w,h). The bs*n conv patch
    output is (bs,d,w,h)
    weight is (k_h*k_w-1   , bond_dim, bond_dim, in_channels)
            + (out_channels, bond_dim, bond_dim, in_channels)
    # nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
    '''
    def __init__(self,in_channels,out_channels,kernel_size=3,stride=1,padding=0,bond_dim=3,init_std=1e-9,fixed_bias=True,**kargs):
        super().__init__()
        self.k_h,self.k_w = _pair(kernel_size)
        self.kernel_size  = kernel_size
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.stride       = stride
        self.padding      = padding
        self.fixed_bias   = fixed_bias

        self.bd           = bond_dim

        assert (self.k_h*self.k_w)%2==1

        self.hn = half_num = (self.k_h*self.k_w-1)//2
        bias_mat = torch.eye(self.bd).unsqueeze(-1).repeat(1,1,self.in_channels)
        self.left_tensors = nn.Parameter(init_std * torch.randn(half_num,self.bd,self.bd,self.in_channels)+ bias_mat)
        self.cent_tensors = nn.Parameter(init_std * torch.randn(self.out_channels,self.bd,self.bd,self.in_channels)+ bias_mat)
        self.rigt_tensors = nn.Parameter(init_std * torch.randn(half_num,self.bd,self.bd,self.in_channels)+ bias_mat)

    def forward(self, input_data):
        # the input data shape is (B,C,W,H)
        # expand to convolution patch
        embedded_data = get_conv_patch(input_data,self.kernel_size,self.stride,self.padding)# (B,C,W,H,k_w,k_h)
        B,C , W, H ,k_w,k_h = embedded_data.shape
        embedded_data = embedded_data.permute(0,2,3,4,5,1).reshape(B*W*H,k_w*k_h,C)#i.e.(NB,K,C)

        left_tensors = torch.einsum('wijp,nwp->wnij',self.left_tensors,embedded_data[:,:self.hn])#i.e. (K,NB,b,b)
        cent_tensors = torch.einsum('wijp, np->wnij',self.cent_tensors,embedded_data[:,self.hn])#i.e.  (T,NB,b,b)
        rigt_tensors = torch.einsum('wijp,nwp->wnij',self.rigt_tensors,embedded_data[:,-self.hn:])#i.e.(K,NB,b,b)

        left_tensors = get_chain_contraction(left_tensors) #i.e. (NB,b,b)
        rigt_tensors = get_chain_contraction(rigt_tensors) #i.e. (NB,b,b)

        tensor  = torch.einsum('bip,obpl,bli->bo',left_tensors,cent_tensors,rigt_tensors)
        # (NB,b,b) <-> (T,NB,b,b) <-> (NB,b,b) ==> (NB,T)
        tensor  = tensor.view(B, W, H, self.out_channels).permute(0,3,1,2)#(B,C,W,H)
        return tensor
class embedConv2dMPS_even(nn.Module):
    '''
    input  is (bs,d,w,h). The bs*n conv patch
    output is (bs,d,w,h)
    weight is (k_h,k_w, bond_dim, bond_dim, in_channels)
            + (1      , bond_dim, bond_dim,out_channels)
    # nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
    '''
    def __init__(self,in_channels,out_channels,kernel_size=3,stride=1,padding=0,bond_dim=3,init_std=1e-9,fixed_bias=True,**kargs):
        super().__init__()
        self.k_h,self.k_w = _pair(kernel_size)
        self.kernel_size  = kernel_size
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.stride       = stride
        self.padding      = padding
        self.fixed_bias   = fixed_bias

        self.bd           = bond_dim

        assert (self.k_h*self.k_w)%2==0

        self.hn = half_num = (self.k_h*self.k_w)//2
        bias_mat = torch.eye(self.bd).unsqueeze(-1).repeat(1,1,self.in_channels)
        self.left_tensors = nn.Parameter(init_std * torch.randn(half_num,self.bd,self.bd,self.in_channels)+ bias_mat)
        self.cent_tensors = nn.Parameter(init_std * torch.randn(self.out_channels,self.bd,self.bd)+ torch.eye(self.bd))
        self.rigt_tensors = nn.Parameter(init_std * torch.randn(half_num,self.bd,self.bd,self.in_channels)+ bias_mat)


    def forward(self, input_data):
        # the input data shape is (B,C,W,H)
        # expand to convolution patch
        embedded_data = get_conv_patch(input_data,self.kernel_size,self.stride,self.padding)# (B,C,W,H,k_w,k_h)
        B,C , W, H ,k_w,k_h = embedded_data.shape
        embedded_data = embedded_data.permute(0,2,3,4,5,1).reshape(B*W*H,k_w*k_h,C)#i.e.(NB,K,C)

        left_tensors = torch.einsum('wijp,nwp->wnij',self.left_tensors,embedded_data[:,:self.hn])#i.e. (K,NB,b,b)
        rigt_tensors = torch.einsum('wijp,nwp->wnij',self.rigt_tensors,embedded_data[:,-self.hn:])#i.e.(K,NB,b,b)

        left_tensors = get_chain_contraction(left_tensors) #i.e. (NB,b,b)
        rigt_tensors = get_chain_contraction(rigt_tensors) #i.e. (NB,b,b)

        tensor  = torch.einsum('bip,opl,bli->bo',left_tensors,cent_tensors,rigt_tensors)
        # (NB,b,b) <-> (T,b,b) <-> (NB,b,b) ==> (NB,T)
        tensor  = tensor.view(B, W, H, self.out_channels).permute(0,3,1,2)#(B,C,W,H)
        return tensor
def embedConv2d(in_channels,out_channels,kernel_size,stride=1,padding=0,bond_dim=3,**kargs):
    moduler =  embedConv2dMPS_odd if np.prod(_pair(kernel_size))%2 == 1 else embedConv2dMPS_even
    return  moduler(in_channels,out_channels,
                                kernel_size=kernel_size,stride=stride,padding=padding,
                                bond_dim=bond_dim,**kargs)
MPSCNNPool={"CaMPS":Conv2dAMPS,'eCMPS':embedConv2d}
def cnn2mpscnn(module,type='eCMPS'):
    module_output = module
    if isinstance(module, nn.Conv2d):
        if np.prod(_pair(module.kernel_size))==1:
            module_output = module
        else:
            module_output = MPSCNNPool[type](in_channels =module.in_channels ,
                                        out_channels=module.out_channels,
                                        kernel_size =module.kernel_size ,
                                        stride      =module.stride      ,
                                        padding     =module.padding     ,
                                        dilation    =module.dilation    ,
                                        bias        =False if module.bias is None else True)
    for name, child in module.named_children():
        module_output.add_module(name, cnn2mpscnn(child,type))
    del module
    return module_output
