import numpy as np
import torch
import torch.nn.functional as F

def complex_mul(tensor_1: torch.Tensor,tensor_2: torch.Tensor,mode='cc')-> torch.Tensor:
    '''
    :param tensor_1(2) [...,2] for real part and image part
    '''
    if mode == 'cc':
        assert tensor_1.shape[-1]==2
        assert tensor_2.shape[-1]==2
        real1,imag1=tensor_1[...,0],tensor_1[...,1]
        real2,imag2=tensor_2[...,0],tensor_2[...,1]
        return torch.stack([real1 * real2 - imag1 * imag2, real1 * imag2 + imag1 * real2], dim = -1)
    elif mode=='cr':
        assert tensor_1.shape[-1]==2
        real1,imag1=tensor_1[...,0],tensor_1[...,1]
        real2      =tensor_2
        return torch.stack([real1 * real2, imag1 * real2], dim = -1)
    elif mode=='rc':
        assert tensor_2.shape[-1]==2
        real1,imag1=tensor_2[...,0],tensor_2[...,1]
        real2      =tensor_1
        return torch.stack([real1 * real2, imag1 * real2], dim = -1)
    else:
        raise NotImplementedError

def complex_mm(tensor_1: torch.Tensor,tensor_2: torch.Tensor,mode='cc')-> torch.Tensor:
    if mode == 'cc':
        assert tensor_1.shape[-1]==2
        assert tensor_2.shape[-1]==2
        real1,imag1=tensor_1[...,0],tensor_1[...,1]
        real2,imag2=tensor_2[...,0],tensor_2[...,1]
        return torch.stack([real1.mm(real2) - imag1.mm(imag2), real1.mm(imag2) + imag1.mm(real2)], dim = -1)
    elif mode=='cr':
        assert tensor_1.shape[-1]==2
        real1,imag1=tensor_1[...,0],tensor_1[...,1]
        real2      =tensor_2
        return torch.stack([real1.mm(real2), imag1.mm(real2)], dim = -1)
    elif mode=='rc':
        assert tensor_1.shape[-1]==2
        real1,imag1=tensor_2[...,0],tensor_2[...,1]
        real2      =tensor_1
        return torch.stack([real1.mm(real2), imag1.mm(real2)], dim = -1)
    else:
        raise NotImplementedError

def complex_mv(matrix: torch.Tensor,vector: torch.Tensor,mode='cc')-> torch.Tensor:
    if mode == 'cc':
        assert matrix.shape[-1]==2
        assert vector.shape[-1]==2
        real1,imag1=matrix[...,0],matrix[...,1]
        real2,imag2=vector[...,0],vector[...,1]
        return torch.stack([real1.mv(real2) - imag1.mv(imag2), real1.mv(imag2) + imag1.mv(real2)], dim = -1)
    elif mode=='cr':
        assert matrix.shape[-1]==2
        real1,imag1=matrix[...,0],matrix[...,1]
        real2      =vector
        return torch.stack([real1.mv(real2), imag1.mv(real2)], dim = -1)
    else:
        raise NotImplementedError

def complex_div(tensor_1: torch.Tensor,tensor_2: torch.Tensor)-> torch.Tensor:
    if mode == 'cc':
        assert tensor_1.shape[-1]==2
        assert tensor_2.shape[-1]==2
        a,b=tensor_1[...,0],tensor_1[...,1]
        c,d=tensor_2[...,0],tensor_2[...,1]
        Denominator = c**2+d**2
        return torch.stack([(a * c + b * d)/Denominator, (b*c-a*d)/Denominator], dim = -1)
    elif mode=='cr':
        assert tensor_1.shape[-1]==2
        a,b=tensor_1[...,0],tensor_1[...,1]
        c  =tensor_2
        return torch.stack([a/c,b/c], dim = -1)
    else:
        raise NotImplementedError

def complex_conj(tensor_1: torch.Tensor)-> torch.Tensor:
    assert tensor_1.shape[-1]==2
    real1,imag1=tensor_1[...,0],tensor_1[...,1]
    imag1=-imag1
    return torch.stack([real1,imag1], dim = -1)

def complex_polar(tensor: torch.Tensor)-> torch.Tensor:
    assert tensor.shape[-1]==2
    real,imag=tensor[...,0],tensor[...,1]
    radius = torch.norm(tensor,dim=-1)
    angles = torch.atan(real/imag)
    return torch.stack([radius,angles],dim=-1)

def complex_exp(tensor: torch.Tensor,angle_unit=1)-> torch.Tensor:
    assert tensor.shape[-1]==2
    factor,angles=tensor[...,0],tensor[...,1]
    radius = torch.exp(factor)
    angles = angles*angle_unit
    direct = torch.stack([angles.cos(),angles.sin()],dim=-1)
    return complex_mul(direct,radius,'cr')

def complex_polar_ln(tensor: torch.Tensor):
    assert tensor.shape[-1]==2
    real,imag=tensor[...,0],tensor[...,1]
    radius = torch.norm(tensor,dim=-1).log()
    angles = torch.atan(real/imag)
    return radius,angles

def complex_tch2np(tch: torch.Tensor)->np.ndarray:
    assert tch.shape[-1]==2
    out=tch.detach().numpy()
    return out[...,0]+1j*out[...,1]

def complex_np2tch(npx:np.ndarray)-> torch.Tensor:
    real = torch.Tensor(np.real(npx))
    imag = torch.Tensor(np.imag(npx))
    return torch.stack([real,imag],dim=-1)

def complex_conv2d(inputs,filters,bias=None,**kargs):
    assert len(inputs.shape)==5
    assert len(filters.shape)==5
    assert inputs.shape[-1]==2
    assert filters.shape[-1]==2

    convfun = lambda x,w,b:F.conv2d(x,w,b,**kargs)
    x_r,x_i=inputs[...,0],inputs[...,1]
    w_r,w_i=filters[...,0],filters[...,1]
    b_r=b_i=None
    if bias is not None:
        assert bias.shape[-1]==2
        b_r,b_i = bias[...,0],bias[...,1]

    o_r = convfun(x_r,w_r,b_r) - convfun(x_i,w_i,None)
    o_i = convfun(x_r,w_i,b_i) + convfun(x_i,w_r,None)

    ### another implement
    ##  but with very slow performance
    # o_r = F.conv3d(_inputs*torch.Tensor([1,-1]),_filter,stride=(stride,stride,1),padding=(padding,padding,0))
    # o_i = F.conv3d(_inputs,_filter.flip(-1),stride=(stride,stride,1),padding=(padding,padding,0))
    return torch.stack([o_r, o_i], dim = -1)

def complex_conv1d(inputs,filters,bias=None,**kargs):
    assert len(inputs.shape)==4
    assert len(filters.shape)==4
    assert inputs.shape[-1]==2
    assert filters.shape[-1]==2

    convfun = lambda x,w,b:F.conv1d(x,w,b,**kargs)
    x_r,x_i=inputs[...,0],inputs[...,1]
    w_r,w_i=filters[...,0],filters[...,1]
    b_r=b_i=None
    if bias is not None:
        assert bias.shape[-1]==2
        b_r,b_i = bias[...,0],bias[...,1]

    o_r = convfun(x_r,w_r,b_r) - convfun(x_i,w_i,None)
    o_i = convfun(x_r,w_i,b_i) + convfun(x_i,w_r,None)

    return torch.stack([o_r, o_i], dim = -1)

def complex_tanh(tensor:torch.Tensor)-> torch.Tensor:
    x,y  = tensor.split(1,dim=-1)
    x = 2*x
    y = 2*y
    n = y.cos() + x.cosh() + 1e-8
    #real = x.sinh()/n
    #imag = y.sin()/n
    return torch.cat([x.sinh()/n, y.sin()/n], dim = -1)

def complex_sigmoid(tensor:torch.Tensor)-> torch.Tensor:
    x,y  = tensor.split(1,dim=-1)
    x = torch.exp(-x)
    a = 1+x*y.cos()
    b = x*y.sin()
    n = a**2+b**2+ 1e-8
    return torch.cat([a/n, b/n], dim = -1)


def complexize(tensor: torch.Tensor)-> torch.Tensor:
    '''
    real to complex
    '''
    if tensor.shape[-1] == 2:return tensor
    imag = torch.zeros_like(tensor)
    return torch.stack([tensor,imag],-1)
