from . import complex_operation as  C
from .complex_scalar import ComplexScalar
import numpy as np
import torch
import re



class ComplexTensor(torch.Tensor):


    @staticmethod
    def __new__(cls, x):
        # required pytorch tensor shape (...,2)
        # required numpy complex tensor (...)
        if isinstance(x, np.ndarray) and 'complex' in str(x.dtype):
            x = C.complex_np2tch(x)
        if not isinstance(x,torch.Tensor):
            print('now type is '+str(type(x)))
            raise Exception('Only support np.ndarray or torch.Tensor')
        assert x.shape[-1]==2
        x=x.float()
        # init new t
        new_t = super().__new__(cls, x)
        return new_t

    @property
    def real(self):
        result = self[...,0]
        return result

    @property
    def imag(self):
        result = self[...,1]
        return result

    @property
    def radius(self):
        result = self.norm(dim=-1)
        return result

    @property
    def phase(self):
        real   = self.real
        imag   = self.imag
        result = torch.atan(imag/real)
        return result

    @property
    def grad(self):
        g = self._grad
        real = g[...,0]
        imag = -g[...,1]
        result = torch.stack([real,imag],dim=-1)
        result.__class__ = ComplexTensor
        return result

    def is_complex(self):
        return True

    def __str__(self):
        return self.__repr__()

    def conj(self):
        real =  self.real
        imag = -self.imag
        result = torch.stack([real,imag],dim=-1)
        result.__class__ = ComplexTensor
        return result

    def exp(self):
        result = C.complex_exp(self)
        result.__class__ = ComplexTensor
        return result

    def log(self):
        real = self.radius.log()
        imag = self.phase
        result = torch.stack([real,imag],dim=-1)
        result.__class__ = ComplexTensor
        return result

    def sum(self, *args):
        real_sum = self.real.sum(*args)
        imag_sum = self.imag.sum(*args)
        return ComplexScalar(real_sum, imag_sum)

    def mean(self, *args):
        real_mean = self.real.mean(*args)
        imag_mean = self.imag.mean(*args)
        return ComplexScalar(real_mean, imag_mean)

    def abs(self):
        return self.norm(dim=-1)
    
    def add(self,m):
        if isinstance(c,ComplexTensor):
            result = self+m
        else:
            if m.shape[-1]==1:m=m[...,0]
            if len(self.shape) == len(m.shape)+1:
                real =  self.real + m
                imag =  self.imag
                result = torch.stack([real,imag],dim=-1)
            else:
                raise
        result.__class__ = ComplexTensor
        return result

    def sub(self,c):
        shapecl= len(self.shape)
        shapeml= len(c.shape)
        if shapecl == shapeml:
            result = self - c
        elif shapecl == shapeml+1:
            real =  self.real - c
            imag =  self.imag
            result = torch.stack([real,imag],dim=-1)
        result.__class__ = ComplexTensor
        return result

    def div(self,m):
        shapecl= len(self.shape)
        shapeml= len(m.shape)
        if shapecl == shapeml:sign='cc'
        elif shapecl == shapeml+1:sign='cr'
        result = C.complex_div(self,m,sign)
        result.__class__ = ComplexTensor
        return result

    def mul(self,m):
        shapecl= len(self.shape)
        shapeml= len(m.shape)
        if shapecl == shapeml:sign='cc'
        elif shapecl == shapeml+1:sign='cr'
        result = C.complex_mul(self,m,sign)
        result.__class__ = ComplexTensor
        return result

    def mm(self,m):
        shapecl= len(self.shape)
        shapeml= len(m.shape)
        if shapecl == shapeml:sign='cc'
        elif shapecl == shapeml+1:sign='cr'
        result = C.complex_mm(self,m,sign)
        result.__class__ = ComplexTensor
        return result

    def mv(self,v):
        shapevl= len(v.shape)
        if   shapevl == 2:sign='cc'
        elif shapevl == 1:sign='cr'
        result = C.complex_mv(self,v,sign)
        result.__class__ = ComplexTensor
        return result

    def numpy(self):
        return C.complex_tch2np(self)

    def __repr__(self):
        # use numpy to print for us
        # strings = np.asarray([f'({a}{"+" if b > 0 else "-"}{abs(b)}j)' for a, b in zip(real, imag)])
        #strings = np.asarray([complex(a,b) for a, b in zip(real, imag)]).astype(np.complex64)
        strings = C.complex_tch2np(self)
        strings = strings.__repr__()
        strings = re.sub('array', 'tensor', strings)
        return strings
