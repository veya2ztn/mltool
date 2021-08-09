import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
import numpy as np
class AMPS(nn.Module):
    def __init__(self, n=20, bond_dim=5, phys_dim = 2, std=1e-8):
        super(AMPS, self).__init__()
        self.register_buffer('bias_mat', torch.eye(4).unsqueeze(-1).repeat(1,1,phys_dim))
        # bias_mat: which is realy important when n>>1
        self.tensors = nn.Parameter(std * torch.randn(n, n, bond_dim, bond_dim, phys_dim)+self.bias_mat)
        # Initialize AMPS model parameters, which is a (n, n, D, D, 2) tensor
        # In AMPS workflow, bias_mat is not trainable
        # In AMPS workflow, the lower triangle part i,i+1: do not involve train.
        # Initialize masks
        # which is equaly to multiply a mask bellow
        # self.register_buffer('mask', self.tensors.data.clone())
        # self.mask.fill_(1)
        # for i in range(n):self.mask[i, i + 1:, :, :] = 0
        self.n = n
        self.bond_dim = bond_dim
        self.std = std

    def forward(self, data):
        """
        Input: data/spin configurations, shape: (bs, n)
        Output: log prob of each sample, shape: (bs,)
        """

        #self.tensors.data *= self.mask
        bs = data.shape[0]
        assert data.shape[-1] == 2
        # local embedding feature map, x_j -> [x_j, 1-x_j]
        # for MPS normlization requirement, the
        # embedded_data = torch.stack([data, 1.0 - data], dim=2)  # (bs, n, 2)
        embedded_data = data
        logx_hat      = torch.zeros_like(embedded_data)


        idx  = 0
        #paper way
        logx_hat[:, 0, :] = F.log_softmax(self.tensors[0, 0, 0, 0, :], dim=0)
        mats = torch.einsum('nlri,bi->nblr', self.tensors[idx:, idx, :, :, :], embedded_data[:, idx, :])
        #(n,D,D,2) <-> (B,2) ==> (n,B,D,D)
        left_vec = mats[:, :, 0, :]#(n,B,D,D) -> (n,B,D)
        # notice to simplify data store, here we set all tensor-block as (D,D,2) but the corner
        # tensor is only (D,2) so we take the first row of the first regist tensor
        for idx in range(1, self.n):
            left_vec=left_vec[1:]
            logits = torch.einsum('br, ri->bi', left_vec[0],self.tensors[idx, idx, :, 0 , :])
            # [(n-i,bs,D) ->(bs,D)] <-> [(n,n,D,D,2)->(D,D,2)] -> [(bs,D,2)]
            logx_hat[:, idx, :] = F.log_softmax(logits, dim=1)
            mats     = torch.einsum('nlri,bi->nblr', self.tensors[idx:, idx, :, :, :], embedded_data[:, idx, :])
            #(n-i,D,D,2) <-> (B,2) ==> (n-i,B,D,D)
            left_vec = torch.einsum('nbr,nbrk->nbk', left_vec, mats) #(n-i,bs,D)
            # compute p(s_1 | s_0) and so on

        # upper processing equal to below; Theoretically, use `for` loop will save half computing source, but
        # TODO: need to compare the GPU matrix acceleration



        # compute log prob
        log_prob = logx_hat[:, :, 0] * data + logx_hat[:, :, 1] * (1.0 - data)

        return log_prob.sum(-1)

    def sample(self, bs, random_start=False):
        """
        Sample images/spin configurations
        """

        self.tensors.data *= self.mask

        device = self.tensors.device
        samples = torch.empty([bs, self.n], device=device)

        # if random_start = True, force s_1 = -1/+1 randomly
        if random_start:
            samples[:, 0] = torch.randint(2, size=(bs, ), dtype=torch.float, device=device)
        else:
            samples[:, 0] = 0.

        for idx in range(self.n - 1):
            if idx == 0:
                # sample s_2 from p(s_2 | s_1)
                embedded_data = torch.stack([samples[:, 0], 1.0 - samples[:, 0]], dim=1)  # (bs, 2)
                mats = torch.einsum('nlri,bi->nblr', self.tensors[:, 0, :, :, :] , embedded_data)
                left_vec = mats[:, :, 0, :].unsqueeze(2)  # (n, bs, 1, D)
                logits = torch.einsum('nblr, nri->nbli', left_vec,
                                      (self.tensors[:, 1, :, :, :] )[:, :, 0, :]).squeeze(2)
                samples[:, 1] = torch.bernoulli(torch.softmax(logits[idx + 1, :, :], dim=1)[:, 0])
            else:
                # then sample s_3 from  p(s_3 | s_1, s_2) and so on
                embedded_data = torch.stack([samples[:, idx], 1.0 - samples[:, idx]], dim=1)  # (bs, 2)
                mats = torch.einsum('nlri,bi->nblr', self.tensors[:, idx, :, :, :] , embedded_data)
                #                 left_vec = torch.bmm(left_vec, mats)  # (bs, 1, D)
                left_vec = torch.einsum('nblr,nbrk->nblk', left_vec, mats)  # (n, bs, 1, D)
                logits = torch.einsum('nblr,nri->nbli', left_vec,
                                      (self.tensors[:, idx + 1, :, :, :] )[:, :, 0, :]).squeeze(2)
                samples[:, idx + 1] = torch.bernoulli(torch.softmax(logits[idx + 1, :, :], dim=1)[:, 0])

        return samples

class AMPS_FAST(nn.Module):
    '''
    this version may fast, but will cost much more memory
    '''
    def __init__(self, n=20, bond_dim=5, phys_dim = 2, std=1e-8):
        super(AMPS_FAST, self).__init__()
        self.register_buffer('bias_mat', torch.eye(4).unsqueeze(-1).repeat(1,1,phys_dim))
        #self.left_vec = nn.Parameter(std * torch.randn(         n,        1, bond_dim, phys_dim)+self.bias_mat)
        self.upslice  = torch.tril_indices(n-1,n-1)
        self.lwslice  = torch.triu_indices(n-1,n-1,offset=1)
        self.tri_up   = nn.Parameter(std * torch.randn((n-1)*n//2, bond_dim, bond_dim, phys_dim)+self.bias_mat)
        self.diag     = nn.Parameter(std * torch.randn(         n, bond_dim          , phys_dim)+self.bias_mat)

        # In AMPS workflow, the upper triangle part i,i+1: do not involve train.
        # Initialize masks
        # which is equaly to multiply a mask bellow
        # self.register_buffer('mask', self.tensors.data.clone())
        # self.mask.fill_(1)
        # for i in range(n):self.mask[i, i + 1:, :, :] = 0

        self.n = n
        self.bond_dim = bond_dim
        self.phys_dim = phys_dim
        self.std = std

    def forward(self, data):
        """
        Input: data/spin configurations, shape: (bs, n)
        Output: log prob of each sample, shape: (bs,)
        """
        embedded_data = torch.stack([data, 1.0 - data], dim=2)  # (bs, n, 2)
        n        = self.n
        b        = self.bond_dim

        upslice  = self.upslice
        lwslice  = self.lwslice
        tensor   = torch.zeros(n-1,n-1,b,b,phys_dim).type_as(self.tri_up)
        tensor[self.upslice] = self.tri_up
        # Contraction physics bond
        tensor = torch.einsum('nmlri,bmi->mnblr', tensor, embedded_data[:,:-1])
        # (n-1,m-1,D,D,2) <-> (B,m-1,2) -> (m-1,n-1,B,D,D) :: this should be the biggest cache matrix in our scheme
        # cause we swith the `m` slice and `n` slice, so the slice swith too.
        tensor[lwslice[1],lwslice[0]] = bias_mat[...,0]
        # set as Identity, this introduce unnecessary computing
        # but benifit from paraller accelaration
        # Contraction virtual bond

        # Contraction virtual bond
        size   = int(tensor.shape[0])
        while size > 1:
            half_size = size // 2
            nice_size = 2 * half_size
            leftover = tensor[nice_size:]
            tensor = torch.einsum("mnbik,mnbkj->mnbij",tensor[0:nice_size:2], tensor[1:nice_size:2])
            #(m/2,n,B,D,D),(m/2,n,B,D,D) <-> (m/2,n,B,D,D)
            tensor = torch.concatenate([tensor, leftover], axis=0)
            size   = half_size + int(size % 2 == 1)
        tensor = tensor.squeeze()[...,0,:]
        #(1,n-1,B,D,D) -> (n-1,B,D,D)  -> (n-1,B,D)
        tensor = torch.einsum('nbd,ndp->nbp',tensor,self.diag[1:])
        #(n-1, B, D) <-> (n-1, D, 2)  -> (n-1,B,2)
        tensor = torch.cat([diag[0][0].expand_as(tensor[0]).unsqueeze(0),tensor])
        # (n,D,2)->(,2)->(B,2)->(B,1,2) + (B,n-1,2)-> (n,B,2)
        logx_hat = F.log_softmax(tensor, dim=-1).transpose(1,0)

        log_prob = logx_hat[:, :, 0] * data + logx_hat[:, :, 1] * (1.0 - data)
        return log_prob.sum(-1)

class AMPSShare(nn.Module):
    def __init__(self, n=784, bond_dim=10, phys_dim=2,std=1e-8):
        super(AMPSShare, self).__init__()
        # Initialize AMPS model parameters, which is a (n, D, D, 2) tensor
        self.register_buffer('bias_mat', torch.eye(4).unsqueeze(-1).repeat(1,1,phys_dim))
        # bias_mat: which is realy important when n>>1
        self.tensors = nn.Parameter(std * torch.randn(n, bond_dim, bond_dim, phys_dim)+self.bias_mat)
        # Set attributes
        self.n = n
        self.bond_dim = bond_dim
        self.std = std

    def forward(self, data):
        """
        Input: data/spin configurations, shape: (bs, n)
        Output: log prob of each sample, shape: (bs,)
        """

        bs = data.shape[0]
        # local feature map, x_j -> [x_j, 1-x_j]
        embedded_data = torch.stack([data, 1.0 - data], dim=2)  # (bs, n, 2)

        logx_hat = torch.zeros_like(embedded_data)

        logx_hat[:, 0, :] = F.log_softmax(self.tensors[0, 0, 0], dim=0)
        mats = torch.einsum('lri,bi->blr', self.tensors[0] , embedded_data[:, 0, :])
        left_vec = mats[:, 0:1, :]  # (bs, 1, D)

        for idx in range(1, self.n):
            # compute p(s_2 | s_1) and so on
            logits = torch.einsum('br, ri->bi', left_vec,self.tensors[idx,:,0]).squeeze(1)
            #(bs,D) <-> [(D,D,2)]->[(D,2)]
            logx_hat[:, idx, :] = F.log_softmax(logits, dim=1)
            mats = torch.einsum('lri,bi->blr', self.tensors[idx, :, :, :] , embedded_data[:, idx, :])
            #(D,D,2) <-> (bs,2) ->(bs,D,D)
            left_vec = torch.bmm(left_vec, mats)  # (bs, 1, D)
            ## (bs, 1, D)
        # compute log prob
        log_prob = logx_hat[:, :, 0] * data + logx_hat[:, :, 1] * (1.0 - data)

        return log_prob.sum(-1)

    def sample(self, bs, random_start=False):
        """
        Sample images/spin configurations
        """

        device = self.tensors.device
        samples = torch.empty([bs, self.n], device=device)

        # if random_start = True, force s_1 = -1/+1 randomly
        if random_start:
            samples[:, 0] = torch.randint(2, size=(bs, ), dtype=torch.float, device=device)
        else:
            samples[:, 0] = 0.

        for idx in range(self.n - 1):
            if idx == 0:
                # sample s_2 from p(s_2 | s_1)
                embedded_data = torch.stack([samples[:, 0], 1.0 - samples[:, 0]], dim=1)  # (bs, 2)
                mats = torch.einsum('lri,bi->blr', self.tensors[0, :, :, :] , embedded_data)
                left_vec = mats[:, 0, :].unsqueeze(1)  # (bs, 1, D)
                logits = torch.einsum('blr, ri->bli', left_vec,
                                      (self.tensors[1, :, :, :] )[:, 0, :]).squeeze(1)
                samples[:, 1] = torch.bernoulli(torch.softmax(logits, dim=1)[:, 0])
            else:
                # then sample s_3 from  p(s_3 | s_1, s_2) and so on
                embedded_data = torch.stack([samples[:, idx], 1.0 - samples[:, idx]], dim=1)  # (bs, 2)
                mats = torch.einsum('lri,bi->blr', self.tensors[idx, :, :, :] , embedded_data)
                left_vec = torch.bmm(left_vec, mats)  # (bs, 1, D)
                logits = torch.einsum('blr, ri->bli', left_vec,
                                      (self.tensors[idx + 1, :, :, :] )[:, 0, :]).squeeze(1)
                samples[:, idx + 1] = torch.bernoulli(torch.softmax(logits, dim=1)[:, 0])

        return samples

class Conv2dAMPS(nn.Module):
    '''
    input  is (bs,d,w,h). The bs*n conv patch
    output is (bs,d,w,h)
    # nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
    '''
    def __init__(self,in_channels,out_channels,kernel_size=2,stride=1,padding=0,init_std=1e-9,fixed_bias=True,device='cuda:0'):
        super().__init__()
        self.k_h,self.k_w = _pair(kernel_size)
        self.kernel_size  = kernel_size
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.stride       = stride
        self.padding      = padding
        self.fixed_bias   = fixed_bias
        self.device       = device


        bias_mat = torch.eye(self.out_channels)#[D,D]
        if fixed_bias>0:
            self.register_buffer(name='bias_mat', tensor=bias_mat)
        else:
            print('fixed_bias==False')
            self.register_parameter(name='bias_mat', param=nn.Parameter(bias_mat))
        shape        = [self.k_h*self.k_w, self.out_channels, self.out_channels, self.in_channels]
        bias_mat     = torch.eye(self.out_channels).unsqueeze(-1).repeat(1,1,self.in_channels)
        self.tensors = nn.Parameter(init_std * torch.randn(shape)+ bias_mat)

    @staticmethod
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

    def forward(self, input_data):
        # the input data shape is (B,C,W,H)
        # expand to convolution patch
        embedded_data = self.get_conv_patch(input_data,self.kernel_size,self.stride,self.padding)# (B,C,W,H,k_w,k_h)
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
        size   = int(tensor.shape[0])
        while size > 1:
            half_size = size // 2
            nice_size = 2 * half_size
            leftover  = tensor[nice_size:]
            tensor    = torch.einsum("mbik,mbkj->mbij",tensor[0:nice_size:2], tensor[1:nice_size:2])
            #(k/2,NB,D,D),(k/2,NB,D,D) <-> (k/2,NB,D,D)
            tensor   = torch.cat([tensor, leftover], axis=0)
            size     = half_size + int(size % 2 == 1)
        tensor  = tensor.squeeze(0)[:,0]#(NB,D,D)->(NB,D)
        #tensor  = torch.einsum("mbii->mbi",tensor)
        tensor  = tensor.view(B, W, H, self.out_channels).permute(0,3,1,2)#(B,C,W,H)
        return tensor
