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
