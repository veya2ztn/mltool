import torch
import numpy as np
import gc
import torch

def get_model_para_num(model):
    i=0
    j=0
    for p in model.parameters():
        i+=1
        j+=np.prod(p.shape)
    print("Totally {} parameters, totally {} numbers".format(i,j))
    return i,j

def get_model_para_detail(model,mode='standard'):
    i=0
    j=0
    for name,p in model.named_parameters():
        if mode == 'standard':
            _='Count all parameters'
        elif mode == 'graded':
            #only count parameter need optimilize
            if p.grad is None:continue
        print("{:40} {:5} {}".format(name,np.prod(p.shape),p.shape))
        i+=np.prod(p.shape)
        j+=1
    print("Totally {} parameters, totally {} numbers".format(j,i))

def mem_report():
    '''Report the memory usage of the tensor.storage in pytorch
    Both on CPUs and GPUs are reported'''

    def _mem_report(tensors, mem_type):
        '''Print the selected tensors of type
        There are two major storage types in our major concern:
            - GPU: tensors transferred to CUDA devices
            - CPU: tensors remaining on the system memory (usually unimportant)
        Args:
            - tensors: the tensors of specified type
            - mem_type: 'CPU' or 'GPU' in current implementation '''
        print('Storage on %s' %(mem_type))
        print('-'*LEN)
        total_numel = 0
        total_mem = 0
        visited_data = []
        for tensor in tensors:
            if tensor.is_sparse:
                continue
            # a data_ptr indicates a memory block allocated
            data_ptr = tensor.storage().data_ptr()
            if data_ptr in visited_data:
                continue
            visited_data.append(data_ptr)

            numel = tensor.storage().size()
            total_numel += numel
            element_size = tensor.storage().element_size()
            mem = numel*element_size /1024/1024 # 32bit=4Byte, MByte
            total_mem += mem
            element_type = type(tensor).__name__
            size = tuple(tensor.size())

            print('%s\t\t%s\t\t%.2f' % (
                element_type,
                size,
                mem) )
        print('-'*LEN)
        print('Total Tensors: %d \tUsed Memory Space: %.2f MBytes' % (total_numel, total_mem) )
        print('-'*LEN)

    LEN = 65
    print('='*LEN)
    objects = gc.get_objects()
    print('%s\t%s\t\t\t%s' %('Element type', 'Size', 'Used MEM(MBytes)') )
    tensors = [obj for obj in objects if torch.is_tensor(obj)]
    cuda_tensors = [t for t in tensors if t.is_cuda]
    host_tensors = [t for t in tensors if not t.is_cuda]
    _mem_report(cuda_tensors, 'GPU')
    _mem_report(host_tensors, 'CPU')
    print('='*LEN)


def set_dropout(model, drop_rate=0.1):
    for name, child in model.named_children():
        if isinstance(child, torch.nn.Dropout):
            child.p = drop_rate
        set_dropout(child, drop_rate=drop_rate)


def relu2leackrelu(module,_slope=0.2):
    module_output = module
    if isinstance(module, torch.nn.ReLU):
        module_output = torch.nn.LeakyReLU(inplace=True,negative_slope=_slope)
    for name, child in module.named_children():
        module_output.add_module(name, relu2leackrelu(child,_slope))
    del module
    return module_output

def print_bn_mean_var(model,gpu):
    for name, child in model.named_children():
        #if isinstance(child, torch.nn.BatchNorm2d):
        if isinstance(child, torch.nn.SyncBatchNorm):
            rm = child.running_mean.mean()
            rv = child.running_var.mean()
            print("{} {} {} {}".format(gpu,name,rm,rv))
        print_bn_mean_var(child,gpu)

def print_linear_k_b(model,gpu):
    for name, child in model.named_children():
        #if isinstance(child, torch.nn.BatchNorm2d):
        if isinstance(child, torch.nn.Linear):
            rm = child.weight.mean()
            rv = child.bias.mean()
            print("{} {} {} {}".format(gpu,name,rm,rv))
        print_linear_k_b(child,gpu)

def print_conv_k_b(model,gpu):
    for name, child in model.named_children():
        #if isinstance(child, torch.nn.BatchNorm2d):
        if isinstance(child, torch.nn.Conv2d):
            rm = child.weight.mean()
            rv = child.bias.mean() if child.bias is not None else None
            print("{} {} {} {}".format(gpu,name,rm,rv))
        print_conv_k_b(child,gpu)
import torch.nn as nn
def init_weights(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        #torch.nn.init.constant_(m.weight,0.5)
        if m.bias is not None:torch.nn.init.zeros_(m.bias)
    if isinstance(m, nn.Linear):
        #torch.nn.init.constant_(m.weight,0.5)
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:m.bias.data.fill_(0.01)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.constant_(m.weight,0.5)
        if m.bias is not None:
            m.bias.data.fill_(0)
    if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                torch.nn.init.xavier_uniform_(param.data)
                #torch.nn.init.constant_(param.data,0.5)
            elif 'weight_hh' in name:
                torch.nn.init.xavier_uniform_(param.data)
                #torch.nn.init.constant_(param.data,0.5)
            elif 'bias' in name:
                param.data.fill_(0)
