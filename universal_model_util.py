import torch
import numpy as np

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
