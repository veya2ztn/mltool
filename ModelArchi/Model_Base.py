import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class BasicModule(torch.nn.Module):
    def __init__(self,,output_type):
        super().__init__()
        self.in_size     = 16*16
        self.out_put_fea = out_put_fea
        self.output_type = output_type
        self.mulnum = 2 if output_type is'complex' else 1
        self.out_features= self.mulnum*self.out_put_fea

        self.test_metric  = None
        self.loss_metric  = None
        self.optimaizer   = None

    def forward(self,x,target=None):
        raise NotImplementedError
        return y,loss

    def train(self,x_train,y_train):
        y_pred,loss = self.forward(x_train,y_train)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return y_pred,loss

    def fit(self,x_train,y_train):
        return self.train(x_train,y_train)

    def predict(self,x):
        with torch.no_grad():
            return self.forward(x)

    def test(self,x,target=None):
        with torch.no_grad():
            return self.forward(x,target)

    def save_to(self,path):
        torch.save(self.state_dict(),path)

    def load_from(self,path):
        self.load_state_dict(torch.load(path))
