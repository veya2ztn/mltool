import torch
import torch.nn as nn
import numpy as np

import copy
import torch.nn.functional as F

class MultiBoxLoss(nn.Module):
    def __init__(self):
        self.lambdaa=10
        super(MultiBoxLoss, self).__init__()

    def forward(self, predictions, targets):
        #print('+++++++++++++++++++++++++++++++++++++++++++++++++++')
        cout, rout = predictions
        if cout.shape[0]==2:cout=cout.transpose(1,0)
        if rout.shape[0]==4:rout=rout.transpose(1,0)
        assert cout.shape[1]==2
        assert rout.shape[1]==4
        """ class """
        class_pred, class_target = cout, targets[:, 0].long()
        pos_index , neg_index    = list(np.where(class_target == 1)[0]), list(np.where(class_target == 0)[0])
        pos_num, neg_num         = len(pos_index), len(neg_index)
        class_pred, class_target = class_pred[pos_index + neg_index], class_target[pos_index + neg_index]

        closs = F.cross_entropy(class_pred, class_target, reduction='none')
        closs = torch.div(torch.sum(closs), 64)

        """ regression """
        reg_pred   = rout
        reg_target = targets[:, 1:]
        rloss = F.smooth_l1_loss(reg_pred, reg_target, reduction='none') #1445, 4
        rloss = torch.div(torch.sum(rloss, dim = 1), 4)
        rloss = torch.div(torch.sum(rloss[pos_index]), 16)

        lambdaa=self.lambdaa
        loss = closs + lambdaa*rloss
        return closs, rloss, loss, reg_pred, reg_target, pos_index, neg_index

def adjust_learning_rate(lr, optimizer, epoch, gamma=0.1):
    """Sets the learning rate to the initial LR decayed 0.9 every 50 epochs"""
    lr = lr * (0.9 ** (epoch // 1))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr
