import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

class BiLSTM(nn.Module):
    def __init__(self,embedding_dim,n_hidden,num_classes):
        super().__init__()
        self.embedding_dim= embedding_dim
        self.n_hidden     = n_hidden
        self.num_classes  = num_classes
        self.lstm         = nn.LSTM(self.embedding_dim, self.n_hidden, bidirectional=True)
        self.out          = nn.Linear(self.n_hidden * 2, self.num_classes)
    # lstm_output : [batch_size, n_step, self.n_hidden * num_directions(=2)], F matrix
    def attention_net(self, lstm_output, final_state):
        hidden = final_state.view(-1, self.n_hidden * 2, 1)   # hidden : [batch_size, self.n_hidden * num_directions(=2), 1(=n_layer)]
        attn_weights = torch.bmm(lstm_output, hidden).squeeze(2) # attn_weights : [batch_size, n_step]
        soft_attn_weights = F.softmax(attn_weights, 1)
        # [batch_size, self.n_hidden * num_directions(=2), n_step] * [batch_size, n_step, 1] = [batch_size, self.n_hidden * num_directions(=2), 1]
        context = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        return context, soft_attn_weights # context : [batch_size, self.n_hidden * num_directions(=2)]

    def forward(self, X):
        # input : [batch_size, len_seq, self.embedding_dim]
        if len(X.shape)==2:X=X.unsqueeze(1)
        input = X.permute(1, 0, 2) # input : [len_seq, batch_size, self.embedding_dim]

        #hidden_state = Variable(torch.zeros(1*2, len(X), self.n_hidden)) # [num_layers(=1) * num_directions(=2), batch_size, self.n_hidden]
        #cell_state   = Variable(torch.zeros(1*2, len(X), self.n_hidden)) # [num_layers(=1) * num_directions(=2), batch_size, self.n_hidden]

        # final_hidden_state, final_cell_state : [num_layers(=1) * num_directions(=2), batch_size, self.n_hidden]
        # output, (final_hidden_state, final_cell_state) = self.lstm(input, (hidden_state, cell_state))
        output, (final_hidden_state, final_cell_state) = self.lstm(input)
        output = output.permute(1, 0, 2) # output : [batch_size, len_seq, self.n_hidden]
        attn_output, attention = self.attention_net(output, final_hidden_state)
        #return self.out(attn_output), attention # model : [batch_size, self.num_classes], attention : [batch_size, n_step]
        return self.out(attn_output) # model : [batch_size, self.num_classes], attention : [batch_size, n_step]

class BiLSTM2(nn.Module):

    def __init__(self,embedding_dim,n_hidden,num_classes):
        super().__init__()
        self.embedding_dim= embedding_dim
        self.n_hidden     = n_hidden
        self.num_classes  = num_classes
        self.lstm         = nn.LSTM(self.embedding_dim, self.n_hidden, bidirectional=True)
        self.out          = nn.Linear(self.n_hidden * 2, self.num_classes)

    def forward(self, X):
        if len(X.shape)==2:X=X.unsqueeze(1)
        input = X.permute(1, 0, 2) # input : [len_seq, batch_size, self.embedding_dim]
        output, (final_hidden_state, final_cell_state) = self.lstm(input)
        y = self.out(output[-1])
        log_probs = F.log_softmax(y,dim=-1)
        return log_probs
