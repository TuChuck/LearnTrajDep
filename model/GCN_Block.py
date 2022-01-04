import torch.nn as nn
import torch
from torch.nn.parameter import Parameter
import GCN

import itertools

class GCN_Block(nn.Module):
    def __init__(self, input_feature, hidden_feature, p_dropout, n_separate = 1, num_stage=1, node_n=48):
        super(GCN_Block, self).__init__()

        self.n_separate = n_separate

        self.gcnbs = []
        for i in range(n_separate):
            self.gcnbs.append(GCN(input_feature, hidden_feature, p_dropout=p_dropout, num_stage=num_stage, node_n=node_n))

        self.gcnbs = nn.ModuleList(self.gcnbs)

    def forward(self, x):
        # y_ = [[] for i in range(self.n_separate)]
        
        for i in range(self.n_separate):
            if i == 0:
                y = self.gcnbs[i](x[i])
                x_ = x[i]
            else:
                y = torch.cat((y,self.gcnbs[i](x[i])),dim=-1)
                x_ = torch.cat((x_,x[i]),dim=-1)
        # y = list(itertools.chain(*y_))

        return y