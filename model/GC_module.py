import torch.nn as nn
import torch
from torch.nn.parameter import Parameter
import GC_Block

import itertools

class GC_module(nn.Module):
    def __init__(self, hidden_feature, p_dropout, n_separate = 1, node_n=48):
        super(GC_module,self).__init__()

        self.hidden_feature = hidden_feature
        self.n_separate = n_separate
        self.block_indices = list(reversed(range(n_separate)))

        self.gcbs = []
        for i in range(self.n_separate):
            self.gcbs.append(GC_Block(hidden_feature,p_dropout=p_dropout, node_n=node_n))

        self.conv1ds = []
        self.bns = []
        self.dos = []
        for i in range(self.n_separate-1):
            self.conv1ds.append(nn.Conv1d(hidden_feature * 2, hidden_feature,kernel_size=1))
            self.bns.append(nn.BatchNorm1d(node_n * hidden_feature))
            self.dos.append(nn.Dropout(p_dropout))

        self.gcbs = nn.ModuleList(self.gcbs)
        self.conv1ds = nn.ModuleList(self.conv1ds)
        self.bns = nn.ModuleList(self.bns)
        self.dos = nn.ModuleList(self.dos)

        self.act_f = nn.Tanh()

    def forward(self, x):
        y = []
        block_indices = list(reversed(range(self.n_separate)))
        
        _idx = block_indices.pop()
        y.append(self.gcbs[_idx](x[_idx]))

        while(len(block_indices) != 0):
            _idx = block_indices.pop()
            ### convolution
            conv_x = self.conv1ds[_idx-1](torch.cat((x[_idx],y[_idx-1]),dim=-1).transpose(1,2)).transpose(1,2)
            ### batch-normalization
            b, n, f = conv_x.shape
            conv_x = self.bns[_idx-1](conv_x.reshape(b,-1)).reshape(b,n,f)
            ### activation function
            conv_x = self.act_f(conv_x)
            ### dropout
            conv_x = self.dos[_idx-1](conv_x)

            y.append(self.gcbs[_idx](conv_x))

        return y