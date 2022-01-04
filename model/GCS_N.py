import torch.nn as nn
import torch
from torch.nn.parameter import Parameter
import GraphConvolution
import GC_module

import itertools

class GCS_N(nn.Module):
    def __init__(self, input_feature, hidden_feature, p_dropout, n_separate = 2,num_stage = 1, node_n=48):
        """
        Define GCS_N(Graph Convolution Separation Network) that work in separated method

        :param input_feature: num of input feature
        :param hidden_feature: num of hidden feature
        :param p_dropout: drop out prob.
        :param n_separate : num of  GCN modules
        :param num_stage: number of residual blocks
        :param node_n: number of nodes in graph
        """
        super(GCS_N,self).__init__()

        self.gcms = []  # gc modules
        self.pre_gcs = []
        self.pre_bns = []
        self.pre_dos = []
        self.post_gcs  = []
        self.n_separate = n_separate
        self.num_stage = num_stage
        
        for i in range(n_separate):
            self.pre_gcs.append(GraphConvolution(input_feature, hidden_feature, node_n=node_n))
            self.pre_bns.append(nn.BatchNorm1d(node_n * hidden_feature))
            self.pre_dos.append(nn.Dropout(p_dropout))
            self.post_gcs.append(GraphConvolution(hidden_feature, input_feature, node_n=node_n))

        for i in range(num_stage):
            self.gcms.append(GC_module(hidden_feature,
                                       p_dropout=p_dropout, 
                                       n_separate = n_separate,
                                       node_n=node_n))
            
        self.gcms = nn.ModuleList(self.gcms)
        self.pre_gcs = nn.ModuleList(self.pre_gcs)
        self.pre_bns = nn.ModuleList(self.pre_bns)
        self.pre_dos = nn.ModuleList(self.pre_dos)
        self.post_gcs = nn.ModuleList(self.post_gcs)

        self.act_f = nn.Tanh()

    def forward(self,x):
        y = [] 
        F = []

        for i in range(self.n_separate):
            F_ = self.pre_gcs[i](x[i])
            b, n,f = F_.shape
            F_ = self.pre_bns[i](F_.view(b,-1)).view(b,n,f)
            F_ = self.act_f(F_)
            F_ = self.pre_dos[i](F_)
            F.append(F_)

        for i in range(self.num_stage):
            F = self.gcms[i](F)

        for i in range(self.n_separate):
            if i == 0:
                y = self.post_gcs[i](F[i])
                x_ = x[i]
            else:
                y = torch.cat((y,self.post_gcs[i](F[i])),dim=-1)
                x_ = torch.cat((x_,x[i]),dim=-1)

        return x_ + y  