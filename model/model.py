#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function
# from torch._C import T

import torch.nn as nn
import torch
from torch.nn.parameter import Parameter
import math

import itertools

class GraphConvolution(nn.Module):
    """
    adapted from : https://github.com/tkipf/gcn/blob/92600c39797c2bfb61a508e52b88fb554df30177/gcn/layers.py#L132
    """

    def __init__(self, in_features, out_features, bias=True, node_n=48):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.att = Parameter(torch.FloatTensor(node_n, node_n))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.att.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(self.att, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GC_Block(nn.Module):
    def __init__(self, in_features, p_dropout, bias=True, node_n=48):
        """
        Define a residual block of GCN
        """
        super(GC_Block, self).__init__()
        self.in_features = in_features
        self.out_features = in_features

        self.gc1 = GraphConvolution(in_features, in_features, node_n=node_n, bias=bias)
        self.bn1 = nn.BatchNorm1d(node_n * in_features)

        self.gc2 = GraphConvolution(in_features, in_features, node_n=node_n, bias=bias)
        self.bn2 = nn.BatchNorm1d(node_n * in_features)

        self.do = nn.Dropout(p_dropout)
        self.act_f = nn.Tanh()

    def forward(self, x):
        y = self.gc1(x)
        b, n, f = y.shape
        y = self.bn1(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)

        y = self.gc2(y)
        b, n, f = y.shape
        y = self.bn2(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)

        return y + x

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, input_feature, hidden_feature, p_dropout, num_stage=1, node_n=48):
        """
        Define GCN that learn Adjacent matrix
        """
        super(GCN, self).__init__()
        self.num_stage = num_stage

        self.gc1 = GraphConvolution(input_feature, hidden_feature, node_n=node_n)
        self.bn1 = nn.BatchNorm1d(node_n * hidden_feature)

        self.gcbs = []
        for i in range(num_stage):
            self.gcbs.append(GC_Block(hidden_feature, p_dropout=p_dropout, node_n=node_n))

        self.gcbs = nn.ModuleList(self.gcbs)

        self.gc7 = GraphConvolution(hidden_feature, input_feature, node_n=node_n)

        self.do = nn.Dropout(p_dropout)
        self.act_f = nn.Tanh()

    def forward(self, x):
        y = self.gc1(x)   # x.shape => [bs, njoint*3,  input_n + output_n]
        b, n, f = y.shape
        y = self.bn1(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)

        for i in range(self.num_stage):
            y = self.gcbs[i](y)

        y = self.gc7(y)
        y = y + x

        return y

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

class GCN_Block_NonRes(nn.Module):
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

        return x_+y

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

class GCS_N_NonRes(nn.Module):
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

        return y  
        

        



        

        