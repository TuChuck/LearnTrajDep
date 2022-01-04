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

import GraphConvolution
import GC_Block
import GCN
import GC_module
import GCN_Block
import GCN_Block_NonRes
import GCS_N
import GCS_N_NonRes

class ModelFn(nn.Module):
    def __init__(self, model_name, option) -> None:
        super(ModelFn,self).__init__()
        self.input_feature = option.dct_n
        self.hidden_feature = option.linear_size
        self.p_dropout = option.dropout
        self.num_stage = option.num_stage
        self.node_n = option.node_n
        self.n_separate = option.num_separate

        self.model = self.switch(model_name)
        

    def switch(self,model_name):
        model = {"GCN":GCN(input_feature=self.input_feature,
                           hidden_feature=self.hidden_feature,
                           p_dropout=self.p_dropout,
                           num_stage=self.num_stage,
                           node_n=self.node_n),

                 "GCN_Block": GCN_Block(input_feature=self.input_feature,
                           hidden_feature=self.hidden_feature,
                           p_dropout=self.p_dropout,
                           num_stage=self.num_stage,
                           node_n=self.node_n,
                           n_separate=self.n_separate),

                 "GCN_Block_NonRes":GCN_Block_NonRes(
                           input_feature=self.input_feature,
                           hidden_feature=self.hidden_feature,
                           p_dropout=self.p_dropout,
                           num_stage=self.num_stage,
                           node_n=self.node_n,
                           n_separate=self.n_separate),

                 "GCS_N":GCS_N(input_feature=self.input_feature,
                           hidden_feature=self.hidden_feature,
                           p_dropout=self.p_dropout,
                           num_stage=self.num_stage,
                           node_n=self.node_n,
                           n_separate=self.n_separate),

                 "GCS_N_NonRes":GCS_N_NonRes(input_feature=self.input_feature,
                           hidden_feature=self.hidden_feature,
                           p_dropout=self.p_dropout,
                           num_stage=self.num_stage,
                           node_n=self.node_n,
                           n_separate=self.n_separate)}.get(model_name,"GCN")

        return model

def test_swich(model_name):
    option={}
    option.input_feature = 20
    option.hidden_feature = 256
    option.p_dropout = 0.5
    option.num_stage = 12
    option.node_n = 48
    option.n_separate = 2

    model = ModelFn(model_name,option)
    
    X = torch.FloatTensor(16,option.input_feature,option.node_n)
    print(model(X).size())

if __name__ == '__main__':
    test_swich('GCN')