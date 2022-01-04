#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import argparse
from utils.utils import create_dir_tree
from pprint import pprint


class Options:
    def __init__(self, is_demo = False):
        self.parser = argparse.ArgumentParser()
        self.opt = None
        self.is_demo = is_demo
        self._file = None

    def _initial(self):
        # ===============================================================
        #                     General options
        # ===============================================================
        self.parser.add_argument('--data_dir', type=str, default='/home/wei/Downloads/h3.6m/dataset/', help='path to H36M dataset')
        self.parser.add_argument('--data_dir_3dpw', type=str, default='/home/wei/Downloads/3DPW/sequenceFiles/', help='path to 3DPW dataset')
        self.parser.add_argument('--data_dir_cmu', type=str, default='/home/wei/Downloads/cmu_mocap/', help='path to CMU dataset')
        self.parser.add_argument('--exp', type=str, default='test', help='ID of experiment')
        self.parser.add_argument('--ckpt', type=str, default='checkpoint/', help='path to save checkpoint')
        self.parser.add_argument('--model_prefix', type=str, default='')
        self.parser.add_argument('--_file', type=str, default=None)
        # ===============================================================
        #                     Model options
        # ===============================================================
        self.parser.add_argument('--max_norm', dest='max_norm', action='store_true',
                                 help='maxnorm constraint to weights')
        self.parser.add_argument('--linear_size', type=int, default=256, help='size of each model layer')
        self.parser.add_argument('--num_stage', type=int, default=12, help='# layers in linear model')
        self.parser.add_argument('--num_separate', type=int, default=1, help="parameter that separate input sequence in sepd method")
        self.parser.add_argument('--model', type=str, default="GCN", help="choise model")
        # ===============================================================
        #                     Running options
        # ===============================================================
        self.parser.add_argument('--lr', type=float, default=5.0e-4)
        self.parser.add_argument('--lr_decay', type=int, default=2, help='every lr_decay epoch do lr decay')
        self.parser.add_argument('--lr_gamma', type=float, default=0.96)
        self.parser.add_argument('--input_n', type=int, default=10, help='observed seq length')
        self.parser.add_argument('--output_n', type=int, default=25, help='future seq length')
        self.parser.add_argument('--dct_n', type=int, default=35, help='number of DCT coeff. preserved for 3D')
        self.parser.add_argument('--actions', type=str, default='all', help='path to save checkpoint')
        self.parser.add_argument('--epochs', type=int, default=50)
        self.parser.add_argument('--dropout', type=float, default=0.5,
                                 help='dropout probability, 1.0 to make no dropout')
        self.parser.add_argument('--train_batch', type=int, default=16)
        self.parser.add_argument('--test_batch', type=int, default=128)
        self.parser.add_argument('--job', type=int, default=10, help='subprocesses to use for data loading')
        self.parser.add_argument('--is_load', type=str, default = None, help='path to load existing model')
        self.parser.add_argument('--sample_rate', type=int, default=2, help='frame sampling rate')
        self.parser.add_argument('--is_norm_dct', dest='is_norm_dct', action='store_true', help='whether to normalize the dct coeff')
        self.parser.add_argument('--is_norm', dest='is_norm', action='store_true', help='whether to normalize the angles/3d coordinates')

        self.parser.set_defaults(max_norm=True)
        # self.parser.set_defaults(is_load=False)
        # self.parser.set_defaults(is_norm_dct=True)
        # self.parser.set_defaults(is_norm=True)

    def _print(self):
        print("\n==================Options=================")
        pprint(vars(self.opt), indent=4)
        print("==========================================\n")

    def parse(self):
        self._initial()
        self.opt = self.parser.parse_args()
        # do some pre-check
        ckpt = os.path.join(self.opt.ckpt, self.opt.exp)
        if not os.path.isdir(ckpt):
            os.makedirs(ckpt)
        self.opt.ckpt = ckpt
        self._print()
        
        if not self.is_demo:
            self.opt._file = self.opt.model+"_in{:d}_out{:d}_dctn{:d}_sepd{:d}_linearSize{:d}_epoch{:d}".format(
                self.opt.input_n, 
                self.opt.output_n, 
                self.opt.dct_n, 
                self.opt.num_separate, 
                self.opt.linear_size,
                self.opt.epochs)

            create_dir_tree(os.path.join(self.opt.model_prefix,self.opt._file))
            config_path = os.path.join(self.opt.model_prefix,self.opt._file,'config','config.json')

            with open(config_path, 'w') as file_:
                json.dump(vars(self.opt), file_, indent=4)

        return self.opt
