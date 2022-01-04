#!/bin/bash

python main.py --input_n=10  \
               --output=10   \
               --dct_n=20    \
               --data_dir="../../3.public_dataset/h3.6m/dataset/"\
               --model_prefix=./model_params\
               --num_separate=2\
               --actions=all\
               --linear_size=256\
               --job=4 \
               --model="GCN_Block"
