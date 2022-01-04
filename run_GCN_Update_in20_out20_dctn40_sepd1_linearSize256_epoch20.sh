#!/bin/bash

python main.py --input_n=20  \
               --output=20   \
               --dct_n=40    \
               --data_dir="../../3.public_dataset/h3.6m/dataset/"\
               --model_prefix=./model_params\
               --num_separate=1\
               --actions=all\
               --linear_size=256\
               --job=4 \
               --model="GCN_Update" \
               --epochs=20
