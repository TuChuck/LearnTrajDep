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
               --is_load="./checkpoint/test/ckpt_main_GCN_in10_out10_dctn20_sepd1_last.pth.tar" \
               --epochs=50
