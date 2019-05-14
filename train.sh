#!/bin/bash

python3.6 main.py --mode "train" \
               --hidden-size 512 \
               --batch-size 512 \
               --vbatch-size 4096 \
               --epoch 40 \
               --data-root data/ \
               --save log_dgx/log_relu_512_do_e_q_v_cls \
               --log-freq 50 \
               --wemb-init data/glove_pretrained_300.npy \
