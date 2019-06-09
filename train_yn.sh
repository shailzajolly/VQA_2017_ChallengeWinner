#!/bin/bash

python3.6 yn_net.py --mode "train" \
               --hidden-size 512 \
               --batch-size 256 \
               --vbatch-size 4096 \
               --epoch 500 \
               --data-root data/ \
               --save log_dgx/trash/log_tr_orig_testyn \
               --log-freq 2000 \
               --wemb-init data/glove_pretrained_300.npy \
