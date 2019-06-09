#!/bin/bash

python3.6 main.py --mode "eval" \
               --hidden-size 512 \
               --batch-size 512 \
               --vbatch-size 4096 \
               --epoch 40 \
               --data-root data/ \
               --resume log_dgx/log_relu_512_do_e_q_v_cls/ckpts/model_30.pth.tar \
               --save log_dgx/trash/log_tr_orig_te_yn/ \
	       --log-freq 50 \
               --wemb-init data/glove_pretrained_300.npy \
