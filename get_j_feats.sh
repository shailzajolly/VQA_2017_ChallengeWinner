#!/bin/bash

python3.6 get_joint_feats.py --mode "eval" \
               --hidden-size 512 \
               --batch-size 4096 \
               --vbatch-size 4096 \
               --epoch 40 \
               --data-root /b_test/jolly/VQA_Bottom-up/data_yesno/non_yn/ \
               --save log_dgx/yn_exps \
               --log-freq 50 \
	       --resume log_dgx/log_non_yes_no/best.pth.tar \
               --wemb-init /b_test/jolly/VQA_Bottom-up/data_yesno/non_yn/glove_pretrained_300.npy \
