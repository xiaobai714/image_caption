#!/bin/bash

/home/srwpf/anaconda2/bin/python eval.py\
    --dump_images 0 --num_images 5000 \
    --model log_fc/model-best.pth \
    --infos_path log_fc/infos_fc-best.pkl\
    --language_eval 1
