#!/bin/bash

/home/ubuntu/anaconda2/bin/python eval.py\
    --dump_images 0\
    --model log_fc/model-best.pth\
    --infos_path log_fc/infos_fc-best.pkl\
    --num_images 1000\
    --language_eval 1


