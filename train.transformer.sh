#!/bin/bash

TOOLPATH1=/home/ubuntu/cv/Image_caption_and_face_recognition/image-caption/coco-caption
TOOLPATH2=/home/ubuntu/cv/Image_caption_and_face_recognition/image-caption/cider
export PYTHONPATH=$TOOLPATH2:$TOOLPATH1:$PYTHONPATH

/home/ubuntu/anaconda2/bin/python train.py --id fc\
    --caption_model transformer \
    --input_json data/cocotalk.json \
    --input_fc_dir data/cocotalk_fc \
    --input_att_dir data/cocotalk_att \
    --input_label_h5 data/cocotalk_label.h5\
    --batch_size 32 --learning_rate 5e-5 \
    --learning_rate_decay_start 0 \
    --scheduled_sampling_start 0 \
    --checkpoint_path log_fc\
    --save_checkpoint_every 1000\
    --val_images_use 1000\
    --max_epochs 30 \
    --despective_model_on 0\
    --start_from  log_fc\
    --checkpoint_path log_fc\
    --id  fc-best
