#!/bin/bash

/home/ubuntu/anaconda2/bin/python eval.py\
    --model log_fc/model-best.pth\
    --infos_path log_fc/infos_fc-best.pkl\
    --num_images 10\
    --image_folder /home/ubuntu/cv/Image_caption_and_face_recognition/dataset_joint_learning\


