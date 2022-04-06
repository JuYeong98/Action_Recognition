#!/bin/bash

for file in ../AI_HUB_1.03/Action/main_train_img/*
do
   #echo ${file:37}
   python 3.argparse_inference_gpu1.py --test_video ${file:37} --gpu 1
   break
done