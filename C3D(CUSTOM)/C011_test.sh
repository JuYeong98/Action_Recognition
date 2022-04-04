#!/bin/bash

for file in ../ALL_DATA/mp4/Action/train/A01/*
do
   echo ${file:29:39}
   python 3.argparse_inference.py --test_video ${file:29:39} --gpu 0
donecd