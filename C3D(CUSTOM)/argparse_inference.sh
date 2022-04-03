#!/bin/bash

for file in ../ALL_VIDEO/test_video/C011/*
do
   # echo ${file:24:36}
   python test.py ${file:24:36} 
done