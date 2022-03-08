# coding=utf8
from models import c3d_model
from tensorflow.keras.optimizers import SGD
import tensorflow as tf
import numpy as np
import cv2
import os
import json
from PIL import Image,ImageDraw,ImageFont
from datetime import datetime
from glob import glob
from tqdm import tqdm
import time

import argparse
  
def test(test_video, gpu):
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
    os.environ["CUDA_VISIBLE_DEVICES"]= str(gpu)    
    video_path = '/workspace/NIA/NIA_AI_DATASET_2021-ST-GCAE/E2ON/source/ALL_DATA/mp4/Action/train/'
    #영상이 저장되어 있는 경로
    # test_video='C021/C021_A17_SY16_P01_B07_01DBS.mp4'
    print(test_video)
    
        
    #테스트 할 영상
    data_path = '/workspace/NIA/NIA_AI_DATASET_2021-ST-GCAE/E2ON/source/AI_HUB_1.03/Action'
    data_path2= '/workspace/NIA/NIA_AI_DATASET_2021-ST-GCAE/E2ON/source/AI_HUB_1.03'
    #학습 데이터가 저장되어 있는 경로

    test_date=str(datetime.today().month) +'.'+ str(datetime.today().day)  
    main_action=os.path.dirname(test_video)
    video_name=os.path.basename(test_video) 

    if not os.path.exists(data_path+'/test_'+test_date):
        os.mkdir(data_path+'/test_'+test_date)
    if not os.path.exists(data_path+'/test_'+test_date+'/'+main_action):
        os.mkdir(data_path+'/test_'+test_date+'/'+main_action)
    print(data_path+'/test_'+test_date+'/'+main_action)

    
    fm=open(data_path2+'/input/main/index.txt', 'r')
    main_names = fm.readlines()


    fw = open(data_path+'/test_'+test_date+'/'+main_action+'/'+video_name.split('.')[0]+'_test.txt', 'w')

    # init model
    model = c3d_model()
    lr = 0.005
    sgd = SGD(lr=lr, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
   

    cap = cv2.VideoCapture(video_path+test_video)
    fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(3)) 
    height = int(cap.get(4)) 
    fcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    out = cv2.VideoWriter(data_path+'/test_'+test_date+'/'+main_action+'/'+video_name.split('.')[0]+'_test.mp4', fcc, 20, (width*2, height))

    clip = []
    main_count_list = [0 for i in range(len(main_names))]
    # sub_count_list = [[0 for i in range(len(sub_names))] for j in range(len(json_data[0]['block_information']))]
    scene=0

    start = time.time()
    for i in tqdm(range(fps)):
        ret, frame = cap.read()
        black_img=np.zeros((height,width,3),dtype=np.uint8)
        if ret:
            tmp = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            clip.append(cv2.resize(tmp, (171, 128)))
            if len(clip) == 16:
                inputs = np.array(clip).astype(np.float32)
                inputs = np.expand_dims(inputs, axis=0)
                inputs[..., 0] -= 99.9
                inputs[..., 1] -= 92.1
                inputs[..., 2] -= 82.6
                inputs[..., 0] /= 65.8
                inputs[..., 1] /= 62.3
                inputs[..., 2] /= 60.3
                inputs = inputs[:,:,8:120,30:142,:]
                inputs = np.transpose(inputs, (0, 2, 3, 1, 4))

                # model.load_weights(data_path+'/main_result/epoch10/weights_c3d.h5', by_name=True)
                model.load_weights(data_path2+'/main_result/epoch10_temp_weights_c3d.h5', by_name=True)
                # checkpoint_path = "training_10/cp-{epoch:04d}.ckpt"
                # checkpoint_dir = os.path.dirname(checkpoint_path)
                # latest = tf.train.latest_checkpoint(checkpoint_dir)
                # model.load_weights(latest)
                
                pred_main = model.predict(inputs)
                # print(pred_main.shape)
                
                main_label = np.argmax(pred_main[0])
                
                main_count_list[main_label]=main_count_list[main_label]+1
                
              
                cv2.putText(black_img, main_names[main_label].split(' ')[1].strip()+" prob: %.4f" % pred_main[0][main_label], (20, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
                    
                fw.write(main_names[main_label].split(' ')[1].strip()+" prob: %.4f" % pred_main[0][main_label]+'\n')
                
                clip.pop(0)
            add_frame_img = cv2.hconcat((frame, black_img))
