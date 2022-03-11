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
#from get_videos import A
import argparse
  
def test():
    #사용할 GPU 설정
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
    os.environ["CUDA_VISIBLE_DEVICES"]= '4'    
    #problem = open(data_path2+'/error_list.txt', 'w') #에러나는 파일 이름 저장



    #영상/이미지 파일 경로 가지고 오기
    
    
    data_path = '/workspace/NIA/NIA_AI_DATASET_2021-ST-GCAE/E2ON/source/AI_HUB_1.03/Action/main_train_img/A23_MID'
    videos = os.listdir(data_path)
    #테스트 리스트 경로, 결과 파일 저장 경로 설정
    data_path2= '/workspace/NIA/NIA_AI_DATASET_2021-ST-GCAE/E2ON/source/AI_HUB_1.03/Action'
    
    test_date=str(datetime.today().month) +'.'+ str(datetime.today().day)  
    if not os.path.exists(data_path2+'/test_'+test_date):
        os.mkdir(data_path2+'/test_'+test_date)
    if not os.path.exists(data_path2+'/test_'+test_date+'/'):
        os.mkdir(data_path2+'/test_'+test_date+'/')
    print(data_path2+'/test_'+test_date+'/')
    print()

    # Action 정보 (20개)
    fm=open('/workspace/NIA/NIA_AI_DATASET_2021-ST-GCAE/E2ON/source/AI_HUB_1.03/input/main/index.txt', 'r')
    main_names = fm.readlines()
    print(len(videos))


    # 저장된 모델 로드
    model = tf.keras.models.load_model('/workspace/NIA/NIA_AI_DATASET_2021-ST-GCAE/E2ON/source/AI_HUB_1.03/main_result/epoch10_temp_weights_c3d.h5')
      

    for i in range(0,len(videos)):
        imgs = glob(data_path +'/'+videos[i]+'/*.jpg')
        
        print(len(imgs))

        #main_action=os.path.dirname(test_video)
        video_name=videos[i].split('/')[-1] 
        print(video_name)
        main_action = video_name[0:4]
        
        
        # 프레임당 결과 저장 파일 생성
        fw = open(data_path2+'/test_'+test_date+'/A23/'+video_name.split('.')[0]+'_test.txt', 'w')
        

        fps = len(imgs)
        clip = []
        main_count_list = [0 for i in range(len(main_names))]
        scene=0
    
        start = time.time()
        for i in tqdm(range(fps)):
            try:
                frame = cv2.imread(imgs[i],cv2.IMREAD_COLOR)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            except:
                print('error on image index:' + str(i)+'/'+ str(fps))
            else:    
                #if True:
                clip.append(frame)
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

                    # 프레임의 액션 추론
                    pred_main = model.predict(inputs)
                    main_label = np.argmax(pred_main[0])

                    main_count_list[main_label]=main_count_list[main_label]+1 
                    fw.write(main_names[main_label].split(' ')[1].strip()+" prob: %.4f" % pred_main[0][main_label]+'\n')
                        
                    clip.pop(0)
        
        fw.close()        
        

        # 영상 1개당 결과 통합 저장 파일 생성
        print("ㅠㅠ")        
        ftw = open(data_path2+'/test_'+test_date+'/A23/'+video_name.split('.')[0]+'_total.txt', 'w')
        ftw.write(video_name+'\n')
        ftw.write(main_action+' 영상 '+str(fps-15)+' 프레임 중 ')
        main_mode_label = np.argmax(main_count_list)    
        ftw.write(main_names[main_mode_label].split(' ')[-1].strip()+" 검출 "+str(main_count_list[main_mode_label])+" 프레임 ")
        main_frame_prod=main_count_list[main_mode_label]/(fps-15)*100

        ftw.write(str(int(main_frame_prod))+'%\n')
        for corr_main_label in range(len(main_names)):
            if video_name==main_names[corr_main_label].split(' ')[-1].strip()!=main_names[main_mode_label].split(' ')[-1].strip():
                main_frame_prod=main_count_list[corr_main_label]/(fps-15)*100
                ftw.write('\t\t\t\t\t\t\t'+main_names[corr_main_label].split(' ')[-1].strip()+" 검출 "+str(main_count_list[corr_main_label])+" 프레임 "+str(int(main_frame_prod))+'%\n')


    
        ftw.write('\n\n')
        ftw.close()

        end = time.time()
        print(f"{end-start} sec")
    #problem.close()    


test()