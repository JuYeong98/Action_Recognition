import shutil
import os
import random
import cv2
import json

def move_file(path1):
    action_list = os.listdir(path1+'/all') #수정
    action_list.sort()
    if not os.path.exists(path1+'/test_video/'):
        os.mkdir(path1+'/test_video/')
    if not os.path.exists(path1+'/train_video/'):
        os.mkdir(path1+'/train_video/')
    for action in action_list:
        video_list = os.listdir(path1+'/all/'+action) #수정
        video_list.sort()
        if not os.path.exists(path1+'/test_video/'+action):
            os.mkdir(path1+'/test_video/'+action)
        if not os.path.exists(path1+'/train_video/'+action):
            os.mkdir(path1+'/train_video/'+action)
        for video in video_list:
            cnt=random.randint(1, 10)
            if cnt>2:
                print(str(cnt)+'1')
                shutil.move(path1+'/video'+action+'/'+video, path1+'/train_video/'+action+'/'+video)
            else:
                print(str(cnt)+'2')
                shutil.move(path1+'/video'+action+'/'+video, path1+'/test_video/'+action+'/'+video)

def main_video2img(path1, path2):
    if not os.path.exists(path2+'/main_train_img'):
        os.mkdir(path2+'/main_train_img')
    if not os.path.exists(path2+'/main_test_img'):
        os.mkdir(path2+'/main_test_img')
    if not os.path.exists(path2+'/main_val_img'):
        os.mkdir(path2+'/main_val_img')   

    train_action_list = os.listdir(path1+'/train')#train   변경
    val_action_list = os.listdir(path1+'/val') # val       추가
    test_action_list = os.listdir(path1+'/test') #test     변경
    train_action_list.sort()
    val_action_list.sort()
    test_action_list.sort()
    train_action_list = train_action_list[12:13]
    print(train_action_list)
    
    for train_action in train_action_list:  #action_list 는 A01 ~A31
        if not os.path.exists(path2+'/main_train_img/'+train_action):
            os.mkdir(path2+'/main_train_img/'+train_action)
            os.mkdir(path2+'/main_val_img'+train_action)
            os.mkdir(path2+'/main_test_img/'+train_action)
        video_list = os.listdir(path1+'/train/'+train_action)  #변경
        video_list.sort()
        for video in video_list:
            prefix = video.split('.')[0]
            if not os.path.exists(path2+'/main_train_img/'+train_action+'/'+prefix):
                os.mkdir(path2+'/main_train_img/'+train_action+'/'+prefix)
                #print(prefix)
            cap = cv2.VideoCapture(path1+'/train/'+train_action+'/'+video)
            print(path1+'/train/'+train_action+'/'+video)
            fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print(fps)
            for i in range(fps):
                ret, frame = cap.read()
                #print(ret)
                if ret:
                    frame = cv2.resize(frame, (171, 128))
                    cv2.imwrite(path2+'/main_train_img/' + train_action + '/' + prefix + '/'+str(i+1)+'.jpg',frame)
              
    """
    for test_action in test_action_list:          
        video_list = os.listdir(path1+'/test/'+test_action)
        video_list.sort()
        for video in video_list:
            prefix = video.split('.')[0]
            if not os.path.exists(path2+'/main_test_img/'+test_action+'/'+prefix):
                os.mkdir(path2+'/main_test_img/'+test_action+'/'+prefix)
            #cap = cv2.VideoCapture(path1+'/test/'+test_action+'/'+video) #수정
            #print(path1+'/test/'+test_action+'/'+video)  #수정
            #fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            #print(fps)
            #for i in range(fps):
            #    ret, frame = cap.read()
            #    if ret:
            #        frame = cv2.resize(frame, (171, 128))
            #        cv2.imwrite(path2+'/main_test_img/' + test_action + '/' + prefix + '/'+str(i+1)+'.jpg',frame)
    """              
    print(val_action_list) 
    #for val in val_action_list:
        #os.mkdir(path2+'/main_val_img/'+val)
        
        
    """
    for val_action in val_action_list:  #action_list 는 A01 ~A31
        if not os.path.exists(path2+'/main_val_img/'+val_action):
            os.mkdir(path2+'/main_val_img/'+val_action)
            print(val_action)
            
        video_list = os.listdir(path1+'/val/'+val_action)  #변경
        video_list.sort()
        print(video_list)
        for video in video_list:
            prefix = video.split('.')[0]
            if not os.path.exists(path2+'/main_val_img/'+val_action+'/'+prefix):
                os.mkdir(path2+'/main_val_img/'+val_action+'/'+prefix)
            cap = cv2.VideoCapture(path1+'/val/'+val_action+'/'+video)
            print(path1+'/val/'+val_action+'/'+video)
            fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print(fps)
            for i in range(fps):
                ret, frame = cap.read()
                if ret:
                    
                    frame = cv2.resize(frame, (171, 128))
                    cv2.imwrite(path2+'/main_val_img/' + val_action + '/' + prefix + '/'+str(i+1)+'.jpg',frame)                
    """          

def sub_video2img(path1, path2):
    if not os.path.exists(path2+'/sub_train_img'):
        os.mkdir(path2+'/sub_train_img')
    if not os.path.exists(path2+'/sub_test_img'):
        os.mkdir(path2+'/sub_test_img')

    train_action_list = os.listdir(path1+'/train_video')
    test_action_list = os.listdir(path1+'/test_video')
    train_action_list.sort()
    test_action_list.sort()

    for train_action in train_action_list:
        video_list = os.listdir(path1+'/train_video/'+train_action)
        video_list.sort()
        for video in video_list:
            prefix = video.split('.')[0]

            json_data=[]
            print(path1+'/json/' + train_action +'/'+ prefix+'_blockinfo.json')
            with open(path1+'/json/' + train_action +'/'+ prefix+'_blockinfo.json', 'r') as f:
                json_data.append(json.load(f))
            for i in range (len(json_data[0]['block_information'])):
                subaction=json_data[0]['block_information'][i]['block_detail']
                if not os.path.exists(path2+'/sub_train_img/'+subaction):
                    os.mkdir(path2+'/sub_train_img/'+subaction)
                    os.mkdir(path2+'/sub_test_img/'+subaction)
                
            cap = cv2.VideoCapture(path1+'/train_video/'+train_action+'/'+video)
            print(path1+'/train_video/'+train_action+'/'+video)
            fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print(fps)
            
            for i in range(fps):
                ret, frame = cap.read()
                if ret:
                    frame = cv2.resize(frame, (171, 128))
                    for j in range (len(json_data[0]['block_information'])):
                        if(int(json_data[0]['block_information'][j]['start_frame_index'])==i):
                            subaction = json_data[0]['block_information'][j]['block_detail']

                    save_name = path2+'/sub_train_img/'+subaction + '/' + prefix+'_'+subaction + '/'
                    if not os.path.exists(save_name):
                        os.mkdir(save_name)
                    cv2.imwrite(save_name+str(i+1)+'.jpg',frame)

    for test_action in test_action_list:
        video_list = os.listdir(path1+'/test_video/'+test_action)
        video_list.sort()
        for video in video_list:
            prefix = video.split('.')[0]

            json_data=[]
            print(path1+'/json/' + test_action +'/'+ prefix+'_blockinfo.json')
            with open(path1+'/json/' + test_action +'/'+ prefix+'_blockinfo.json', 'r') as f:
                json_data.append(json.load(f))

            cap = cv2.VideoCapture(path1+'/test_video/'+test_action+'/'+video)
            print(path1+'/test_video/'+test_action+'/'+video)
            fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print(fps)
            
            for i in range(fps):
                ret, frame = cap.read()
                if ret:
                    frame = cv2.resize(frame, (171, 128))
                    for j in range (len(json_data[0]['block_information'])):
                        if(int(json_data[0]['block_information'][j]['start_frame_index'])==i):
                            subaction = json_data[0]['block_information'][j]['block_detail']
                    
                    save_name = path2+'/sub_test_img/' +subaction+'/'+ prefix+'_'+subaction + '/'
                    if not os.path.exists(save_name):
                        os.mkdir(save_name)
                    cv2.imwrite(save_name+str(i+1)+'.jpg',frame)  
                                 

def sub_action_search(sub_action):
    if sub_action=='A01':
            sub_action_name='Child is alone.'
    elif sub_action=='A05':
        sub_action_name='Stroller left unattended.'
    elif sub_action=='A06':
        sub_action_name='Push the stroller hard.'
    elif sub_action=='A07':
        sub_action_name='Hold the stroller with your feet.'
    elif sub_action=='A08':
        sub_action_name='Pulled the stroller down hard.'
    elif sub_action=='A14':
        sub_action_name='Adult throws a child.'
    elif sub_action=='A16':
        sub_action_name='Overturn the stroller.'
    elif sub_action=='A17':
        sub_action_name='Pacing around.'
    elif sub_action=='A18':
        sub_action_name='Trying to open the door lock.'
    elif sub_action=='A19':
        sub_action_name='kick the door.'
    elif sub_action=='A20':
        sub_action_name='Trying to look inside the door.'
    elif sub_action=='A21':
        sub_action_name='Knocked on the door.'
    elif sub_action=='A29':
        sub_action_name='Hitting with tools.'
    elif sub_action=='A30':
        sub_action_name='Stealing packages.'
    elif sub_action=='A31':
        sub_action_name='Hanging around in front of a car.'
    elif sub_action=='N0':
        sub_action_name='Normal behavior.'
    elif sub_action=='N1':
        sub_action_name='Normal behavior.'
    elif sub_action=='SY13':
        sub_action_name='A kid is in a stroller.'
    elif sub_action=='SY14':
        sub_action_name='A child is walking around.'
    elif sub_action=='SY15':
        sub_action_name='A person is pacing around the door.'
    elif sub_action=='SY16':
        sub_action_name='A person is standing around the door.'
    elif sub_action=='SY17':
        sub_action_name='A person is sitting around the door.'
    elif sub_action=='SY25':
        sub_action_name='A person is pacing in front of the package.'
    elif sub_action=='SY26':
        sub_action_name='A person is standing around the delivery.'
    elif sub_action=='SY28':
        sub_action_name='A person standing in front of a car.'
    elif sub_action=='SY29':
        sub_action_name='A person is standing around a car.'
    elif sub_action=='SY30':
        sub_action_name='A person is sitting around a car.'
    elif sub_action=='SY31':
        sub_action_name='A person is leaning against the door.'
    elif sub_action=='SY32':
        sub_action_name='A person is leaning against a wall (or pole).'
    return sub_action_name

def index_gen(path2):
    if not os.path.exists(path2+'/input'):
        os.mkdir(path2+'/input')
    if not os.path.exists(path2+'/input/main'):
        os.mkdir(path2+'/input/main')
    #if not os.path.exists(path2+'/input/sub'):
        #os.mkdir(path2+'/input/sub')

    ftrain =  open(path2+'/../input/main'+'/index.txt', "w")
    #ftest =  open(path2+'/input/sub'+'/index.txt', "w")
        
    main_action_list = os.listdir(path2+'/main_train_img')
    #sub_action_list = os.listdir(path2+'/sub_train_img')
    main_action_list.sort()
    #sub_action_list.sort()
    print(main_action_list)


    for i, main_action in enumerate (main_action_list):
        ftrain.write(str(i)+': '+main_action + '\n')

    #for i, sub_action in enumerate (sub_action_list):
        #sub_action_name=sub_action_search(sub_action)
        #ftest.write(str(i)+': '+sub_action + ': '+sub_action_name+'\n')

def makefile(path2):    
    fm_train =  open(path2+'/../input/main/trainfile.txt', "w")
    fm_test =  open(path2+'/../input/main/testfile.txt', "w")
    fm_val =  open(path2+'/../input/main/valfile.txt', "w")
    #fs_train =  open(path2+'/input/sub/trainfile.txt', "w")
    #fs_test =  open(path2+'/input/sub/testfile.txt', "w")

    main_action_list = os.listdir(path2+'/main_train_img')
    #sub_action_list = os.listdir(path2+'/sub_train_img')

    main_action_list.sort()
    #sub_action_list.sort()

    for i, main_action in enumerate(main_action_list):
        train_img_list = os.listdir(path2+'/main_train_img/'+main_action)  
        test_img_list = os.listdir(path2+'/main_test_img/'+main_action)
        val_img_list = os.listdir(path2+'/main_val_img/'+main_action)
        train_img_list.sort()
        test_img_list.sort()
        val_img_list.sort()
        for val_img in val_img_list:
            fm_val.write('main_val_img/'+main_action+'/'+val_img + " " + str(i) + "\n")
        
        for train_img in train_img_list:
            fm_train.write('main_train_img/'+main_action+'/'+train_img + " " + str(i) + "\n")
        for test_img in test_img_list:
            fm_test.write('main_test_img/'+main_action+'/'+test_img + " " + str(i) + "\n")
    """ sub 부분은 지움
    for i, sub_action in enumerate(sub_action_list):
        train_img_list = os.listdir(path2+'/sub_train_img/'+sub_action)
        test_img_list = os.listdir(path2+'/sub_test_img/'+sub_action)
        train_img_list.sort()
        test_img_list.sort()
        for train_img in train_img_list:
            fs_train.write('sub_train_img/'+sub_action+'/'+train_img + " " + str(i) + "\n")
        for test_img in test_img_list:
            fs_test.write('sub_test_img/'+sub_action+'/'+test_img + " " + str(i) + "\n")
    """        
        
def file2list(path2):
    fmr_train = open(path2+'/../input/main/'+'trainfile.txt',mode='r')
    #fmr_test  = open(path2+'/input/main/'+'testfile.txt',mode='r')
    fmr_val  = open(path2+'/../input/main/'+'valfile.txt',mode='r')
    #fsr_train = open(path2+'/input/sub/'+'trainfile.txt',mode='r')
    #fsr_test  = open(path2+'/input/sub/'+'testfile.txt',mode='r')

    main_train_list = fmr_train.readlines()
    #main_test_list = fmr_test.readlines()
    main_val_list = fmr_val.readlines()
    #sub_train_list = fsr_train.readlines()
    #sub_test_list = fsr_test.readlines()

    fmw_train = open(path2+'/../input/main/'+'train_list.txt', 'w')
    #fmw_test = open(path2+'/input/main/'+'test_list.txt', 'w')
    fmw_val = open(path2+'/../input/main/'+'val_list.txt', 'w')
    #fsw_train = open(path2+'/input/sub/'+'test_list.txt', 'w')
    #fsw_test = open(path2+'/input/sub/'+'test_list.txt', 'w')

    clip_length = 16
    
    for line in main_train_list:
        images = os.listdir(path2+'/'+line.split(' ')[0])
        images.sort()
        nb = len(images) // clip_length
        for i in range(nb):
            fmw_train.write(line.split(' ')[0]+' '+ str(i*clip_length+1)+' '+line.split(' ')[-1])
    """
    for line in main_test_list:
        images = os.listdir(path2+'/'+line.split(' ')[0])
        images.sort()
        nb = len(images) // clip_length
        for i in range(nb):
            fmw_test.write(line.split(' ')[0]+' '+ str(i*clip_length+1)+' '+line.split(' ')[-1])
            
    """        
    for line in main_val_list:
        images = os.listdir(path2+'/'+line.split(' ')[0])
        images.sort()
        nb = len(images) // clip_length
        for i in range(nb):
            fmw_val.write(line.split(' ')[0]+' '+ str(i*clip_length+1)+' '+line.split(' ')[-1])        
"""
    for line in sub_train_list:
        images = os.listdir(path2+'/'+line.split(' ')[0])
        images.sort()
        nb = len(images) // clip_length
        for i in range(nb):
            fsw_train.write(line.split(' ')[0]+' '+ str(i*clip_length+1)+' '+line.split(' ')[-1])

    for line in sub_test_list:
        images = os.listdir(path2+'/'+line.split(' ')[0])
        images.sort()
        nb = len(images) // clip_length
        for i in range(nb):
            fsw_test.write(line.split(' ')[0]+' '+ str(i*clip_length+1)+' '+line.split(' ')[-1])
"""            
"""
def main():
    #video_path, data_path 만 수정
    video_path='/workspace/NIA/NIA_AI_DATASET_2021-ST-GCAE/E2ON/source/ALL' #영상이 저장되어있는 경로
    data_path='/workspace/NIA/NIA_AI_DATASET_2021-ST-GCAE/E2ON/source/ALL_E2ON_Data_12.13' #학습데이터가 저장될 경로

    if not os.path.exists(data_path):
            os.mkdir(data_path)

    #move_file(video_path)
    main_video2img(video_path,data_path)
    #sub_video2img(video_path,data_path)
    #index_gen(data_path)
    #makefile(data_path)
    #file2list(data_path)

if __name__ == '__main__':
     main()
"""

video_path='/workspace/NIA/NIA_AI_DATASET_2021-ST-GCAE/E2ON/source/ALL_DATA/mp4/Action' #영상이 저장되어있는 경로  경로 변경
data_path='/workspace/NIA/NIA_AI_DATASET_2021-ST-GCAE/E2ON/source/AI_HUB_1.03/Action' #학습데이터가 저장될 경로    경로 변경

if not os.path.exists(data_path):
    os.mkdir(data_path)

#move_file(video_path)
main_video2img(video_path,data_path)     
#index_gen(data_path)
#makefile(data_path)
#file2list(data_path)