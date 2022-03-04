import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
# os.environ["CUDA_VISIBLE_DEVICES"]= "2"  
# 세션 사용시.
# session = tf.Session(config=config)
from models import c3d_model

from schedules import onetenth_4_8_12
import numpy as np
import random
import cv2
import random
import matplotlib
matplotlib.use('AGG')
import glob
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.optimizers import SGD,Adam
from keras.utils import np_utils
from tensorflow.keras.callbacks import ModelCheckpoint
import pickle
import time
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True

def plot_history(history, result_dir):
    plt.plot(history.history['accuracy'], marker='.')
   # plt.plot(history.history['val_accuracy'], marker='.')
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.grid()
    #plt.legend(['accuracy', 'val_accuracy'], loc='lower right')
    plt.savefig(os.path.join(result_dir, 'model_acc.png'))
    plt.close()

    plt.plot(history.history['loss'], marker='.')
    #plt.plot(history.history['val_loss'], marker='.')
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid()
    plt.legend(['loss'], loc='upper right')
    plt.savefig(os.path.join(result_dir, 'model_loss.png'))
    plt.close()

  
def process_batch(lines,img_path,train=True):
    num = len(lines)
    batch = np.zeros((num,16,112,112,3),dtype='float32')
    labels = np.zeros(num,dtype='int')
    error_list=[]
    for i in range(num):
        path = lines[i].split(' ')[0]
        label = lines[i].split(' ')[-1]
        symbol = lines[i].split(' ')[1]
        label = label.strip('\n')
        label = int(label)
        symbol = int(symbol)-1
        # print(path)
        imgs = os.listdir(img_path+path)
        imgs.sort(key=str.lower)
        if train:
            crop_x = random.randint(0, 15)
            crop_y = random.randint(0, 58)
            is_flip = random.randint(0, 1)
            for j in range(16):
                img = imgs[symbol + j]

                try:
                    image = cv2.imread(img_path + path + '/' + img)      #변경      
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)        
                except:
                    error_list.append(img_path + path + '/' + img)       #변경
                else:
                    
                    image = cv2.resize(image, (171, 128))
                    if is_flip == 1:
                        image = cv2.flip(image, 1)
                    batch[i][j][:][:][:] = image[crop_x:crop_x + 112, crop_y:crop_y + 112, :]
            labels[i] = label
        else:
            for j in range(16):
                img = imgs[symbol + j]
                
                try:
                    image = cv2.imread(img_path + path + '/' + img)       #변경
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                except:
                    error_list.append(img_path + path + '/' + img)       #변경
                else:
                    
                    image = cv2.resize(image, (171, 128))
                    batch[i][j][:][:][:] = image[8:120, 30:142, :]
            labels[i] = label
    with open('error_list.pkl','wb') as f:
        pickle.dump(error_list ,f)        
    return batch, labels

def preprocess(inputs):
    inputs[..., 0] -= 99.9
    inputs[..., 1] -= 92.1
    inputs[..., 2] -= 82.6
    inputs[..., 0] /= 65.8
    inputs[..., 1] /= 62.3
    inputs[..., 2] /= 60.3
    # inputs /=255.
    # inputs -= 0.5
    # inputs *=2.
    return inputs

def generator_train_batch(train_txt,batch_size,num_classes,img_path):

    ff = open(train_txt, 'r')
    lines = ff.readlines()
    num = len(lines)
    while True:
        new_line = []
        index = [n for n in range(num)]
        random.shuffle(index)
        for m in range(num):
            new_line.append(lines[index[m]])
        for i in range(int(num/batch_size)):
            a = i*batch_size
            b = (i+1)*batch_size
            x_train, x_labels = process_batch(new_line[a:b],img_path,train=True)
            x = preprocess(x_train)
            y = np_utils.to_categorical(np.array(x_labels), num_classes)
            x = np.transpose(x, (0,2,3,1,4))
            yield x, y

def generator_val_batch(val_txt,batch_size,num_classes,img_path):
    f = open(val_txt, 'r')
    lines = f.readlines()
    num = len(lines)
    while True:
        new_line = []
        index = [n for n in range(num)]
        random.shuffle(index)
        for m in range(num):
            new_line.append(lines[index[m]])
        for i in range(int(num / batch_size)):
            a = i * batch_size
            b = (i + 1) * batch_size
            y_test,y_labels = process_batch(new_line[a:b],img_path,train=False)
            x = preprocess(y_test)
            x = np.transpose(x,(0,2,3,1,4))
            y = np_utils.to_categorical(np.array(y_labels), num_classes)
            yield x, y

def main():
    # data_path = '../ALL_E2ON_Data_12.13/'   경로 변경 전
    data_path = '/workspace/NIA/NIA_AI_DATASET_2021-ST-GCAE/E2ON/source/AI_HUB_1.03/Action/'  # 경로 변경 후
    data_path2= '/workspace/NIA/NIA_AI_DATASET_2021-ST-GCAE/E2ON/source/AI_HUB_1.03/'    #경로 변경 후
    #학습데이터가 저장되어있는 경로로 설정
    field='main' 
    #학습할 분야가 메인(main)/서브(sub) 중에서 선택

    if not os.path.exists(data_path2+'/result'): 
        os.mkdir(data_path2+'/result')

    train_list=data_path2+'input/'+field+'/train_tmp_list.txt'            
    test_list=data_path2+'input/'+field+'/val_list.txt'

    ftrain = open(train_list, 'r')
    lines = ftrain.readlines()
    ftrain.close()
    train_samples = len(lines)

    ftest = open(test_list, 'r')
    lines = ftest.readlines()
    ftest.close()
    val_samples = len(lines)

    

    num_classes = 20    #101 ==> 20으로 변경
    batch_size = 128     
    epochs = 10
    lr = 0.005
    sgd = SGD(lr=lr, momentum=0.9, nesterov=True)
    #adam = Adam(learning_rate=lr)
    strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0","/gpu:1","/gpu:2","/gpu:3","/gpu:4","/gpu:5","/gpu:6","/gpu:7"])
    #filename = './checkpoint/checkpoint-epoch-{}-batch-{}-traial-001-with-validation.h5'.format(epochs, batch_size)
    checkpoint_path = "training_10/1.25_cp-{epoch:04d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    checkpoint = ModelCheckpoint(checkpoint_path, monitor='loss', verbose=1, save_best_only=False, mode="auto", period=1)
    with strategy.scope():
        model = c3d_model()
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.save_weights(checkpoint_path.format(epoch=0))
    
    model.summary()
    
    history = model.fit_generator(generator_train_batch(train_list, batch_size, num_classes,data_path),
                                steps_per_epoch=train_samples // batch_size, epochs=epochs, callbacks=[onetenth_4_8_12(lr),checkpoint])
    data_path2 = '/workspace/NIA/NIA_AI_DATASET_2021-ST-GCAE/E2ON/source/AI_HUB_1.03/'  #경로 변경
    if not os.path.exists(data_path2+field+'_result'+'/epoch'+str(epochs)+'/'): #경로 변경
        os.mkdir(data_path2+field+'_result'+'/epoch'+str(epochs)+'/')   #경로 변경 
    model.save(data_path2+field+'_result'+'/epoch'+str(epochs)+'_1.25_temp_weights_c3d.h5')  # 경로 변경
    
    plot_history(history, data_path2+field+'_result'+'/1.25_epoch'+str(epochs)+'/')
    #save_history(history, data_path+field+'_result'+'/epoch'+str(epochs)+'/')
    #model.save_weights(data_path+field+'_result'+'/epoch'+str(epochs)+'/weights_c3d.h5')

if __name__ == '__main__':
    main()