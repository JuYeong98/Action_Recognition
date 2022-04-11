
import json
import tensorflow as tf
from tensorflow import keras
from imutils import paths
from PIL import Image as im
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import imageio
import cv2
import os
import glob
import time
import random
import pickle
train_df = pd.DataFrame()     #train 파일 내의 모든 json 경로와 라벨값 
test_df = pd.DataFrame()      #test 파일 내의 모든 json 경로와 라벨값    
val_df = pd.DataFrame()       #val 파일 내의 모든 json 경로와 라벨값   

MAX_IMAGE_WIDTH = 320
MAX_IMAGE_HEIGHT = 180
MAX_IMAGE_SIZE = MAX_IMAGE_WIDTH * MAX_IMAGE_HEIGHT
RESCALE=6

MAX_IMAGE_WIDTH = 320
MAX_IMAGE_HEIGHT = 180
RESCALE = 6
BATCH_SIZE = 64
EPOCHS = 10

MAX_SEQ_LENGTH = 20
NUM_FEATURES = 2048
json_path ='/workspace/NIA/NIA_AI_DATASET_2021-ST-GCAE/E2ON/source/ALL_DATA/json/'  #기본 json 폴더 위치
base_path='/workspace/NIA/NIA_AI_DATASET_2021-ST-GCAE/E2ON/source/AI_HUB_1.03/input/main/'   #text 파일 위치

def get_file_list(check):        #파일 리스트와 라벨 리스트를 동시에 얻는 함수 
    file_list = []
    label_index = []
    if check == 'valfile.txt':
        path = base_path + check
    elif check =='testfile.txt':
        path = base_path + check
    elif check =='trainfile.txt':
        path = base_path + check

    ff = open(path ,'r')
    lines = ff.readlines()
    num = len(lines)    
    for line in lines:
        label = line.split(' ')[1]
        line = line.split(' ')[0]
        file_list.append(json_path+line+'.json')
        label_index.append(label)
    return file_list , label_index

a ,b = get_file_list('trainfile.txt')
train_df['file_name'] =a
train_df['label'] =b
c,d = get_file_list('testfile.txt')
test_df['file_name'] =c
test_df['label'] =d
e,f = get_file_list('valfile.txt')
val_df['file_name'] =e
val_df['label'] =f

test_end = len(test_df)
train_end=len(train_df)


def make_bitmap(a,b):
    #A = random.randint(0,10000)
    targetList = [0] * MAX_IMAGE_SIZE
    for i in range(len(a)):
        setBoldPoint(targetList, a[i], b[i])
    #targetArray = np.array(targetList,dtype = np.uint8)
    targetArray = np.fromiter(targetList, dtype=np.uint8)
    targetArray= np.reshape(targetArray, (MAX_IMAGE_HEIGHT,MAX_IMAGE_WIDTH))
    data = im.fromarray(targetArray)
    data = data.convert('RGB')
    #a = np.array(data)
    #print(a.shape)
    #data.save('./img/'+str(A)+'.png')
    # saving the final output as a PNG file
    data.save('gfg_dummy_pic.png')

    return data          

def setBoldPoint(targetList, x,y):
    if x<0 or y <0 or x >= MAX_IMAGE_WIDTH or y >= MAX_IMAGE_HEIGHT:
        return 
    if y-1 >= 0 and x-1 >= 0 : #x와 y가 1보다 크면
        targetList[MAX_IMAGE_WIDTH * (y-1) + (x-1)] = 255
    if y-1 >= 0 :  #y좌표가 1보다 크면
        targetList[MAX_IMAGE_WIDTH * (y-1) + (x)] = 255
    if y-1 >= 0 and x+1 < MAX_IMAGE_WIDTH:    
        targetList[MAX_IMAGE_WIDTH * (y-1) + (x+1)] = 255
    if x-1 >= 0 :
        targetList[MAX_IMAGE_WIDTH * (y) + (x-1)] = 255
    targetList[MAX_IMAGE_WIDTH * (y) + (x)] = 255
    if x+1 < MAX_IMAGE_WIDTH:
        targetList[MAX_IMAGE_WIDTH * (y) + (x+1)] = 255
    if y+1 < MAX_IMAGE_HEIGHT and x-1 >= 0 :
        targetList[MAX_IMAGE_WIDTH * (y+1) + (x-1)] = 255
    if y+1 < MAX_IMAGE_HEIGHT:
        targetList[MAX_IMAGE_WIDTH * (y+1) + (x)] = 255    
    if y+1 < MAX_IMAGE_HEIGHT and  x+1 < MAX_IMAGE_WIDTH:
        targetList[MAX_IMAGE_WIDTH * (y+1) + (x+1)] = 255
    return targetList

def extract_skeleton(start,end,dataframe, choice):  #json에서 skeleton point를 뽑는다. 
    s_time = time.time()
    train_file = '/workspace/NIA/NIA_AI_DATASET_2021-ST-GCAE/E2ON/source/ALL_DATA/json/Action/test/A01/C011_A01_SY02_P07_S03_02DBS.json'
    #train_json = dataframe['file_name'].to_list()
    train_json = []
    train_json.append(train_file)
    empty_list_x=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    empty_list_y=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    image_frame_list= []
    total_file_object_list=[]
    start = 0
    end =1
    for i in range(start, end): #전체에서 
        print(train_json[i]+' json 파일에서 이미지 전처리중')
        with open(train_json[i],'r') as f: #동영상 하나
            json_file = json.load(f)
        image_frame_list=[]    
        for frame in json_file['file'][0]['frames']:  #동영상 하나
            point1 = []
            point2 = []
            point3 = []
            point4 = []

            person_length = len(frame['persons'])
            if person_length ==0:
                data = make_bitmap(empty_list_x+empty_list_x,empty_list_y+empty_list_y)
            elif person_length==1:
                a=frame['persons'][0]['keypoints']
                point1 = [int(float(point.split(',')[0])//RESCALE) for point in a]
                point2 = [int(float(point.split(',')[1])//RESCALE) for point in a]
                data = make_bitmap(point1+empty_list_x , point2+empty_list_y)
            else:
                a=frame['persons'][0]['keypoints']
                b=frame['persons'][1]['keypoints'] 

                point1 = [int(float(point.split(',')[0])//RESCALE) for point in a]
                point2 = [int(float(point.split(',')[1])//RESCALE) for point in a]
                point3 = [int(float(point.split(',')[0])//RESCALE) for point in b]
                point4 = [int(float(point.split(',')[1])//RESCALE) for point in b]

                data = make_bitmap(point1+point3,point2+point4)     
            image_frame_list.append(data)
        print(train_json[i])
        name = train_json[i].split('/')[-1]
        name=name.split('.')[0]       
        #with open('./pickle_'+choice+'/'+name+'.pkl', 'wb') as f : 
             #pickle.dump(image_frame_list , f)
        #total_file_object_list.append(image_frame_list)
        #print(total_file_object_list)    
        e_time = time.time()
        print(e_time - s_time)
    #return total_file_object_list

#total_file_object_list = extract_skeleton(0,train_end,train_df,'train')
#extract_skeleton(0,train_end,train_df,'train')   
#total_file_test_object_list = extract_skeleton(0,test_end,test_df, 'test')
extract_skeleton(0,test_end,train_df, 'test')
#print(len(total_file_object_list))
#rint(len(total_file_object_list[0]))   
#print(train_df['file_name'].tolist())

"""

def build_feature_extractor():
    feature_extractor = keras.applications.ResNet50(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(MAX_IMAGE_HEIGHT, MAX_IMAGE_WIDTH, 3),
    )
    preprocess_input = keras.applications.resnet.preprocess_input

    inputs = keras.Input((MAX_IMAGE_HEIGHT, MAX_IMAGE_WIDTH, 3))
    preprocessed = preprocess_input(inputs)

    outputs = feature_extractor(preprocessed)
    return keras.Model(inputs, outputs, name="feature_extractor")


feature_extractor = build_feature_extractor()

label_processor = tf.keras.layers.experimental.preprocessing.StringLookup(
    num_oov_indices=0, vocabulary=np.unique(train_df["label"])
)


def prepare_all_videos(df, check):
    s_time = time.time()
    num_samples = len(df)
    json_paths = df["file_name"].values.tolist()
    #json_paths = json_paths[0:5]
    labels = df["label"].values
    labels = label_processor(labels[..., None]).numpy()

    # `frame_masks` and `frame_features` are what we will feed to our sequence model.
    # `frame_masks` will contain a bunch of booleans denoting if a timestep is
    # masked with padding or not.
    frame_masks = np.zeros(shape=(num_samples, MAX_SEQ_LENGTH), dtype="bool")
    frame_features = np.zeros(
        shape=(num_samples, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32"
    )
    #['a', 'b', 'c']
    # For each video.
    if check =='train':
        pkl_list = glob.glob('/workspace/NIA/NIA_AI_DATASET_2021-ST-GCAE/E2ON/source/CNN_RNN_model/pickle_train/*.pkl')
    elif check =='test':
        pkl_list = glob.glob('/workspace/NIA/NIA_AI_DATASET_2021-ST-GCAE/E2ON/source/CNN_RNN_model/pickle_test/*.pkl')    
       
    for idx, path in enumerate(pkl_list): #7193개의 아이템이 있는 리스트    path 한  동영상내의 이미지 모음(3600개)  
        e_time = time.time()
        print(e_time-s_time)
        # Gather all
        # its frames and add a batch dimension.
        #print(path)
        pkl_file_name = path.split('/')[-1]
        with open(path,'rb') as f:
            data = pickle.load(f)
        
        bitmap_object_np = np.array([np.array(file) for file in data])    
        print(bitmap_object_np.shape)
        frames = bitmap_object_np[None, ...] 

        # Initialize placeholders to store the masks and features of the current video.
        temp_frame_mask = np.zeros(shape=(1, MAX_SEQ_LENGTH,), dtype="bool")
        temp_frame_features = np.zeros(
            shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32"
        )

        # Extract features from the frames of the current video.
        for i, batch in enumerate(frames):
            video_length = batch.shape[0]
            length = min(MAX_SEQ_LENGTH, video_length)
            for j in range(length):
                temp_frame_features[i, j, :] = feature_extractor.predict(
                    batch[None, j, :]
                )
            temp_frame_mask[i, :length] = 1  # 1 = not masked, 0 = masked

        frame_features[idx,] = temp_frame_features.squeeze()
        frame_masks[idx,] = temp_frame_mask.squeeze()
    #print(idx) 
    return frame_features, frame_masks, labels

train_frame_features,train_frame_masks, train_labels = prepare_all_videos(train_df, 'train')
with open('./pkl_train/'+'train_frame_features.pkl', 'wb') as f : 
    pickle.dump(train_frame_features ,f)
with open('./pkl_train/'+'train_frame_masks.pkl', 'wb') as f : 
    pickle.dump(train_frame_masks ,f)
with open('./pkl_train/'+'train_labels.pkl', 'wb') as f : 
    pickle.dump(train_labels ,f)
    
test_frame_features,test_frame_masks, test_labels = prepare_all_videos(test_df, 'test')
with open('./pkl_test/'+'test_frame_features.pkl', 'wb') as f : 
    pickle.dump(test_frame_features ,f)
with open('./pkl_test/'+'test_frame_masks.pkl', 'wb') as f : 
    pickle.dump(test_frame_masks ,f)
with open('./pkl_test/'+'test_labels.pkl', 'wb') as f : 
    pickle.dump(test_labels ,f)
        

with open('/workspace/NIA/NIA_AI_DATASET_2021-ST-GCAE/E2ON/source/CNN_RNN_model/pkl_train/train_frame_features.pkl','rb') as f:
    tr_frame_features = pickle.load(f)
with open('/workspace/NIA/NIA_AI_DATASET_2021-ST-GCAE/E2ON/source/CNN_RNN_model/pkl_train/train_frame_masks.pkl','rb') as f:
    tr_frame_masks = pickle.load(f)
with open('/workspace/NIA/NIA_AI_DATASET_2021-ST-GCAE/E2ON/source/CNN_RNN_model/pkl_train/train_labels.pkl','rb') as f:
    tr_frame_label = pickle.load(f)
train_data = (tr_frame_features,tr_frame_masks)
train_labels = tr_frame_label

with open('/workspace/NIA/NIA_AI_DATASET_2021-ST-GCAE/E2ON/source/CNN_RNN_model/pkl_test/test_frame_features.pkl','rb') as f:
    te_frame_features = pickle.load(f)
print(type(te_frame_features))    
with open('/workspace/NIA/NIA_AI_DATASET_2021-ST-GCAE/E2ON/source/CNN_RNN_model/pkl_test/test_frame_masks.pkl','rb') as f:
    te_frame_masks = pickle.load(f)
print(type(te_frame_masks))    
with open('/workspace/NIA/NIA_AI_DATASET_2021-ST-GCAE/E2ON/source/CNN_RNN_model/pkl_test/test_labels.pkl','rb') as f:
    te_label = pickle.load(f)
print(type(te_label))
test_data=(te_frame_features,te_frame_masks)
test_labels =te_label



# Utility for our sequence model.
def get_sequence_model():
    
    class_vocab = label_processor.get_vocabulary()

    frame_features_input = keras.Input((MAX_SEQ_LENGTH, NUM_FEATURES))
    mask_input = keras.Input((MAX_SEQ_LENGTH,), dtype="bool")

    # Refer to the following tutorial to understand the significance of using `mask`:
    # https://keras.io/api/layers/recurrent_layers/gru/
    x = keras.layers.GRU(16, return_sequences=True)(
        frame_features_input, mask=mask_input
    )
    x = keras.layers.GRU(8)(x)
    x = keras.layers.Dropout(0.4)(x)
    x = keras.layers.Dense(8, activation="relu")(x)
    output = keras.layers.Dense(len(class_vocab), activation="softmax")(x)

    rnn_model = keras.Model([frame_features_input, mask_input], output)

    rnn_model.compile(
        loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    return rnn_model

# Utility for running experiments.
def run_experiment():
    filepath = "/workspace/NIA/NIA_AI_DATASET_2021-ST-GCAE/E2ON/source/CNN_RNN_model/tmp/video_classifier"
    file_name = '/1_08cp_10.ckpt'
    checkpoint = keras.callbacks.ModelCheckpoint(filepath+file_name, save_weights_only=True, monitor = 'loss' , save_best_only=True, verbose=1)

    seq_model = get_sequence_model()
    history = seq_model.fit(
        [train_data[0], train_data[1]],
        train_labels,
        #validation_split=0.3,
        epochs=EPOCHS,
        callbacks=[checkpoint],
    )

    seq_model.load_weights(filepath+file_name)
    _, accuracy = seq_model.evaluate([test_data[0], test_data[1]], test_labels)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")

    return history, seq_model
_, sequence_model = run_experiment()

def prepare_single_video(frames):
    #print(frames)
    frames = frames[None, ...]
    frame_mask = np.zeros(shape=(1, MAX_SEQ_LENGTH,), dtype="bool")
    frame_features = np.zeros(shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32")

    for i, batch in enumerate(frames):
        video_length = batch.shape[0]
        length = min(MAX_SEQ_LENGTH, video_length)
        for j in range(length):
            frame_features[i, j, :] = feature_extractor.predict(batch[None, j, :])
        frame_mask[i, :length] = 1  # 1 = not masked, 0 = masked

    return frame_features, frame_mask


txt =[]
def sequence_prediction(path):
    
    class_vocab = label_processor.get_vocabulary()
    #print(class_vocab)
    frames = path
    #print(frames)
    frame_features, frame_mask = prepare_single_video(frames)
    probabilities = sequence_model.predict([frame_features, frame_mask])[0]

    for i in np.argsort(probabilities)[::-1]:
        print(f"  {class_vocab[i]}: {probabilities[i] * 100:5.2f}%")
        print(f"  {class_vocab[1]}: {probabilities[1] * 100:5.2f}%")
        print(f"  {class_vocab[2]}: {probabilities[2] * 100:5.2f}%")
        print(f"  {class_vocab[3]}: {probabilities[3] * 100:5.2f}%")
        print(f"  {class_vocab[4]}: {probabilities[4] * 100:5.2f}%")
        txt.append(class_vocab[i]+'\n')
        break

    return frames





#print(test_videos)
test_list= glob.glob('/workspace/NIA/NIA_AI_DATASET_2021-ST-GCAE/E2ON/source/CNN_RNN_model/pickle_test/*.pkl')
for i in range(1):
    print(i)
    try:
        with open(test_list[i],'rb') as f:
            test_videos = pickle.load(f)

    except:
        pass
    else:
        test_video = np.array([np.array(a) for a in test_videos])
        test_frames = sequence_prediction(test_video)
with open('result.txt', 'w') as file:    # hello.txt 파일을 쓰기 모드(w)로 열기
    file.writelines(txt)








"""


"""
def to_gif(images):
    converted_images = images.astype(np.uint8)
    imageio.mimsave("animation.gif", converted_images, fps=10)
    return embed.embed_file("animation.gif")



print(f"Frame features in train set: {train_data[0].shape}")
print(f"Frame masks in train set: {train_data[1].shape}")

_, sequence_model = run_experiment()   #학습


test_video = np.random.choice(test_df["video_name"].values.tolist())  
print(f"Test video path: {test_video}")
test_frames = sequence_prediction(test_video)  #테스트
#to_gif(test_frames[:MAX_SEQ_LENGTH])

"""