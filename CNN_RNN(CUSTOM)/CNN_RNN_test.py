    
import json
import tensorflow as tf
from tensorflow import keras
from imutils import paths
from PIL import Image as im
import matplotlib.pyplot as plt
#from pkl_video_classification_test_end import
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
#os.environ["CUDA_VISIBLE_DEVICES"]='2'

json_path ='./input_data/'  #기본 json 폴더 위치
base_path='./input_data/'   #text 파일 위치

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0"
def get_file_list(check):        #파일 리스트와 라벨 리스트를 동시에 얻는 함수 
    ff = open('./input_data/trainfile.txt' ,'r')
    lines = ff.readlines()
    num = len(lines)    
    label_index = []
    for line in lines:
        label = line.split(' ')[1]
        line = line.split(' ')[0]
        label_index.append(label)
    return  label_index

a = get_file_list('trainfile.txt')
train_df = pd.DataFrame()
train_df['label'] =get_file_list('trainfile.txt')



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


def extract_skeleton(path):  #json에서 skeleton point를 뽑는다. 
    empty_list_x=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    empty_list_y=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    image_frame_list= []
    with open(path,'r') as f: #동영상 하나
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
        
    return image_frame_list




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
    output = keras.layers.Dense(21, activation="softmax")(x)
    
    rnn_model = keras.Model([frame_features_input, mask_input], output)

    rnn_model.compile(
        loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    return rnn_model

# Utility for running experiments.

def prepare_single_video(frames , feature_extractor):
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

def sequence_prediction(path , sequence_model):
    class_vocab = label_processor.get_vocabulary()

    frames = path

    frame_features, frame_mask = prepare_single_video(frames , feature_extractor)
    probabilities = sequence_model.predict([frame_features, frame_mask])[0]
    print(probabilities)
    for i in np.argsort(probabilities)[::-1]:
        #print(i)
        #print(f"  {class_vocab[i]}: {probabilities[i] * 100:5.2f}%")
        
        return int(class_vocab[i])-1
    
def CNN_RNN_test(path ) :

    test_file = extract_skeleton(path)   
    with open('./input_data/train_frame_features.pkl','rb') as f:
        tr_frame_features = pickle.load(f)
    print(tr_frame_features.shape)    
    with open('./input_data/train_frame_masks.pkl','rb') as f:
        tr_frame_masks = pickle.load(f)
    print(tr_frame_masks.shape)    
    with open('./input_data/train_labels.pkl','rb') as f:
        tr_frame_label = pickle.load(f)
    print(tr_frame_label.shape)    
    train_data = (tr_frame_features,tr_frame_masks)
    train_labels = tr_frame_label                                                                                                                                               

    #b=  ['', '1\n', '10\n', '11\n', '12\n', '13\n', '14\n', '15\n', '16\n', '17\n', '18\n', '19\n', '2\n', '3\n', '4\n', '5\n', '6\n', '7\n', '8\n', '9\n']
    
    label_processor = tf.keras.layers.experimental.preprocessing.StringLookup(
        num_oov_indices=0, vocabulary=np.unique(train_df['label'])
    )

    #print(len(label_processor.get_vocabulary()))
    #print(label_processor.get_vocabulary())
    feature_extractor = build_feature_extractor()
    #sequence_model = run_experiment(train_data , train_labels , test_data , test_labels, label_processor)
    sequence_model = get_sequence_model()
    sequence_model.load_weights('./input_data/seq_model.h5')
    file= np.array([np.array(a) for a in test_file])
    test_frames = sequence_prediction(file ,sequence_model )  #테스트
    print(test_frames)
    return test_frames
    
#CNN_RNN_test('/workspace/NIA/NIA_AI_DATASET_2021-ST-GCAE/E2ON/source/intergrate_model/testfile2/C042_A31_SY32_P09_S07_02NAS.json')
    