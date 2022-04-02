import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score , top_k_accuracy_score
import datetime
import joblib


def calculate_F1(matrix, f):  #precision , recall ,F1 score을 계산하는 함수 
    precision = []
    recall= []
    for i in range(len(matrix)):
        answer = matrix[i][i]
        col_sum=matrix.sum(axis= 0)[i]  #precision
        row_sum=matrix.sum(axis=1)[i]   #recall
        precision.append(answer/col_sum)
        recall.append(answer/row_sum)
    precision_value = sum(precision) / len(precision)
    print('precision_value: ', precision_value)
    recall_value = sum(recall) / len(recall)
    print('recall_value: ', recall_value)
    F1 = (2*precision_value*recall_value) / (precision_value + recall_value)
    f.write('Precision: '+str(precision_value)+'\n')
    f.write('Recall: '+str(recall_value)+'\n')
    f.write('F1-Score: '+str(F1)+'\n\n')
    #print(F1)
    return F1  
def num_to_category(input): #카테고리 인덱스 ==> 문자열 
    if input ==0:
        return '아동방임-방임(C011)'
    elif input ==1:
        return '아동학대-신체학대(C012)'
    elif input ==2:
        return '주거침임-문(C021)'
    elif input ==3:
        return '폭행/강도-흉기(C031)'
    elif input ==4:
        return '폭행강도-위협행동(C032)'
    elif input ==5:
        return '절도-문앞(C041)'
    elif input ==6:
        return '절도-주차장(C042)'
        
def M3(input1, input2 , label , names , now):
    top_5_dict = {'아동방임-방임(C011)':0,'아동학대-신체학대(C012)':0,'주거침임-문(C021)':0,'폭행/강도-흉기(C031)':0, '폭행강도-위협행동(C032)':0,'절도-문앞(C041)':0,'절도-주차장(C042)':0}
    time = str(now)
    #print(now)
    CATEGORY = names
    CATEGORY = CATEGORY[0:4]

    file_name = names
    print('json 이상행동 분류 알고리즘 종료\n')

    f =open('./test_log/'+CATEGORY+'/'+file_name+'_M3_'+time[0:3]+'_.txt', 'w')  #최종 로그 파일 생성 
    
    f.write('Start time : '+str(now)+'\n\n') #시작시간
    X_test = [[0 for col in range(2)] for row in range(len(input1))] #입력으로 들어온 input1값을 SVM에서 사용할 수 있도록 shape 을 맞춰줌
    
    for i in range(len(input1)):
        X_test[i][0] = input1[i]
        X_test[i][1] = input2[i]
    X_test = np.array(X_test)    
    y_test = np.array(label)
    sc =  joblib.load('./input_data/Standard_Scaler.pkl') #이미 정규화된 scaler를 가지고 옴
    clf = joblib.load('./input_data/SVM_Model.pkl') #train data로 이미 학습된 SVM을 가지고 옴
    X_test_std = sc.transform(X_test) #X_test 를 정규화
    y_pred = clf.predict(X_test_std) #예측
    probs = clf.predict_proba(X_test_std)
    n=0
    for key in top_5_dict:
        top_5_dict[key] = probs[0][n]
        n=n+1
  
    top_5_dict_sorted = sorted(top_5_dict.items(),reverse=True, key =lambda item:item[1])
    
    count =0
    # print(top_k_accuracy_score(y_test, probs, k=5))
    print('이상행동 분류 완료')
    for i in range(len(input1)): #입력데이터가 리스트이기 때문에  for를 통해서 반복해서 예측
        print('\n실제 관측된 행동: ',num_to_category(y_test[i]))
        answer = num_to_category(y_test[i])
        print('모델이 예측한 행동: ',num_to_category(y_pred[i]))
        pred = num_to_category(y_pred[i])
        if y_test[i] == y_pred[i]:
            f.write('Target file: \n'+names+',\n모델 예측 : '+pred+'  \n실제 정답: '+answer+'\n\n')
            count +=1
        else:
            f.write('Target file: \n'+names+',\n모델 예측 : '+pred+'  \n실제 정답: '+answer+'\n\n')
            print(names[i]+'는 틀렸습니다.')    
    end = datetime.datetime.now()
    print('\n이상행동 분류 모델 예측 확률')
    num =0
    for key, value in top_5_dict_sorted:
        print(key , ': ',int(value*100))
        num = num+1
        if num==5:
            break
    f.write('Finish time: '+str(end))        
            
        
    
#M3([4],[5],[5],'C041_A30_SY32_P07_S08_01NBS.mp4','now')