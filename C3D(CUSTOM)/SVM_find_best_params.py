import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
import datetime
import joblib

def calculate_F1(matrix, f):
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
    print('recall_value : ', recall_value)
    F1 = (2*precision_value*recall_value) / (precision_value + recall_value)
    f.write('Precision :'+str(precision_value)+'\n')
    f.write('Recall :'+str(recall_value)+'\n')
    f.write('F1-Score :'+str(F1)+'\n\n')
    #print(F1)
    return F1    


def SVM(input1, input2 , label ):
    #################해당 파일에서는 C3D모델과 CNN_RNN모델의 결곽값을 통해서 최고의 정확도를 보이는 파라미터를 찾습니다.
    #################해당 모델의 가중치 값과 표준화값은 pkl파일로 저장되어 추후 SVM모델에서 불러와집니다.
    f =open('SVM_LOG.txt', 'w')
    now = datetime.datetime.now()
    print(str(now))
    f.write('실행 시간 : '+str(now))
    f.write('\n전체 데이터로 best parameter 탐색\n')
    train_df = pd.read_excel('./input_data/train_output.xlsx', engine='openpyxl')
    train_df_range = len(train_df)
    train_data = [[0 for col in range(2)] for row in range(train_df_range)]

    train_coulmn_1 =train_df['C3D_Prediceted_Category_Label'].values.tolist()  #C3D 예측 라벨
    train_coulmn_2= train_df['CNN_RNN_Predicted_Category_label'].values.tolist() #CNN_RNN 예측 라벨
    train_label=train_df['label'].values.tolist()  #실제 정답
    
    for i in range(train_df_range):
        train_data[i][0] = train_coulmn_1[i]
        train_data[i][1] = train_coulmn_2[i]
    train_data = np.array(train_data)
    train_label = np.array(train_label)  #타입 변환

    val_df = pd.read_excel('./input_data/val_output.xlsx', engine='openpyxl')
    val_df_range = len(val_df)
    val_data = [[0 for col in range(2)] for row in range(val_df_range)]

    val_coulmn_1 =val_df['C3D_Prediceted_Category_Label'].values.tolist()  #C3D 예측 라벨
    val_coulmn_2= val_df['CNN_RNN_Predicted_Category_label'].values.tolist() #CNN_RNN 예측 라벨
    val_label=val_df['label'].values.tolist()  #실제 정답

    for i in range(val_df_range):
        val_data[i][0] = val_coulmn_1[i]
        val_data[i][1] = val_coulmn_2[i]
    val_data = np.array(val_data)
    val_label = np.array(val_label)  #타입 변환

    test_df = pd.read_excel('./input_data/test_output.xlsx', engine='openpyxl')
    test_df_range = len(test_df)
    test_data = [[0 for col in range(2)] for row in range(test_df_range)]

    test_coulmn_1 =test_df['C3D_Prediceted_Category_Label'].values.tolist()  #C3D 예측 라벨
    test_coulmn_2= test_df['CNN_RNN_Predicted_Category_label'].values.tolist() #CNN_RNN 예측 라벨
    test_label=test_df['label'].values.tolist()  #실제 정답
    names = test_df['videoname'].values.tolist()
    
    for i in range(test_df_range):
        test_data[i][0] = test_coulmn_1[i]
        test_data[i][1] = test_coulmn_2[i]
    test_data = np.array(test_data)
    test_label = np.array(test_label)  #타입 변환
    X_train = train_data
    y_train = train_label
    
    X_val = val_data
    y_val = val_label
    
    X_test =test_data
    y_test =test_label
    
    print(X_train.shape)
    """
    empty =[]
    input_list = [input1 , input2]
    empty.append(input_list)
    """
    sc = StandardScaler()
    sc.fit(X_train)
    
    X_train_std = sc.transform(X_train)
    X_val_std =sc.transform(X_val)
    X_test_std = sc.transform(X_test)
    #input_std = sc.transform(empty)
    joblib.dump(sc,'./input_data/Standard_Scaler.pkl')
    print("표준화된 특성1 범위: ", "[", min(X_train_std[:, 0]), ",", max(X_train_std[:, 0]), "]")
    print("표준화된 특성2 범위: ", "[", min(X_train_std[:, 1]), ",", max(X_train_std[:, 1]), "]")
    
    from sklearn.svm import SVC

    best_score = 0
    best_parameter={}

    gammas = [0.001,0.01,0.1,1,10,100]
    Cs = [0.1,1,10,100,1000,10000]
    f.write('**This part is train part**\n')
    for g in gammas:
        for c in Cs:
            svc = SVC(gamma=g, C=c ,probability=True)
            svc = svc.fit(X_train_std, y_train)
            joblib.dump(svc, 'SVM_MODEL.pkl')
            score = svc.score(X_train_std, y_train)
            train_pred = svc.predict(X_train_std)
            preds = svc.predict_proba(X_train_std)
            print(preds)
            print('gamma : ',g,'   C:',c,'  일때')
            F1 = calculate_F1(confusion_matrix(y_train , train_pred),f)
            print('F1 score : ',F1,'\n\n')
            print(confusion_matrix(y_train, train_pred))
            #print(confusion_matrix(y_val, val_pred))
            #print(score)
            #joblib.dump(svc , './input_data/SVM_Model_C-'+str(c)+'_gamma-'+str(g)+'.pkl')   #가중치 저장
            f.write('SVM parameter      gamma : '+str(g))
            f.write('    C :'+str(c)+'   ')
            f.write('    Accuracy : '+ str(score)+'\n')
            if score > best_score:
                best_score = score
                best_parameter = {'C':c, 'gamma':g}
                print(best_parameter)
        f.write('\n\n')        
    y_pred=[]
    f.write('\n\nIn train, Best Parameter is C :'+str(best_parameter['C'])+'   and gamma : '+str(best_parameter['gamma'])+'\n\n\n')
    f.write('**This part is validation part**\n')
    
    for g in gammas:
        for c in Cs:
            svc = SVC(gamma=g, C=c).fit(X_train_std, y_train)
            score = svc.score(X_val_std, y_val)
            val_pred = svc.predict(X_val_std)
            print('gamma : ',g,'   C:',c,'  일때')
            F1 = calculate_F1(confusion_matrix(y_val , val_pred),f)
            print('F1 score : ',F1,'\n\n')

            print(confusion_matrix(y_val, val_pred))
            #print(score)
            joblib.dump(svc , './input_data/SVM_Model_C-'+str(c)+'_gamma-'+str(g)+'.pkl')   #가중치 저장
            f.write('SVM parameter      gamma : '+str(g))
            f.write('    C :'+str(c)+'   ')
            f.write('\nAccuracy : '+ str(score)+'\n')
            if score > best_score:
                best_score = score
                best_parameter = {'C':c, 'gamma':g}
                print(best_parameter)
        f.write('\n\n')        
                
    print(best_parameter['C'])
    print(best_parameter['gamma'])  
    f.write('\n\nIn validation Best Parameter is C :'+str(best_parameter['C'])+'   and gamma : '+str(best_parameter['gamma'])+'\n')          
    clf = SVC(kernel='rbf', gamma=best_parameter['gamma'], C = best_parameter['C']).fit(X_train_std, y_train)
    #clf
    y_pred = clf.predict(X_test_std)
    print("예측된 라벨:", y_pred)
    score = clf.score(X_test_std, y_test)
    #print("ground-truth 라벨:", y_test)
    #print("prediction accuracy: {:.2f}".format(score)) # 예측 정확도
    #print(confusion_matrix(y_test,y_pred))
    f.write('예측 정확도 : '+str(score)+'\n')
    count =0
    for i in range(len(names)):
        if str(y_pred[i]) == str(y_test[i]):
            f.write(names[i]+'는 맞았습니다. \n')
            count +=1
        else:
            f.write(names[i]+'는 틀렸습니다. \n')    
    f.write(str(len(test_df))+'개 중 '+str(count)+'개 맞았습니다.\n')       
    end = datetime.datetime.now()
    f.write('실행 종료 시간 : '+str(end))
    joblib.dump(clf , './input_data/SVM_Model.pkl')   #가중치 저장
    """            

    print("예측된 라벨:", y_pred)
    #print("ground-truth 라벨:", y_test)
    print(label)
    print("prediction accuracy: {:.2f}".format(np.mean(y_pred == label))) # 예측 정확도
    #cm =confusion_matrix(y_test , y_pred )
    return y_pred
    """
SVM('1','2','3')    