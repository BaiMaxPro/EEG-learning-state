import os
from sklearn import svm
from sklearn.model_selection import KFold,train_test_split
import numpy as np
import pandas as pd
from sklearn import metrics
import pickle

import Load

from NoiseEliminate import NoisE
from EEG_feature_extraction import generate_feature_vectors_from_samples,matrix_from_csv_file,generate_feature_vectors_from_data

def svc_classification(EEGfile,save=False):
    dataMat = Load.csv_to_pandas(EEGfile)
    iter = 1
    # x = dataMat.loc[1:,['theta_0','theta_1','theta_2','theta_3','alpha_0','alpha_1','alpha_2','alpha_3','beta_0','beta_1','beta_2','beta_3']]
    x = dataMat.iloc[1:,:-1]
    # x = dataMat.loc[1:,['Alpha','Beta']]
    # x = dataMat.iloc[1:,:5]
    x=x.values.tolist()
    y = dataMat.iloc[1:,-1]
    y=y.values.tolist()
    x,y = np.array(x),np.array(y)
    print(x.shape,y)
    print(dataMat)
    dataList = dataMat.values.tolist()
    dataList = dataList[1:]
    avg_train = 0
    avg_test = 0
    clf = svm.SVC(C=113, kernel="rbf",max_iter=-1)
    kf = KFold(n_splits=10)
    for train, test in kf.split(dataList):
        datalen = len(x)
        x_train = x[train]
        y_train = y[train]
        clf.fit(x_train, y_train)
        pre_train = clf.predict(x[train, :])    #得到模型关于训练数据集的分类结果
        pre_test = clf.predict(x[test, :])      #得到模型关于测试数据集的分类结果
        if save:
            s=pickle.dumps(clf)
            f=open(f'model/svm-T-{iter}.model', "wb+")
            f.write(s)
            f.close()

        print("The "+str(iter)+"th cross validation:")
        train_acc = metrics.accuracy_score(y_train, pre_train)
        test_acc =  metrics.accuracy_score(y[test], pre_test)
        print("Train Accuracy:%.4f" % train_acc + \
              "\tTest  Accuracy:%.4f\n" % test_acc)
        iter = iter + 1
        avg_train += train_acc
        avg_test += test_acc
    avg_train = avg_train/int(iter-1)
    avg_test = avg_test/int(iter-1)
    print("Avg Train Accuracy:%.4f" % avg_train +"\t Avg Test  Accuracy:%.4f\n" % avg_test)

def Mreload(modelfile):
    f2=open(modelfile,'rb')
    s2=f2.read()
    model1=pickle.loads(s2)
    return model1

def LoadData(file):
    data = Load.csv_to_pandas(file)
    # print(path+'/'+file)
    NE_Data = NoisE()
    NEdata = NE_Data.eliminat_differential(data)
    fs = 256
    cutoff = 64
    BLNEdata = Load.butter_lowpass_filter(NEdata, cutoff, fs, 6)
    Delta,Theta,Alpha,Beta,Gamma = Load.wave_Processing(BLNEdata)
    mdata = pd.DataFrame({
            'Delta':Delta,
            'Theta':Theta,
            'Alpha':Alpha,
            'Beta':Beta,
            'Gamma':Gamma
        })
    return mdata

def LoadFeature(file):
    data = matrix_from_csv_file(file)
    # data = Load.butter_lowpass_filter(data,64,256)
    vectors, header = generate_feature_vectors_from_data(data, 
                                                        nsamples = 150, 
                                                        period = 1.,
                                                        state = None,
                                                        cols_to_ignore = None)
    FINAL_MATRIX = vectors
    return FINAL_MATRIX
def predict(dataMat):
    x = dataMat.loc[1:,['Alpha','Beta']]
    x=x.values.tolist()
    clf = Mreload('svm-mark-9.model')
    res = clf.predict(x)
    sum = np.sum(res)
    rate = sum/len(res)
    return rate*100

def Fpredict(data):
    model = Mreload('./model/svm-T-6.model')
    res = model.predict(data[1:])
    # print(res)
    sum = np.sum(res)
    rate = sum/len(res)/2
    return rate*100

def Score(EEGfile):
    '''
    输入为EEG信号文件目录，输出评定分数
    '''
    data = LoadData(EEGfile)
    score = float(predict(data))
    print("学习状态评定分数：",score)
    return score

def ScoreF(EEGfile):
    data = LoadFeature(EEGfile)
    # print(data)
    score = float(Fpredict(data))
    print("学习状态评定分数：",score)
    return score

if __name__ == '__main__':
    # Score('EEG_recording_2021-04-11-17.44.44.csv')
    svc_classification('train_data_J_t_1.csv')
    # ScoreF('./dataset/test/10sec.csv')

    # path = './EEG_DATA/original_data'
    # files = os.listdir(path)
    # for file in files:
    #     ScoreF(path+'/'+file)

    # path = './EEG_DATA/original_data'
    # files = os.listdir(path)
    # for file in files:
    #     print(file)
    #     ScoreF(path+'/'+file)

    


