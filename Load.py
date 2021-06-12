import os
import os, sys

import csv
import pandas as pd
import pywt
import numpy as np 
from scipy.signal import butter, lfilter, freqz

from EEG_feature_extraction import generate_feature_vectors_from_samples
from NoiseEliminate import  NoisE
# from muselsl import stream

# muses = stream.list_muses()
# stream.stream(muses[0]['address'])

def csv_to_dict(filename):
    data = {}
    with open(filename, 'r') as csvfile:
        rdr = csv.reader(csvfile)
        headers = next(rdr, None)

        for col_mun,header_name in enumerate(headers):
            print(col_mun)
            data[header_name] = [row[col_mun] for row in rdr]
            print([row[col_mun] for row in rdr][1:10])
            print(data[header_name][0:10])
    return data

def csv_to_pandas(filename):
    '''
    将csv文件按列存入字典
    '''
    # data={}
    rdr = pd.read_csv(filename)
    # headers = rdr.columns
    # for header in headers:
    #     data[header] = list(rdr[header])
    # return data
    return rdr

def wave_Processing(data):
    sampleData = data
    wavelet='db4' #选取的小波基函数
    X = range(len(sampleData))
    wave =pywt.wavedec(sampleData, wavelet, level=4)
    #小波重构
    Delta = pywt.waverec(np.multiply(wave,[1, 0, 0, 0, 0]).tolist(),wavelet)#0-4hz重构小波(δ节律))
    Theta = pywt.waverec(np.multiply(wave, [0, 1, 0, 0, 0]).tolist(), wavelet)#4-8hz重构小波（θ节律）
    Alpha = pywt.waverec(np.multiply(wave, [0, 0, 1, 0, 0]).tolist(), wavelet)#8-16hz重构小波（α节律）
    Beta = pywt.waverec(np.multiply(wave, [0, 0, 0, 1, 0]).tolist(), wavelet)#16-32hz重构小波（β节律）
    Gamma = pywt.waverec(np.multiply(wave, [0, 0, 0, 0, 1]).tolist(), wavelet)#32-64hz重构小波（γ节律）

    return Delta,Theta,Alpha,Beta,Gamma

def wave_Packet(data):
    wp = pywt.WaveletPacket(data, wavelet='db10', mode='symmetric', maxlevel=9)
    nodeArr=np.array([node.path for node in wp.get_level(9, 'freq')])#获取第九层节点数组 256组
    #定义一个空的小波来接收重构后的信号
    new_Delta = pywt.WaveletPacket(data=None, wavelet='db10', mode='symmetric')
    new_Theta = pywt.WaveletPacket(data=None, wavelet='db10', mode='symmetric')
    new_Alpha = pywt.WaveletPacket(data=None, wavelet='db10', mode='symmetric')
    new_Beta = pywt.WaveletPacket(data=None, wavelet='db10', mode='symmetric')
    new_Gamma = pywt.WaveletPacket(data=None, wavelet='db10', mode='symmetric')
    for j in range(2, 11):
        cunrrentNode = nodeArr[j]
        new_Delta[cunrrentNode] = wp[cunrrentNode]
    Delta = new_Delta.reconstruct(update=True)
    for j in range(15, 28):
        cunrrentNode = nodeArr[j]
        new_Theta[cunrrentNode] = wp[cunrrentNode]
    Theta = new_Theta.reconstruct(update=True)
    for j in range(31, 53):
        cunrrentNode = nodeArr[j]
        new_Alpha[cunrrentNode] = wp[cunrrentNode]
    Alpha = new_Alpha.reconstruct(update=True)
    for j in range(56, 122):
        cunrrentNode = nodeArr[j]
        new_Beta[cunrrentNode] = wp[cunrrentNode]
    Beta = new_Beta.reconstruct(update=True)
    for j in range(128, 255):
        cunrrentNode = nodeArr[j]
        new_Gamma[cunrrentNode] = wp[cunrrentNode]
    Gamma = new_Gamma.reconstruct(update=True)

    return Delta,Theta,Alpha,Beta,Gamma



def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    ncutoff = cutoff / nyq
    b, a = butter(order, ncutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def mark_data():
    for mark in [2,0,1]:
        if mark == 2 :
            path = 'EEG_DATA/original_learning'
            # path = 'EEG_DATA/Learning'
            head = 1
        elif mark == 0:
            path = 'EEG_DATA/original_not_learning'
            # path = 'EEG_DATA/Not_Learning'
            head = 2
        else:
            path = 'EEG_DATA/original_data'
            head = 2

        files = os.listdir(path)
        
        print(f'processing {path} data, mark = {mark}')
        for file in files:
            data = csv_to_pandas(path+'/'+file)
            print(path+'/'+file)
            # NE_Data = NoisE()
            # NEdata = NE_Data.eliminat_differential(data)
            TP9 = np.array(data.loc[1:,['TP9']].values.tolist()).flatten()
            TP10 = np.array(data.loc[1:,['TP10']].values.tolist()).flatten()
            AF7 = np.array(data.loc[1:,['AF7']].values.tolist()).flatten()
            AF8 = np.array(data.loc[1:,['AF8']].values.tolist()).flatten()
            fs = 256
            cutoff = 64
            # BLNEdata = butter_lowpass_filter(NEdata, cutoff, fs, 6)
            # Delta,Theta,Alpha,Beta,Gamma = wave_Packet(BLNEdata)
            # Delta,Theta,Alpha,Beta,Gamma = wave_Processing(BLNEdata)
            # mdata = pd.DataFrame({
            #         'Delta':Delta,
            #         'Theta':Theta,
            #         'Alpha':Alpha,
            #         'Beta':Beta,
            #         'Gamma':Gamma,
            #         'Mark': pd.Series(mark, index=list(range(len(Delta))))
            #     })
            BLNEdata_9 = butter_lowpass_filter(TP9, cutoff, fs, 6)
            BLNEdata_10 = butter_lowpass_filter(TP10, cutoff, fs, 6)
            BLNEdata_7 = butter_lowpass_filter(AF7, cutoff, fs, 6)
            BLNEdata_8 = butter_lowpass_filter(AF8, cutoff, fs, 6)
            BLNEdata_9 = [i for i in BLNEdata_9]
            BLNEdata_10 = [i for i in BLNEdata_10]
            BLNEdata_7 = [i for i in BLNEdata_7]
            BLNEdata_8 = [i for i in BLNEdata_8]
            Delta_9,Theta_9,Alpha_9,Beta_9,Gamma_9 = wave_Processing(BLNEdata_9)
            Delta_10,Theta_10,Alpha_10,Beta_10,Gamma_10 = wave_Processing(BLNEdata_10)
            Delta_7,Theta_7,Alpha_7,Beta_7,Gamma_7 = wave_Processing(BLNEdata_7)
            Delta_8,Theta_8,Alpha_8,Beta_8,Gamma_8 = wave_Processing(BLNEdata_8)
            mdata = pd.DataFrame({
                    'Delta_0':Delta_9,
                    'Delta_1':Delta_10,
                    'Delta_2':Delta_7,
                    'Delta_3':Delta_8,
                    'Theta_0':Theta_9,
                    'Theta_1':Theta_10,
                    'Theta_2':Theta_7,
                    'Theta_3':Theta_8,
                    'Alpha_0':Alpha_9,
                    'Alpha_1':Alpha_10,
                    'Alpha_2':Alpha_7,
                    'Alpha_3':Alpha_8,
                    'Beta_0':Beta_9,
                    'Beta_1':Beta_10,
                    'Beta_2':Beta_7,
                    'Beta_3':Beta_8,
                    'Gamma_0':Gamma_9,
                    'Gamma_1':Gamma_10,
                    'Gamma_2':Gamma_7,
                    'Gamma_3':Gamma_8,
                    'Label': pd.Series(mark, index=list(range(len(Delta_9))))
                })
            mdata.to_csv('train_data_J.csv',mode='a',header=False if head >1 else True,index=False)
            # mdata.to_csv('train_data_J_t.csv',mode='a',header=False if head >1 else True,index=False)
            head += 1

def mark_data_time(directory_path, output_file, cols_to_ignore):
    	# Initialise return matrix
	FINAL_MATRIX = None
	
	for x in os.listdir(directory_path):

		# Ignore non-CSV files
		if not x.lower().endswith('.csv'):
			continue
		
		# For safety we'll ignore files containing the substring "test". 
		# [Test files should not be in the dataset directory in the first place]
		if 'test' in x.lower():
			continue
		try:
			name, state, _ = x[:-4].split('-')
		except:
			print ('Wrong file name', x)
			sys.exit(-1)
		if state.lower() == 'concentrating':
			state = 2.
		elif state.lower() == 'neutral':
			state = 1.
		elif state.lower() == 'relaxed':
			state = 0.
		else:
			print ('Wrong file name', x)
			sys.exit(-1)
			
		print ('Using file', x)
		full_file_path = directory_path  +   '/'   + x
		vectors, header = generate_feature_vectors_from_samples(file_path = full_file_path, 
														        nsamples = 150, 
																period = 1.,
																state = state,
																cols_to_ignore = cols_to_ignore)		
		
		if FINAL_MATRIX is None:
			FINAL_MATRIX = vectors
		else:
			FINAL_MATRIX = np.vstack( [ FINAL_MATRIX, vectors ] )

	np.random.shuffle(FINAL_MATRIX)
	
	np.savetxt(output_file, FINAL_MATRIX, delimiter = ',',
			header = ','.join(header), 
			comments = '')

	return None


if __name__ == '__main__':
    # data = csv_to_pandas('EEG_recording_2021-02-06-02.02.51.csv')
    # print(type(data))
    # mark_data()
    # path1 = './dataset/original_data'
    # mark_data_time(path1,'train_data.csv',-1)

    # print(csv_to_pandas('train_data.csv').shape)
    # print(csv_to_pandas('data.csv'))

    data = csv_to_pandas('train_data_J.csv')
    data = data.sample(10000)
    data.to_csv('train_data_J_t_1.csv')
