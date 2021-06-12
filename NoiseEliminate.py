
import numpy as np
from numba import jit 
import logging
from logging.handlers import RotatingFileHandler
from scipy.fftpack import fft,ifft


logging.basicConfig(
    level=logging.INFO, 
    filename='log', 
    filemode='w', 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

wave_HZ={ #脑电信号频带范围
    'min_HZ' : 0.1,'max_HZ' : 100,
    'delta_min' : 0.5,'delta_max' : 3,
    'theta_min' : 4,'theta_max' : 7,
    'alpha_min' : 8,'alpha_max' : 12,
    'beta_min' : 15,'beta_max' : 30,
    'gamma_min' : 30
}

class NoisE:
    def __init__(self):
        logging.info("Noise is eliminating")
    
    def Fourier(self, raw_data):
        '''
        使用傅里叶变化,显示信号频带
        '''
        fft_y = fft(raw_data)
        data=np.abs(fft_y)/1000
        return data
    
    # def eliminat(self, raw_data):
    #     '''
    #     多组电极信号，利用中位值消除噪声信号 #信号分析有误，弃用
    #     '''
    #     data = []
    #     for num, wave_value in enumerate(raw_data['timestamps']):
    #         single_value = []
    #         for header in raw_data:
    #             if header == 'timestamps': continue
    #             single_value.append(raw_data[header][num])
    #         # print(single_value)
    #         median = np.median(single_value)
    #         # print(median,1.5*median,0.5*median)
    #         final_value = [value for value in single_value if abs(value) <= abs(1.2*median) and abs(value) >= abs(0.8*median)]
    #         # print("final_value:",final_value)
    #         if not final_value: 
    #             data.append(median)
    #         else:
    #             data.append(np.mean(final_value))
    #         logging.info(f'single_value:{single_value}\tmediaan:{median}\tfinal_value:{final_value}')
    #     return data

    def eliminat_differential(self, raw_data):
        '''
        利用两组差分信号，消除信道噪声
        '''
        data = []
        for num, wave_value in enumerate(raw_data['timestamps']): #num :data longth
            sum = raw_data['TP9'][num] - raw_data['AF7'][num] + raw_data['AF8'][num] - raw_data['TP10'][num]
            single_value = sum /4
            data.append(single_value)
            # logging.info(f'TP9:{raw_data['TP9'][num]},AF7:{raw_data['AF7'][num]},data:{single_value}')
        return data
    def eliminat_horizontal(self,raw_data):
        pass

if __name__ == "__main__":
    NE_Data = NoisE()
    print(wave_HZ)

