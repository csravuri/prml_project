#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 14:03:35 2018

@author: Chandra Sekhar Ravuri
"""

import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np

# generate corrletion matrix for two dictionaries for devices
def correlation_matrix(device1, device2):
    
    total = []
    for i1 in range(device1.shape[0]):
        
        row = []
        for i2 in range(device2.shape[0]):
            row.append(np.correlate(device1[i1,:], device2[i2,:]))
            
            
        total.append(row)
            
            
    total = np.array(total, np.float32).squeeze()
    
    return total

dict_folder = '/home/hadoop1/Documents/prml/project/UKDALE/dictionary_20000_2itr_mean/' # '/' at the end is importatnt

dish_washer = sio.loadmat(dict_folder+'house2_dish_washer_dictionary.mat')['dish_washer'][0][1]
kettle = sio.loadmat(dict_folder+'house2_kettle_dictionary.mat')['kettle'][0][1]
microwave = sio.loadmat(dict_folder+'house2_microwave_dictionary.mat')['microwave'][0][1]
rice_cooker = sio.loadmat(dict_folder+'house2_rice_cooker_dictionary.mat')['rice_cooker'][0][1]
running_machine = sio.loadmat(dict_folder+'house2_running_machine_dictionary.mat')['running_machine'][0][1]
washing_machine = sio.loadmat(dict_folder+'house2_washing_machine_dictionary.mat')['washing_machine'][0][1]


#
#
#plt.figure()
#for i1 in range(10):
#    plt.plot(dish_washer[i1,:])
#    
#    
#plt.legend(range(1,11))
#plt.title('20000 samples 2 iterations MEAN')
#plt.show()


kettle_dish = abs(correlation_matrix(kettle, washing_machine))

kettle_dish[kettle_dish < 0.5] = 0

