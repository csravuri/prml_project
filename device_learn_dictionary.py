#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 15:29:21 2018

@author: Chandra Sekhar Ravuri
"""

# This is to learn a dictionary from the given device level power usage
# this dictionary used to estimate power from aggregate data

import pandas as pd
import numpy as np
from sklearn import decomposition
from time import time
import scipy.io as sio


def Learn_Dict(device, length, overlap):
    
    X_device = []
    for i1 in range(0,len(device)-length, overlap):
        
        X_device.append(device[i1:i1+length, 1])
    
    
    X_device = np.array(X_device, np.float32)
    
    sum_ele = np.sum(X_device,axis=1)
    
    X_device = np.delete(X_device,[sum_ele<5000],axis=0)
    
    np.random.shuffle(X_device)
    X_device = X_device[:6000,:]
    #X_device = X_device - np.reshape(np.mean(X_device,axis=1),(-1,1))
    D_device = decomposition.dict_learning(X_device, n_components=500, 
                                           alpha= 0.7, max_iter = 20, return_n_iter=True,
                                           n_jobs=1, method='lars',tol=1e-8)
    
    return D_device
    
    
tim = time()
length = 300 # number of components to dictinary
overlap = 10
### Load the device data 

kettle_data = pd.read_table('/home/hadoop1/Documents/prml/project/data/UK_DALE/DATA/house_2/channel_8.dat', sep=' ').values
rice_cooker_data = pd.read_table('/home/hadoop1/Documents/prml/project/data/UK_DALE/DATA/house_2/channel_9.dat', sep=' ').values
running_machine_data = pd.read_table('/home/hadoop1/Documents/prml/project/data/UK_DALE/DATA/house_2/channel_10.dat', sep=' ').values
washing_machine_data = pd.read_table('/home/hadoop1/Documents/prml/project/data/UK_DALE/DATA/house_2/channel_12.dat', sep=' ').values
dish_washer_data = pd.read_table('/home/hadoop1/Documents/prml/project/data/UK_DALE/DATA/house_2/channel_13.dat', sep=' ').values
microwave_data = pd.read_table('/home/hadoop1/Documents/prml/project/data/UK_DALE/DATA/house_2/channel_15.dat', sep=' ').values



D_kettle = Learn_Dict(kettle_data, 300, 10)
sio.savemat('/home/hadoop1/Documents/prml/project/UKDALE/dictionary_NMF/house2_kettle_dictionary.mat',{'kettle':D_kettle})
print('kettle learned and saved', int(time()-tim), 'Sec')
"""
D_washing_machine = Learn_Dict(washing_machine_data, 300, 10)
sio.savemat('/home/hadoop1/Documents/prml/project/UKDALE/dictionary_NMF/house2_washing_machine_dictionary.mat',{'washing_machine':D_washing_machine})
print('washing_machine learned and saved', int(time()-tim), 'Sec')

D_rice_cooker = Learn_Dict(rice_cooker_data, 300, 10)
sio.savemat('/home/hadoop1/Documents/prml/project/UKDALE/dictionary_NMF/house2_rice_cooker_dictionary.mat',{'rice_cooker':D_rice_cooker})
print('rice_cooker learned and saved', int(time()-tim), 'Sec')


D_running_machine = Learn_Dict(running_machine_data, 300, 10)
sio.savemat('/home/hadoop1/Documents/prml/project/UKDALE/dictionary_NMF/house2_running_machine_dictionary.mat',{'running_machine':D_running_machine})
print('running_machine learned and saved', int(time()-tim), 'Sec')



D_dish_washer = Learn_Dict(dish_washer_data, 300, 10)
sio.savemat('/home/hadoop1/Documents/prml/project/UKDALE/dictionary_NMF/house2_dish_washer_dictionary.mat',{'dish_washer':D_dish_washer})
print('dish_washer learned and saved', int(time()-tim), 'Sec')

D_microwave = Learn_Dict(microwave_data, 300, 10)
sio.savemat('/home/hadoop1/Documents/prml/project/UKDALE/dictionary_NMF/house2_microwave_dictionary.mat',{'microwave':D_microwave})
print('microwave learned and saved', int(time()-tim), 'Sec')




#D_device = decomposition.DictionaryLearning(n_components=500, 
#                                            max_iter=2).fit(X_device)


print('Dude this took',int(time()-tim),'Sec')



#import matplotlib.pyplot as plt

#%%

#plt.figure()
#plt.plot(D_device[1][:10,3])
#plt.show()

"""