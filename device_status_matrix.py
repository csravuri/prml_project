#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 00:02:16 2018

@author: Chandra Sekhar Ravuri
"""
###include
# this file will make a marix of day wise device ON - OFF - Missing data

# Matrix value:
# ON = 1
# OFF = 0
# Miss = -1



import numpy as np
import matplotlib.pyplot as plt
from time import time
from datetime import datetime
import pandas as pd
import scipy.io as sio


def make_matrix(data, thresh, duration):
    
    ############# Convert Time stamp to YYYYMMDDHHmmss format #######
    data[:,0] = [datetime.utcfromtimestamp(i).strftime("%Y%m%d%H%M%S") for i in data[:,0]]
    
    
    
    ############ Collecting dates YYYYMMDD format ###############
    lst = np.unique(data[:,0]//1000000) # list of all dates
    
    ########### listing day wise ###########################
    day_data = []
    
    for i in lst:
        a = data[data[:,0]//1000000==i]
        day_data.append(a)
    
    
    ########## Time generation day wise ######################
    min_inter = np.arange(0000,5959,duration*100)
    hr_inter = np.arange(000000,235959,10000)
    
    time_axis = np.array([min_inter+i for i in hr_inter]).flatten()
    
    ########### matrix with day data ############################
    # DeviceOn=1 ; DeviceOff=0 ; MissingData=-1
    day_mat = - np.ones((len(lst), int(1440/duration)))
    
    
    for i1 in range(len(day_data)):
        
        day_wise = day_data[i1]
        day_wise[:,0] = day_wise[:,0]%1000000
        for i2 in range(len(time_axis)-1):
            
            
            aa = day_wise[day_wise[:,0]>=time_axis[i2]]
            aa = aa[aa[:,0]< time_axis[i2+1],1]
            
            
            if (len(aa)):
                
                if (aa.max() > thresh):
                    day_mat[i1,i2] = 1
                    
                else:
                    day_mat[i1,i2] = 0
    
    return day_mat[:,:-1]


#%%


def data_finder(data, numb):
    data[:,0] = [datetime.utcfromtimestamp(i).strftime("%Y%m%d%H%M%S") for i in data[:,0]]
    
    device = list(data[:,0]//1000000)
    ############ Collecting dates YYYYMMDD format ###############
    lst = np.unique(data[:,0]//1000000) # list of all dates
    dev_ind = []
    for i1 in lst[-numb-1:]:
        dev_ind.append(device.index(i1)) 
    
    final_data = []
    for i1 in range(len(xx[-55:])-2):
        final_data.append(data[dev_ind[i1]:dev_ind[i1+1],1])
        
    return final_data, xx


#%%


st_time = time() # to calculate running time

#####################################################################################
########################   Kettle       ############################################# 
#####################################################################################

kettle_data = pd.read_table('/home/hadoop1/Documents/prml/project/data/UK_DALE/DATA/house_2/channel_8.dat', sep=' ')
rice_cooker_data = pd.read_table('/home/hadoop1/Documents/prml/project/data/UK_DALE/DATA/house_2/channel_9.dat', sep=' ')
running_machine_data = pd.read_table('/home/hadoop1/Documents/prml/project/data/UK_DALE/DATA/house_2/channel_10.dat', sep=' ')
washing_machine_data = pd.read_table('/home/hadoop1/Documents/prml/project/data/UK_DALE/DATA/house_2/channel_12.dat', sep=' ')
dish_washer_data = pd.read_table('/home/hadoop1/Documents/prml/project/data/UK_DALE/DATA/house_2/channel_13.dat', sep=' ')
microwave_data = pd.read_table('/home/hadoop1/Documents/prml/project/data/UK_DALE/DATA/house_2/channel_15.dat', sep=' ')


#
#kettle_mat = make_matrix(kettle_data.values, 50, 5)
#rice_cooker_mat = make_matrix(rice_cooker_data.values, 20, 5)
#running_machine_mat = make_matrix(running_machine_data.values, 50, 5)
#washing_machine_mat = make_matrix(washing_machine_data.values, 50, 5)
#dish_washer_mat = make_matrix(dish_washer_data.values, 20, 5)
#microwave_mat = make_matrix(microwave_data.values, 50, 5)


kettle_test, kettle_ind = data_finder(kettle_data.values, 47)
rice_cooker_test, rice_cooker_ind = data_finder(rice_cooker_data.values, 37)
running_machine_test, running_machine_ind = data_finder(running_machine_data.values, 46)
washing_machine_test, washing_machine_ind = data_finder(washing_machine_data.values, 37)
dish_washer_test, dish_washer_ind = data_finder(dish_washer_data.values, 37)
microwave_test, microwave_ind = data_finder(microwave_data.values, 46)



#sio.savemat('/home/hadoop1/Documents/prml/project/UKDALE/device_status_house2.mat',
#            {'kettle':kettle_mat,
#             'rice_cooker':rice_cooker_mat,
#             'running_machine':running_machine_mat,
#             'washing_machine':washing_machine_mat,
#             'dish_washer':dish_washer_mat,
#             'microwave':microwave_mat,
#             'description':'This is UKDALE house 2, day wise device status matrix'})


# test data
sio.savemat('/home/hadoop1/Documents/prml/project/UKDALE/device_test_commondays_house2.mat',
            {'kettle':kettle_test,
             'rice_cooker':rice_cooker_test,
             'running_machine':running_machine_test,
             'washing_machine':washing_machine_test,
             'dish_washer':dish_washer_test,
             'microwave':microwave_test,
             'description':'This is UKDALE house 2, day wise device test data and common days in xx',
             'days':xx})



print(time() - st_time)

print('Done! but wait untill files are saved')


"""
############# Save the data with human readable format #############
np.save('/home/hadoop1/Documents/prml/project/data/UK_DALE/DATA/house_2/converted_data/channel_'+str(chnum)+'.npy',data)

try:
    za = sio.loadmat('/home/hadoop1/Documents/prml/project/data/UK_DALE/DATA/house_2/matrix/house2.mat')
    za.update({devicename:day_mat})
    sio.savemat('/home/hadoop1/Documents/prml/project/data/UK_DALE/DATA/house_2/matrix/house2', za, appendmat=True, do_compression=True)
    
except:
    sio.savemat('/home/hadoop1/Documents/prml/project/data/UK_DALE/DATA/house_2/matrix/house2',{devicename:day_mat}, appendmat=True, do_compression=True)

"""






