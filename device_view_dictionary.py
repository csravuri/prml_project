#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 16:00:56 2018

@author: Chandra Sekhar Ravuri
"""
# this code is redundent check its useful ness before use it

# This to load the dictionaries and veiw their basis vectors


import scipy.io as sio
import matplotlib.pyplot as plt
import time


dish_washer = sio.loadmat('/home/hadoop1/Documents/prml/project/UKDALE/dictionary2000_2itr/house2_dish_washer_dictionary.mat')['dish_washer'][0][1]
kettle = sio.loadmat('/home/hadoop1/Documents/prml/project/UKDALE/dictionary2000_2itr/house2_kettle_dictionary.mat')['kettle'][0][1]
microwave = sio.loadmat('/home/hadoop1/Documents/prml/project/UKDALE/dictionary2000_2itr/house2_microwave_dictionary.mat')['microwave'][0][1]
rice_cooker = sio.loadmat('/home/hadoop1/Documents/prml/project/UKDALE/dictionary2000_2itr/house2_rice_cooker_dictionary.mat')['rice_cooker'][0][1]
running_machine = sio.loadmat('/home/hadoop1/Documents/prml/project/UKDALE/dictionary2000_2itr/house2_running_machine_dictionary.mat')['running_machine'][0][1]
washing_machine = sio.loadmat('/home/hadoop1/Documents/prml/project/UKDALE/dictionary2000_2itr/house2_washing_machine_dictionary.mat')['washing_machine'][0][1]


plt.close('all')

plt.figure()
for i1 in range(10):
    plt.plot(dish_washer[i1,:])
    
plt.legend(range(1,11))
plt.title('200 samples 2 iterations')
plt.show()


##

dish_washer = sio.loadmat('/home/hadoop1/Documents/prml/project/UKDALE/dictionary20000_2itr/house2_dish_washer_dictionary.mat')['dish_washer'][0][1]
kettle = sio.loadmat('/home/hadoop1/Documents/prml/project/UKDALE/dictionary20000_2itr/house2_kettle_dictionary.mat')['kettle'][0][1]
microwave = sio.loadmat('/home/hadoop1/Documents/prml/project/UKDALE/dictionary20000_2itr/house2_microwave_dictionary.mat')['microwave'][0][1]
rice_cooker = sio.loadmat('/home/hadoop1/Documents/prml/project/UKDALE/dictionary20000_2itr/house2_rice_cooker_dictionary.mat')['rice_cooker'][0][1]
running_machine = sio.loadmat('/home/hadoop1/Documents/prml/project/UKDALE/dictionary20000_2itr/house2_running_machine_dictionary.mat')['running_machine'][0][1]
washing_machine = sio.loadmat('/home/hadoop1/Documents/prml/project/UKDALE/dictionary20000_2itr/house2_washing_machine_dictionary.mat')['washing_machine'][0][1]




plt.figure()
for i1 in range(10):
    plt.plot(dish_washer[i1,:])
    
plt.legend(range(1,11))
plt.title('20000 samples 2 iterations')
plt.show()


###


dish_washer = sio.loadmat('/home/hadoop1/Documents/prml/project/UKDALE/dictionary_20000_2itr_mean/house2_dish_washer_dictionary.mat')['dish_washer'][0][1]
kettle = sio.loadmat('/home/hadoop1/Documents/prml/project/UKDALE/dictionary_20000_2itr_mean/house2_kettle_dictionary.mat')['kettle'][0][1]
microwave = sio.loadmat('/home/hadoop1/Documents/prml/project/UKDALE/dictionary_20000_2itr_mean/house2_microwave_dictionary.mat')['microwave'][0][1]
rice_cooker = sio.loadmat('/home/hadoop1/Documents/prml/project/UKDALE/dictionary_20000_2itr_mean/house2_rice_cooker_dictionary.mat')['rice_cooker'][0][1]
running_machine = sio.loadmat('/home/hadoop1/Documents/prml/project/UKDALE/dictionary_20000_2itr_mean/house2_running_machine_dictionary.mat')['running_machine'][0][1]
washing_machine = sio.loadmat('/home/hadoop1/Documents/prml/project/UKDALE/dictionary_20000_2itr_mean/house2_washing_machine_dictionary.mat')['washing_machine'][0][1]




plt.figure()
for i1 in range(10):
    plt.plot(dish_washer[i1,:])
    
    
plt.legend(range(1,11))
plt.title('20000 samples 2 iterations MEAN')
plt.show()







