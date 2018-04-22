#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 02:49:38 2018

@author: sai
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 22:29:14 2018

@author: sai
"""

from cvxpy import *
import numpy as np

import scipy.io as sio


#%%    load the dictionary

dict_folder = '/media/sai/F070BD0C70BCDA94/ThinkSpace/dictionary_6000_20itr_mean/' # '/' at the end is importatnt

psi_dish_washer = sio.loadmat(dict_folder+'house2_dish_washer_dictionary.mat')['dish_washer'][0][1]
psi_kettle = sio.loadmat(dict_folder+'house2_kettle_dictionary.mat')['kettle'][0][1]
psi_microwave = sio.loadmat(dict_folder+'house2_microwave_dictionary.mat')['microwave'][0][1]
psi_rice_cooker = sio.loadmat(dict_folder+'house2_rice_cooker_dictionary.mat')['rice_cooker'][0][1]
psi_running_machine = sio.loadmat(dict_folder+'house2_running_machine_dictionary.mat')['running_machine'][0][1]
psi_washing_machine = sio.loadmat(dict_folder+'house2_washing_machine_dictionary.mat')['washing_machine'][0][1]


# device level consumption data
device_folder='/media/sai/F070BD0C70BCDA94/ThinkSpace/Device_Usage/'

dev_dish_washer = np.load(device_folder+'Activation_Dishwasher_3days.npy')
dev_kettle = np.load(device_folder+'Activation_Kettle_3days.npy')
dev_running_machine= np.load(device_folder+'Activation_RunningMachine_3days.npy')
dev_washing_machine = np.load(device_folder+'Activation_WashingMachine_3days.npy')

A = np.concatenate((psi_dish_washer.T, psi_kettle.T, psi_running_machine.T, psi_washing_machine.T),axis=1)

agg_data = dev_dish_washer + dev_kettle + dev_running_machine + dev_washing_machine

agg_data = agg_data - np.reshape(np.mean(agg_data,axis=1),(-1,1))

for i1 in range(agg_data.shape[0]): # to select day wise
    
    a = []
    for i2 in range(0,agg_data.shape[1],300):
        
        b = np.reshape(agg_data[i1,i2:i2+300],(-1,1))
        
        # Construct the problem.
        
        x = Variable(A.shape[1])   # coefficient vactor
        
        objective=Minimize(norm(x,1))
        
        constraints = [A*x-b == 0]
        
        prob = Problem(objective, constraints)
        
        # The optimal objective is returned by prob.solve().
        result = prob.solve()
        
        #The optimal value for x is stored in x.value.
        x=x.value
        
        a.append(x)

    a =np.array(a).squeeze()
    
    x1 = []
    x2 = []
    x3 = []
    x4 = []
    for j1 in range(a.shape[0]):
        
        x1.extend(np.matmul(psi_dish_washer.T, a[j1,:500]))
        x2.extend(np.matmul(psi_kettle.T, a[j1,500:1000]))
        x3.extend(np.matmul(psi_running_machine.T, a[j1,1000:1500]))
        x4.extend(np.matmul(psi_washing_machine.T, a[j1,1500:]))
        
        
        
        
        
import matplotlib.pyplot as plt

plt.figure()
plt.plot(x1)
plt.plot(dev_dish_washer[0,:])
plt.show()



    

"""
# Problem data.
m = 300    # sample length
n = 500    # No. of basis in dictionary

A = np.random.randn(m, n)  # It indicates the dictionary learnt with 300 rows and 500 columns.

b = np.random.randn(m)  # It is the sum total of appliance level power.

# Construct the problem.

x = Variable(n)   # coefficient vactor

objective=Minimize(norm(x,1))

constraints = [A*x-b == 0]

prob = Problem(objective, constraints)

# The optimal objective is returned by prob.solve().
result = prob.solve()

#The optimal value for x is stored in x.value.
x=x.value
"""
