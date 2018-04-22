#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 19:40:47 2018

@author: Chandra Sekhar Ravuri
"""

# to make bassian of three types
# 1. Bayesian as normal from matrix by counting number of minitues ON time
# 2. multyply ON time/total_time to above distribution
# 3. devide data to 20% and 80% prior with 20% and distribution with 80% then 
#       multyply the prior(got from 20%) to 80% data it become posterior 


import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

plt.close('all')

devies = sio.loadmat('/home/hadoop1/Documents/prml/project/data/UK_DALE/DATA/house_2/matrix/house2_1min.mat')

dev_lst = list(devies.keys())[3:]


plots = [] # normal bayesian
plots2 = [] # 20% and 80% bayesian
for dev in dev_lst:
    
    device = devies[dev][:,:-1]
    
    mask = device.copy()
    mask[device > -1] = 0
    mask = np.abs(mask)
    
    active = np.invert(mask.any(axis=1))
    
    device2 = device[active,:]
    
    plot_on = np.sum(device2,axis=0, dtype='float32')
    
    plots.append(plot_on)
    
    # 3rd bayesian
    
    device = np.random.permutation(device2)
    
    ind = int(len(device)*0.2)
    
    dev_pri = device[:ind,:]
    dev_pos = device[ind:,:]
    
    plot_20 = np.sum(dev_pri, axis=0, dtype='float32')
    plot_80 = np.sum(dev_pos, axis=0, dtype='float32')
    
    
    plots2.append((plot_20/float(np.sum(plot_20)))*(plot_80/float(np.sum(plot_80))))
    
    
    
    
    
    
    
#dish_washer

plot11 = plots[0]/float(np.sum(plots[0]))
plot12 = 0.5*plots[0]/float(np.sum(plots[0]))
plot13 = plots2[0]/float(np.sum(plots2[0]))

plt.figure()
plt.plot(plot11)
plt.plot(plot12)
plt.plot(plot13)
plt.title(dev_lst[0])
plt.legend(['Normal', 'Multiplied by constant', 'Postirier'])
plt.show()

#kettle

plot21 = plots[1]/float(np.sum(plots[1]))
plot22 = 0.5*plots[1]/float(np.sum(plots[1]))
plot23 = plots2[1]/float(np.sum(plots2[1]))

plt.figure()
plt.plot(plot21)
plt.plot(plot22)
plt.plot(plot23)
plt.title(dev_lst[1])
plt.legend(['Normal', 'Multiplied by constant', 'Postirier'])
plt.show()


#rice_cooker

plot31 = plots[2]/float(np.sum(plots[2]))
plot32 = 0.5*plots[2]/float(np.sum(plots[2]))
plot33 = plots2[2]/float(np.sum(plots2[2]))

plt.figure()
plt.plot(plot31)
plt.plot(plot32)
plt.plot(plot33)
plt.title(dev_lst[2])
plt.legend(['Normal', 'Multiplied by constant', 'Postirier'])
plt.show()

#running_machine

plot41 = plots[3]/float(np.sum(plots[3]))
plot42 = 0.5*plots[3]/float(np.sum(plots[3]))
plot43 = plots2[3]/float(np.sum(plots2[3]))

plt.figure()
plt.plot(plot41)
plt.plot(plot42)
plt.plot(plot43)
plt.title(dev_lst[3])
plt.legend(['Normal', 'Multiplied by constant', 'Postirier'])
plt.show()

#washing_machine

plot51 = plots[4]/float(np.sum(plots[4]))
plot52 = 0.5*plots[4]/float(np.sum(plots[4]))
plot53 = plots2[4]/float(np.sum(plots2[4]))

plt.figure()
plt.plot(plot51)
plt.plot(plot52)
plt.plot(plot53)
plt.title(dev_lst[4])
plt.legend(['Normal', 'Multiplied by constant', 'Postirier'])
plt.show()

#microwave

plot61 = plots[5]/float(np.sum(plots[5]))
plot62 = 0.5*plots[5]/float(np.sum(plots[5]))
plot63 = plots2[5]/float(np.sum(plots2[5]))

plt.figure()
plt.plot(plot61)
plt.plot(plot62)
plt.plot(plot63)
plt.title(dev_lst[5])
plt.legend(['Normal', 'Multiplied by constant', 'Postirier'])
plt.show()







