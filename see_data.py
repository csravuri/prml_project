#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 23:43:12 2018

@author: Chandra Sekhar Ravuri
"""

import numpy as np
import matplotlib.pyplot as plt
from time import time
from datetime import datetime, timedelta
import pandas as pd


duration = 5 # in minutes
strt_time = 000000 # HHmmSS
############ Load the data file #######################
#data1 = np.genfromtxt('/home/hadoop1/Documents/prml/project/data/UK_DALE/DATA/house_1/channel_1.dat')
data = pd.read_table('/home/hadoop1/Documents/prml/project/data/UK_DALE/DATA/house_2/channel_8.dat',sep=' ')
data = np.array(data)


# for making(not required)
#data = data1[:10000]

st_time = time()

############# Convert Time stamp to YYYYMMDDHHmmss format #######
data[:,0] = [datetime.utcfromtimestamp(i).strftime("%Y%m%d%H%M%S") for i in data[:,0]]

#for i in range(len(data[:,0])):
#    data[i,0] = datetime.utcfromtimestamp(data[i,0]).strftime("%Y%m%d%H%M%S")

#print(time() - st_time)

############ Collecting dates YYYYMMDD format ###############
lst = np.unique(data[:,0]//1000000) # list of all dates

print('Total '+str(len(lst))+' days')


########### listing day wise ###########################
day_data = []

for i in lst:
    a = data[data[:,0]//1000000==i]
    day_data.append(a)



########## Time generation day wise ######################
min_inter = np.arange(strt_time,5959,500)
hr_inter = np.arange(000000,235959,10000)

time_axis = np.array([min_inter+i for i in hr_inter]).flatten()

# save days list also ( lst ) 
########### matrix with day data ############################
# DeviceOn=1 ; DeviceOff=0 ; MissingData=-1
day_mat = - np.ones((len(lst), int(1440/duration)))


for i1 in range(len(day_data)):
    
    day_wise = day_data[i1]
    for i2 in range(len(time_axis)-1):
        
        aa = []
        for i in range(len(day_wise[:,0])):
            if (day_wise[i,0]%1000000 >= time_axis[i2] and day_wise[i,0]%1000000 < time_axis[i2+1]):
                aa.append(day_wise[i,1])
        
        aa = np.array(aa)
        
        if (len(aa) == 0):
            day_mat[i1,i2] = 0
            
        elif (aa.max() > 10):
            day_mat[i1,i2] = 1
        
             
print(time() - st_time)
print('Done!')












#ax = data[data[:,0]%1000000 < time_axis[1],0]

#aa = day_wise[1 <= day_wise[:,0]%1000000 < 10]




#aa = [day_wise[i,1] for i in day_wise[:,0]%1000000 if(i>=1 and i<20)]

#plt.figure()
#plt.plot(data[:50000,0],data[:50000,1])
#plt.show()


#data[:,0] = data[:,0]//1000000

#a = data[data[:,0]//1000000==data[0,0]//1000000]


'''
print(a.shape)
plt.figure()
plt.plot(a[:,0],a[:,1])
plt.show()
'''

#print(time.strftime("%Y %m %d %H %M %S","15465465"))
#print(time.ctime())
#print(time.clock())


#print(datetime.fromtimestamp(1175412949).strftime("%Y%m%d%H%M%S")) ## YYYYMMDDHHmmss


