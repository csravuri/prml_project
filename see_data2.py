#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 23:43:12 2018

@author: Chandra Sekhar Ravuri
"""

'''
import numpy as np
import matplotlib.pyplot as plt
from time import time
from datetime import datetime
import pandas as pd
import scipy.io as sio

st_time = time() # to calculate running time

############## Basic parameters #####################
chnum = 10 # channel number of device identifier
thresh = 20 # Threshod power for a device considered to be ON
devicename = 'dish_washer'

duration = 5 # in minutes

############ Load the data file #######################
#data1 = np.genfromtxt('/home/hadoop1/Documents/prml/project/data/UK_DALE/DATA/house_1/channel_1.dat')
data = pd.read_table('/home/hadoop1/Documents/prml/project/data/UK_DALE/DATA/house_2/channel_'+str(chnum)+'.dat',sep=' ')
data = np.array(data)

"""
########### For intution ##########################
plt.figure()
plt.plot(data[:,1])
plt.show()


"""
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
        
             
print(time() - st_time)

print('Done! but wait untill files are saved')

############# Save the data with human readable format #############
np.save('/home/hadoop1/Documents/prml/project/data/UK_DALE/DATA/house_2/converted_data/channel_'+str(chnum)+'.npy',data)

try:
    za = sio.loadmat('/home/hadoop1/Documents/prml/project/data/UK_DALE/DATA/house_2/matrix/house2.mat')
    za.update({devicename:day_mat})
    sio.savemat('/home/hadoop1/Documents/prml/project/data/UK_DALE/DATA/house_2/matrix/house2', za, appendmat=True, do_compression=True)
    
except:
    sio.savemat('/home/hadoop1/Documents/prml/project/data/UK_DALE/DATA/house_2/matrix/house2',{devicename:day_mat}, appendmat=True, do_compression=True)


'''



#%%%
    
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np

aa = sio.loadmat('/home/hadoop1/Documents/prml/project/data/UK_DALE/DATA/house_2/matrix/house2.mat')


dish_washer = aa['dish_washer'][:,:-1]
time_min = range(0,1434,5)

observed_data = dish_washer.copy()
observed_data[observed_data>-1] = 1
observed_data[observed_data<1] = 0
plt_obser = np.sum(observed_data,axis=0)

on_data = dish_washer.copy()
on_data[on_data<1] = 0
plt_on = np.sum(on_data,axis=0)

plt.close('all')

rc('xtick',labelsize=18)
rc('ytick',labelsize=18)

plt.figure()
#plt.stem(time_min, plt_obser, markerfmt=':')
plt.bar(time_min, plt_on)
plt.legend(['Obeserved days', 'ON days'],fontsize=18)
plt.xlabel('Time in Minutes(Whole day)',size=18)
plt.ylabel('Frequency (Number of days)',size =18)
plt.title('Dish Washer',size=18)
plt.show()












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


