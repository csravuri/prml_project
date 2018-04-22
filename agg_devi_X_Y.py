#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 01:19:10 2018

@author: Chandra Sekhar Ravuri
"""

# this is extract X and Y from agrregate data and appliance data respectively
# X1 = vectors from agreegate data 
# Y1 = vector from device level data
# ON_OFF = ON/OFF condition

################# Importing Libraries  #######################################
import pandas as pd
import numpy as np
from time import time 


################   Loading Data   #############################################

agg_data = pd.read_table('/home/hadoop1/Documents/prml/project/data/UK_DALE/DATA/house_1/channel_8.dat', sep=' ').values#[:15000,:]

dev_data = pd.read_table('/home/hadoop1/Documents/prml/project/data/UK_DALE/DATA/house_2/channel_8.dat', sep=' ').values#[:5000,:]

tim = time()

p = 60 # 1 min
M = 300 # 5 min
w = 20 # power in Watts
visin = 5 # search visinity

agg_list = list(agg_data[:,0])

X1 = []
Y1 = []
ON_OFF = []

for i1 in range(0,dev_data.shape[0]-M,p):
    
    print(i1)
    y1 = dev_data[i1:i1+M,:]
    
    num = range(y1[0,0]-visin, y1[0,0]+visin) # num[visin] is original(interst) from dev_data
    
    valid = np.isin(num,agg_data[:,0])
    print(valid)
    if (not valid.any()):
        print('bla')
        continue
    
    if (valid[visin]):
        x_ind = agg_list.index(num[visin])
        
        X1.append(agg_data[x_ind:x_ind+M,1].T)
        Y1.append(y1[:,1].T)
        if(y1[:,1].max() >= w):
            ON_OFF.append(1)
        else:
            ON_OFF.append(0)
        
    else:
        x_ind = agg_list.index(num[list(valid).index(True)])
        
        X1.append(agg_data[x_ind:x_ind+M,1].T)
        Y1.append(y1[:,1].T)
        if(y1[:,1].max() >= w):
            ON_OFF.append([1])
        else:
            ON_OFF.append([0])
        
        
    agg_list = agg_list[x_ind-visin:]
    #print(y1.shape)


print('Total time taken',int(time()-tim))
print(len(X1),len(Y1),len(ON_OFF))


np.save('agregate.npy',np.array(X1))
np.save('device_kattle.npy',np.array(Y1))
np.save('device_kattle_ON_OFF.npy',np.array(ON_OFF))



"""
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


########### For intution ##########################
plt.figure()
plt.plot(data[:,1])
plt.show()



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


#print(time.strftime("%Y %m %d %H %M %S","15465465"))
#print(time.ctime())
#print(time.clock())


#print(datetime.fromtimestamp(1175412949).strftime("%Y%m%d%H%M%S")) ## YYYYMMDDHHmmss


"""
