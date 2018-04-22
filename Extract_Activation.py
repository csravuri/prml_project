# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 13:32:32 2018

@author: SmartGridData
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io as sio

x=np.load('./Data/Processed/house_1_kt.npy')
x=np.load('./Data/Processed/house_1_wm.npy')
x=np.load('./Data/Processed/house_1_dw.npy')
x=np.load('./Data/Processed/house_1_mw.npy')

x=pd.read_csv('./Data/Processed/channel_10.dat',delimiter=' ') # Running machine
x=np.array(x)


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Activation for appliance Kettle %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
x=x[0:1000000,1]
x_diff=np.diff(x)

Start_Approx=np.array(np.where(x_diff>=1050))
End_Approx=np.array(np.where(x_diff<=-1050))

Min=np.min((len(Start_Approx[0,:]),len(End_Approx[0,:])))

Start=np.reshape(Start_Approx[0,0:Min],(Min,1))

End=np.reshape(End_Approx[0,0:Min],(Min,1))

Switching_Points=np.concatenate((Start,End),axis=1)

Switching_Points[:,0]=Switching_Points[:,0]-5

Switching_Points[:,1]=Switching_Points[:,1]+5

df=np.diff(Switching_Points,axis=1)

Act_Kettle=[]
for i in range(220):
    Act_Kettle.append(x[Switching_Points[i,0]:Switching_Points[i,1]])
    
np.save('./Appliance_Activations/Act_Kettle',Act_Kettle)       

#%%%%%%%%%%%%%%%%%%%%%%%%%% Activation for appliance Washing machine, Dishwasher, Running machine %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
x=x[0:100000000,1]

# Washing Machine threshold1 = 400
# Washing Machine threshold2 = 50

# Dishwasher threshold1 = 400
# Dishwasher threshold2 = 50

# Running Machine threshold1 = 12
# Running machine threshold2 = 10


threshold1=12

threshold2=10

Index=np.array(np.where(x>threshold1))

Index_Diff=np.diff(Index)
  
Index_Greater=np.array(np.where(Index_Diff>100))

Switching_Points=np.zeros((len(Index_Greater[0,:]),2))

Switching_Points[1:,0]=Index_Greater[1,0:-1]
Switching_Points[1:,0]=Switching_Points[1:,0]


Switching_Points[:,1]=Index_Greater[1,:]
Switching_Points[:,1]=Switching_Points[:,1]

#t1=Index[0,int(Switching_Points[i,0]):int(Switching_Points[i,1])]

Act_WashingMachine=[]
Activation_Index=[]
for i in range(np.size(Switching_Points[:,0])):
    Activation_Index.append(Index[0,int(Switching_Points[i,0]):int(Switching_Points[i,1])])
    if np.array(Activation_Index[i].shape) > threshold2 : 
       Act_WashingMachine.append(x[Index[0,int(Switching_Points[i,0]):int(Switching_Points[i,1])]])
       
Act_Appliance=[]
for i in range(np.size(Act_WashingMachine)):
    temp1=np.size(Act_WashingMachine[i])
    temp2=np.zeros((temp1+10))
    temp2[5:-5]=Act_WashingMachine[i]
    Act_Appliance.append(temp2)

       
np.save('./Appliance_Activations/Act_RunningMachine', Act_Appliance)
 
sio.savemat('./Appliance_Activations/Act_RunningMachine.mat', {'vect':Act_Appliance}) 


#%% plotting 
 for i in range(8,16):
    plt.figure()
    plt.plot(Act_Appliance[i])    
    







    