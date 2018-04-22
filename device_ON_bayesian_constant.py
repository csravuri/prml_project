#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 13:12:27 2018

@author: Chandra Sekhar Ravuri
"""
# this for constant prior 

import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, f1_score
from prettytable import PrettyTable
import itertools

################  Functions #########################################

def ON_bayesian(device, duration, thresh):
    
    # device = matrix which is output of device_status_matrix.py aka ON OFF Missing data matrix
    # duration = time duration in minutes for state to be decided wether ON OFF
    # thresh = threshold to PMF to decide device is ON or OFF
    
    #################   Time axis  #####################
    time_min = np.arange(1, device.shape[-1]+1)*duration
    
    
    
    mask = device.copy()
    mask[device > -1] = 0
    mask = np.abs(mask)
    
    #### days were whole day data collected  #######
    active = np.invert(mask.any(axis=1))
    
    device = device[active,:]
    
#    np.random.shuffle(device)
    
    ind = int(len(device)*0.3)
    
    dev_test_org = device[-ind:,:]#.flatten() # test data
    
    previous_prior = np.sum(device[:ind,:], axis=0, dtype='float32')  #1 # constant prior
    previous_prior = previous_prior/previous_prior.sum()
    
    
    dev_test_pred = []
    Acc = []
    F1Score = []
    Preci = []
    for itr in range(ind):
        
        likely_hood = np.sum(device[ind:-ind+itr,:], axis=0, dtype='float32')
        
        dev_test_day = likely_hood * previous_prior
        
        dev_test_day[dev_test_day >= thresh] = 1
        dev_test_day[dev_test_day < thresh] = 0
        
        dev_test_pred.append(dev_test_day)
        
        Acc.append(round(accuracy_score(dev_test_org[itr,:], dev_test_day), 4))
        F1Score.append(round(f1_score(dev_test_org[itr,:], dev_test_day), 4))
        Preci.append(round(precision_score(dev_test_org[itr,:], dev_test_day), 4))
        
        
            
    dev_test_pred = np.array(dev_test_pred, np.float32).flatten()
    
    conf_mat = confusion_matrix(dev_test_org.flatten(), dev_test_pred)
    
    CM_Acc_F1_Prec = [conf_mat, Acc, F1Score, Preci,]
    
    # this will give day wise what time data collected
    """
    observed_data = device.copy()
    observed_data[observed_data > -1] = 1
    observed_data[observed_data < 1] = 0
    plt_obser = np.sum(observed_data,axis=0)
    
    
    # this will give day wise what time device is ON
    on_data = device.copy()
    on_data[on_data < 1] = 0
    plt_on = np.sum(on_data,axis=0)
    """
    
    return time_min, previous_prior, CM_Acc_F1_Prec


# this below function obtained from
# http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title,size=18)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label',size=18)
    plt.xlabel('Predicted label',size=18)


#########################################################################


matrix_data = sio.loadmat('/home/hadoop1/Documents/prml/project/UKDALE/device_status_house2.mat')



#%%
#
#duration =5
#thresh= 0.0005
#device = matrix_data['kettle']

#%%





thresh = 0.0005

# to get bayesian plots and 
kettle_time, kettle_plot, kettle_CM_Acc_F1_Prec = ON_bayesian(matrix_data['kettle'], 5, thresh)
rice_cooker_time, rice_cooker_plot, rice_cooker_CM_Acc_F1_Prec = ON_bayesian(matrix_data['rice_cooker'], 5, thresh)
running_machine_time, running_machine_plot, running_machine_CM_Acc_F1_Prec = ON_bayesian(matrix_data['running_machine'], 5, thresh)
washing_machine_time, washing_machine_plot, washing_machine_CM_Acc_F1_Prec = ON_bayesian(matrix_data['washing_machine'], 5, thresh)
dish_washer_time, dish_washer_plot, dish_washer_CM_Acc_F1_Prec = ON_bayesian(matrix_data['dish_washer'], 5, thresh)
microwave_time, microwave_plot, microwave_CM_Acc_F1_Prec = ON_bayesian(matrix_data['microwave'], 5, thresh)

#### Results display  ###########################################
print('Threshold = %f'%thresh)

print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++')

plot_confusion_matrix(kettle_CM_Acc_F1_Prec[0],classes=['OFF','ON'], title='kettle confussion matrix')
plt.show()


plt.figure()
plt.plot(kettle_CM_Acc_F1_Prec[1])
plt.plot(kettle_CM_Acc_F1_Prec[2])
plt.plot(kettle_CM_Acc_F1_Prec[3])
plt.title('Parameters on test days')
plt.xlabel('Days')
plt.ylabel('Parameter')
plt.legend(['Accuracy','F1 Score','Precision'])
plt.show()


print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++')

plt.figure()
plot_confusion_matrix(rice_cooker_CM_Acc_F1_Prec[0],classes=['OFF','ON'], title='rice_cooker confussion matrix')
plt.show()


print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++')
plt.figure()
plot_confusion_matrix(running_machine_CM_Acc_F1_Prec[0],classes=['OFF','ON'], title='running_machine confussion matrix')
plt.show()


print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++')
plt.figure()
plot_confusion_matrix(washing_machine_CM_Acc_F1_Prec[0],classes=['OFF','ON'], title='washing_machine confussion matrix')
plt.show()



print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++')
plt.figure()
plot_confusion_matrix(dish_washer_CM_Acc_F1_Prec[0],classes=['OFF','ON'], title='dish_washer confussion matrix')
plt.show()


print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++')
plt.figure()
plot_confusion_matrix(microwave_CM_Acc_F1_Prec[0],classes=['OFF','ON'], title='microwave confussion matrix')
plt.show()



plt.close('all')

rc('xtick',labelsize=18)
rc('ytick',labelsize=18)

plt.figure()
plt.bar(kettle_time, kettle_plot)
plt.xlabel('Time in Minutes(Whole day)',size=18)
plt.ylabel('Frequency (Number of days)',size =18)
plt.title('final prior Bayesian for Kettle',size=18)
plt.savefig('/home/hadoop1/Documents/prml/project/UKDALE/plots/kettle_time.jpg')
plt.show()


plt.figure()
plt.bar(rice_cooker_time, rice_cooker_plot)
plt.xlabel('Time in Minutes(Whole day)',size=18)
plt.ylabel('Frequency (Number of days)',size =18)
plt.title('final prior Bayesian for Rice coocker',size=18)
plt.savefig('/home/hadoop1/Documents/prml/project/UKDALE/plots/rice_coocker_time.jpg')
plt.show()


plt.figure()
plt.bar(running_machine_time, running_machine_plot)
plt.xlabel('Time in Minutes(Whole day)',size=18)
plt.ylabel('Frequency (Number of days)',size =18)
plt.title('final prior Bayesian for Running machine',size=18)
plt.savefig('/home/hadoop1/Documents/prml/project/UKDALE/plots/running_machine_time.jpg')
plt.show()


plt.figure()
plt.bar(washing_machine_time, washing_machine_plot)
plt.xlabel('Time in Minutes(Whole day)',size=18)
plt.ylabel('Frequency (Number of days)',size =18)
plt.title('final prior Bayesian for Washing machine',size=18)
plt.savefig('/home/hadoop1/Documents/prml/project/UKDALE/plots/washing_machine_time.jpg')
plt.show()


plt.figure()
plt.bar(dish_washer_time, dish_washer_plot)
plt.xlabel('Time in Minutes(Whole day)',size=18)
plt.ylabel('Frequency (Number of days)',size =18)
plt.title('final prior Bayesian for Dishwasher',size=18)
plt.savefig('/home/hadoop1/Documents/prml/project/UKDALE/plots/dish_washer_time.jpg')
plt.show()


plt.figure()
plt.bar(microwave_time, microwave_plot)
plt.xlabel('Time in Minutes(Whole day)',size=18)
plt.ylabel('Frequency (Number of days)',size =18)
plt.title('final prior Bayesian for Microwave',size=18)
plt.savefig('/home/hadoop1/Documents/prml/project/UKDALE/plots/microwave_time.jpg')
plt.show()
