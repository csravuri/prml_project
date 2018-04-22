#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 14:26:36 2018

@author: Chandra Sekhar Ravuri
"""

# this for constant prior and update also

import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, f1_score
from prettytable import PrettyTable
import itertools

################  Functions #########################################

def ON_bayesian_constant(device, duration, thresh):
    
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
        
        #likely_hood = likely_hood/likely_hood.sum()
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

def ON_bayesian_update(device, duration, thresh):
    
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
        
        #likely_hood = likely_hood/likely_hood.sum()
        
        dev_test_day = likely_hood * previous_prior
        
        dev_test_day = dev_test_day/dev_test_day.sum()
        
        previous_prior = dev_test_day.copy()
        #previous_prior = previous_prior/previous_prior.sum()
        
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
    plt.title(title,size=15)
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
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


#########################################################################


matrix_data = sio.loadmat('/home/hadoop1/Documents/prml/project/UKDALE/device_status_house2.mat')




#
#duration =5
#thresh= 0.0005
#device = matrix_data['kettle']







thresh = 0.005

# to get bayesian plots and Constant
kettle_time_constant, kettle_plot_constant, kettle_CM_Acc_F1_Prec_constant = ON_bayesian_constant(matrix_data['kettle'], 5, thresh)
rice_cooker_time_constant, rice_cooker_plot_constant, rice_cooker_CM_Acc_F1_Prec_constant = ON_bayesian_constant(matrix_data['rice_cooker'], 5, thresh)
running_machine_time_constant, running_machine_plot_constant, running_machine_CM_Acc_F1_Prec_constant = ON_bayesian_constant(matrix_data['running_machine'], 5, thresh)
washing_machine_time_constant, washing_machine_plot_constant, washing_machine_CM_Acc_F1_Prec_constant = ON_bayesian_constant(matrix_data['washing_machine'], 5, thresh)
dish_washer_time_constant, dish_washer_plot_constant, dish_washer_CM_Acc_F1_Prec_constant = ON_bayesian_constant(matrix_data['dish_washer'], 5, thresh)
microwave_time_constant, microwave_plot_constant, microwave_CM_Acc_F1_Prec_constant = ON_bayesian_constant(matrix_data['microwave'], 5, thresh)


# to get bayesian plots and Updated
kettle_time_update, kettle_plot_update, kettle_CM_Acc_F1_Prec_update = ON_bayesian_update(matrix_data['kettle'], 5, thresh)
rice_cooker_time_update, rice_cooker_plot_update, rice_cooker_CM_Acc_F1_Prec_update = ON_bayesian_update(matrix_data['rice_cooker'], 5, thresh)
running_machine_time_update, running_machine_plot_update, running_machine_CM_Acc_F1_Prec_update = ON_bayesian_update(matrix_data['running_machine'], 5, thresh)
washing_machine_time_update, washing_machine_plot_update, washing_machine_CM_Acc_F1_Prec_update = ON_bayesian_update(matrix_data['washing_machine'], 5, thresh)
dish_washer_time_update, dish_washer_plot_update, dish_washer_CM_Acc_F1_Prec_update = ON_bayesian_update(matrix_data['dish_washer'], 5, thresh)
microwave_time_update, microwave_plot_update, microwave_CM_Acc_F1_Prec_update = ON_bayesian_update(matrix_data['microwave'], 5, thresh)




#%%
#### Results display  ###########################################
print('Threshold = %f'%thresh)


########## Metrics ploting  ################################
rc('xtick',labelsize=10)
rc('ytick',labelsize=10)


print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++')

plt.figure(1)
plt.subplot(221)
plt.plot(kettle_CM_Acc_F1_Prec_constant[1])
plt.plot(kettle_CM_Acc_F1_Prec_update[1])
plt.title('Accuracy on test days for kettle', size=15)
plt.xlabel('Days', size=15)
plt.ylabel('Accuracy', size=15)
plt.legend(['Costant prior','Updating Prior'])
plt.show()


plt.figure(2)
plt.subplot(221)
plt.plot(kettle_CM_Acc_F1_Prec_constant[2])
plt.plot(kettle_CM_Acc_F1_Prec_update[2])
plt.title('F1 Score on test days for kettle', size=15)
plt.xlabel('Days', size=15)
plt.ylabel('F1 Score', size=15)
plt.legend(['Costant prior','Updating Prior'])
plt.show()

plt.figure(3)
plt.subplot(221)
plt.plot(kettle_CM_Acc_F1_Prec_constant[3])
plt.plot(kettle_CM_Acc_F1_Prec_update[3])
plt.title('Precision on test days for kettle', size=15)
plt.xlabel('Days', size=15)
plt.ylabel('Precision', size=15)
plt.legend(['Costant prior','Updating Prior'])
plt.show()


#print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++')
#
#plt.figure()
#plt.plot(rice_cooker_CM_Acc_F1_Prec_constant[1])
#plt.plot(rice_cooker_CM_Acc_F1_Prec_update[1])
#plt.title('Accuracy on test days for rice_cooker', size=15)
#plt.xlabel('Days', size=15)
#plt.ylabel('Accuracy', size=15)
#plt.legend(['Costant prior','Updating Prior'])
#plt.show()
#
#
#plt.figure()
#plt.plot(rice_cooker_CM_Acc_F1_Prec_constant[2])
#plt.plot(rice_cooker_CM_Acc_F1_Prec_update[2])
#plt.title('F1 Score on test days for rice_cooker', size=15)
#plt.xlabel('Days', size=15)
#plt.ylabel('F1 Score', size=15)
#plt.legend(['Costant prior','Updating Prior'])
#plt.show()
#
#plt.figure()
#plt.plot(rice_cooker_CM_Acc_F1_Prec_constant[3])
#plt.plot(rice_cooker_CM_Acc_F1_Prec_update[3])
#plt.title('Precision on test days for rice_cooker', size=15)
#plt.xlabel('Days', size=15)
#plt.ylabel('Precision', size=15)
#plt.legend(['Costant prior','Updating Prior'])
#plt.show()


print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++')

plt.figure(1)
plt.subplot(222)
plt.plot(running_machine_CM_Acc_F1_Prec_constant[1])
plt.plot(running_machine_CM_Acc_F1_Prec_update[1])
plt.title('Accuracy on test days for running_machine', size=15)
plt.xlabel('Days', size=15)
plt.ylabel('Accuracy', size=15)
plt.legend(['Costant prior','Updating Prior'])
plt.show()


plt.figure(2)
plt.subplot(222)
plt.plot(running_machine_CM_Acc_F1_Prec_constant[2])
plt.plot(running_machine_CM_Acc_F1_Prec_update[2])
plt.title('F1 Score on test days for running_machine', size=15)
plt.xlabel('Days', size=15)
plt.ylabel('F1 Score', size=15)
plt.legend(['Costant prior','Updating Prior'])
plt.show()

plt.figure(3)
plt.subplot(222)
plt.plot(running_machine_CM_Acc_F1_Prec_constant[3])
plt.plot(running_machine_CM_Acc_F1_Prec_update[3])
plt.title('Precision on test days for running_machine', size=15)
plt.xlabel('Days', size=15)
plt.ylabel('Precision', size=15)
plt.legend(['Costant prior','Updating Prior'])
plt.show()


print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++')

plt.figure(1)
plt.subplot(223)
plt.plot(washing_machine_CM_Acc_F1_Prec_constant[1])
plt.plot(washing_machine_CM_Acc_F1_Prec_update[1])
plt.title('Accuracy on test days for washing_machine', size=15)
plt.xlabel('Days', size=15)
plt.ylabel('Accuracy', size=15)
plt.legend(['Costant prior','Updating Prior'])
plt.show()


plt.figure(2)
plt.subplot(223)
plt.plot(washing_machine_CM_Acc_F1_Prec_constant[2])
plt.plot(washing_machine_CM_Acc_F1_Prec_update[2])
plt.title('F1 Score on test days for washing_machine', size=15)
plt.xlabel('Days', size=15)
plt.ylabel('F1 Score', size=15)
plt.legend(['Costant prior','Updating Prior'])
plt.show()

plt.figure(3)
plt.subplot(223)
plt.plot(washing_machine_CM_Acc_F1_Prec_constant[3])
plt.plot(washing_machine_CM_Acc_F1_Prec_update[3])
plt.title('Precision on test days for washing_machine', size=15)
plt.xlabel('Days', size=15)
plt.ylabel('Precision', size=15)
plt.legend(['Costant prior','Updating Prior'])
plt.show()


print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++')

plt.figure(1)
plt.subplot(224)
plt.plot(dish_washer_CM_Acc_F1_Prec_constant[1])
plt.plot(dish_washer_CM_Acc_F1_Prec_update[1])
plt.title('Accuracy on test days for dish_washer', size=15)
plt.xlabel('Days', size=15)
plt.ylabel('Accuracy', size=15)
plt.legend(['Costant prior','Updating Prior'])
plt.show()


plt.figure(2)
plt.subplot(224)
plt.plot(dish_washer_CM_Acc_F1_Prec_constant[2])
plt.plot(dish_washer_CM_Acc_F1_Prec_update[2])
plt.title('F1 Score on test days for dish_washer', size=15)
plt.xlabel('Days', size=15)
plt.ylabel('F1 Score', size=15)
plt.legend(['Costant prior','Updating Prior'])
plt.show()

plt.figure(3)
plt.subplot(224)
plt.plot(dish_washer_CM_Acc_F1_Prec_constant[3])
plt.plot(dish_washer_CM_Acc_F1_Prec_update[3])
plt.title('Precision on test days for dish_washer', size=15)
plt.xlabel('Days', size=15)
plt.ylabel('Precision', size=15)
plt.legend(['Costant prior','Updating Prior'])
plt.show()


print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++')

#plt.figure()
#plt.plot(microwave_CM_Acc_F1_Prec_constant[1])
#plt.plot(microwave_CM_Acc_F1_Prec_update[1])
#plt.title('Accuracy on test days for microwave', size=15)
#plt.xlabel('Days', size=15)
#plt.ylabel('Accuracy', size=15)
#plt.legend(['Costant prior','Updating Prior'])
#plt.show()
#
#
#plt.figure()
#plt.plot(microwave_CM_Acc_F1_Prec_constant[2])
#plt.plot(microwave_CM_Acc_F1_Prec_update[2])
#plt.title('F1 Score on test days for microwave', size=15)
#plt.xlabel('Days', size=15)
#plt.ylabel('F1 Score', size=15)
#plt.legend(['Costant prior','Updating Prior'])
#plt.show()
#
#plt.figure()
#plt.plot(microwave_CM_Acc_F1_Prec_constant[3])
#plt.plot(microwave_CM_Acc_F1_Prec_update[3])
#plt.title('Precision on test days for microwave', size=15)
#plt.xlabel('Days', size=15)
#plt.ylabel('Precision', size=15)
#plt.legend(['Costant prior','Updating Prior'])
#plt.show()





#%%

############ Confusion matrices ####################


plt.figure()
plot_confusion_matrix(kettle_CM_Acc_F1_Prec_constant[0],classes=['OFF','ON'], title='kettle confussion matrix with constant prior')
plt.show()

plt.figure()
plot_confusion_matrix(rice_cooker_CM_Acc_F1_Prec_constant[0],classes=['OFF','ON'], title='rice_cooker confussion matrix with constant prior')
plt.show()


plt.figure()
plot_confusion_matrix(running_machine_CM_Acc_F1_Prec_constant[0],classes=['OFF','ON'], title='running_machine confussion matrix with constant prior')
plt.show()


plt.figure()
plot_confusion_matrix(washing_machine_CM_Acc_F1_Prec_constant[0],classes=['OFF','ON'], title='washing_machine confussion matrix with constant prior')
plt.show()



plt.figure()
plot_confusion_matrix(dish_washer_CM_Acc_F1_Prec_constant[0],classes=['OFF','ON'], title='dish_washer confussion matrix with constant prior')
plt.show()


plt.figure()
plot_confusion_matrix(microwave_CM_Acc_F1_Prec_constant[0],classes=['OFF','ON'], title='microwave confussion matrix with constant prior')
plt.show()

####

plt.figure()
plot_confusion_matrix(kettle_CM_Acc_F1_Prec_update[0],classes=['OFF','ON'], title='kettle confussion matrix with update prior')
plt.show()

plt.figure()
plot_confusion_matrix(rice_cooker_CM_Acc_F1_Prec_update[0],classes=['OFF','ON'], title='rice_cooker confussion matrix with update prior')
plt.show()


plt.figure()
plot_confusion_matrix(running_machine_CM_Acc_F1_Prec_update[0],classes=['OFF','ON'], title='running_machine confussion matrix with update prior')
plt.show()


plt.figure()
plot_confusion_matrix(washing_machine_CM_Acc_F1_Prec_update[0],classes=['OFF','ON'], title='washing_machine confussion matrix with update prior')
plt.show()



plt.figure()
plot_confusion_matrix(dish_washer_CM_Acc_F1_Prec_update[0],classes=['OFF','ON'], title='dish_washer confussion matrix with update prior')
plt.show()


plt.figure()
plot_confusion_matrix(microwave_CM_Acc_F1_Prec_update[0],classes=['OFF','ON'], title='microwave confussion matrix with update prior')
plt.show()




############ end Confusion matrix ##############




#%% basian

#plt.close('all')


plt.figure()
plt.bar(kettle_time_constant, kettle_plot_constant)
plt.xlabel('Time in Minutes(Whole day)',size=15)
plt.ylabel('Frequency (Number of days)',size =18)
plt.title('constant prior Bayesian for Kettle',size=15)
#plt.savefig('/home/hadoop1/Documents/prml/project/UKDALE/plots/kettle_time.jpg')
plt.show()


plt.figure()
plt.bar(rice_cooker_time_constant, rice_cooker_plot_constant)
plt.xlabel('Time in Minutes(Whole day)',size=15)
plt.ylabel('Frequency (Number of days)',size =18)
plt.title('constant prior Bayesian for Rice coocker',size=15)
#plt.savefig('/home/hadoop1/Documents/prml/project/UKDALE/plots/rice_coocker_time.jpg')
plt.show()


plt.figure()
plt.bar(running_machine_time_constant, running_machine_plot_constant)
plt.xlabel('Time in Minutes(Whole day)',size=15)
plt.ylabel('Frequency (Number of days)',size =18)
plt.title('constant prior Bayesian for Running machine',size=15)
#plt.savefig('/home/hadoop1/Documents/prml/project/UKDALE/plots/running_machine_time.jpg')
plt.show()


plt.figure()
plt.bar(washing_machine_time_constant, washing_machine_plot_constant)
plt.xlabel('Time in Minutes(Whole day)',size=15)
plt.ylabel('Frequency (Number of days)',size =18)
plt.title('constant prior Bayesian for Washing machine',size=15)
#plt.savefig('/home/hadoop1/Documents/prml/project/UKDALE/plots/washing_machine_time.jpg')
plt.show()


plt.figure()
plt.bar(dish_washer_time_constant, dish_washer_plot_constant)
plt.xlabel('Time in Minutes(Whole day)',size=15)
plt.ylabel('Frequency (Number of days)',size =18)
plt.title('constant prior Bayesian for Dishwasher',size=15)
#plt.savefig('/home/hadoop1/Documents/prml/project/UKDALE/plots/dish_washer_time.jpg')
plt.show()


plt.figure()
plt.bar(microwave_time_constant, microwave_plot_constant)
plt.xlabel('Time in Minutes(Whole day)',size=15)
plt.ylabel('Frequency (Number of days)',size =18)
plt.title('constant prior Bayesian for Microwave',size=15)
#plt.savefig('/home/hadoop1/Documents/prml/project/UKDALE/plots/microwave_time.jpg')
plt.show()




#####

plt.figure()
plt.plot(kettle_time_update, kettle_plot_update)
plt.xlabel('Time in Minutes(Whole day)',size=15)
plt.ylabel('Frequency (Number of days)',size =18)
plt.title('updating prior Bayesian for Kettle',size=15)
#plt.savefig('/home/hadoop1/Documents/prml/project/UKDALE/plots/kettle_time.jpg')
plt.show()


plt.figure()
plt.plot(rice_cooker_time_update, rice_cooker_plot_update)
plt.xlabel('Time in Minutes(Whole day)',size=15)
plt.ylabel('Frequency (Number of days)',size =18)
plt.title('updating prior Bayesian for Rice coocker',size=15)
#plt.savefig('/home/hadoop1/Documents/prml/project/UKDALE/plots/rice_coocker_time.jpg')
plt.show()


plt.figure()
plt.plot(running_machine_time_update, running_machine_plot_update)
plt.xlabel('Time in Minutes(Whole day)',size=15)
plt.ylabel('Frequency (Number of days)',size =18)
plt.title('updating prior Bayesian for Running machine',size=15)
#plt.savefig('/home/hadoop1/Documents/prml/project/UKDALE/plots/running_machine_time.jpg')
plt.show()


plt.figure()
plt.plot(washing_machine_time_update, washing_machine_plot_update)
plt.xlabel('Time in Minutes(Whole day)',size=15)
plt.ylabel('Frequency (Number of days)',size =18)
plt.title('updating prior Bayesian for Washing machine',size=15)
#plt.savefig('/home/hadoop1/Documents/prml/project/UKDALE/plots/washing_machine_time.jpg')
plt.show()


plt.figure()
plt.plot(dish_washer_time_update, dish_washer_plot_update)
plt.xlabel('Time in Minutes(Whole day)',size=15)
plt.ylabel('Frequency (Number of days)',size =18)
plt.title('updating prior Bayesian for Dishwasher',size=15)
#plt.savefig('/home/hadoop1/Documents/prml/project/UKDALE/plots/dish_washer_time.jpg')
plt.show()


plt.figure()
plt.plot(microwave_time_update, microwave_plot_update)
plt.xlabel('Time in Minutes(Whole day)',size=15)
plt.ylabel('Frequency (Number of days)',size =18)
plt.title('updating prior Bayesian for Microwave',size=15)
#plt.savefig('/home/hadoop1/Documents/prml/project/UKDALE/plots/microwave_time.jpg')
plt.show()


