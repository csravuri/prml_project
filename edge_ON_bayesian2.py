#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 02:03:42 2018

@author: Chandra Sekhar Ravuri
"""

#    edge bayesian 
#    device ON OFF state for different edge magnitudes and finds prior from
#    that data. this prior is used to test form aggregate data



import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, f1_score
from prettytable import PrettyTable
import itertools

################  Functions #########################################

def ON_bayesian(device, step_size, thresh):
    
    # device = matrix which is output of device_status_matrix.py aka ON OFF Missing data matrix
    # step_size = time step_size in minutes for state to be decided wether ON OFF
    # thresh = threshold to PMF to decide device is ON or OFF
    
    #################   Edge axis  #####################
    edge_len = np.arange(1, device.shape[-1]+1)*step_size
    
    
    
    mask = device.copy()
    mask[device > -1] = 0
    mask = np.abs(mask)
    
    #### activations were data collected  #######
    active = np.invert(mask.any(axis=1))
    
    device = device[active,:]
    
    #np.random.shuffle(device)
    
    ind = int(len(device)*0.5)
    
    
    prior = np.sum(device[:ind,:], axis=0, dtype='float32')
    
    prior = prior/prior.sum()
    
    likely_hood = np.sum(device[ind:,:], axis=0, dtype='float32')
    
    likely_hood = likely_hood/likely_hood.sum()
    
    
    
    
    return edge_len, prior, likely_hood


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


matrix_data = sio.loadmat('/home/hadoop1/Documents/prml/project/data/UK_DALE/DATA/house_2/matrix/edge_house_2_matrix.mat')

agg_data = sio.loadmat('/home/hadoop1/Documents/prml/project/UKDALE/edge_7days_data/appliance_aggegate_data_7days.mat')

#%%
thresh = 0.0005

# to get bayesian plots and 
kettle_edge, kettle_prior, kettle_likely = ON_bayesian(matrix_data['kettle'], 50, thresh)
running_machine_edge, running_machine_plot, running_machine_CM_Acc_F1_Prec = ON_bayesian(matrix_data['running_machine'], 50, thresh)
washing_machine_edge, washing_machine_plot, washing_machine_CM_Acc_F1_Prec = ON_bayesian(matrix_data['washing_machine'], 50, thresh)
dish_washer_edge, dish_washer_plot, dish_washer_CM_Acc_F1_Prec = ON_bayesian(matrix_data['dish_washer'], 50, thresh)







plt.figure()
plt.plot(kettle_edge, kettle_prior)
plt.xlabel('Edge magnitude',size=18)
plt.ylabel('Frequency (Number of activations)',size =18)
plt.title('final prior Bayesian for Kettle',size=18)
#plt.savefig('/home/hadoop1/Documents/prml/project/UKDALE/bayesian_plots_house2/kettle.jpg')
plt.show()





"""
#### Results display  ###########################################
print('Threshold = %f'%thresh)

print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++')

plot_confusion_matrix(kettle_CM_Acc_F1_Prec[0],classes=['OFF','ON'], title='kettle confussion matrix')
plt.show()

kettle_paras = PrettyTable(['Parameter', 'Value'])
kettle_paras.add_row(['Accuracy', round(kettle_CM_Acc_F1_Prec[1],4)])
kettle_paras.add_row(['F1 Score', round(kettle_CM_Acc_F1_Prec[2],4)])
kettle_paras.add_row(['Precision', round(kettle_CM_Acc_F1_Prec[3],4)])
kettle_paras.align['Parameter'] = 'l'

print(kettle_paras)


print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++')

plot_confusion_matrix(running_machine_CM_Acc_F1_Prec[0],classes=['OFF','ON'], title='running_machine confussion matrix')
plt.show()

running_machine_paras = PrettyTable(['Parameter', 'Value'])
running_machine_paras.add_row(['Accuracy', round(running_machine_CM_Acc_F1_Prec[1],4)])
running_machine_paras.add_row(['F1 Score', round(running_machine_CM_Acc_F1_Prec[2],4)])
running_machine_paras.add_row(['Precision', round(running_machine_CM_Acc_F1_Prec[3],4)])
running_machine_paras.align['Parameter'] = 'l'

print(running_machine_paras)

print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++')

plot_confusion_matrix(washing_machine_CM_Acc_F1_Prec[0],classes=['OFF','ON'], title='washing_machine confussion matrix')
plt.show()

washing_machine_paras = PrettyTable(['Parameter', 'Value'])
washing_machine_paras.add_row(['Accuracy', round(washing_machine_CM_Acc_F1_Prec[1],4)])
washing_machine_paras.add_row(['F1 Score', round(washing_machine_CM_Acc_F1_Prec[2],4)])
washing_machine_paras.add_row(['Precision', round(washing_machine_CM_Acc_F1_Prec[3],4)])
washing_machine_paras.align['Parameter'] = 'l'

print(washing_machine_paras)


print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++')

plot_confusion_matrix(dish_washer_CM_Acc_F1_Prec[0],classes=['OFF','ON'], title='dish_washer confussion matrix')
plt.show()

dish_washer_paras = PrettyTable(['Parameter', 'Value'])
dish_washer_paras.add_row(['Accuracy', round(dish_washer_CM_Acc_F1_Prec[1],4)])
dish_washer_paras.add_row(['F1 Score', round(dish_washer_CM_Acc_F1_Prec[2],4)])
dish_washer_paras.add_row(['Precision', round(dish_washer_CM_Acc_F1_Prec[3],4)])
dish_washer_paras.align['Parameter'] = 'l'

print(dish_washer_paras)


plt.close('all')

rc('xtick',labelsize=18)
rc('ytick',labelsize=18)

plt.figure()
plt.bar(kettle_edge, kettle_plot)
plt.xlabel('Edge magnitude',size=18)
plt.ylabel('Frequency (Number of activations)',size =18)
plt.title('final prior Bayesian for Kettle',size=18)
#plt.savefig('/home/hadoop1/Documents/prml/project/UKDALE/bayesian_plots_house2/kettle.jpg')
plt.show()




plt.figure()
plt.bar(running_machine_edge, running_machine_plot)
plt.xlabel('Edge magnitude',size=18)
plt.ylabel('Frequency (Number of activations)',size =18)
plt.title('final prior Bayesian for Running machine',size=18)
plt.show()


plt.figure()
plt.bar(washing_machine_edge, washing_machine_plot)
plt.xlabel('Edge magnitude',size=18)
plt.ylabel('Frequency (Number of activations)',size =18)
plt.title('final prior Bayesian for Washing machine',size=18)
plt.show()


plt.figure()
plt.bar(dish_washer_edge, dish_washer_plot)
plt.xlabel('Edge magnitude',size=18)
plt.ylabel('Frequency (Number of activations)',size =18)
plt.title('final prior Bayesian for Dishwasher',size=18)
plt.show()





"""